import os
import sys
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple
from ase.io import read
from ase import Atoms
from pymatgen.core import Structure, Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from latinn.external.eval import Evaluator
from mkite_catalysis.runners.coverage import CoverageGenerator
from latinn.data.utils.preprocess import prepare_dset


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Performs Monte Carlo sampling for a given facet and molecule"
    )

    parser.add_argument(
        "json", type=str, help="Path to the JSON file that describes the job"
    )
    parser.add_argument("-b", "--batch_size", type=int, default=200, help="Output file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file")

    return parser.parse_args()


def load_evaluator(model_folder, batch_size: int = 200):
    # Find where the checkpoint is located
    ckpt_options = os.listdir(os.path.join(model_folder, "checkpoints"))
    ckpt_options = sorted([x for x in ckpt_options if x.startswith("epoch_")])
    ckpt_path = os.path.join(model_folder, "checkpoints", ckpt_options[-1])

    # find where the hyperparameters file is located
    hparams_path = os.path.join(model_folder, "csv", "version_0", "hparams.yaml")
    with open(hparams_path, "r") as f:
        hparams = yaml.safe_load(f)

    data_hparams = {**hparams["data"], "batch_size": batch_size}
    ev = Evaluator(ckpt_path, data_hparams)
    return ev


def load_covgen(cfg):
    surf = Structure.from_dict(cfg["surface"])
    mol = Molecule.from_dict(cfg["molecule"])

    covgen = CoverageGenerator(surf, mol, **cfg["kwargs"])
    return covgen


def dset_from_indices(
    indices: List[List[int]], covgen, sites: List[Tuple[float, float, float]]
):
    dset = []
    for comb in indices:
        new_locs = [sites[i] for i in comb]
        new_surf = covgen.adsorb_on_sites(new_locs)
        at = AseAtomsAdaptor.get_atoms(new_surf)
        dset.append(at)

    return dset


def log(msg):
    print(f"[SAMPLER]: {msg}")


def main():
    args = parse_arguments()

    if args.output is not None and os.path.exists(args.output):
        log(f"output file {args.output} already exists. Exiting...")
        sys.exit()

    with open(args.json, "r") as f:
        cfg = json.load(f)

    log("loading model")
    ev = load_evaluator(cfg["model_folder"], args.batch_size)

    log("preparing structures")
    covgen = load_covgen(cfg["covgen"])
    sites = covgen.get_sites()
    dists = covgen.get_distances(sites)

    def preprocess(dset: List[Atoms]):
        return prepare_dset(
            dset, adsorbate_atoms=cfg["adsorbate_atoms"], max_dist=cfg["max_dist"]
        )

    log("preparing the sampling process")
    temps = cfg["temperature"]
    n_moves = cfg["num_moves"]
    if isinstance(temps, float):
        temps = np.full((n_moves,), temps)

    prev_indices = covgen.get_large_combinations(
        cfg["num_adsorbates"], cfg["num_configs"], dists
    )
    prev_dset = dset_from_indices(prev_indices, covgen, sites)
    prev_dset = preprocess(prev_dset)
    pred, _ = ev.evaluate(prev_dset)
    prev_energy = pred.cpu().numpy()

    results = [
        {"move": 0, "replica": i, "atoms": at.todict(), "energy": en}
        for i, (at, en) in enumerate(zip(prev_dset, prev_energy))
    ]
    for move, T in enumerate(temps, 1):
        log(f"MC move: {move}")
        swaps = [
            covgen.swap_indices(comb, num_configs=1, dists=dists)[0]
            for comb in prev_indices
        ]
        dset = dset_from_indices(swaps, covgen, sites)
        dset = preprocess(dset)
        energy, _ = ev.evaluate(dset)
        energy = energy.cpu().numpy()

        rand = np.random.uniform(size=(len(energy)))
        accept = np.exp(-(energy - prev_energy) / T) > rand

        prev_dset = [at if accept[i] else prev_dset[i] for i, at in enumerate(dset)]
        prev_energy = [
            en if accept[i] else prev_energy[i] for i, en in enumerate(energy)
        ]
        results += [
            {"replica": i, "move": move, "atoms": at.todict(), "energy": en}
            for i, (at, en) in enumerate(zip(prev_dset, prev_energy))
        ]

    log("saving results")
    df = pd.DataFrame(results)

    if args.output is not None:
        df.to_json(args.output)


if __name__ == "__main__":
    main()
