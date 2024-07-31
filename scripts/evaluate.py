import os
import time
import yaml
import argparse
import numpy as np
import pandas as pd
from ase.io import read
from ase import Atoms
from latinn.external.eval import Evaluator


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluates a given dataset using the model"
    )

    parser.add_argument(
        "dset", type=str, help="Path to the dataset file. Often an XYZ file."
    )
    parser.add_argument(
        "model_folder", type=str, help="Path to the file containing all hyperparameters"
    )
    parser.add_argument("-b", "--batch_size", type=int, default=300, help="Output file")
    parser.add_argument(
        "--swa",
        action="store_true",
        default=False,
        help="If set, uses the model with SWA to evaluate the dataset",
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file")

    return parser.parse_args()


def load_evaluator(model_folder, batch_size: int = 200, swa: bool = True):
    # Find where the checkpoint is located
    ckpt_options = os.listdir(os.path.join(model_folder, "checkpoints"))

    if swa:
        ckpt_options = sorted([x for x in ckpt_options if x.startswith("last")])
    else:
        ckpt_options = sorted([x for x in ckpt_options if x.startswith("epoch_")])

    ckpt_path = os.path.join(model_folder, "checkpoints", ckpt_options[-1])

    # find where the hyperparameters file is located
    hparams_path = os.path.join(model_folder, "csv", "version_0", "hparams.yaml")
    with open(hparams_path, "r") as f:
        hparams = yaml.safe_load(f)

    hparams["data"] = {**hparams["data"], "batch_size": batch_size}
    ev = Evaluator(ckpt_path, hparams, swa=swa)
    return ev


def log(msg):
    print(f"EVAL: {msg}")


def main():
    args = parse_arguments()

    log("loading dataset")
    dset = read(args.dset, index=":")
    nads = np.array([at.get_array("adsorbate").max() for at in dset])

    log("loading model")
    ev = load_evaluator(args.model_folder, args.batch_size, args.swa)

    start = time.time()
    log("evaluating dataset")
    pred, true = ev.evaluate(dset)
    end = time.time()
    delta = end - start

    log(f"evaluated {len(dset)} configs in {delta} s")

    pred = pred.cpu().numpy()
    true = true.cpu().numpy()

    log("saving results")
    results = []
    for i, (p, n) in enumerate(zip(pred, nads)):
        at = dset[i]
        results.append({"pred": p, "nads": n, **at.info})

    df = pd.DataFrame(results)

    if args.output is not None:
        df.to_csv(args.output)


if __name__ == "__main__":
    main()
