import json
import itertools
import argparse
from ase.io import read
from ase import Atoms
from dscribe.descriptors import SOAP


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prepare the dataset by labeling the adsorbates"
    )

    parser.add_argument(
        "dset", type=str, help="Path to the dataset file. Often an XYZ file."
    )
    parser.add_argument(
        "-r",
        "--r_cut",
        type=float,
        default=5.0,
        help="Cutoff radius (in Å)",
    )
    parser.add_argument(
        "-n",
        "--n_max",
        type=int,
        default=8,
        help="Number of radial basis functions",
    )
    parser.add_argument(
        "-l",
        "--l_max",
        type=int,
        default=6,
        help="Maximum degree of spherical harmonics",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.3,
        help="Standard deviation of Gaussians",
    )
    parser.add_argument(
        "-p",
        "--pairs",
        type=str,
        default=None,
        help="List of symbols to be included in the descriptor",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    dset = read(args.dset, index=":")
    species = []
    periodic = False
    for at in dset:
        species += at.get_chemical_symbols()
        periodic |= at.get_pbc().any()

    species = list(set(species))

    soap = SOAP(
        species=species,
        periodic=periodic,
        r_cut=args.r_cut,
        n_max=args.n_max,
        l_max=args.l_max,
        sigma=args.sigma,
    )

    total_feats = soap.get_number_of_features()
    if args.pairs is not None:
        pairs = json.loads(args.pairs)
        locs = [soap.get_location(pair) for pair in pairs]
        lengths = [len(range(*sl.indices(total_feats))) for sl in locs]
        num_features = sum(lengths)

    else:
        num_features = total_feats

    print(f"Number of SOAP features: {num_features}")


if __name__ == "__main__":
    main()
