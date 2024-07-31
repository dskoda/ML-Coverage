import argparse
from ase.io import read, write
from ase import Atoms

from latinn.data.utils.preprocess import prepare_dset


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prepare the dataset by labeling the adsorbates"
    )

    parser.add_argument(
        "dset", type=str, help="Path to the dataset file. Often an XYZ file."
    )
    parser.add_argument(
        "--atoms", type=str, nargs="+", help="List of symbols for the adsorbates."
    )
    parser.add_argument(
        "-d",
        "--max_dist",
        type=float,
        default=1.5,
        help="Maximum distance to consider bonds between atoms.",
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file")

    return parser.parse_args()


def main():
    args = parse_arguments()

    dset = read(args.dset, index=":")

    results = prepare_dset(dset, args.atoms, args.max_dist)

    if args.output is not None:
        write(args.output, results)


if __name__ == "__main__":
    main()
