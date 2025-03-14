# Comprehensive sampling of coverage effects in catalysis by leveraging generalization in neural network models

This repository contains all the data, plots, scripts, and notebooks to reproduce the manuscript:

D. Schwalbe-Koda, N. Govindarajan, and J. Varley. "Comprehensive sampling of coverage effects in catalysis by leveraging generalization in neural network models". Digital Discovery 4, 234-251 (2025). DOI: [10.1039/D4DD00328D](https://doi.org/10.1039/D4DD00328D)

- All the raw data for the computational analysis is found in the `data` folder.
- The Jupyter Notebooks in `nbs` contain all the code required to reproduce the analysis and the plots shown in the manuscript.
- The `latinn` folder contains the Python code implementing the lateral interaction model and the SOAP-based NN described in the manuscript.

## Installing and running

To reproduce the results from the manuscript, first create a new Python environment using your preferred virtual environment (e.g., `venv` or `conda`).
Then, clone this repository and install it with

```bash
git clone git@github.com:dskoda/ML-Coverage.git
cd ML-Coverage
pip install -e .
```

This should install all dependencies (see [pyproject.toml](pyproject.toml)) to reproduce the data in the manuscript.
For full reproducibility, all packages used when producing the results of this work are given in the [environment.txt](environment.txt) file.

To download the raw data that has all the results for this paper (and the required data for analysis), simply run

```bash
chmod +x download.sh
./download.sh
```

in the root of the repository.
While some of the data is already available in the repository, most of the raw data is too large for GitHub.
Thus, part of the raw data that reproduces the paper is hosted on Zenodo for persistent storage (DOI: [10.5281/zenodo.13801296](https://doi.org/10.5281/zenodo.13801296)).

## Data Description

After downloading the raw data folder, the results will exhibit all data from the paper, including:

- Data analysis for reproducing the plots in the paper
- Train/test splits for the MACE model used in the paper
- Scripts to evaluate the models
- Pre-trained models used to perform the analysis in the manuscript

## Code Description

- `nbs/`: Jupyter notebooks to reproduce all the results from the manuscript. The notebooks build on the downloaded data.
- `scripts/`: Scripts for data preprocessing, sampling, and evaluation.
- `latinn/`: Python module with utilities, models, data handling, and training routines for the lateral interaction model and the SOAP-based NN in the manuscript
- `exps/`: Bash scripts to run specific training routines for the lateral interaction model and the SOAP-based NN.

## Citing

This data has been produced for the following paper:

```bibtex
@article{schwalbe2025comprehensive,
  title={Comprehensive sampling of coverage effects in catalysis by leveraging generalization in neural network models},
  author={Schwalbe-Koda, Daniel and Govindarajan, Nitish and Varley, Joel B},
  journal={Digital Discovery},
  volume={4},
  number={1},
  pages={234--251},
  year={2025},
}
```

## License

The code from this repository is distributed under the [BSD-3 Clause License](LICENSE.md).
