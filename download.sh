#!/bin/bash
# Download the entire data for the project
# (hosted on Zenodo). This allows large data
# files to be downloaded without hosting too
# much on the GitHub repo

cd data

# Downloading main data
wget https://zenodo.org/records/13801296/files/cov-public.tar.gz

tar -zxf cov-public.tar.gz

cd ..
