#!/bin/bash

ROOT_DIR=$(dirname $PWD)

DATA="711"

for i in {01..06}
do

python3 $ROOT_DIR/latinn/train.py \
    data.data_file="$ROOT_DIR/data/${DATA}-train.xyz" \
    +data.val_file="$ROOT_DIR/data/${DATA}-val.xyz" \
    +data.test_file="$ROOT_DIR/data/${DATA}-test.xyz" \
    data=soap \
    model=soap \
    callbacks=default \
    callbacks.swa.swa_lrs=0.0003 \
    model.net.hidden_size=800 \
    model.optimizer.amsgrad=False \
    model.net.batch_norm=False \
    data.cutoff=5.0 \
    task_name="soap" \
    paths.job_dir=$i \
    tags="['soap']" \

done
