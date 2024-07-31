#!/bin/bash

ROOT_DIR=$(dirname $PWD)

DATA="711"

for i in {01..06}
do

python3 $ROOT_DIR/latinn/train.py \
    data.data_file="$ROOT_DIR/data/${DATA}-train.xyz" \
    +data.val_file="$ROOT_DIR/data/${DATA}-val.xyz" \
    +data.test_file="$ROOT_DIR/data/${DATA}-test.xyz" \
    data.batch_size=40 \
    model=interaction \
    callbacks=default \
    callbacks.swa.swa_lrs=0.0003 \
    callbacks.swa.swa_epoch_start=700 \
    model.net.hidden_size=800 \
    model.optimizer.amsgrad=False \
    model.net.batch_norm=False \
    trainer.min_epochs=1000 \
    trainer.max_epochs=1200 \
    task_name="lateral" \
    paths.job_dir=$i \
    tags="['lateral']" \

done
