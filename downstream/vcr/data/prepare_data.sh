#!/usr/bin/env bash

export NUM_FOLDS=64

VCR_DIRECTORY = ''

# Training
mkdir -p logs
parallel -j $(nproc --all) --will-cite "python prepare_data.py -fold {1} -num_folds ${NUM_FOLDS} -base_fn gs://merlot/data/vcr/train -image_dir ${VCR_DIRECTORY}/vcr1images/ -annotations_file  ${VCR_DIRECTORY}/annotation/train.jsonl > logs/trainlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))

parallel -j $(nproc --all) --will-cite "python prepare_data.py -fold {1} -num_folds ${NUM_FOLDS} -base_fn gs://merlot/data/vcr/val -image_dir ${VCR_DIRECTORY}/vcr1images/ -annotations_file  ${VCR_DIRECTORY}/annotation/val.jsonl  > logs/vallog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))