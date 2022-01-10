#!/usr/bin/env bash

export NUM_FOLDS=64

mkdir -p bbox
mkdir -p bbox/train
mkdir -p bbox/train/answer
mkdir -p bbox/train/rationale
mkdir -p bbox/val
mkdir -p bbox/val/answer
mkdir -p bbox/val/rationale

# Training
mkdir -p bbox_logs
mkdir -p bbox_logs/answer
mkdir -p bbox_logs/rationale

parallel -j $(nproc --all) --will-cite "python draw_bbox.py -fold {1} -num_folds ${NUM_FOLDS} -split train -mode answer > bbox_logs/answer/trainlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))

parallel -j $(nproc --all) --will-cite "python draw_bbox.py -fold {1} -num_folds ${NUM_FOLDS} -split train -mode rationale > bbox_logs/rationale/trainlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))

parallel -j $(nproc --all) --will-cite "python draw_bbox.py -fold {1} -num_folds ${NUM_FOLDS} -split val -mode answer > bbox_logs/answer/vallog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))

parallel -j $(nproc --all) --will-cite "python draw_bbox.py -fold {1} -num_folds ${NUM_FOLDS} -split val -mode rationale > bbox_logs/rationale/vallog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))