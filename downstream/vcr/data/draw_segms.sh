#!/usr/bin/env bash

export NUM_FOLDS=64

mkdir -p segm
mkdir -p segm/train
mkdir -p segm/train/answer
mkdir -p segm/train/rationale
mkdir -p segm/val
mkdir -p segm/val/answer
mkdir -p segm/val/rationale

# Training
mkdir -p segm_logs
mkdir -p segm_logs/answer
mkdir -p segm_logs/rationale

parallel -j $(nproc --all) --will-cite "python draw_segms.py -fold {1} -num_folds ${NUM_FOLDS} -split train -mode answer > segm_logs/answer/trainlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))

parallel -j $(nproc --all) --will-cite "python draw_segms.py -fold {1} -num_folds ${NUM_FOLDS} -split train -mode rationale > segm_logs/rationale/trainlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))

parallel -j $(nproc --all) --will-cite "python draw_segms.py -fold {1} -num_folds ${NUM_FOLDS} -split val -mode answer > segm_logs/answer/vallog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))

parallel -j $(nproc --all) --will-cite "python draw_segms.py -fold {1} -num_folds ${NUM_FOLDS} -split val -mode rationale > segm_logs/rationale/vallog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))