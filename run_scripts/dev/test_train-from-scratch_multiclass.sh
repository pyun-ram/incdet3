#!/bin/bash
CUDA_VISIBLE_DEVICES='0' python3 main.py \
    --tag incdet-dev-train-from-scratch-multi \
    --cfg configs/dev/train_from_scratch_multiclass.py \
    --mode train