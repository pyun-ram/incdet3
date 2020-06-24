#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python3 main.py --tag incdet-dev-train-from-scratch-one \
    --cfg configs/dev/train_from_scratch_oneclass.py \
    --mode train