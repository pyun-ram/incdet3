#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200919-expkitti2+seq-ewc5 \
    --cfg-path configs/exp-kitti-2+seq/ewc5.py \
    --mode train
