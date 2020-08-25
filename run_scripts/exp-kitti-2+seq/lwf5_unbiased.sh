#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200825-expkitti2+seq-lwf5_unbiased \
    --cfg-path configs/exp-kitti-2+seq/lwf5_unbiased.py \
    --mode train
