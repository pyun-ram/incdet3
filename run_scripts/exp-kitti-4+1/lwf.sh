#!/bin/bash
# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --tag 20200818-expkitti4+1-lwf \
#     --cfg-path configs/exp-kitti-4+1/lwf.py \
#     --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200825-expkitti4+1-lwf_unbiased \
    --cfg-path configs/exp-kitti-4+1/lwf_unbiased.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200825-expkitti4+1-lwf_threshold \
    --cfg-path configs/exp-kitti-4+1/lwf_threshold.py \
    --mode train
