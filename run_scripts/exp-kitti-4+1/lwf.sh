#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200818-expkitti4+1-lwf \
    --cfg-path configs/exp-kitti-4+1/lwf.py \
    --mode train
