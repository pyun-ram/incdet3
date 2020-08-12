#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200812-expkitti4+1-lwf \
    --cfg-path configs/exp-kitti-4+1/lwf.py \
    --mode train
