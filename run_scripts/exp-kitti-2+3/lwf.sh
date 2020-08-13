#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200813-expkitti2+3-lwf \
    --cfg-path configs/exp-kitti-2+3/lwf.py \
    --mode train
