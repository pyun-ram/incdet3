#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200813-expkitti2+3-train_class2 \
    --cfg-path configs/exp-kitti-2+3/train_class2.py \
    --mode train
