#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200907-expkitti2+seq-ewc3 \
    --cfg-path configs/exp-kitti-2+seq/ewc3.py \
    --mode train
