#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200913-expkitti2+seq-ewc4 \
    --cfg-path configs/exp-kitti-2+seq/ewc4.py \
    --mode train
