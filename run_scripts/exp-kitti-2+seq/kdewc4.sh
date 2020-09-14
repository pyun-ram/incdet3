#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200913-expkitti2+seq-kdewc4 \
    --cfg-path configs/exp-kitti-2+seq/kdewc4.py \
    --mode train
