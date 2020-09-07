#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200907-expkitti2+seq-kdewc5 \
    --cfg-path configs/exp-kitti-2+seq/kdewc5.py \
    --mode train
