#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200921-expkitti2+seq-kdewc3 \
    --cfg-path configs/exp-kitti-2+seq/kdewc3.py \
    --mode train
