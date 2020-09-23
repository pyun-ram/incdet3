#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200923-expkitti2+seq-pseudoewc3 \
    --cfg-path configs/exp-kitti-2+seq/pseudoewc3.py \
    --mode train
