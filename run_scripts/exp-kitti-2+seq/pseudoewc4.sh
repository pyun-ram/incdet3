#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200925-expkitti2+seq-pseudoewc4 \
    --cfg-path configs/exp-kitti-2+seq/pseudoewc4.py \
    --mode train
