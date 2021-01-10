#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20201008-expkitti2+seq-mas4 \
    --cfg-path configs/exp-kitti-2+seq/mas4.py \
    --mode train
