#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20201008-expkitti2+seq-mas3 \
    --cfg-path configs/exp-kitti-2+seq/mas3.py \
    --mode train
