#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20201008-expkitti2+seq-kdmas5 \
    --cfg-path configs/exp-kitti-2+seq/kdmas5.py \
    --mode train
