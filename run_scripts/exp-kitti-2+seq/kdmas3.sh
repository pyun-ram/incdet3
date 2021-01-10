#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20201008-expkitti2+seq-kdmas3 \
    --cfg-path configs/exp-kitti-2+seq/kdmas3.py \
    --mode train
