#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200921-expkitti4+1-ewc \
    --cfg-path configs/exp-kitti-4+1/ewc.py \
    --mode train
