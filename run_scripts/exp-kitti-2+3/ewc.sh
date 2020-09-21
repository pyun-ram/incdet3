#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200921-expkitti2+3-ewc \
    --cfg-path configs/exp-kitti-2+3/ewc.py \
    --mode train