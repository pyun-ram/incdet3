#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200921-expkitti4+1-kdewc \
    --cfg-path configs/exp-kitti-4+1/kdewc.py \
    --mode train
