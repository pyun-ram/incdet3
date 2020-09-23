#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200923-expkitti4+1-pseudoewc \
    --cfg-path configs/exp-kitti-4+1/pseudoewc.py \
    --mode train
