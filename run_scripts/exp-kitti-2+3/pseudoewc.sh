#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200923-expkitti2+3-pseudoewc \
    --cfg-path configs/exp-kitti-2+3/pseudoewc.py \
    --mode train