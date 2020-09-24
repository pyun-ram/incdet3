#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200925-expkitti2+3-kdewc \
    --cfg-path configs/exp-kitti-2+3/kdewc.py \
    --mode train