#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20201008-expkitti4+1-mas \
    --cfg-path configs/exp-kitti-4+1/mas.py \
    --mode train
