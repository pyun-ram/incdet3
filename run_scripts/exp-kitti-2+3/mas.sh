#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20201008-expkitti2+3-mas \
    --cfg-path configs/exp-kitti-2+3/mas.py \
    --mode train