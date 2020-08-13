#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200813-expkitti2+3-finetuning \
    --cfg-path configs/exp-kitti-2+3/finetuning.py \
    --mode train
