#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200818-expkitti4+1-finetuning \
    --cfg-path configs/exp-kitti-4+1/finetuning.py \
    --mode train
