#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200815-expkitti2+seq-finetuning4 \
    --cfg-path configs/exp-kitti-2+seq/finetuning4.py \
    --mode train
