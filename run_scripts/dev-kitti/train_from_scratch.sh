#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200811-dev-kitti-train_train_scratch \
    --cfg-path configs/dev-kitti/train_from_scratch.py \
    --mode train