#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200812-dev-kitti-train_class7 \
    --cfg-path configs/dev-kitti/train_class7.py \
    --mode train