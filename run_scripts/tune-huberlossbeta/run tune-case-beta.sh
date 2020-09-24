#!/bin/bash
case=$1
beta=$2
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200924-tune-kitti2+3-${case}-huberlossbeta${beta}-cv0 \
    --cfg-path configs/tune-huberlossbeta/${case}-${beta}-cv0.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200924-tune-kitti2+3-${case}-huberlossbeta${beta}-cv1 \
    --cfg-path configs/tune-huberlossbeta/${case}-${beta}-cv1.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200924-tune-kitti2+3-${case}-huberlossbeta${beta}-cv2 \
    --cfg-path configs/tune-huberlossbeta/${case}-${beta}-cv2.py \
    --mode train
