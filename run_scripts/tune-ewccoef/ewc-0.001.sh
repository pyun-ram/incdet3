#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200919-tune-kitti2+3-ewc-ewccoef0.001anchor-cv0 \
    --cfg-path configs/tune-ewccoef/ewc-0.001anchor-cv0.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200919-tune-kitti2+3-ewc-ewccoef0.001anchor-cv1 \
    --cfg-path configs/tune-ewccoef/ewc-0.001anchor-cv1.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200919-tune-kitti2+3-ewc-ewccoef0.001anchor-cv2 \
    --cfg-path configs/tune-ewccoef/ewc-0.001anchor-cv2.py \
    --mode train
