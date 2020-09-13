#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200913-tune-kitti2+3-ewc-ewccoef0.01anchor-cv0 \
    --cfg-path configs/tune-ewccoef/ewc-0.01anchor-cv0.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200913-tune-kitti2+3-ewc-ewccoef0.01anchor-cv1 \
    --cfg-path configs/tune-ewccoef/ewc-0.01anchor-cv1.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200913-tune-kitti2+3-ewc-ewccoef0.01anchor-cv2 \
    --cfg-path configs/tune-ewccoef/ewc-0.01anchor-cv2.py \
    --mode train
