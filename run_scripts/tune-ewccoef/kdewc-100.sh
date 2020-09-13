#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200913-tune-kitti2+3-kdewc-ewccoef100anchor-cv0 \
    --cfg-path configs/tune-ewccoef/kdewc-100anchor-cv0.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200913-tune-kitti2+3-kdewc-ewccoef100anchor-cv1 \
    --cfg-path configs/tune-ewccoef/kdewc-100anchor-cv1.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200913-tune-kitti2+3-kdewc-ewccoef100anchor-cv2 \
    --cfg-path configs/tune-ewccoef/kdewc-100anchor-cv2.py \
    --mode train
