#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200907-tune-kitti2+3-kdewc-ewccoef0.1anchor-cv0 \
    --cfg-path configs/tune-ewccoef/kdewc-0.1anchor-cv0.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200907-tune-kitti2+3-kdewc-ewccoef0.1anchor-cv1 \
    --cfg-path configs/tune-ewccoef/kdewc-0.1anchor-cv1.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200907-tune-kitti2+3-kdewc-ewccoef0.1anchor-cv2 \
    --cfg-path configs/tune-ewccoef/kdewc-0.1anchor-cv2.py \
    --mode train
