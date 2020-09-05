#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200905-tune-ewcsigma-kitti2+3-ewc-cv0-100 \
    --cfg-path configs/tune-regsigmaprior/ewc-cv0-100.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200905-tune-ewcsigma-kitti2+3-ewc-cv1-100 \
    --cfg-path configs/tune-regsigmaprior/ewc-cv1-100.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200905-tune-ewcsigma-kitti2+3-ewc-cv2-100 \
    --cfg-path configs/tune-regsigmaprior/ewc-cv2-100.py \
    --mode train
