#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200905-tune-ewcsigma-kitti2+3-kdewc-cv0-10 \
    --cfg-path configs/tune-regsigmaprior/kdewc-cv0-10.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200905-tune-ewcsigma-kitti2+3-kdewc-cv1-10 \
    --cfg-path configs/tune-regsigmaprior/kdewc-cv1-10.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200905-tune-ewcsigma-kitti2+3-kdewc-cv2-10 \
    --cfg-path configs/tune-regsigmaprior/kdewc-cv2-10.py \
    --mode train
