#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200905-tune-ewcsigma-kitti2+3-kdewc-cv0-0.1 \
    --cfg-path configs/tune-regsigmaprior/kdewc-cv0-0.1.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200905-tune-ewcsigma-kitti2+3-kdewc-cv1-0.1 \
    --cfg-path configs/tune-regsigmaprior/kdewc-cv1-0.1.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200905-tune-ewcsigma-kitti2+3-kdewc-cv2-0.1 \
    --cfg-path configs/tune-regsigmaprior/kdewc-cv2-0.1.py \
    --mode train
