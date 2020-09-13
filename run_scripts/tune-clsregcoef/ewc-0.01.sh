#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200912-tune-clsregcoef-kitti2+3-ewc-cv0-0.01 \
    --cfg-path configs/tune-clsregcoef/ewc-cv0-0.01.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200912-tune-clsregcoef-kitti2+3-ewc-cv1-0.01 \
    --cfg-path configs/tune-clsregcoef/ewc-cv1-0.01.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200912-tune-clsregcoef-kitti2+3-ewc-cv2-0.01 \
    --cfg-path configs/tune-clsregcoef/ewc-cv2-0.01.py \
    --mode train