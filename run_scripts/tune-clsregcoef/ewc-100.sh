#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200912-tune-clsregcoef-kitti2+3-ewc-cv0-100 \
    --cfg-path configs/tune-clsregcoef/ewc-cv0-100.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200912-tune-clsregcoef-kitti2+3-ewc-cv1-100 \
    --cfg-path configs/tune-clsregcoef/ewc-cv1-100.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200912-tune-clsregcoef-kitti2+3-ewc-cv2-100 \
    --cfg-path configs/tune-clsregcoef/ewc-cv2-100.py \
    --mode train