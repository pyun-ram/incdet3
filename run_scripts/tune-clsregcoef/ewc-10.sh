#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200912-tune-clsregcoef-kitti2+3-ewc-cv0-10 \
    --cfg-path configs/tune-clsregcoef/ewc-cv0-10.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200912-tune-clsregcoef-kitti2+3-ewc-cv1-10 \
    --cfg-path configs/tune-clsregcoef/ewc-cv1-10.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200912-tune-clsregcoef-kitti2+3-ewc-cv2-10 \
    --cfg-path configs/tune-clsregcoef/ewc-cv2-10.py \
    --mode train