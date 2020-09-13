#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200912-tune-clsregcoef-kitti2+3-kdewc-cv0-0.1 \
    --cfg-path configs/tune-clsregcoef/kdewc-cv0-0.1.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200912-tune-clsregcoef-kitti2+3-kdewc-cv1-0.1 \
    --cfg-path configs/tune-clsregcoef/kdewc-cv1-0.1.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200912-tune-clsregcoef-kitti2+3-kdewc-cv2-0.1 \
    --cfg-path configs/tune-clsregcoef/kdewc-cv2-0.1.py \
    --mode train