#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200912-tune-reg2coef-kitti2+3-ewc-cv0-10 \
    --cfg-path configs/tune-reg2coef/ewc-cv0-10.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200912-tune-reg2coef-kitti2+3-ewc-cv1-10 \
    --cfg-path configs/tune-reg2coef/ewc-cv1-10.py \
    --mode train

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200912-tune-reg2coef-kitti2+3-ewc-cv2-10 \
    --cfg-path configs/tune-reg2coef/ewc-cv2-10.py \
    --mode train