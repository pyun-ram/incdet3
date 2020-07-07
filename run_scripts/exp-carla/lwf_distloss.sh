#!/bin/bash
GPUID="0"
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July07-expcarla-lwf-cv0 \
    --cfg-path configs/exp-carla/lwf_distloss_cv0.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July07-expcarla-lwf-cv1 \
    --cfg-path configs/exp-carla/lwf_distloss_cv1.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July07-expcarla-lwf-cv2 \
    --cfg-path configs/exp-carla/lwf_distloss_cv2.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July07-expcarla-lwf-cv3 \
    --cfg-path configs/exp-carla/lwf_distloss_cv3.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July07-expcarla-lwf-cv4 \
    --cfg-path configs/exp-carla/lwf_distloss_cv4.py \
    --mode train