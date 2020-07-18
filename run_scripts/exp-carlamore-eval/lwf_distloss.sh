#!/bin/bash
GPUID="0"
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlamore-lwf-cv0 \
    --cfg-path configs/exp-carlamore/lwf_distloss_cv0.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlamore-lwf-cv1 \
    --cfg-path configs/exp-carlamore/lwf_distloss_cv1.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlamore-lwf-cv2 \
    --cfg-path configs/exp-carlamore/lwf_distloss_cv2.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlamore-lwf-cv3 \
    --cfg-path configs/exp-carlamore/lwf_distloss_cv3.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlamore-lwf-cv4 \
    --cfg-path configs/exp-carlamore/lwf_distloss_cv4.py \
    --mode train