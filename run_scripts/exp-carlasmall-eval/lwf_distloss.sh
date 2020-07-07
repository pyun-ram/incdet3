#!/bin/bash
GPUID="0"

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July04-expcarlasmall-lwf-cv1 \
    --cfg-path configs/exp-carlasmall-eval/lwf_distloss_cv1.py \
    --mode test

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July04-expcarlasmall-lwf-cv2 \
    --cfg-path configs/exp-carlasmall-eval/lwf_distloss_cv2.py \
    --mode test

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July04-expcarlasmall-lwf-cv3 \
    --cfg-path configs/exp-carlasmall-eval/lwf_distloss_cv3.py \
    --mode test