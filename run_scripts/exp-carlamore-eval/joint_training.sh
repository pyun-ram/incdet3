#!/bin/bash
GPUID="0"
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlamore-jointtraining-cv0 \
    --cfg-path configs/exp-carlamore/jointtraining_cv0.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlamore-jointtraining-cv1 \
    --cfg-path configs/exp-carlamore/jointtraining_cv1.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlamore-jointtraining-cv2 \
    --cfg-path configs/exp-carlamore/jointtraining_cv2.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlamore-jointtraining-cv3 \
    --cfg-path configs/exp-carlamore/jointtraining_cv3.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlamore-jointtraining-cv4 \
    --cfg-path configs/exp-carlamore/jointtraining_cv4.py \
    --mode train