#!/bin/bash
GPUID="0"
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap1-jointtraining-cv0-cyc \
    --cfg-path configs/exp-carla-p1/jointtraining_cv0_cyc.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap1-jointtraining-cv1-cyc \
    --cfg-path configs/exp-carla-p1/jointtraining_cv1_cyc.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap1-jointtraining-cv2-cyc \
    --cfg-path configs/exp-carla-p1/jointtraining_cv2_cyc.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap1-jointtraining-cv3-cyc \
    --cfg-path configs/exp-carla-p1/jointtraining_cv3_cyc.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap1-jointtraining-cv4-cyc \
    --cfg-path configs/exp-carla-p1/jointtraining_cv4_cyc.py \
    --mode train