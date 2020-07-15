#!/bin/bash
GPUID="0"
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap3-lwf-cv0-cyc \
    --cfg-path configs/exp-carla-p3/lwf_distloss_cv0_cyc.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap3-lwf-cv1-cyc \
    --cfg-path configs/exp-carla-p3/lwf_distloss_cv1_cyc.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap3-lwf-cv2-cyc \
    --cfg-path configs/exp-carla-p3/lwf_distloss_cv2_cyc.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap3-lwf-cv3-cyc \
    --cfg-path configs/exp-carla-p3/lwf_distloss_cv3_cyc.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap3-lwf-cv4-cyc \
    --cfg-path configs/exp-carla-p3/lwf_distloss_cv4_cyc.py \
    --mode train
