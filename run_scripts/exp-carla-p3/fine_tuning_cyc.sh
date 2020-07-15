#!/bin/bash
GPUID="0"
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap3-finetune-cv0-cyc \
    --cfg-path configs/exp-carla-p3/finetuning_cv0_cyc.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap3-finetune-cv1-cyc \
    --cfg-path configs/exp-carla-p3/finetuning_cv1_cyc.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap3-finetune-cv2-cyc \
    --cfg-path configs/exp-carla-p3/finetuning_cv2_cyc.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap3-finetune-cv3-cyc \
    --cfg-path configs/exp-carla-p3/finetuning_cv3_cyc.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap3-finetune-cv4-cyc \
    --cfg-path configs/exp-carla-p3/finetuning_cv4_cyc.py \
    --mode train