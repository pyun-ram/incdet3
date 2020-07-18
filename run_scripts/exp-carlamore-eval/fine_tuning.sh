#!/bin/bash
GPUID="0"
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlamore-finetune-cv0 \
    --cfg-path configs/exp-carlamore/finetuning_cv0.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlamore-finetune-cv1 \
    --cfg-path configs/exp-carlamore/finetuning_cv1.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlamore-finetune-cv2 \
    --cfg-path configs/exp-carlamore/finetuning_cv2.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlamore-finetune-cv3 \
    --cfg-path configs/exp-carlamore/finetuning_cv3.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlamore-finetune-cv4 \
    --cfg-path configs/exp-carlamore/finetuning_cv4.py \
    --mode train