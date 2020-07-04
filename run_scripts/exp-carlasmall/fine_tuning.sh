#!/bin/bash
GPUID="0"
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July04-expcarlasmall-finetune-cv1 \
    --cfg-path configs/exp-carlasmall/fine_tuning_cv1.py \
    --mode train
# CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
#     --tag July04-expcarlasmall-finetune-cv1 \
#     --cfg-path configs/exp-carlasmall/fine_tuning_cv1.py \
#     --mode test

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July04-expcarlasmall-finetune-cv2 \
    --cfg-path configs/exp-carlasmall/fine_tuning_cv2.py \
    --mode train
# CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
#     --tag July04-expcarlasmall-finetune-cv2 \
#     --cfg-path configs/exp-carlasmall/fine_tuning_cv2.py \
#     --mode test

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July04-expcarlasmall-finetune-cv3 \
    --cfg-path configs/exp-carlasmall/fine_tuning_cv3.py \
    --mode train
# CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
#     --tag July04-expcarlasmall-finetune-cv3 \
#     --cfg-path configs/exp-carlasmall/fine_tuning_cv3.py \
#     --mode test