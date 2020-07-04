#!/bin/bash
GPUID="0"
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July04-expcarlasmall-jointtraining-cv1 \
    --cfg-path configs/exp-carlasmall/joint_training_cv1.py \
    --mode train
# CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
#     --tag July04-expcarlasmall-jointtraining-cv1 \
#     --cfg-path configs/exp-carlasmall/joint_training_cv1.py \
#     --mode test

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July04-expcarlasmall-jointtraining-cv2 \
    --cfg-path configs/exp-carlasmall/joint_training_cv2.py \
    --mode train
# CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
#     --tag July04-expcarlasmall-jointtraining-cv2 \
#     --cfg-path configs/exp-carlasmall/joint_training_cv2.py \
#     --mode test

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July04-expcarlasmall-jointtraining-cv3 \
    --cfg-path configs/exp-carlasmall/joint_training_cv3.py \
    --mode train
# CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
#     --tag July04-expcarlasmall-jointtraining-cv3 \
#     --cfg-path configs/exp-carlasmall/joint_training_cv3.py \
#     --mode test