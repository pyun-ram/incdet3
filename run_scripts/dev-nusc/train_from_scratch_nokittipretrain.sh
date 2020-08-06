#!/bin/bash
GPUID=$1
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag 20200805-train_from_scratch_nokittipretrain \
    --cfg configs/dev-nusc/20200805-train_from_scratch_nokittipretrain.py \
    --mode train