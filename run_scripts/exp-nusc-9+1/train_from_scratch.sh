#!/bin/bash
GPUID="0"
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July22-expnusc-9+1-train_from_scratch \
    --cfg-path configs/exp-nusc-9+1/train_from_scratch.py \
    --mode train