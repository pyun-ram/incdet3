#!/bin/bash
GPUID="0"
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July22-expnusc-trainall \
    --cfg-path configs/exp-nusc-trainall/train_from_scratch.py \
    --mode train