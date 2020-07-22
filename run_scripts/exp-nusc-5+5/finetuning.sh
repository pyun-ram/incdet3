#!/bin/bash
GPUID="0"
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July22-expnusc-5+5-finetuning \
    --cfg-path configs/exp-nusc-5+5/finetuning.py \
    --mode train