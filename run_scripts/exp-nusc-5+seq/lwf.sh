#!/bin/bash
GPUID="0"
for i in {6..10}
do
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July22-expnusc-5+seq-lwf-${i} \
    --cfg-path configs/exp-nusc-5+seq/lwf-${i}.py \
    --mode train
done