#!/bin/bash
regcoef=$1

for cv in 0 1 2; do
CUDA_VISIBLE_DEVICES=$2 python3 main.py \
    --tag 20201005-tuneregcoef-kitti2+3-${regcoef}-kdmas-cv${cv} \
    --cfg-path configs/tune-regcoef/kdmas-${regcoef}-cv${cv}.py \
    --mode train
done

for cv in 0 1 2; do
CUDA_VISIBLE_DEVICES=$2 python3 main.py \
    --tag 20201005-tuneregcoef-kitti2+3-${regcoef}-mas-cv${cv} \
    --cfg-path configs/tune-regcoef/mas-${regcoef}-cv${cv}.py \
    --mode train
done