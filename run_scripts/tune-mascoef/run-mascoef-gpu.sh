#!/bin/bash
mascoef=$1

for cv in 0 1 2; do
CUDA_VISIBLE_DEVICES=$2 python3 main.py \
    --tag 20201007-tunemascoef-kitti2+3-${mascoef}-kdmas-cv${cv} \
    --cfg-path configs/tune-mascoef/kdmas-${mascoef}-cv${cv}.py \
    --mode train
done

for cv in 0 1 2; do
CUDA_VISIBLE_DEVICES=$2 python3 main.py \
    --tag 20201007-tunemascoef-kitti2+3-${mascoef}-mas-cv${cv} \
    --cfg-path configs/tune-mascoef/mas-${mascoef}-cv${cv}.py \
    --mode train
done