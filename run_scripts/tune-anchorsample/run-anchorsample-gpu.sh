#!/bin/bash
anchorsample=$1
for cv in 0 1 2; do
CUDA_VISIBLE_DEVICES=$2 python3 main.py \
    --tag 20201005-tuneanchorsample-kitti2+3-${anchorsample}-mas-cv${cv} \
    --cfg-path configs/tune-anchorsample/mas-${anchorsample}-cv${cv}.py \
    --mode train
done

for cv in 0 1 2; do
CUDA_VISIBLE_DEVICES=$2 python3 main.py \
    --tag 20201005-tuneanchorsample-kitti2+3-${anchorsample}-kdmas-cv${cv} \
    --cfg-path configs/tune-anchorsample/kdmas-${anchorsample}-cv${cv}.py \
    --mode train
done