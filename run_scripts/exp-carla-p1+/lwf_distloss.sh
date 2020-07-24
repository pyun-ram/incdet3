#!/bin/bash
GPUID="0"
for reuse_tag in "reuse" "noreuse"
do
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July24-expcarlap1+-lwf-cv0-${reuse_tag}_bias_32 \
    --cfg-path configs/exp-carla-p1+/lwf_distloss_cv0_${reuse_tag}_bias_32.py \
    --mode train

mkdir saved_weights/incdet-saveweights/
cp saved_weights/July24-expcarlap1+-lwf-cv0-${reuse_tag}_bias_32/IncDetMain-20000.tckpt \
    saved_weights/incdet-saveweights/IncDetExpCARLAP1+-lwf-cv0-${reuse_tag}-bias-32-20000.tckpt

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July24-expcarlap1+-lwf-cv0-cyc-${reuse_tag}_bias_32 \
    --cfg-path configs/exp-carla-p1+/lwf_distloss_cv0_cyc_${reuse_tag}_bias_32.py \
    --mode train

done