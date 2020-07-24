#!/bin/bash
GPUID=$1
for i in {1..4}
do
for reuse_tag in "reuse" "noreuse"
do
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July24-expcarlap1+-lwf-cv${i}-${reuse_tag}_bias_32 \
    --cfg-path configs/exp-carla-p1+/lwf_distloss_cv${i}_${reuse_tag}_bias_32.py \
    --mode train

mkdir saved_weights/incdet-saveweights/
cp saved_weights/July24-expcarlap1+-lwf-cv${i}-${reuse_tag}_bias_32/IncDetMain-20000.tckpt \
    saved_weights/incdet-saveweights/IncDetExpCARLAP1+-lwf-cv${i}-${reuse_tag}-bias-32-20000.tckpt

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July24-expcarlap1+-lwf-cv${i}-cyc-${reuse_tag}_bias_32 \
    --cfg-path configs/exp-carla-p1+/lwf_distloss_cv${i}_cyc_${reuse_tag}_bias_32.py \
    --mode train

done
done