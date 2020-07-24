#!/bin/bash
GPUID="0"
for reuse_tag in "reuse" "noreuse"
do
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July24-expcarlap1+-jointtrain-cv0-${reuse_tag} \
    --cfg-path configs/exp-carla-p1+/jointtraining_cv0_${reuse_tag}.py \
    --mode train

mkdir saved_weights/incdet-saveweights/
cp saved_weights/July24-expcarlap1+-jointtrain-cv0-${reuse_tag}/IncDetMain-20000.ckpt \
    saved_weights/incdet-saveweights/IncDetExpCARLAP1+-jointtrain-cv0-${reuse_tag}-20000.tckpt

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July24-expcarlap1+-jointtrain-cv0-cyc-${reuse_tag} \
    --cfg-path configs/exp-carla-p1+/jointtraining_cv0_cyc_${reuse_tag}.py \
    --mode train

done