#!/bin/bash
GPUID=$1
for i in {1..4}
do
for reuse_tag in "reuse" "noreuse"
do
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July24-expcarlap1+-jointtrain-cv${i}-${reuse_tag} \
    --cfg-path configs/exp-carla-p1+/jointtraining_cv${i}_${reuse_tag}.py \
    --mode train

mkdir saved_weights/incdet-saveweights/
cp saved_weights/July24-expcarlap1+-jointtrain-cv${i}-${reuse_tag}/IncDetMain-20000.tckpt \
    saved_weights/incdet-saveweights/IncDetExpCARLAP1+-jointtrain-cv${i}-${reuse_tag}-20000.tckpt

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July24-expcarlap1+-jointtrain-cv${i}-cyc-${reuse_tag} \
    --cfg-path configs/exp-carla-p1+/jointtraining_cv${i}_cyc_${reuse_tag}.py \
    --mode train

done
done