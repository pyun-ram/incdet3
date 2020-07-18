#!/bin/bash
GPUID="0"
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap3-jointtraining-cv0 \
    --cfg-path configs/exp-carla-p3/jointtraining_cv0.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap3-jointtraining-cv1 \
    --cfg-path configs/exp-carla-p3/jointtraining_cv1.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap3-jointtraining-cv2 \
    --cfg-path configs/exp-carla-p3/jointtraining_cv2.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap3-jointtraining-cv3 \
    --cfg-path configs/exp-carla-p3/jointtraining_cv3.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap3-jointtraining-cv4 \
    --cfg-path configs/exp-carla-p3/jointtraining_cv4.py \
    --mode train

for i in {0..4}
do
cp saved_weights/July14-expcarlap3-jointtraining-cv$i/IncDetMain-20000.ckpt \
   saved_weights/incdet-saveweights/IncDetExpCARLAP3-jointtrain-cv$i-20000.tckpt
done