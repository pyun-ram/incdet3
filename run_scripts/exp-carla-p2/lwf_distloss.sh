#!/bin/bash
GPUID="0"
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap2-lwf-cv0 \
    --cfg-path configs/exp-carla-p2/lwf_distloss_cv0.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap2-lwf-cv1 \
    --cfg-path configs/exp-carla-p2/lwf_distloss_cv1.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap2-lwf-cv2 \
    --cfg-path configs/exp-carla-p2/lwf_distloss_cv2.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap2-lwf-cv3 \
    --cfg-path configs/exp-carla-p2/lwf_distloss_cv3.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap2-lwf-cv4 \
    --cfg-path configs/exp-carla-p2/lwf_distloss_cv4.py \
    --mode train

for i in {0..4}
do
cp saved_weights/July14-expcarlap2-lwf-cv$i/IncDetMain-20000.ckpt \
   saved_weights/incdet-saveweights/IncDetExpCARLAP2-lwf-cv$i-20000.tckpt
done