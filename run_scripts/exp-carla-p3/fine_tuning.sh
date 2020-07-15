#!/bin/bash
GPUID="0"
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap3-finetune-cv0 \
    --cfg-path configs/exp-carla-p3/finetuning_cv0.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap3-finetune-cv1 \
    --cfg-path configs/exp-carla-p3/finetuning_cv1.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap3-finetune-cv2 \
    --cfg-path configs/exp-carla-p3/finetuning_cv2.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap3-finetune-cv3 \
    --cfg-path configs/exp-carla-p3/finetuning_cv3.py \
    --mode train

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap3-finetune-cv4 \
    --cfg-path configs/exp-carla-p3/finetuning_cv4.py \
    --mode train

for i in {0..4}
do
cp saved_weights/July14-expcarlap3-finetune-cv$i/IncDetMain-20000.ckpt \
   saved_weights/incdet-saveweights/IncDetExpCARLAP3-finetune-cv$i-20000.tckpt
done