#!/bin/bash
GPUID="0"
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July07-expcarla-car \
    --cfg-path configs/exp-carla/train_from_scratch_car.py \
    --mode train
# CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
#     --tag July07-expcarla-car \
#     --cfg-path configs/exp-carla/train_from_scratch_car.py \
#     --mode test
mkdir -p saved_weights/incdet-saveweights
cp saved_weights/July07-expcarla-car/IncDetMain-15000.ckpt \
   saved_weights/incdet-saveweights/IncDetCarExpCARLA-15000.tckpt