#!/bin/bash
GPUID="0"
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlap2-car \
    --cfg-path configs/exp-carla-p2/train_from_scratch_car.py \
    --mode train
mkdir -p saved_weights/incdet-saveweights
cp saved_weights/July14-expcarlap2-car/IncDetMain-15000.ckpt \
   saved_weights/incdet-saveweights/IncDetCarExpCARLAP2-15000.tckpt