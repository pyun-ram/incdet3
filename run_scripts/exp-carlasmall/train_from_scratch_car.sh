#!/bin/bash
GPUID="0"
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July04-expcarlasmall-car \
    --cfg-path configs/exp-carlasmall/train_from_scratch_car.py \
    --mode train
# CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
#     --tag July04-expcarlasmall-car \
#     --cfg-path configs/exp-carlasmall/train_from_scratch_car.py \
#     --mode test
mkdir -p saved_weights/incdet-saveweights
cp saved_weights/July04-expcarlasmall-car/IncDetMain-5000.ckpt \
   saved_weights/incdet-saveweights/IncDetCarExpCARLASmall-2000.tckpt