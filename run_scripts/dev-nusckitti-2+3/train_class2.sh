#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20201219-expnusckitti2+3-train_class2-pretrained_filtergt \
    --cfg-path configs/dev-nusckitti-2+3/train_class2.py \
    --mode train
