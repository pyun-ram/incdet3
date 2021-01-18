#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20201222-expnusckitti2+3-train_class2-merge \
    --cfg-path configs/dev-nusckitti-2+3/train_class2_merge.py \
    --mode train
