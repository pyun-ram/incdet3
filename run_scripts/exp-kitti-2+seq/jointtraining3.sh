#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200815-expkitti2+seq-jointtraining3 \
    --cfg-path configs/exp-kitti-2+seq/jointtraining3.py \
    --mode train
