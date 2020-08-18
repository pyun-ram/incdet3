#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200818-expkitti2+3-jointtraining \
    --cfg-path configs/exp-kitti-2+3/jointtraining.py \
    --mode train
