#!/bin/bash
python3 main.py \
    --tag 20201007-masweights-class2 \
    --cfg configs/exp-kitti-2+3/train_class2_mas.py \
    --mode compute_mas_weights
