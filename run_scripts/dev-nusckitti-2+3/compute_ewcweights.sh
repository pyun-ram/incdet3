#!/bin/bash
python3 main.py \
    --tag 20200919-ewcweights-compute_terms \
    --cfg configs/exp-kitti-2+3/train_class2_ewc.py \
    --mode compute_ewc_weights
