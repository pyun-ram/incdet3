#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200921-ewcweights-kitti2+seq-kdewc4 \
    --cfg configs/exp-kitti-2+seq/kdewc4_ewc.py \
    --mode compute_ewc_weights

echo "DONE.Please upload the FIM to S3."