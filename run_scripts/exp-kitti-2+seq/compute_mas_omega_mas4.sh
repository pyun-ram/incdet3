#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20201008-masweights-kitti2+seq-mas4 \
    --cfg configs/exp-kitti-2+seq/mas4_mas.py \
    --mode compute_mas_weights

echo "DONE.Please upload the FIM to S3."