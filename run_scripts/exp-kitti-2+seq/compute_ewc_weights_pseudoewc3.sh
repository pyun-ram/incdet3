#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200925-ewcweights-kitti2+seq-pseudoewc3 \
    --cfg configs/exp-kitti-2+seq/pseudoewc3_ewc.py \
    --mode compute_ewc_weights

echo "DONE.Please upload the FIM to S3."