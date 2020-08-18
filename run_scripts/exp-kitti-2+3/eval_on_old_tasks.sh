#!/bin/bash
# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --tag 20200818-expkitti2+3-A1to2old-eval \
#     --cfg-path configs/exp-kitti-2+3/A1to2old-eval.py \
#     --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200818-expkitti2+3-B3to5lwfold-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5lwfold-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200818-expkitti2+3-B3to5jointtrainingold-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5jointtrainingold-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200818-expkitti2+3-B3to5finetuningold-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5finetuningold-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200818-expkitti2+3-A1to5old-eval \
    --cfg-path configs/exp-kitti-2+3/A1to5old-eval.py \
    --mode test