#!/bin/bash
# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --tag 20200817-expkitti2+seq-A1to2old-eval \
#     --cfg-path configs/exp-kitti-2+seq/A1to2old-eval.py \
#     --mode test

# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --tag 20200817-expkitti2+seq-B3to5lwfold-eval \
#     --cfg-path configs/exp-kitti-2+seq/B3to5lwfold-eval.py \
#     --mode test

# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --tag 20200817-expkitti2+seq-B3to5jointtrainingold-eval \
#     --cfg-path configs/exp-kitti-2+seq/B3to5jointtrainingold-eval.py \
#     --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200817-expkitti2+seq-B3to5finetuningold-eval \
    --cfg-path configs/exp-kitti-2+seq/B3to5finetuningold-eval.py \
    --mode test

# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --tag 20200817-expkitti2+seq-A1to5old-eval \
#     --cfg-path configs/exp-kitti-2+seq/A1to5old-eval.py \
#     --mode test