#!/bin/bash
# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --tag 20200818-expkitti4+1-A1to4old-eval \
#     --cfg-path configs/exp-kitti-4+1/A1to4old-eval.py \
#     --mode test

# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --tag 20200818-expkitti4+1-B5lwfold-eval \
#     --cfg-path configs/exp-kitti-4+1/B5lwfold-eval.py \
#     --mode test

# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --tag 20200818-expkitti4+1-B5jointtrainingold-eval \
#     --cfg-path configs/exp-kitti-4+1/B5jointtrainingold-eval.py \
#     --mode test

# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --tag 20200818-expkitti4+1-B5finetuningold-eval \
#     --cfg-path configs/exp-kitti-4+1/B5finetuningold-eval.py \
#     --mode test

# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --tag 20200818-expkitti4+1-A1to5old-eval \
#     --cfg-path configs/exp-kitti-4+1/A1to5old-eval.py \
#     --mode test


# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --tag 20200825-expkitti4+1-B5lwfunbiasedold-eval \
#     --cfg-path configs/exp-kitti-4+1/B5lwfunbiasedold-eval.py \
#     --mode test

# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --tag 20200825-expkitti4+1-B5lwfthresholdold-eval \
#     --cfg-path configs/exp-kitti-4+1/B5lwfthresholdold-eval.py \
#     --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200913-expkitti4+1-B5ewcold-eval \
    --cfg-path configs/exp-kitti-4+1/B5ewcold-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200913-expkitti4+1-B5kdewcold-eval \
    --cfg-path configs/exp-kitti-4+1/B5kdewcold-eval.py \
    --mode test