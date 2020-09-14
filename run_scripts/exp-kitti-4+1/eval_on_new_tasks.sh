#!/bin/bash
# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --tag 20200818-expkitti4+1-A1to4new-eval \
#     --cfg-path configs/exp-kitti-4+1/A1to4new-eval.py \
#     --mode test

# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --tag 20200818-expkitti4+1-B5lwfnew-eval \
#     --cfg-path configs/exp-kitti-4+1/B5lwfnew-eval.py \
#     --mode test

# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --tag 20200818-expkitti4+1-B5jointtrainingnew-eval \
#     --cfg-path configs/exp-kitti-4+1/B5jointtrainingnew-eval.py \
#     --mode test

# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --tag 20200818-expkitti4+1-B5finetuningnew-eval \
#     --cfg-path configs/exp-kitti-4+1/B5finetuningnew-eval.py \
#     --mode test

# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --tag 20200818-expkitti4+1-A1to5new-eval \
#     --cfg-path configs/exp-kitti-4+1/A1to5new-eval.py \
#     --mode test

# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --tag 20200825-expkitti4+1-B5lwfunbiasednew-eval \
#     --cfg-path configs/exp-kitti-4+1/B5lwfunbiasednew-eval.py \
#     --mode test

# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --tag 20200825-expkitti4+1-B5lwfthresholdnew-eval \
#     --cfg-path configs/exp-kitti-4+1/B5lwfthresholdnew-eval.py \
#     --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200913-expkitti4+1-B5ewcnew-eval \
    --cfg-path configs/exp-kitti-4+1/B5ewcnew-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200913-expkitti4+1-B5kdewcnew-eval \
    --cfg-path configs/exp-kitti-4+1/B5kdewcnew-eval.py \
    --mode test