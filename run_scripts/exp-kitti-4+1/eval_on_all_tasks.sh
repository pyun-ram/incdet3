#!/bin/bash
# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --tag 20200818-expkitti4+1-A1to4all-eval \
#     --cfg-path configs/exp-kitti-4+1/A1to4all-eval.py \
#     --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200818-expkitti4+1-B5lwfall-eval \
    --cfg-path configs/exp-kitti-4+1/B5lwfall-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200818-expkitti4+1-B5jointtrainingall-eval \
    --cfg-path configs/exp-kitti-4+1/B5jointtrainingall-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200818-expkitti4+1-B5finetuningall-eval \
    --cfg-path configs/exp-kitti-4+1/B5finetuningall-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200818-expkitti4+1-A1to5all-eval \
    --cfg-path configs/exp-kitti-4+1/A1to5all-eval.py \
    --mode test
