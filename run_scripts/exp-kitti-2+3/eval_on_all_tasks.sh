#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200815-expkitti2+3-B3to5lwfall-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5lwfall-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200815-expkitti2+3-B3to5jointtrainingall-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5jointtrainingall-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200815-expkitti2+3-B3to5finetuningall-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5finetuningall-eval.py \
    --mode test

