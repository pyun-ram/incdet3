#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-A1to2all-eval \
    --cfg-path configs/exp-kitti-2+3/A1to2all-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-B3to5lwfall-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5lwfall-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-B3to5jointtrainingall-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5jointtrainingall-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-B3to5finetuningall-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5finetuningall-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-A1to5all-eval \
    --cfg-path configs/exp-kitti-2+3/A1to5all-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-B3to5lwfunbiasedall-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5lwfunbiasedall-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-B3to5lwfthresholdall-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5lwfthresholdall-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-B3to5ewcall-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5ewcall-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-B3to5kdewcall-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5kdewcall-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-B3to5pseudoewcall-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5pseudoewcall-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-B3to5masall-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5masall-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-B3to5kdmasall-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5kdmasall-eval.py \
    --mode test