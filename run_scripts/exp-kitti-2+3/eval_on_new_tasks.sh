#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-A1to2new-eval \
    --cfg-path configs/exp-kitti-2+3/A1to2new-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-B3to5lwfnew-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5lwfnew-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-B3to5jointtrainingnew-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5jointtrainingnew-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-B3to5finetuningnew-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5finetuningnew-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-A1to5new-eval \
    --cfg-path configs/exp-kitti-2+3/A1to5new-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-B3to5lwfunbiasednew-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5lwfunbiasednew-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-B3to5lwfthresholdnew-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5lwfthresholdnew-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-B3to5ewcnew-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5ewcnew-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-B3to5kdewcnew-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5kdewcnew-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-B3to5pseudoewcnew-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5pseudoewcnew-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-B3to5masnew-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5masnew-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+3-B3to5kdmasnew-eval \
    --cfg-path configs/exp-kitti-2+3/B3to5kdmasnew-eval.py \
    --mode test