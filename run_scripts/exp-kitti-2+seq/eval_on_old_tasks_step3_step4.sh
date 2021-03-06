#!/bin/bash
# Fine-tuning
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+seq-B3finetuningold-eval \
    --cfg-path configs/exp-kitti-2+seq/B3finetuningold-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+seq-B4finetuningold-eval \
    --cfg-path configs/exp-kitti-2+seq/B4finetuningold-eval.py \
    --mode test
# Joint-training
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+seq-B3jointtrainingold-eval \
    --cfg-path configs/exp-kitti-2+seq/B3jointtrainingold-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+seq-B4jointtrainingold-eval \
    --cfg-path configs/exp-kitti-2+seq/B4jointtrainingold-eval.py \
    --mode test
## EWC
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+seq-B3ewcold-eval \
    --cfg-path configs/exp-kitti-2+seq/B3ewcold-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+seq-B4ewcold-eval \
    --cfg-path configs/exp-kitti-2+seq/B4ewcold-eval.py \
    --mode test
## MAS
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+seq-B3masold-eval \
    --cfg-path configs/exp-kitti-2+seq/B3masold-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+seq-B4masold-eval \
    --cfg-path configs/exp-kitti-2+seq/B4masold-eval.py \
    --mode test
# KD
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+seq-B3lwfold-eval \
    --cfg-path configs/exp-kitti-2+seq/B3lwfold-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+seq-B4lwfold-eval \
    --cfg-path configs/exp-kitti-2+seq/B4lwfold-eval.py \
    --mode test
## IncDet
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+seq-B3pseudoewcold-eval \
    --cfg-path configs/exp-kitti-2+seq/B3pseudoewcold-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+seq-B4pseudoewcold-eval \
    --cfg-path configs/exp-kitti-2+seq/B4pseudoewcold-eval.py \
    --mode test
# C-KD(EWC)
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+seq-B3kdewcold-eval \
    --cfg-path configs/exp-kitti-2+seq/B3kdewcold-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+seq-B4kdewcold-eval \
    --cfg-path configs/exp-kitti-2+seq/B4kdewcold-eval.py \
    --mode test
# C-KD(MAS)
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+seq-B3kdmasold-eval \
    --cfg-path configs/exp-kitti-2+seq/B3kdmasold-eval.py \
    --mode test

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20210118-expkitti2+seq-B4kdmasold-eval \
    --cfg-path configs/exp-kitti-2+seq/B4kdmasold-eval.py \
    --mode test