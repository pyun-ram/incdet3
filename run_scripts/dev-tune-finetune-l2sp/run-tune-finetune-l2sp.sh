#!/bin/bash

CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-fine-tuning-l2sp-0.2 \
    --cfg-path configs/dev-tune-finetune-l2sp/fine_tuning_l2sp_0.2.py \
    --mode train &

CUDA_VISIBLE_DEVICES="2" python3 main.py \
    --tag incdet-fine-tuning-l2sp-0.1 \
    --cfg-path configs/dev-tune-finetune-l2sp/fine_tuning_l2sp_0.1.py \
    --mode train &

CUDA_VISIBLE_DEVICES="3" python3 main.py \
    --tag incdet-fine-tuning-l2sp-0.05 \
    --cfg-path configs/dev-tune-finetune-l2sp/fine_tuning_l2sp_0.05.py \
    --mode train

CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-fine-tuning-l2sp-0.5 \
    --cfg-path configs/dev-tune-finetune-l2sp/fine_tuning_l2sp_0.5.py \
    --mode train &

CUDA_VISIBLE_DEVICES="2" python3 main.py \
    --tag incdet-fine-tuning-l2sp-0.02 \
    --cfg-path configs/dev-tune-finetune-l2sp/fine_tuning_l2sp_0.02.py \
    --mode train &

CUDA_VISIBLE_DEVICES="3" python3 main.py \
    --tag incdet-fine-tuning-l2sp-0.01 \
    --cfg-path configs/dev-tune-finetune-l2sp/fine_tuning_l2sp_0.01.py \
    --mode train

CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-fine-tuning-l2sp-1.0 \
    --cfg-path configs/dev-tune-finetune-l2sp/fine_tuning_l2sp_1.py \
    --mode train


