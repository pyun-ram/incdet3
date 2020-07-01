#!/bin/bash

CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-fine-tuning-delta-0.1-block03bn2 \
    --cfg-path configs/dev-tune-finetune-delta/fine_tuning_delta_0.1_block03bn2.py \
    --mode train

CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-fine-tuning-delta-0.1-block03conv2 \
    --cfg-path configs/dev-tune-finetune-delta/fine_tuning_delta_0.1_block03conv2.py \
    --mode train

CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-fine-tuning-delta-0.1-block04bn2 \
    --cfg-path configs/dev-tune-finetune-delta/fine_tuning_delta_0.1_block04bn2.py \
    --mode train

CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-fine-tuning-delta-0.1-block04conv2 \
    --cfg-path configs/dev-tune-finetune-delta/fine_tuning_delta_0.1_block04conv2.py \
    --mode train

CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-fine-tuning-delta-0.1-deblocks02 \
    --cfg-path configs/dev-tune-finetune-delta/fine_tuning_delta_0.1_deblocks02.py \
    --mode train