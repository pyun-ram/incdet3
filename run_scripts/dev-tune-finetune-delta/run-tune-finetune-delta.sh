#!/bin/bash

CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-fine-tuning-delta-1.0-block03bn2 \
    --cfg-path configs/dev-tune-finetune-delta/fine_tuning_delta_1.0_block03bn2.py \
    --mode train

CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-fine-tuning-delta-1.0-block03conv2 \
    --cfg-path configs/dev-tune-finetune-delta/fine_tuning_delta_1.0_block03conv2.py \
    --mode train

CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-fine-tuning-delta-1.0-block04bn2 \
    --cfg-path configs/dev-tune-finetune-delta/fine_tuning_delta_1.0_block04bn2.py \
    --mode train

CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-fine-tuning-delta-1.0-block04conv2 \
    --cfg-path configs/dev-tune-finetune-delta/fine_tuning_delta_1.0_block04conv2.py \
    --mode train

CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-fine-tuning-delta-1.0-deblocks02 \
    --cfg-path configs/dev-tune-finetune-delta/fine_tuning_delta_1.0_deblocks02.py \
    --mode train