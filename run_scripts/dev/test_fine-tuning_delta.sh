#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-fine-tuning-delta-mask \
    --cfg-path configs/dev/fine_tuning_delta_mask.py \
    --mode train
