#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-fine-tuning \
    --cfg-path configs/dev/fine_tuning.py \
    --mode train
