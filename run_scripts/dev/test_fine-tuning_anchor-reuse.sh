#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-fine-tuning-anchorreuse \
    --cfg-path configs/dev/fine_tuning_anchor-reuse.py \
    --mode train
