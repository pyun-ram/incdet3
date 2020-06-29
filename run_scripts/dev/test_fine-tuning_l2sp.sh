#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-fine-tuning-l2sp \
    --cfg-path configs/dev/fine_tuning_l2sp.py \
    --mode train
