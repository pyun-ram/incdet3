#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-joint-training \
    --cfg-path configs/dev/joint_training_anchor-reuse.py \
    --mode train