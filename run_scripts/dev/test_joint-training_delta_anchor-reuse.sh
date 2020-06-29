#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-joint-training-delta-anchorreuse \
    --cfg-path configs/dev/joint_training_delta_anchor-reuse.py \
    --mode train