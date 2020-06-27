#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-joint-training-l2sp-anchorreuse \
    --cfg-path configs/dev/joint_training_l2sp_anchor-reuse.py \
    --mode train