#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-joint-training-distloss-anchorreuse \
    --cfg-path configs/dev/joint_training_distloss_anchor-reuse.py \
    --mode train