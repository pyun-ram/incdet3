#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-lwf-distloss-32 \
    --cfg-path configs/dev-tune-lwf-distloss/lwf_distloss-32.py \
    --mode train

CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-lwf-distloss-64 \
    --cfg-path configs/dev-tune-lwf-distloss/lwf_distloss-64.py \
    --mode train

CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-lwf-distloss-128 \
    --cfg-path configs/dev-tune-lwf-distloss/lwf_distloss-128.py \
    --mode train

