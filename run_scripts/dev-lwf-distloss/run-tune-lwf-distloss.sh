#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-lwf-distloss-32-1-2 \
    --cfg-path configs/dev-tune-lwf-distloss/lwf_distloss-32-1.0-2.0.py \
    --mode train

CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-lwf-distloss-64-1-2 \
    --cfg-path configs/dev-tune-lwf-distloss/lwf_distloss-64-1.0-2.0.py \
    --mode train

CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-lwf-distloss-128-1-2 \
    --cfg-path configs/dev-tune-lwf-distloss/lwf_distloss-128-1.0-2.0.py \
    --mode train

CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-lwf-distloss-32-01-02 \
    --cfg-path configs/dev-tune-lwf-distloss/lwf_distloss-32-0.1-0.2.py \
    --mode train

CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-lwf-distloss-64-01-02 \
    --cfg-path configs/dev-tune-lwf-distloss/lwf_distloss-64-0.1-0.2.py \
    --mode train

CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-lwf-distloss-128-01-02 \
    --cfg-path configs/dev-tune-lwf-distloss/lwf_distloss-128-0.1-0.2.py \
    --mode train
