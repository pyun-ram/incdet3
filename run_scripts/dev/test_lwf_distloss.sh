#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python3 main.py \
    --tag incdet-lwf-distloss \
    --cfg-path configs/dev/lwf_distloss.py \
    --mode train
