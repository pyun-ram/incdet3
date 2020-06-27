#!/bin/bash
CUDA_VISIBLE_DEVICES='0' python3 main.py \
    --tag incdet-dev-feature-extraction-reuse \
    --cfg-path configs/dev/feature_extraction_anchor_reuse.py \
    --mode train