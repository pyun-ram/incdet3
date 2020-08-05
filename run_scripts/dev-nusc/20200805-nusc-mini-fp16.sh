#!/bin/bash
./run_scripts/clear_gpus.sh
python3 main.py \
    --tag 20200805-nusc-mini-fp16 \
    --cfg-path configs/dev-nusc/train_from_scratch_fp16.py \
    --mode train

# python3 main.py \
#     --tag incdet3-nusc \
#     --cfg-path configs/dev-nusc/train_from_scratch.py \
#     --mode test