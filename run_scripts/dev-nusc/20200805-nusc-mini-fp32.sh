#!/bin/bash
./run_scripts/clear_gpus.sh
python3 main.py \
    --tag 20200805-nusc-mini-fp32 \
    --cfg-path configs/dev-nusc/train_from_scratch_fp32.py \
    --mode train

# python3 main.py \
#     --tag incdet3-nusc \
#     --cfg-path configs/dev-nusc/train_from_scratch.py \
#     --mode test