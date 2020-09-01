#!/bin/bash
./run_scripts/clear_gpus.sh
python3 main.py \
    --tag 202009001-dev-ewc \
    --cfg configs/dev-kitti/train_class2.py \
    --mode compute_ewc_weights