#!/bin/bash
./run_scripts/clear_gpus.sh
python3 main.py \
    --tag 202009001-dev-ewclwf3 \
    --cfg configs/dev-kitti/lwf3.py \
    --mode train