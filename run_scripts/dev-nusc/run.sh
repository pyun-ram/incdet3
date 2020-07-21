#!/bin/bash
./run_scripts/clear_gpus.sh
python3 main.py \
    --tag incdet3-nusc \
    --cfg-path configs/dev-nusc/train_from_scratch_car.py \
    --mode train