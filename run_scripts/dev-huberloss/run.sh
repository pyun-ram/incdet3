#!/bin/bash
./run_scripts/clear_gpus.sh
# test ewc_measure_distance l2-norm case
# python3 run_scripts/dev-huberloss/test_ewc_measure_distance_l2.py
# test ewc_measure_distance huberloss case
# python3 run_scripts/dev-huberloss/test_ewc_measure_distance_huber.py
# test functionality of integration
python3 main.py \
    --tag 20200924-dev-huberloss \
    --cfg-path configs/dev-huberloss/ewc.py \
    --mode train