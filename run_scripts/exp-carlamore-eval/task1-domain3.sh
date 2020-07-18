#!/bin/bash
GPUID="0"
task=task1
domain=domain3

./run_scripts/clear_gpus.sh
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlarmore-${task}_${domain} \
    --cfg-path configs/exp-carlamore-eval/${task}_${domain}.py \
    --mode test
