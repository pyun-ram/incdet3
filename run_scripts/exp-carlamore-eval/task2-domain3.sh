#!/bin/bash
GPUID="0"
task=task2
domain=domain3

./run_scripts/clear_gpus.sh
for i in {0..4}
do
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlarmore-finetuning_cv${i}_${task}_${domain} \
    --cfg-path configs/exp-carlamore-eval/finetuning_cv${i}_${task}_${domain}.py \
    --mode test

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlarmore-lwf_cv${i}_${task}_${domain} \
    --cfg-path configs/exp-carlamore-eval/lwf_distloss_cv${i}_${task}_${domain}.py \
    --mode test

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July14-expcarlarmore-jointtraining_cv${i}_${task}_${domain} \
    --cfg-path configs/exp-carlamore-eval/jointtraining_cv${i}_${task}_${domain}.py \
    --mode test
done
