#!/bin/bash
task=task2
domain=domain3
for i in {0..4}
do
python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carlamore-eval/finetuning_cv0.py \
    --save-path configs/exp-carlamore-eval/finetuning_cv${i}_${task}_${domain}.py \
    --key-pairs July07-expcarla-finetune-cv0=July07-expcarla-finetune-cv${i}

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carlamore-eval/lwf_distloss_cv0.py \
    --save-path configs/exp-carlamore-eval/lwf_distloss_cv${i}_${task}_${domain}.py \
    --key-pairs July07-expcarla-lwf-cv0=July07-expcarla-lwf-cv${i}

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carlamore-eval/jointtraining_cv0.py \
    --save-path configs/exp-carlamore-eval/jointtraining_cv${i}_${task}_${domain}.py \
    --key-pairs July07-expcarla-jointtraining-cv0=July07-expcarla-jointtraining-cv${i}
done