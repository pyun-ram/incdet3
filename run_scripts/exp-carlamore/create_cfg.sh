#!/bin/bash
for i in {0..4}
do
python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carlamore/finetuning_cv0.py \
    --save-path configs/exp-carlamore/finetuning_cv$i.py \
    --key-pairs CARLA_infos_train0.pkl=CARLA_infos_train$i.pkl,CARLA_infos_val0.pkl=CARLA_infos_val$i.pkl,IncDetCarExpCARLAMore-finetune-cv0-20000.tckpt=IncDetCarExpCARLAMore-finetune-cv$i-20000.tckpt

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carlamore/lwf_distloss_cv0.py \
    --save-path configs/exp-carlamore/lwf_distloss_cv$i.py \
    --key-pairs CARLA_infos_train0.pkl=CARLA_infos_train$i.pkl,CARLA_infos_val0.pkl=CARLA_infos_val$i.pkl,IncDetCarExpCARLAMore-lwf-cv0-20000.tckpt=IncDetCarExpCARLAMore-lwf-cv$i-20000.tckpt

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carlamore/jointtraining_cv0.py \
    --save-path configs/exp-carlamore/jointtraining_cv$i.py \
    --key-pairs CARLA_infos_train0.pkl=CARLA_infos_train$i.pkl,CARLA_infos_val0.pkl=CARLA_infos_val$i.pkl,IncDetCarExpCARLAMore-jointtrain-cv0-20000.tckpt=IncDetCarExpCARLAMore-jointtrain-cv$i-20000.tckpt

done