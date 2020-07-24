#!/bin/bash
for i in {0..4}
for resue_tag in "reuse" "noreuse"
do
do
python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-p1+/finetuning_cv0_${resue_tag}.py \
    --save-path configs/exp-carla-p1+/finetuning_cv$i_${resue_tag}.py \
    --key-pairs CARLA_infos_train0.pkl=CARLA_infos_train$i.pkl,CARLA_infos_val0.pkl=CARLA_infos_val$i.pkl

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-p1+/lwf_distloss_cv0_${resue_tag}.py \
    --save-path configs/exp-carla-p1+/lwf_distloss_cv$i_${resue_tag}.py \
    --key-pairs CARLA_infos_train0.pkl=CARLA_infos_train$i.pkl,CARLA_infos_val0.pkl=CARLA_infos_val$i.pkl

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-p1+/jointtraining_cv0_${resue_tag}.py \
    --save-path configs/exp-carla-p1+/jointtraining_cv$i_${resue_tag}.py \
    --key-pairs CARLA_infos_train0.pkl=CARLA_infos_train$i.pkl,CARLA_infos_val0.pkl=CARLA_infos_val$i.pkl

done
done

# for i in {0..4}
# do
# python3 tools/create_config_files.py \
#     --template-cfg-path configs/exp-carla-p1+/finetuning_cv0_cyc.py \
#     --save-path configs/exp-carla-p1+/finetuning_cv${i}_cyc.py \
#     --key-pairs CARLA_infos_train0.pkl=CARLA_infos_train$i.pkl,CARLA_infos_val0.pkl=CARLA_infos_val$i.pkl,IncDetExpCARLAP1-finetune-cv0-20000.tckpt=IncDetExpCARLAP1-finetune-cv$i-20000.tckpt

# python3 tools/create_config_files.py \
#     --template-cfg-path configs/exp-carla-p1+/lwf_distloss_cv0_cyc.py \
#     --save-path configs/exp-carla-p1+/lwf_distloss_cv${i}_cyc.py \
#     --key-pairs CARLA_infos_train0.pkl=CARLA_infos_train$i.pkl,CARLA_infos_val0.pkl=CARLA_infos_val$i.pkl,IncDetExpCARLAP1-lwf-cv0-20000.tckpt=IncDetExpCARLAP1-lwf-cv$i-20000.tckpt

# python3 tools/create_config_files.py \
#     --template-cfg-path configs/exp-carla-p1+/jointtraining_cv0_cyc.py \
#     --save-path configs/exp-carla-p1+/jointtraining_cv${i}_cyc.py \
#     --key-pairs CARLA_infos_train0.pkl=CARLA_infos_train$i.pkl,CARLA_infos_val0.pkl=CARLA_infos_val$i.pkl,IncDetExpCARLAP1-jointtrain-cv0-20000.tckpt=IncDetExpCARLAP1-jointtrain-cv$i-20000.tckpt
# done