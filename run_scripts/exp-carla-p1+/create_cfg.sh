#!/bin/bash
# for i in {0..4}
# do
# for reuse_tag in "reuse" "noreuse"
# do
# python3 tools/create_config_files.py \
#     --template-cfg-path configs/exp-carla-p1+/finetuning_cv0_${reuse_tag}.py \
#     --save-path configs/exp-carla-p1+/finetuning_cv${i}_${reuse_tag}.py \
#     --key-pairs CARLA_infos_train0.pkl=CARLA_infos_train${i}.pkl,CARLA_infos_val0.pkl=CARLA_infos_val${i}.pkl

# python3 tools/create_config_files.py \
#     --template-cfg-path configs/exp-carla-p1+/lwf_distloss_cv0_${reuse_tag}_bias_32.py \
#     --save-path configs/exp-carla-p1+/lwf_distloss_cv${i}_${reuse_tag}_bias_32.py \
#     --key-pairs CARLA_infos_train0.pkl=CARLA_infos_train${i}.pkl,CARLA_infos_val0.pkl=CARLA_infos_val${i}.pkl

# python3 tools/create_config_files.py \
#     --template-cfg-path configs/exp-carla-p1+/jointtraining_cv0_${reuse_tag}.py \
#     --save-path configs/exp-carla-p1+/jointtraining_cv${i}_${reuse_tag}.py \
#     --key-pairs CARLA_infos_train0.pkl=CARLA_infos_train${i}.pkl,CARLA_infos_val0.pkl=CARLA_infos_val${i}.pkl

# done
# done

for i in {0..4}
do
for reuse_tag in "reuse" "noreuse"
do
python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-p1+/finetuning_cv0_cyc_${reuse_tag}.py \
    --save-path configs/exp-carla-p1+/finetuning_cv${i}_cyc_${reuse_tag}.py \
    --key-pairs CARLA_infos_train0.pkl=CARLA_infos_train${i}.pkl,CARLA_infos_val0.pkl=CARLA_infos_val${i}.pkl,IncDetExpCARLAP1+-finetune-cv0-${reuse_tag}-20000.tckpt=IncDetExpCARLAP1+-finetune-cv${i}-${reuse_tag}-20000.tckpt

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-p1+/lwf_distloss_cv0_cyc_${reuse_tag}_bias_32.py \
    --save-path configs/exp-carla-p1+/lwf_distloss_cv${i}_cyc_${reuse_tag}_bias_32.py \
    --key-pairs CARLA_infos_train0.pkl=CARLA_infos_train${i}.pkl,CARLA_infos_val0.pkl=CARLA_infos_val${i}.pkl,IncDetExpCARLAP1+-lwf-cv0-${reuse_tag}-bias-32-20000.tckpt=IncDetExpCARLAP1+-lwf-cv${i}-${reuse_tag}-bias-32-20000.tckpt

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-p1+/jointtraining_cv0_cyc_${reuse_tag}.py \
    --save-path configs/exp-carla-p1+/jointtraining_cv${i}_cyc_${reuse_tag}.py \
    --key-pairs CARLA_infos_train0.pkl=CARLA_infos_train${i}.pkl,CARLA_infos_val0.pkl=CARLA_infos_val${i}.pkl,IncDetExpCARLAP1+-jointtrain-cv0-${reuse_tag}-20000.tckpt=IncDetExpCARLAP1+-jointtrain-cv${i}-${reuse_tag}-20000.tckpt
done
done