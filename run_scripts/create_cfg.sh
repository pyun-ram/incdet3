!/bin/bash
# after il (target dataset) (best model)
for i in {0..4}
do
python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/lwf_distloss_cv$i.py \
    --save-path configs/exp-carla-eval/lwf_distloss_cv$i-afteril-targetdataset-best.py

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/jointtraining_cv$i.py \
    --save-path configs/exp-carla-eval/jointtraining_cv$i-afteril-targetdataset-best.py

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/finetuning_cv$i.py \
    --save-path configs/exp-carla-eval/finetuning_cv$i-afteril-targetdataset-best.py
done
# after il (target dataset) (last model)
python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/lwf_distloss_cv0.py \
    --save-path configs/exp-carla-eval/lwf_distloss_cv0-afteril-targetdataset-last.py \
    --key-pairs IncDetMain-20000.tckpt=IncDetMain-20000.tckpt

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/lwf_distloss_cv1.py \
    --save-path configs/exp-carla-eval/lwf_distloss_cv1-afteril-targetdataset-last.py \
    --key-pairs IncDetMain-19500.tckpt=IncDetMain-20000.tckpt

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/lwf_distloss_cv2.py \
    --save-path configs/exp-carla-eval/lwf_distloss_cv2-afteril-targetdataset-last.py \
    --key-pairs IncDetMain-18000.tckpt=IncDetMain-20000.tckpt

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/lwf_distloss_cv3.py \
    --save-path configs/exp-carla-eval/lwf_distloss_cv3-afteril-targetdataset-last.py \
    --key-pairs IncDetMain-19500.tckpt=IncDetMain-20000.tckpt

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/lwf_distloss_cv4.py \
    --save-path configs/exp-carla-eval/lwf_distloss_cv4-afteril-targetdataset-last.py \
    --key-pairs IncDetMain-18000.tckpt=IncDetMain-20000.tckpt

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/jointtraining_cv0.py \
    --save-path configs/exp-carla-eval/jointtraining_cv0-afteril-targetdataset-last.py \
    --key-pairs IncDetMain-20000.tckpt=IncDetMain-20000.tckpt

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/jointtraining_cv1.py \
    --save-path configs/exp-carla-eval/jointtraining_cv1-afteril-targetdataset-last.py \
    --key-pairs IncDetMain-19000.tckpt=IncDetMain-20000.tckpt

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/jointtraining_cv2.py \
    --save-path configs/exp-carla-eval/jointtraining_cv2-afteril-targetdataset-last.py \
    --key-pairs IncDetMain-20000.tckpt=IncDetMain-20000.tckpt

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/jointtraining_cv3.py \
    --save-path configs/exp-carla-eval/jointtraining_cv3-afteril-targetdataset-last.py \
    --key-pairs IncDetMain-20000.tckpt=IncDetMain-20000.tckpt

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/jointtraining_cv4.py \
    --save-path configs/exp-carla-eval/jointtraining_cv4-afteril-targetdataset-last.py \
    --key-pairs IncDetMain-16000.tckpt=IncDetMain-20000.tckpt

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/finetuning_cv0.py \
    --save-path configs/exp-carla-eval/finetuning_cv0-afteril-targetdataset-last.py \
    --key-pairs IncDetMain-18500.tckpt=IncDetMain-20000.tckpt

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/finetuning_cv1.py \
    --save-path configs/exp-carla-eval/finetuning_cv1-afteril-targetdataset-last.py \
    --key-pairs IncDetMain-20000.tckpt=IncDetMain-20000.tckpt

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/finetuning_cv2.py \
    --save-path configs/exp-carla-eval/finetuning_cv2-afteril-targetdataset-last.py \
    --key-pairs IncDetMain-20000.tckpt=IncDetMain-20000.tckpt

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/finetuning_cv3.py \
    --save-path configs/exp-carla-eval/finetuning_cv3-afteril-targetdataset-last.py \
    --key-pairs IncDetMain-19500.tckpt=IncDetMain-20000.tckpt

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/finetuning_cv4.py \
    --save-path configs/exp-carla-eval/finetuning_cv4-afteril-targetdataset-last.py \
    --key-pairs IncDetMain-20000.tckpt=IncDetMain-20000.tckpt

# after il (source dataset) (best model)
for i in {0..4}
do
python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/lwf_distloss_cv$i.py \
    --save-path configs/exp-carla-eval/lwf_distloss_cv$i-afteril-sourcedataset-best.py \
    --key-pairs CARLA-TOWN02CARPED=CARLA-TOWN01CAR

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/jointtraining_cv$i.py \
    --save-path configs/exp-carla-eval/jointtraining_cv$i-afteril-sourcedataset-best.py \
    --key-pairs CARLA-TOWN02CARPED=CARLA-TOWN01CAR

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/finetuning_cv$i.py \
    --save-path configs/exp-carla-eval/finetuning_cv$i-afteril-sourcedataset-best.py \
    --key-pairs CARLA-TOWN02CARPED=CARLA-TOWN01CAR
done

# after il (source dataset) (last model)
python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/lwf_distloss_cv0.py \
    --save-path configs/exp-carla-eval/lwf_distloss_cv0-afteril-sourcedataset-last.py \
    --key-pairs IncDetMain-20000.tckpt=IncDetMain-20000.tckpt,CARLA-TOWN02CARPED=CARLA-TOWN01CAR

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/lwf_distloss_cv1.py \
    --save-path configs/exp-carla-eval/lwf_distloss_cv1-afteril-sourcedataset-last.py \
    --key-pairs IncDetMain-19500.tckpt=IncDetMain-20000.tckpt,CARLA-TOWN02CARPED=CARLA-TOWN01CAR

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/lwf_distloss_cv2.py \
    --save-path configs/exp-carla-eval/lwf_distloss_cv2-afteril-sourcedataset-last.py \
    --key-pairs IncDetMain-18000.tckpt=IncDetMain-20000.tckpt,CARLA-TOWN02CARPED=CARLA-TOWN01CAR

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/lwf_distloss_cv3.py \
    --save-path configs/exp-carla-eval/lwf_distloss_cv3-afteril-sourcedataset-last.py \
    --key-pairs IncDetMain-19500.tckpt=IncDetMain-20000.tckpt,CARLA-TOWN02CARPED=CARLA-TOWN01CAR

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/lwf_distloss_cv4.py \
    --save-path configs/exp-carla-eval/lwf_distloss_cv4-afteril-sourcedataset-last.py \
    --key-pairs IncDetMain-18000.tckpt=IncDetMain-20000.tckpt,CARLA-TOWN02CARPED=CARLA-TOWN01CAR

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/jointtraining_cv0.py \
    --save-path configs/exp-carla-eval/jointtraining_cv0-afteril-sourcedataset-last.py \
    --key-pairs IncDetMain-20000.tckpt=IncDetMain-20000.tckpt,CARLA-TOWN02CARPED=CARLA-TOWN01CAR

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/jointtraining_cv1.py \
    --save-path configs/exp-carla-eval/jointtraining_cv1-afteril-sourcedataset-last.py \
    --key-pairs IncDetMain-19000.tckpt=IncDetMain-20000.tckpt,CARLA-TOWN02CARPED=CARLA-TOWN01CAR

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/jointtraining_cv2.py \
    --save-path configs/exp-carla-eval/jointtraining_cv2-afteril-sourcedataset-last.py \
    --key-pairs IncDetMain-20000.tckpt=IncDetMain-20000.tckpt,CARLA-TOWN02CARPED=CARLA-TOWN01CAR

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/jointtraining_cv3.py \
    --save-path configs/exp-carla-eval/jointtraining_cv3-afteril-sourcedataset-last.py \
    --key-pairs IncDetMain-20000.tckpt=IncDetMain-20000.tckpt,CARLA-TOWN02CARPED=CARLA-TOWN01CAR

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/jointtraining_cv4.py \
    --save-path configs/exp-carla-eval/jointtraining_cv4-afteril-sourcedataset-last.py \
    --key-pairs IncDetMain-16000.tckpt=IncDetMain-20000.tckpt,CARLA-TOWN02CARPED=CARLA-TOWN01CAR

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/finetuning_cv0.py \
    --save-path configs/exp-carla-eval/finetuning_cv0-afteril-sourcedataset-last.py \
    --key-pairs IncDetMain-18500.tckpt=IncDetMain-20000.tckpt,CARLA-TOWN02CARPED=CARLA-TOWN01CAR

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/finetuning_cv1.py \
    --save-path configs/exp-carla-eval/finetuning_cv1-afteril-sourcedataset-last.py \
    --key-pairs IncDetMain-20000.tckpt=IncDetMain-20000.tckpt,CARLA-TOWN02CARPED=CARLA-TOWN01CAR

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/finetuning_cv2.py \
    --save-path configs/exp-carla-eval/finetuning_cv2-afteril-sourcedataset-last.py \
    --key-pairs IncDetMain-20000.tckpt=IncDetMain-20000.tckpt,CARLA-TOWN02CARPED=CARLA-TOWN01CAR

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/finetuning_cv3.py \
    --save-path configs/exp-carla-eval/finetuning_cv3-afteril-sourcedataset-last.py \
    --key-pairs IncDetMain-19500.tckpt=IncDetMain-20000.tckpt,CARLA-TOWN02CARPED=CARLA-TOWN01CAR

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/finetuning_cv4.py \
    --save-path configs/exp-carla-eval/finetuning_cv4-afteril-sourcedataset-last.py \
    --key-pairs IncDetMain-20000.tckpt=IncDetMain-20000.tckpt,CARLA-TOWN02CARPED=CARLA-TOWN01CAR

python3 tools/create_config_files.py \
    --template-cfg-path configs/exp-carla-eval/beforeil-targetdataset.py \
    --save-path configs/exp-carla-eval/beforeil-sourcedataset.py \
    --key-pairs CARLA-TOWN02CARPED=CARLA-TOWN01CAR
