#!/bin/bash
data_dir=/data_shared/Docker/IncDet3/data
num_train=10
num_val=2
num_test=5
epoch_num=1
current_map=Town01
with_ped=false

episode=episode_000$epoch_num
source activate carlapy37
cd ./carla/carla-0.8.4/PythonClient/
mkdir $data_dir

# Training Data
for (( i=1; i<=$num_train; i++ )) # 225 per round
do
    if $with_ped
    then
        ./collect_data_from_carla.py -l -i -a --epoch $epoch_num -m $current_map --with-ped
    else
        ./collect_data_from_carla.py -l -i -a --epoch $epoch_num -m $current_map
    fi
    sudo mv ./_out/setup_1/$episode $data_dir/train_$i
done

# Validation Data
for (( i=1; i<=$num_val; i++ )) # 225 per round
do
    if $with_ped
    then
        ./collect_data_from_carla.py -l -i -a --epoch $epoch_num -m $current_map --with-ped
    else
        ./collect_data_from_carla.py -l -i -a --epoch $epoch_num -m $current_map
    fi
    sudo mv ./_out/setup_1/$episode $data_dir/val_$i
done

# Test Data
for (( i=1; i<=$num_test; i++ )) # 225 per round
do
    if $with_ped
    then
        ./collect_data_from_carla.py -l -i -a --epoch $epoch_num -m $current_map --with-ped
    else
        ./collect_data_from_carla.py -l -i -a --epoch $epoch_num -m $current_map
    fi
    sudo mv ./_out/setup_1/$episode $data_dir/test_$i
done

echo "Need to call tools/process_carladata.py to post-process the collected data."