#!/bin/bash
data_dir=/data_shared/Docker/IncDet3/data
num_train=10
num_val=2
num_test=5
epoch_num=1
current_map=Town01
with_ped=true
with_cyc=true

episode=episode_000$epoch_num
source activate carlapy37
cd ./carla/carla-0.8.4/PythonClient/
mkdir $data_dir

# Training Data
for (( i=1; i<=$num_train; i++ )) # 225 per round
do
    params='-l -i -a --epoch '$epoch_num' -m '$current_map
    if $with_ped
    then
        params=$params' --with-ped'
    fi
    if $with_cyc
    then
        params=$params' --with-cyc'
    fi
    ./collect_data_from_carla.py $params
    sudo mv ./_out/setup_1/$episode $data_dir/train_$i
done

# Validation Data
for (( i=1; i<=$num_val; i++ )) # 225 per round
do
    params='-l -i -a --epoch '$epoch_num' -m '$current_map
    if $with_ped
    then
        params=$params' --with-ped'
    fi
    if $with_cyc
    then
        params=$params' --with-cyc'
    fi
    ./collect_data_from_carla.py $params
    sudo mv ./_out/setup_1/$episode $data_dir/val_$i
done

# Test Data
for (( i=1; i<=$num_test; i++ )) # 225 per round
do
    params='-l -i -a --epoch '$epoch_num' -m '$current_map
    if $with_ped
    then
        params=$params' --with-ped'
    fi
    if $with_cyc
    then
        params=$params' --with-cyc'
    fi
    ./collect_data_from_carla.py $params
    sudo mv ./_out/setup_1/$episode $data_dir/test_$i
done

echo "Need to call tools/process_carladata.py to post-process the collected data."
if $with_cyc
then
    echo "remember to add --with-cyc."
fi