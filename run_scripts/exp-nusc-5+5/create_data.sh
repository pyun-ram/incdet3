#!/bin/bash
# python3 tools/create_data_nusc.py nuscenes_data_prep \
#     --root_path /usr/app/data/Nuscenes-Mini/training \
#     --version "v1.0-trainval" \
#     --max_sweeps=1 \
#     --dataset_name="NuScenesDataset"

python3 tools/create_data_nusc.py nuscenes_data_prep \
    --root_path /usr/app/data/Nuscenes-Part1-2/training \
    --version "incdet3-trainval" \
    --max_sweeps=1 \
    --dataset_name="NuScenesDataset"

python3 tools/create_data_nusc.py nuscenes_data_prep \
    --root_path /usr/app/data/Nuscenes-Part3/testing \
    --version "incdet3eval-val" \
    --max_sweeps=1 \
    --dataset_name="NuScenesDataset"