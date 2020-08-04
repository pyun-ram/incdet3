#!/bin/bash
# python3 tools/create_data_nusc.py nuscenes_data_prep \
#     --root_path /usr/app/data/Nuscenes-All \
#     --version "v1.0-trainval" \
#     --max_sweeps=10 \
#     --dataset_name="NuScenesDataset"

python3 tools/create_data_nusc.py nuscenes_data_prep \
    --root_path /usr/app/data/Nuscenes-Mini \
    --version "v1.0-mini" \
    --max_sweeps=10 \
    --dataset_name="NuScenesDataset"