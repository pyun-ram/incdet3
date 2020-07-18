#!/bin/bash
 python3 tools/compute_mAP.py \
    --log-dir logs/July14-expcarlamore/${1}${2} \
    --val-pkl-path /usr/app/data/CARLA-TOWN01CARPEDCYC/CARLA_infos_val${2}.pkl \
    --valid-range -35.2 -40 -1.5 35.2 40 2.6 \
    --valid-classes Car,Pedestrian,Cyclist