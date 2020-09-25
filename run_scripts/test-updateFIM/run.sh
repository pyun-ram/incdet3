#!/bin/bash
# test num_new_classes, num_new_anchor_per_loc
python3 main.py \
    --tag test-updateFIM \
    --cfg-path configs/test-updateFIM/ewc3_ewc.py \
    --mode compute_ewc_weights
# test num_old_classes, num_old_anchor_per_loc
# test expand_old_weights