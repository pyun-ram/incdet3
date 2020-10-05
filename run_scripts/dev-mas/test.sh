#!/bin/bash
./run_scripts/clear_gpus.sh
# PASS test cores
# DONE test params
# DONE test compute_mas_weights sampling strategy all
# DONE test compute_mas_weights sampling strategy unbiased
# DONE test compute_mas_weights sampling strategy biased
# DONE test compute_mas_weights compute_omega_cls_term
# DONE test compute_mas_weights compute_omega_reg_term
# DONE test compute_mas_weights update_mas_term
# DONE test compute_mas_weights newomega
# DONE test compute_mas_weights accumulate old_omegas
python3 main.py \
    --tag 20201003-dev-mas \
    --cfg-path configs/dev-mas/train_class2_mas.py \
    --mode compute_mas_weights