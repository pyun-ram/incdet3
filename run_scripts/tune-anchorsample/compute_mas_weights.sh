#!/bin/bash
for anchorsample in all biased32 biased64 biased128; do
python3 main.py \
    --tag 20201005-masweights-${anchorsample} \
    --cfg configs/tune-anchorsample/train_class2_mas_${anchorsample}.py \
    --mode compute_mas_weights
done