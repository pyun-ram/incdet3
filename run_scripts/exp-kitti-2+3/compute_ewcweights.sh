#!/bin/bash
python3 main.py \
    --tag 20200905-ewcweights-compute_clsterm_regterm \
    --cfg configs/exp-kitti-2+3/train_class2_ewc.py \
    --mode compute_ewc_weights

for reg_prior in 0.1 1 10 100; do
python tools/impose_ewc-regsigmaprior.py \
--cls-term-path saved_weights/20200905-ewcweights-compute_clsterm_regterm/ewc_clsterm-23200.pkl \
--reg-term-path saved_weights/20200905-ewcweights-compute_clsterm_regterm/ewc_regterm-23200.pkl \
--reg-sigma-prior ${reg_prior} \
--output-path saved_weights/20200905-ewcweights-compute_clsterm_regterm/ewc_weights-23200-${reg_prior}.pkl
done