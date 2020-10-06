#!/bin/bash
for reg_coef in 0.3 1 3 10 30; do
python tools/impose_mas-regcoef.py \
--cls-term-path saved_weights/20201005-masweights-biased128/mas_newclsterm-23200.pkl \
--reg-term-path saved_weights/20201005-masweights-biased128/mas_newregterm-23200.pkl \
--reg-coef ${reg_coef} \
--output-path saved_weights/20201006-masweights-tuneregcoef/mas_omega_regcoef${reg_coef}-23200.pkl
done