#!/bin/bash
python3 main.py \
    --tag 20200907-ewcweights-train_class4_ewc \
    --cfg configs/exp-kitti-4+1/train_class4_ewc.py \
    --mode compute_ewc_weights

for reg_prior in 0.1; do
python tools/impose_ewc-regsigmaprior.py \
--cls-term-path saved_weights/20200907-ewcweights-train_class4_ewc/ewc_clsterm-23200.pkl \
--reg-term-path saved_weights/20200907-ewcweights-train_class4_ewc/ewc_regterm-23200.pkl \
--reg-sigma-prior ${reg_prior} \
--output-path saved_weights/20200907-ewcweights-train_class4_ewc/ewc_weights-23200-${reg_prior}.pkl
done

echo "DONE.Please upload the FIM to S3."