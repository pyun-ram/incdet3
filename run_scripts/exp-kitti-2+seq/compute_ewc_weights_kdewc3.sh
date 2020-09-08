#!/bin/bash
python3 main.py \
    --tag 20200907-ewcweights-kitti2+seq-kdewc3 \
    --cfg configs/exp-kitti-2+seq/kdewc3_ewc.py \
    --mode compute_ewc_weights

for reg_prior in 0.1; do
python tools/impose_ewc-regsigmaprior.py \
--cls-term-path saved_weights/20200907-ewcweights-kitti2+seq-kdewc3/ewc_clsterm-26690.pkl \
--reg-term-path saved_weights/20200907-ewcweights-kitti2+seq-kdewc3/ewc_regterm-26690.pkl \
--reg-sigma-prior ${reg_prior} \
--output-path saved_weights/20200907-ewcweights-kitti2+seq-kdewc3/ewc_weights-26690-${reg_prior}.pkl
done

echo "DONE.Please upload the FIM to S3."