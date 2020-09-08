#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200907-ewcweights-kitti2+seq-ewc4 \
    --cfg configs/exp-kitti-2+seq/ewc4_ewc.py \
    --mode compute_ewc_weights

for reg_prior in 0.1; do
python tools/impose_ewc-regsigmaprior.py \
--cls-term-path saved_weights/20200907-ewcweights-kitti2+seq-ewc4/ewc_clsterm-33450.pkl \
--reg-term-path saved_weights/20200907-ewcweights-kitti2+seq-ewc4/ewc_regterm-33450.pkl \
--reg-sigma-prior ${reg_prior} \
--output-path saved_weights/20200907-ewcweights-kitti2+seq-ewc4/ewc_weights-33450-${reg_prior}.pkl
done

echo "DONE.Please upload the FIM to S3."