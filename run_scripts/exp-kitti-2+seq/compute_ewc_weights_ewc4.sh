#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --tag 20200913-ewcweights-kitti2+seq-ewc4 \
    --cfg configs/exp-kitti-2+seq/ewc4_ewc.py \
    --mode compute_ewc_weights

for clsregcoef in 1; do
for reg2coef in 1; do
python tools/impose_ewc-reg2coef-clsregcoef.py \
--cls2term-path saved_weights/20200913-ewcweights-kitti2+seq-ewc4/ewc_cls2term-36572.pkl \
--reg2term-path saved_weights/20200913-ewcweights-kitti2+seq-ewc4/ewc_reg2term-36572.pkl \
--clsregterm-path saved_weights/20200913-ewcweights-kitti2+seq-ewc4/ewc_clsregterm-36572.pkl \
--reg2coef ${reg2coef} \
--clsregcoef ${clsregcoef} \
--output-path saved_weights/20200913-ewcweights-kitti2+seq-ewc4/ewc_weights-36572-reg2coef${reg2coef}-clsregcoef${clsregcoef}.pkl
done
done

echo "DONE.Please upload the FIM to S3."