#!/bin/bash
python3 main.py \
    --tag 20200907-ewcweights-kitti2+seq-ewc3 \
    --cfg configs/exp-kitti-2+seq/ewc3_ewc.py \
    --mode compute_ewc_weights

for reg_prior in 0.1; do
python tools/impose_ewc-regsigmaprior.py \
--cls-term-path saved_weights/20200907-ewcweights-kitti2+seq-ewc3/ewc_clsterm-TBDstepsTBD.pkl \
--reg-term-path saved_weights/20200907-ewcweights-kitti2+seq-ewc3/ewc_regterm-TBDstepsTBD.pkl \
--reg-sigma-prior ${reg_prior} \
--output-path saved_weights/20200907-ewcweights-kitti2+seq-ewc3/ewc_weights-TBDstepsTBD-${reg_prior}.pkl
done

python3 main.py \
    --tag 20200907-ewcweights-kitti2+seq-kdewc3 \
    --cfg configs/exp-kitti-2+seq/kdewc3_ewc.py \
    --mode compute_ewc_weights

for reg_prior in 0.1; do
python tools/impose_ewc-regsigmaprior.py \
--cls-term-path saved_weights/20200907-ewcweights-kitti2+seq-kdewc3/ewc_clsterm-TBDstepsTBD.pkl \
--reg-term-path saved_weights/20200907-ewcweights-kitti2+seq-kdewc3/ewc_regterm-TBDstepsTBD.pkl \
--reg-sigma-prior ${reg_prior} \
--output-path saved_weights/20200907-ewcweights-kitti2+seq-kdewc3/ewc_weights-TBDstepsTBD-${reg_prior}.pkl
done

#############################EWC4&KDEWC4#############################3
python3 main.py \
    --tag 20200907-ewcweights-kitti2+seq-ewc4 \
    --cfg configs/exp-kitti-2+seq/ewc4_ewc.py \
    --mode compute_ewc_weights

for reg_prior in 0.1; do
python tools/impose_ewc-regsigmaprior.py \
--cls-term-path saved_weights/20200907-ewcweights-kitti2+seq-ewc4/ewc_clsterm-TBDstepsTBD.pkl \
--reg-term-path saved_weights/20200907-ewcweights-kitti2+seq-ewc4/ewc_regterm-TBDstepsTBD.pkl \
--reg-sigma-prior ${reg_prior} \
--output-path saved_weights/20200907-ewcweights-kitti2+seq-ewc4/ewc_weights-TBDstepsTBD-${reg_prior}.pkl
done

python3 main.py \
    --tag 20200907-ewcweights-kitti2+seq-kdewc4 \
    --cfg configs/exp-kitti-2+seq/kdewc4_ewc.py \
    --mode compute_ewc_weights

for reg_prior in 0.1; do
python tools/impose_ewc-regsigmaprior.py \
--cls-term-path saved_weights/20200907-ewcweights-kitti2+seq-kdewc4/ewc_clsterm-TBDstepsTBD.pkl \
--reg-term-path saved_weights/20200907-ewcweights-kitti2+seq-kdewc4/ewc_regterm-TBDstepsTBD.pkl \
--reg-sigma-prior ${reg_prior} \
--output-path saved_weights/20200907-ewcweights-kitti2+seq-kdewc4/ewc_weights-TBDstepsTBD-${reg_prior}.pkl
done
echo "DONE.Please upload the FIM to S3."