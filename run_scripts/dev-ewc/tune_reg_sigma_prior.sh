#!/bin/bash
# python3 main.py \
#     --tag 20200905-ewcweights-tune-regsigmaprior \
#     --cfg configs/dev-kitti/train_class2.py \
#     --mode compute_ewc_weights

 python tools/impose_ewc-regsigmaprior.py \
    --cls-term-path saved_weights/20200905-ewcweights-tune-regsigmaprior/ewc_clsterm-23200.pkl \
    --reg-term-path saved_weights/20200905-ewcweights-tune-regsigmaprior/ewc_regterm-23200.pkl \
    --reg-sigma-prior 1 \
    --output-path saved_weights/20200905-ewcweights-tune-regsigmaprior/ewc_weights-23200-post.pkl