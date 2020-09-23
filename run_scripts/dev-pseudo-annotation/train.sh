#!/bin/bash
./run_scripts/clear_gpus.sh
rm -rf /tmp/*
# test no pseudo annotation
# test build network
## - not change cfg.NETWORK
## - network._model is well set up
# test build dataloader
## - not change cfg.TRAINDATA
## - dataloader is well set up
# tmp dir is well set up
# test generate detection results
# test ensemble detections and ground-truth label
# test generated pkl
# test final configs
python3 main.py \
    --tag 20200923-dev-pseudo-annotation \
    --cfg-path configs/dev-pseudo-annotation/test_build_network.py \
    --mode train