#!/bin/bash
apt-get update && apt-get install -y unzip

cd /usr/app/incdet3/saved_weights/
wget --no-check-certificate \
     --no-proxy 'https://pyun-data-hk.s3.ap-east-1.amazonaws.com/IncDet3/Weights/20200815-expkitti2%2Bseq-saved_weights.zip' \
     --output-document=/usr/app/incdet3/saved_weights/incdet-saveweights.zip
unzip incdet-saveweights.zip

cd /usr/app/incdet3/saved_weights/
wget --no-check-certificate \
     --no-proxy 'https://pyun-data-hk.s3.ap-east-1.amazonaws.com/IncDet3/Weights/20200919-ewcweights-kitti2%2Bseq-compute_terms.zip' \
     --output-document=/usr/app/incdet3/saved_weights/incdet-saveweights.zip
unzip incdet-saveweights.zip

cd /usr/app/incdet3/saved_weights/
wget --no-check-certificate \
     --no-proxy 'https://pyun-data-hk.s3.ap-east-1.amazonaws.com/IncDet3/Weights/20200921-expkitti2%2Bseq-weights.zip' \
     --output-document=/usr/app/incdet3/saved_weights/incdet-saveweights.zip
unzip incdet-saveweights.zip