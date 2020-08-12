#!/bin/bash
apt-get update && apt-get install -y unzip

cd /usr/app/incdet3/saved_weights/
wget --no-check-certificate \
     --no-proxy 'https://pyun-data.s3.amazonaws.com/IncDet3/Weights/20200812-expkitti4%2B1-saved_weights.zip' \
     --output-document=/usr/app/incdet3/saved_weights/incdet-saveweights.zip
unzip incdet-saveweights.zip
