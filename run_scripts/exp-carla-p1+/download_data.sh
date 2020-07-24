#!/bin/bash
apt-get update && apt-get install -y unzip
mkdir /usr/app/data
cd /usr/app/data
wget --no-check-certificate \
     --no-proxy 'https://pyun-data.s3.amazonaws.com/IncDet3/Data/CARLA_P1.zip' \
     --output-document=/usr/app/data/CARLA_P1.zip
unzip CARLA_P1.zip

cd /usr/app/incdet3/saved_weights/
wget --no-check-certificate \
     --no-proxy 'https://pyun-data.s3.amazonaws.com/IncDet3/Weights/incdet-saveweights-July14-expcarlaP1-car.zip' \
     --output-document=/usr/app/incdet3/saved_weights/incdet-saveweights.zip
unzip incdet-saveweights.zip

# cd /usr/app/incdet3/saved_weights/
# wget --no-check-certificate \
#      --no-proxy 'https://pyun-data.s3.amazonaws.com/IncDet3/Weights/incdet-saveweights-July14-expcarlaP1-carped.zip' \
#      --output-document=/usr/app/incdet3/saved_weights/incdet-saveweights.zip
# unzip incdet-saveweights.zip
