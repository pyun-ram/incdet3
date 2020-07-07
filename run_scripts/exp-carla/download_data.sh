#!/bin/bash
apt-get update && apt-get install -y unzip
mkdir /usr/app/data
cd /usr/app/data
wget --no-check-certificate \
     --no-proxy 'https://pyun-data.s3.amazonaws.com/IncDet3/Data/CARLA.zip' \
     --output-document=/usr/app/data/CARLA.zip
unzip CARLA.zip

cd /usr/app/incdet3/saved_weights/
wget --no-check-certificate \
     --no-proxy 'https://pyun-data.s3.amazonaws.com/IncDet3/Weights/incdet-saveweights-July07-expcarla-car.zip' \
     --output-document=/usr/app/incdet3/saved_weights/incdet-saveweights.zip
unzip incdet-saveweights.zip