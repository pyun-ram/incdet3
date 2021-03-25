#!/bin/bash
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

conda install -c conda-forge fire
# pip install nuscenes-devkit
apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libfontconfig1
cd /usr/app
# git clone https://git.ram-lab.com/yun/nuscenes-devkit-pyun
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

cd /usr/app
git clone https://github.com/pyun-ram/det3 -b release

source /root/.bashrc
