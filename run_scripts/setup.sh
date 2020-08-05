#!/bin/bash
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

conda install -c conda-forge fire
pip install nuscenes-devkit
apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libfontconfig1
cd /usr/app
git clone https://git.ram-lab.com/yun/nuscenes-devkit-pyun
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
echo "${YELLOW}You need to manually add /usr/app/nuscenes-devkit-pyun/python-sdk to PYTHONPATH${NC}"