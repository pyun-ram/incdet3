#!/bin/bash
conda install -c conda-forge fire
pip install nuscenes-devkit
apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libfontconfig1
cd /usr/app
git clone https://git.ram-lab.com/yun/nuscenes-devkit-pyun
echo "You need to manually add /usr/app/nuscenes-devkit-pyun/python-sdk to PYTHONPATH"