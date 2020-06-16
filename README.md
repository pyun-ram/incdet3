# README

## Collect Data from CARLA 0.8.4
Run CARLA Simulator
```
docker run --gpus all -e NE_DEVICES=0  --rm -p 2000-2002:2000-2002 carlasim/carla:0.8.4
```
Run Python client to collect data
```
cd <root>/carla/
wget https://github.com/carla-simulator/carla/archive/0.8.4.tar.gz
tar -xvf 0.8.4.tar.gz
rm 0.8.4.tar.gz
cd carla-0.8.4

conda create -n carla python=3.5 # 3.7 is not supported
source activate carla
pip3 install -r PythonClient/requirements.txt

cd <root>/carla/
cd carla-0.8.4/PythonClient/
./carla_visualization_multi_lidar.py -l -i -a
```