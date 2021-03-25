# README

This repository is for task-incremental-learning in LiDAR-based 3D object detection.

## Requirements

- python 3.7.3
- CUDA 10.1+
- PyTorch 1.4.0+
- Open3D 0.9
- [det3](https://github.com/pyun-ram/det3)

## Dockerfile

```
# e.g. CUDA10.1
cd Dockerfiles
docker build ./peng-incdet3-pytorch1.4_cu10.1_spconv -t <dockerimage-tag>/incdet3:latest
docker run -it --name incdet3 --gpus all <dockerimage-tag>/incdet3:latest
```

## Usage

The run_scripts provide the shell scripts of exp-kitti-2+3 and exp-kitti-2+seq.
You can run it inside the docker container for both training and evaluation.

You can download our [weights](http://gofile.me/4jm56/DbSl0zfvs) for reproducing the experimental results.
