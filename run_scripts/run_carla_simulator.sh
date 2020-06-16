#!/bin/bash
docker run --gpus all --name carla_simulator -e NE_DEVICES=0  --rm -p 2000-2002:2000-2002 carlasim/carla:0.8.4
