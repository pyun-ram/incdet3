#!/bin/bash
current_map=Town01
docker run --gpus all --name carla_simulator -e NE_DEVICES=0  --rm -p 2000-2002:2000-2002 carlasim/carla:0.8.4 \
    /bin/bash CarlaUE4.sh /Game/Maps/$current_map -carla-server
