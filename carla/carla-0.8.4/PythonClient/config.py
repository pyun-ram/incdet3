from easydict import EasyDict as edict
import numpy as np
import math
__C = edict()
cfg = __C

__C.carla_task_dick = {
    "EPOCH": 1, # 1, 2, 3
    "NUM_EPISODE": 1,
    "FRAME_OF_EPISODE": 225,
    "FRAME_START_OF_EPISODE": 5,
    "FRAME_STEP": 1,
    "SAVE_FOLD": "_out/setup_1"
}

__C.carla_sensor_dict = {
    # x, y, z, roll, pitch, yaw
    "VELO_TOP": (0, 0, 2.4, 0, 0, 0),
    "VELO_LEFT": (0, -0.8, 2.0, math.pi/6, 0, 0),
    "VELO_RIGHT": (0, 0.8, 2.0, -math.pi/6, 0, 0)
}