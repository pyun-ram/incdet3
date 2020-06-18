import numpy as np
from easydict import EasyDict as edict
from mlod.utils import deg2rad

cfg = edict()

cfg.TASK = {
    "valid_range": [-35.2, -40, -1.5, 35.2, 40, 2.6],
    "total_training_steps": 40e3,
}

cfg.TRAIN = {
    "num_epochs": 50,
    "disp_itv": 20, # steps
    # "optimizer_dict":{
    #     "type": "ADAMOptimizer",
    #     "amsgrad": False,
    #     "init_lr": 1e-3, # it will be changed by OneCycle LRscheduler
    #     "weight_decay": 0.01,
    #     "fixed_weight_decay": True,
    # },
    "optimizer_dict":{
        "type": "adam",
        "init_lr": 5e-3,
        "weight_decay": 0.01,
    },
    # "lr_scheduler_dict":{
    #     "type": "OneCycle",
    #     "total_step": cfg.TASK["total_training_steps"],
    #     "lr_max": 2.25e-3,
    #     "moms": [0.95, 0.85],
    #     "div_factor": 10.0,
    #     "pct_start": 0.4,
    # }
    "lr_scheduler_dict":{
        "type": "StepLR",
        "step_size": cfg.TASK["total_training_steps"] * 0.8,
        "gamma": 0.1
    }
}

cfg.VOXELIZER = {
    "type": "VoxelizerV1",
    "@voxel_size": [0.05, 0.05, 0.1],
    "@point_cloud_range": cfg.TASK["valid_range"].copy(),
    "@max_num_points": 5,
    "@max_voxels": 100000
}

cfg.TARGETASSIGNER = {
    "type": "TaskAssignerV1",
    "@classes": ["Car", "Pedestrian"],
    "@feature_map_sizes": None,
    "@positive_fraction": None,
    "@sample_size": 512,
    "@assign_per_class": True,
    "box_coder": {
        "type": "BoxCoderV1",
        "@custom_ndim": 0
    },
    "class_settings_car": {
        "AnchorGenerator": {
            "type": "AnchorGeneratorBEV",
            "@class_name": "Car",
            "@anchor_ranges": cfg.TASK["valid_range"].copy(), # TBD in modify_cfg(cfg)
            "@sizes": [1.6, 3.9, 1.56], # wlh
            "@rotations": [0, 1.57],
            "@match_threshold": 0.6,
            "@unmatch_threshold": 0.45,
        },
        "SimilarityCalculator": {
            "type": "NearestIoUSimilarity"
        }
    },
    "class_settings_pedestrian": {
        "AnchorGenerator": {
            "type": "AnchorGeneratorBEV",
            "@class_name": "Pedestrian",
            "@anchor_ranges": cfg.TASK["valid_range"].copy(), # TBD in modify_cfg(cfg)
            "@sizes": [0.6, 0.8, 1.73], # wlh
            "@rotations": [0, 1.57],
            "@match_threshold": 0.6,
            "@unmatch_threshold": 0.45,
        },
        "SimilarityCalculator": {
            "type": "NearestIoUSimilarity"
        }
    },
}

def modify_cfg(cfg):
    # modify anchor ranges
    for k, v in cfg.TARGETASSIGNER.items():
        if "class_settings_" in k:
            cfg.TARGETASSIGNER[k]["AnchorGenerator"]["@anchor_ranges"] = cfg.TASK["valid_range"].copy()
            cfg.TARGETASSIGNER[k]["AnchorGenerator"]["@anchor_ranges"][2] = 0.0
            cfg.TARGETASSIGNER[k]["AnchorGenerator"]["@anchor_ranges"][-1] = 0.0
