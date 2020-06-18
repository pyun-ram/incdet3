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
    "optimizer_dict":{
        "type": "adam",
        "init_lr": 5e-3,
        "weight_decay": 0.01,
    },
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
    "@classes": ["Car"],
    # "@classes": ["Car", "Pedestrian"],
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
    # "class_settings_pedestrian": {
    #     "AnchorGenerator": {
    #         "type": "AnchorGeneratorBEV",
    #         "@class_name": "Pedestrian",
    #         "@anchor_ranges": cfg.TASK["valid_range"].copy(), # TBD in modify_cfg(cfg)
    #         "@sizes": [0.6, 0.8, 1.73], # wlh
    #         "@rotations": [0, 1.57],
    #         "@match_threshold": 0.6,
    #         "@unmatch_threshold": 0.45,
    #     },
    #     "SimilarityCalculator": {
    #         "type": "NearestIoUSimilarity"
    #     }
    # },
}


cfg.MODEL = {
    "name": "baseline", # baseline, inputfusion, featurefusion
    "resume": "saved_weights/MLOD-CARLACAR-TOP-B/VoxelNet-46850.tckpt",
    "VoxelEncoder": {
        "name": "SimpleVoxel",
        "num_input_features": 3,
    },
    "MiddleLayer":{
        "name": "SpMiddleFHD",
        "use_norm": True,
        "num_input_features": 3,
        "downsample_factor": 8
    },
    "RPN":{
        "name": "RPNV2",
        "use_norm": True,
        "use_groupnorm": False,
        "num_groups": 0,
        "layer_nums": [5],
        "layer_strides": [1],
        "num_filters": [128],
        "upsample_strides": [1],
        "num_upsample_filters": [128],
        "num_input_features": 128,
    },
    "ClassificationLoss":{
        "name": "SigmoidFocalClassificationLoss",
        "alpha": 0.25,
        "gamma": 2.0,
    },
    "LocalizationLoss":{
        "name": "WeightedSmoothL1LocalizationLoss",
        "sigma": 3.0,
        "code_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "codewise": True,
    },
    "num_class": 1,
    "use_sigmoid_score": True,
    "encode_background_as_zeros": True,
    "use_direction_classifier": True,
    "num_direction_bins": 2,
    "encode_rad_error_by_sin": True,
    "post_center_range": cfg.TASK["valid_range"].copy(),
    "nms_class_agnostic": False,
    "direction_limit_offset": 1,
    "sin_error_factor": 1.0,
    "use_rotate_nms": True,
    "multiclass_nms": False,
    "nms_pre_max_sizes": [2000],
    "nms_post_max_sizes": [100],
    "nms_score_thresholds": [0.3], # 0.4 in submit, but 0.3 can get better hard performance
    "nms_iou_thresholds": [0.2],
    "cls_loss_weight": 1.0,
    "loc_loss_weight": 2.0,
    "loss_norm_type": "NormByNumPositives",
    "direction_offset": 0.0,
    "direction_loss_weight": 0.2,
    "pos_cls_weight": 1.0,
    "neg_cls_weight": 1.0,
}

cfg.TRAINDATA = {
    "dataset": "carla", # carla, waymo
    "training": True,
    "batch_size": 6,
    "num_workers": 6,
    "@root_path": "/usr/app/data/CARLA/training/",
    "@info_path": "/usr/app/data/CARLA/CARLA_infos_dev.pkl",
    "@class_names": cfg.TARGETASSIGNER["@classes"].copy(),
    "prep": {
        "@training": True,
        "@augment_dict":
        {
            "p_rot": 0.25,
            "dry_range": [deg2rad(-45), deg2rad(45)],
            "p_tr": 0.25,
            "dx_range": [-1, 1],
            "dy_range": [-1, 1],
            "dz_range": [-0.1, 0.1],
            "p_flip": 0.25,
            "p_keep": 0.25
        },
        "@filter_label_dict":
        {
            "keep_classes": cfg.TARGETASSIGNER["@classes"].copy(),
            "min_num_pts": 5,
            "label_range": cfg.TASK["valid_range"].copy(),
            # [min_x, min_y, min_z, max_x, max_y, max_z] FIMU
        }
    }
}

def modify_cfg(cfg):
    # modify anchor ranges
    for k, v in cfg.TARGETASSIGNER.items():
        if "class_settings_" in k:
            cfg.TARGETASSIGNER[k]["AnchorGenerator"]["@anchor_ranges"] = cfg.TASK["valid_range"].copy()
            cfg.TARGETASSIGNER[k]["AnchorGenerator"]["@anchor_ranges"][2] = 0.0
            cfg.TARGETASSIGNER[k]["AnchorGenerator"]["@anchor_ranges"][-1] = 0.0
