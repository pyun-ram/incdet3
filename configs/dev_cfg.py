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
    # "@classes": ["Car"],
    "@classes": ["Car", "Pedestrian", "Pedestrian2"],
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
    "class_settings_pedestrian2": {
        "AnchorGenerator": {
            "type": "AnchorGeneratorBEV",
            "@class_name": "Pedestrian2",
            "@anchor_ranges": cfg.TASK["valid_range"].copy(), # TBD in modify_cfg(cfg)
            "@sizes": [0.61, 0.81, 1.731], # wlh
            "@rotations": [0, 1.57],
            "@match_threshold": 0.6,
            "@unmatch_threshold": 0.45,
        },
        "SimilarityCalculator": {
            "type": "NearestIoUSimilarity"
        }
    },
}

cfg.TRAINDATA = {
    "dataset": "carla", # carla
    "training": True,
    "batch_size": 1,
    "num_workers": 1,
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
        },
        "@feature_map_size": None # TBD
    }
}


cfg.VALDATA = {
    "dataset": "carla", # carla
    "training": False,
    "batch_size": 1,
    "num_workers": 1,
    "@root_path": "/usr/app/data/CARLA/training/",
    "@info_path": "/usr/app/data/CARLA/CARLA_infos_dev.pkl",
    "@class_names": cfg.TARGETASSIGNER["@classes"].copy(),
    "prep": {
        "@training": False,
        "@augment_dict": None,
        "@filter_label_dict": dict(),
        "@feature_map_size": None # TBD
    }
}

cfg.NETWORK = {
    "@classes_target": ["Car", "Pedestrian", "Pedestrian2"],
    "@classes_source": ["Car", "Pedestrian",],
    "@model_resume_dict": {
        "ckpt_path": "./IncDetMain-2.tckpt",
        "num_classes": 2,
        "num_anchor_per_loc": 4,
        "partially_load_params": [
            "rpn.conv_cls.weight", "rpn.conv_cls.bias",
            "rpn.conv_box.weight", "rpn.conv_box.bias",
            "rpn.conv_dir_cls.weight", "rpn.conv_dir_cls.bias"]
    },
    "@sub_model_resume_dict": {
        "ckpt_path": "./IncDetMain-2.tckpt",
        "num_classes": 2,
        "num_anchor_per_loc": 4,
        "partially_load_params": []
    },
    "@voxel_encoder_dict": {
        "name": "SimpleVoxel",
        "@num_input_features": 4,
    },
    "@middle_layer_dict":{
        "name": "SpMiddleFHD",
        "@use_norm": True,
        "@num_input_features": 4,
        "@output_shape": None, #TBD
        "downsample_factor": 8
    },
    "@rpn_dict":{
        "name": "ResNetRPN",
        "@use_norm": True,
        "@num_class": None, # TBD
        "@layer_nums": [5],
        "@layer_strides": [1],
        "@num_filters": [128],
        "@upsample_strides": [1],
        "@num_upsample_filters": [128],
        "@num_input_features": 128,
        "@num_anchor_per_loc": None, # TBD
        "@encode_background_as_zeros": True,
        "@use_direction_classifier": True,
        "@use_groupnorm": False,
        "@num_groups": 0,
        "@box_code_size": None, # TBD
        "@num_direction_bins": 2,
    },
}
def modify_cfg(cfg):
    # modify anchor ranges
    for k, v in cfg.TARGETASSIGNER.items():
        if "class_settings_" in k:
            cfg.TARGETASSIGNER[k]["AnchorGenerator"]["@anchor_ranges"] = cfg.TASK["valid_range"].copy()
            cfg.TARGETASSIGNER[k]["AnchorGenerator"]["@anchor_ranges"][2] = 0.0
            cfg.TARGETASSIGNER[k]["AnchorGenerator"]["@anchor_ranges"][-1] = 0.0
