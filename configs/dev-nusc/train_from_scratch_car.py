import numpy as np
from easydict import EasyDict as edict
from incdet3.utils import deg2rad

cfg = edict()

cfg.TASK = {
    "valid_range": [-49.6, -49.6, -5, 49.6, 49.6, 3],
    "total_training_steps": 15e3+100,
}

cfg.TRAIN = {
    "train_iter": cfg.TASK["total_training_steps"],
    "num_log_iter": 20,
    "num_val_iter": 1e2,
    "num_save_iter": 1e3,
    "optimizer_dict":{
        "type": "adam",
        "init_lr": 1e-3,
        "weight_decay": 0,
    },
    "lr_scheduler_dict":{
        "type": "StepLR",
        "step_size": cfg.TASK["total_training_steps"] * 0.8,
        "gamma": 0.1
    }
}

cfg.VOXELIZER = {
    "type": "VoxelizerV1",
    "@voxel_size": [0.05, 0.05, 0.2],
    "@point_cloud_range": cfg.TASK["valid_range"].copy(),
    "@max_num_points": 1,
    "@max_voxels": 100000
}

cfg.TARGETASSIGNER = {
    "type": "TaskAssignerV1",
    "@classes": ["car"],
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
            "@class_name": "car",
            "@anchor_ranges": cfg.TASK["valid_range"].copy(), # TBD in modify_cfg(cfg)
            "@sizes": [1.9, 4.6, 1.7], # wlh
            "@rotations": [0, 1.57],
            "@match_threshold": 0.4,
            "@unmatch_threshold": 0.3,
        },
        "SimilarityCalculator": {
            "type": "NearestIoUSimilarity"
        }
    },
}

cfg.TRAINDATA = {
    "dataset": "nusc", # carla
    "training": True,
    "batch_size": 4,
    "num_workers": 4,
    "feature_map_size": [1, 248, 248],
    "@root_path": "/usr/app/data/Nuscenes-Mini/training/",
    "@info_path": "/usr/app/data/Nuscenes-Mini/training/infos_train.pkl",
    "@class_names": cfg.TARGETASSIGNER["@classes"].copy(),
    "prep": {
        "@training": True,
        "@augment_dict":
        {
            "p_rot": 0.25,
            "dry_range": [deg2rad(-20), deg2rad(20)],
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
        "@feature_map_size": None, # TBD in dataloader_builder.py
        "@classes_to_exclude": []
    }
}


cfg.VALDATA = {
    "dataset": "nusc", # carla
    "training": False,
    "batch_size": 1,
    "num_workers": 1,
    "feature_map_size": [1, 248, 248],
    "@root_path": "/usr/app/data/Nuscenes-Mini/training/",
    "@info_path": "/usr/app/data/Nuscenes-Mini/training/infos_val.pkl",
    "@class_names": cfg.TARGETASSIGNER["@classes"].copy(),
    "prep": {
        "@training": False,
        "@augment_dict": None,
        "@filter_label_dict": dict(),
        "@feature_map_size": None # TBD in dataloader_builder.py
    }
}

cfg.TESTDATA = {
    "dataset": "nusc", # carla
    "training": False,
    "batch_size": 1,
    "num_workers": 1,
    "feature_map_size": [1, 248, 248],
    "@root_path": "/usr/app/data/Nuscenes-Mini/training/",
    "@info_path": "/usr/app/data/Nuscenes-Mini/training/infos_val.pkl",
    "@class_names": cfg.TARGETASSIGNER["@classes"].copy(),
    "prep": {
        "@training": False,
        "@augment_dict": None,
        "@filter_label_dict": dict(),
        "@feature_map_size": None # TBD in dataloader_builder.py
    }
}

cfg.NETWORK = {
    "@classes_target": ["car"],
    "@classes_source": None,
    "@model_resume_dict": {
        "ckpt_path": "saved_weights/incdet3-nusc/IncDetMain-15000.tckpt",
        "num_classes": 1,
        "num_anchor_per_loc": 2,
        "partially_load_params": [],
        "ignore_params": []
    },
    "@sub_model_resume_dict": None,
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
        "@num_class": None, # TBD in Network._build_model_and_init()
        "@layer_nums": [5],
        "@layer_strides": [1],
        "@num_filters": [128],
        "@upsample_strides": [1],
        "@num_upsample_filters": [128],
        "@num_input_features": 128,
        "@num_anchor_per_loc": None, # TBD in Network._build_model_and_init()
        "@encode_background_as_zeros": True,
        "@use_direction_classifier": True,
        "@use_groupnorm": False,
        "@num_groups": 0,
        "@box_code_size": 7, # TBD
        "@num_direction_bins": 2,
    },
    "@training_mode": "train_from_scratch",
    "@is_training": None, #TBD
    "@cls_loss_weight": 1.0,
    "@loc_loss_weight": 2.0,
    "@dir_loss_weight": 0.2,
    "@weight_decay_coef": 0.001,
    "@pos_cls_weight": 1.0,
    "@neg_cls_weight": 1.0,
    "@l2sp_alpha_coef": 0.2,
    "@delta_coef": 0.01,
    "@distillation_loss_cls_coef": 0.1,
    "@distillation_loss_reg_coef": 0.2,
    "@num_biased_select": 32,
    "@threshold_delta_fgmask": 0.5,
    "@loss_dict": {
        "ClassificationLoss":{
            "name": "SigmoidFocalClassificationLoss",
            "@alpha": 0.25,
            "@gamma": 2.0,
        },
        "LocalizationLoss":{
            "name": "WeightedSmoothL1LocalizationLoss",
            "@sigma": 3.0,
            "@code_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "@codewise": True,
        },
        "DirectionLoss":{
            "name": "WeightedSoftmaxClassificationLoss",
        },
        "DistillationClassificationLoss":{
            "name": "WeightedSmoothL1LocalizationLoss",
            "@sigma": 1.0,
            "@code_weights": None, # TBD in modify_cfg()
            "@codewise": True,
        },
        "DistillationRegressionLoss":{
            "name": "WeightedSmoothL1LocalizationLoss",
            "@sigma": 1.0,
            "@code_weights": [1.0] * 7,
            "@codewise": True,
        },
    },
    "@hook_layers": [],
    "@distillation_mode": [],
    "@bool_reuse_anchor_for_cls": True,
    "@bool_biased_select_with_submodel": True,
    "@bool_delta_use_mask": False,
    "@bool_oldclassoldanchor_predicts_only": False,
    "@post_center_range": cfg.TASK["valid_range"].copy(),
    "@nms_score_thresholds": [0.2],
    "@nms_pre_max_sizes": [2000],
    "@nms_post_max_sizes": [500],
    "@nms_iou_thresholds": [0.3],
    "@box_coder": None #TBD in main.py
}

def modify_cfg(cfg_):
    # modify anchor ranges
    for k, v in cfg_.TARGETASSIGNER.items():
        if "class_settings_" in k:
            cfg_.TARGETASSIGNER[k]["AnchorGenerator"]["@anchor_ranges"] = cfg_.TASK["valid_range"].copy()
            if "car" in k:
                cfg_.TARGETASSIGNER[k]["AnchorGenerator"]["@anchor_ranges"][2] = -0.93897414
                cfg_.TARGETASSIGNER[k]["AnchorGenerator"]["@anchor_ranges"][-1] = -0.93897414
    # modify num_old_classes for distillation_loss
    key_list = [itm for itm in cfg_.NETWORK["@loss_dict"].keys()]
    if "DistillationClassificationLoss" in key_list:
        num_old_classes = len(cfg_.NETWORK["@classes_source"]) if cfg_.NETWORK["@classes_source"] is not None else 0
        cfg_.NETWORK["@loss_dict"]["DistillationClassificationLoss"]["code_weights"] = [1.0] * num_old_classes

def check_cfg(cfg_):
    assert cfg_.TARGETASSIGNER["class_settings_car"]["AnchorGenerator"]["@anchor_ranges"][2] == -0.93897414
    assert cfg_.TARGETASSIGNER["class_settings_car"]["AnchorGenerator"]["@anchor_ranges"][-1] == -0.93897414
    return True