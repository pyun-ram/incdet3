import numpy as np
from easydict import EasyDict as edict
from incdet3.utils import deg2rad

cfg = edict()

cfg.TASK = {
    "valid_range": [0, -32.0, -5, 52.8, 32.0, 3],
    "total_training_steps": 464 * 50,
    "use_fp16": False,
}

cfg.TRAIN = {
    "train_iter": cfg.TASK["total_training_steps"],
    "num_log_iter": 40,
    "num_val_iter": 901,
    "num_save_iter": 901,
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
    "@max_num_points": 5,
    "@max_voxels": 60000
}

cfg.TARGETASSIGNER = {
    "type": "TaskAssignerV1",
    "@classes": ["car", "pedestrian"],
    "@feature_map_sizes": None,
    "@positive_fraction": None,
    "@sample_size": 512,
    "@assign_per_class": True,
    "box_coder": {
        "type": "BoxCoderV1",
        "@custom_ndim": 0
    },
}

cfg.TRAINDATA = {
    "dataset": "nusc-kitti", # carla, nusc, kitti
    "training": True,
    "batch_size": 4,
    "num_workers": 4,
    "feature_map_size": [1, 160, 132],
    "@root_path": "/usr/app/data/nusc-kitti-mergesweep/training",
    "@info_path": "/usr/app/data/nusc-kitti-mergesweep/KITTI_infos_train.pkl",
    "@class_names": cfg.TARGETASSIGNER["@classes"].copy(),
    "prep": {
        "@training": True,
        "@augment_dict":
        {
            "p_rot": 0.3,
            "dry_range": [deg2rad(-22.5), deg2rad(22.5)],
            "p_tr": 0.3,
            "dx_range": [-1, 1],
            "dy_range": [-1, 1],
            "dz_range": [-0.1, 0.1],
            "p_flip": 0.3,
            "p_keep": 0.1
        },
        "@filter_label_dict":
        {
            "keep_classes": cfg.TARGETASSIGNER["@classes"].copy(),
            ## deactivate min_num_pts filtering
            ## since it is slow when point cloud have too many points
            "min_num_pts": 5,
            "label_range": cfg.TASK["valid_range"].copy(),
            # [min_x, min_y, min_z, max_x, max_y, max_z] FIMU
        },
        "@feature_map_size": None, # TBD in dataloader_builder.py
        "@classes_to_exclude": []
    },
    "prep_infos": {
        "@valid_range": cfg.TASK["valid_range"],
        "@target_classes": ["car", "pedestrian"]
    }
}


cfg.VALDATA = {
    "dataset": "nusc-kitti", # carla
    "training": False,
    "batch_size": 1,
    "num_workers": 5,
    "feature_map_size": [1, 160, 132],
    "@root_path": "/usr/app/data/nusc-kitti-mergesweep/training",
    "@info_path": "/usr/app/data/nusc-kitti-mergesweep/KITTI_infos_train.pkl",
    "@class_names": cfg.TARGETASSIGNER["@classes"].copy(),
    "prep": {
        "@training": False,
        "@augment_dict": None,
        "@filter_label_dict": dict(),
        "@feature_map_size": None # TBD in dataloader_builder.py
    },
    "prep_infos": {
        "@valid_range": cfg.TASK["valid_range"],
        "@target_classes": ["car", "pedestrian"]
    }
}

cfg.TESTDATA = {
    "dataset": "nusc-kitti", # carla
    "training": False,
    "batch_size": 1,
    "num_workers": 5,
    "feature_map_size": [1, 160, 132],
    "@root_path": "/usr/app/data/nusc-kitti-mergesweep/training",
    "@info_path": "/usr/app/data/nusc-kitti-mergesweep/KITTI_infos_train.pkl",
    "@class_names": cfg.TARGETASSIGNER["@classes"].copy(),
    "prep": {
        "@training": False,
        "@augment_dict": None,
        "@filter_label_dict": dict(),
        "@feature_map_size": None # TBD in dataloader_builder.py
    },
}

cfg.NETWORK = {
    "@classes_target": ["car", "pedestrian"],
    "@classes_source": None,
    "@model_resume_dict": None,
    "@sub_model_resume_dict": None,
    "@voxel_encoder_dict": {
        "name": "SimpleVoxel",
        "@num_input_features": 5,
    },
    "@middle_layer_dict":{
        "name": "SpMiddleFHD",
        "@use_norm": True,
        "@num_input_features": 5,
        "@output_shape": None, #TBD
        "downsample_factor": 8
    },
    "@rpn_dict":{
        "name": "ResNetRPN",
        "@use_norm": True,
        "@num_class": None, # TBD in Network._build_model_and_init()
        "@layer_nums": [5, 5],
        "@layer_strides": [1, 2],
        "@num_filters": [128, 256],
        "@upsample_strides": [1, 2],
        "@num_upsample_filters": [256, 256],
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
    "@weight_decay_coef": 0.01,
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
    "@nms_score_thresholds": [0.3],
    "@nms_pre_max_sizes": [1000],
    "@nms_post_max_sizes": [100],
    "@nms_iou_thresholds": [0.01],
    "@box_coder": None #TBD in main.py
}

def modify_cfg(cfg_):
    # add class_settings
    class_list = cfg_.TARGETASSIGNER["@classes"]
    # TODO: TBD for nusc datast
    class2anchor_range = {
        "car": [itm if i not in [2, 5] else -0.6
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "bus": [itm if i not in [2, 5] else  -1.0
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "truck": [itm if i not in [2, 5] else 0
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "pedestrian": [itm if i not in [2, 5] else -0.6
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "construction_vehicle": [itm if i not in [2, 5] else 0.6
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "traffic_cone": [itm if i not in [2, 5] else -0.6
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "bicycle": [itm if i not in [2, 5] else -1.0
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "motorcycle": [itm if i not in [2, 5] else -1.0
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "trailer": [itm if i not in [2, 5] else -0.6
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "barrier": [itm if i not in [2, 5] else 0
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
    }
    class2anchor_size = {
        "car": [1.97, 4.63, 1.74],
        "truck": [2.51, 6.93, 2.84],
        "construction_vehicle": [2.85, 6.37, 3.19],
        "bus": [2.94, 10.5, 3.47],
        "trailer": [2.90, 12.29, 3.87],
        "barrier": [2.53, 0.50, 0.98],
        "motorcycle": [0.77, 2.11, 1.47],
        "bicycle": [0.60, 1.70, 1.28],
        "pedestrian": [0.67, 0.73, 1.77],
        "traffic_cone": [0.41, 0.41, 1.07],
    }
    class2anchor_match_th = {
        "car": 0.6,
        "truck": 0.6,
        "construction_vehicle": 0.6,
        "bus": 0.6,
        "trailer": 0.6,
        "barrier": 0.35,
        "motorcycle": 0.35,
        "bicycle": 0.35,
        "pedestrian": 0.35,
        "traffic_cone": 0.35,
    }
    class2anchor_unmatch_th = {
        "car": 0.45,
        "truck": 0.45,
        "construction_vehicle": 0.45,
        "bus": 0.45,
        "trailer": 0.45,
        "barrier": 0.2,
        "motorcycle": 0.2,
        "bicycle": 0.2,
        "pedestrian": 0.2,
        "traffic_cone": 0.2,
    }
    for cls in class_list:
        key = f"class_settings_{cls}"
        if key in cfg_.TARGETASSIGNER.keys():
            continue
        value = {
            "AnchorGenerator": {
                "type": "AnchorGeneratorBEV",
                "@class_name": cls,
                "@anchor_ranges": class2anchor_range[cls], # TBD in modify_cfg(cfg)
                "@sizes": class2anchor_size[cls], # wlh
                "@rotations": [0, 1.57],
                "@match_threshold": class2anchor_match_th[cls],
                "@unmatch_threshold": class2anchor_unmatch_th[cls],
            },
            "SimilarityCalculator": {
                "type": "NearestIoUSimilarity"
            }
        }
        cfg_.TARGETASSIGNER[key] = value
    # modify anchor ranges
    # for k, v in cfg_.TARGETASSIGNER.items():
    #     if "class_settings_" in k:
    #         cfg_.TARGETASSIGNER[k]["AnchorGenerator"]["@anchor_ranges"] = cfg_.TASK["valid_range"].copy()
    #         if "car" in k:
    #             cfg_.TARGETASSIGNER[k]["AnchorGenerator"]["@anchor_ranges"][2] = -0.93897414
    #             cfg_.TARGETASSIGNER[k]["AnchorGenerator"]["@anchor_ranges"][-1] = -0.93897414
    # modify num_old_classes for distillation_loss
    key_list = [itm for itm in cfg_.NETWORK["@loss_dict"].keys()]
    if "DistillationClassificationLoss" in key_list:
        num_old_classes = len(cfg_.NETWORK["@classes_source"]) if cfg_.NETWORK["@classes_source"] is not None else 0
        cfg_.NETWORK["@loss_dict"]["DistillationClassificationLoss"]["code_weights"] = [1.0] * num_old_classes

def check_cfg(cfg_):
    assert cfg_.TARGETASSIGNER["class_settings_car"]["AnchorGenerator"]["@anchor_ranges"][2] == -0.6
    assert cfg_.TARGETASSIGNER["class_settings_car"]["AnchorGenerator"]["@anchor_ranges"][-1] == -0.6
    return True