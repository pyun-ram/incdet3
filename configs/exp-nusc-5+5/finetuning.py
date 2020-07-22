import numpy as np
from easydict import EasyDict as edict
from incdet3.utils import deg2rad

cfg = edict()

cfg.TASK = {
    "valid_range": [-40, -40, -5, 40, 40, 3],
    "total_training_steps": 50e3,
    "continue_training_steps": 50e3
}

cfg.TRAIN = {
    "train_iter": (cfg.TASK["total_training_steps"] +
                   cfg.TASK["continue_training_steps"]),
    "num_log_iter": 40,
    "num_val_iter": 2e3,
    "num_save_iter": 2e3,
    "optimizer_dict":{
        "type": "adam",
        "init_lr": 1e-4,
        "weight_decay": 0,
    },
    "lr_scheduler_dict":{
        "type": "StepLR",
        "step_size": (cfg.TASK["total_training_steps"] +
                      cfg.TASK["continue_training_steps"] * 0.8),
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
    "@classes": ["car", "pedestrian", "barrier", "truck", "traffic_cone",
        "trailer", "construction_vehicle", "motorcycle", "bicycle", "bus"],
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
    "dataset": "nusc", # carla
    "training": True,
    "batch_size": 3,
    "num_workers": 6,
    "feature_map_size": [1, 200, 200],
    "@root_path": "/usr/app/data/Nuscenes-Part1-2/training/",
    "@info_path": "/usr/app/data/Nuscenes-Part1-2/training/infos_train.pkl",
    "@class_names": cfg.TARGETASSIGNER["@classes"].copy(),
    "prep": {
        "@training": True,
        "@augment_dict":
        {
            "p_rot": 0.1,
            "dry_range": [deg2rad(-20), deg2rad(20)],
            "p_tr": 0.1,
            "dx_range": [-1, 1],
            "dy_range": [-1, 1],
            "dz_range": [-0.1, 0.1],
            "p_flip": 0.4,
            "p_keep": 0.4
        },
        "@filter_label_dict":
        {
            "keep_classes": cfg.TARGETASSIGNER["@classes"].copy(),
            "min_num_pts": 5,
            "label_range": cfg.TASK["valid_range"].copy(),
            # [min_x, min_y, min_z, max_x, max_y, max_z] FIMU
        },
        "@feature_map_size": None, # TBD in dataloader_builder.py
        "@classes_to_exclude": ["car", "pedestrian", "barrier", "truck", "traffic_cone"]
    }
}


cfg.VALDATA = {
    "dataset": "nusc", # carla
    "training": False,
    "batch_size": 1,
    "num_workers": 1,
    "feature_map_size": [1, 200, 200],
    "@root_path": "/usr/app/data/Nuscenes-Part1-2/training/",
    "@info_path": "/usr/app/data/Nuscenes-Part1-2/training/infos_val.pkl",
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
    "feature_map_size": [1, 200, 200],
    "@root_path": "/usr/app/data/Nuscenes-Part3/testing/",
    "@info_path": "/usr/app/data/Nuscenes-Part3/testing/infos_val.pkl",
    "@class_names": cfg.TARGETASSIGNER["@classes"].copy(),
    "prep": {
        "@training": False,
        "@augment_dict": None,
        "@filter_label_dict": dict(),
        "@feature_map_size": None # TBD in dataloader_builder.py
    }
}

cfg.NETWORK = {
    "@classes_target": ["car", "pedestrian", "barrier", "truck", "traffic_cone",
        "trailer", "construction_vehicle", "motorcycle", "bicycle", "bus"],
    "@classes_source": None,
    "@model_resume_dict": {
        "ckpt_path": "saved_weights/July22-expnusc-5+5-train_from_scratch/IncDetMain-50000.tckpt",
        "num_classes": 5,
        "num_anchor_per_loc": 10,
        "partially_load_params": [
            "rpn.conv_cls.weight", "rpn.conv_cls.bias",
            "rpn.conv_box.weight", "rpn.conv_box.bias",
            "rpn.conv_dir_cls.weight", "rpn.conv_dir_cls.bias",
        ],
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
    "@training_mode": "fine_tuning",
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
    "@num_biased_select": 64,
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
    "@bool_reuse_anchor_for_cls": False,
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
    # add class_settings
    class_list = cfg_.TARGETASSIGNER["@classes"]
    class2anchor_range = {
        "car": [itm if i not in [2, 5] else -0.93897414
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "pedestrian": [itm if i not in [2, 5] else  -0.73911038
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "barrier": [itm if i not in [2, 5] else -1.27247968
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "truck": [itm if i not in [2, 5] else -0.37937912
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "traffic_cone": [itm if i not in [2, 5] else -1.27868911
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "trailer": [itm if i not in [2, 5] else 0.22228277
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "construction_vehicle": [itm if i not in [2, 5] else -0.08168083
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "motorcycle": [itm if i not in [2, 5] else -0.99194854
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "bicycle": [itm if i not in [2, 5] else -1.03743013
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "bus":[itm if i not in [2, 5] else -0.0715754
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
    }
    class2anchor_size = {
        "car": [1.95017717, 4.60718145, 1.72270761],
        "pedestrian": [0.66344886, 0.7256437, 1.75748069],
        "barrier": [2.49008838, 0.48578221, 0.98297065],
        "truck": [2.4560939, 6.73778078, 2.73004906],
        "traffic_cone": [0.39694519, 0.40359262, 1.06232151],
        "trailer": [2.87427237, 12.01320693, 3.81509561],
        "construction_vehicle": [2.73050468, 6.38352896, 3.13312415],
        "motorcycle": [0.76279481, 2.09973778, 1.44403034],
        "bicycle": [0.60058911, 1.68452161, 1.27192197],
        "bus": [2.94046906, 11.1885991, 3.47030982]
    }
    class2anchor_match_th = {
        "car": 0.4,
        "pedestrian": 0.5,
        "barrier": 0.3,
        "truck": 0.5,
        "traffic_cone": 0.5,
        "trailer": 0.5,
        "construction_vehicle": 0.4,
        "motorcycle": 0.2,
        "bicycle": 0.2,
        "bus": 0.5,
    }
    class2anchor_unmatch_th = {
        "car": 0.3,
        "pedestrian": 0.35,
        "barrier": 0.2,
        "truck": 0.35,
        "traffic_cone": 0.35,
        "trailer": 0.35,
        "construction_vehicle": 0.3,
        "motorcycle": 0.15,
        "bicycle": 0.15,
        "bus": 0.35,
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
    assert cfg_.TARGETASSIGNER["class_settings_car"]["AnchorGenerator"]["@anchor_ranges"][2] == -0.93897414
    assert cfg_.TARGETASSIGNER["class_settings_car"]["AnchorGenerator"]["@anchor_ranges"][-1] == -0.93897414
    return True