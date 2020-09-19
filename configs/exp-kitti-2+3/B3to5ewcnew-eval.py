import numpy as np
from easydict import EasyDict as edict
from incdet3.utils import deg2rad

cfg = edict()

cfg.TASK = {
    "valid_range": [0, -32.0, -3, 52.8, 32.0, 1],
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
    "@voxel_size": [0.05, 0.05, 0.1],
    "@point_cloud_range": cfg.TASK["valid_range"].copy(),
    "@max_num_points": 5,
    "@max_voxels": 20000
}

cfg.TARGETASSIGNER = {
    "type": "TaskAssignerV1",
    "@classes": ["Car", "Pedestrian", "Cyclist", "Van", "Truck"],
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
    "dataset": "kitti", # carla, nusc, kitti
    "training": True,
    "batch_size": 8,
    "num_workers": 8,
    "feature_map_size": [1, 160, 132],
    "@root_path": "/usr/app/data/KITTI/training",
    "@info_path": "/usr/app/data/KITTI/KITTI_infos_train.pkl",
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
        "@target_classes": ["Car", "Pedestrian", "Cyclist", "Van"]
    }
}


cfg.VALDATA = {
    "dataset": "kitti", # carla
    "training": False,
    "batch_size": 1,
    "num_workers": 5,
    "feature_map_size": [1, 160, 132],
    "@root_path": "/usr/app/data/KITTI/training",
    "@info_path": "/usr/app/data/KITTI/KITTI_infos_val.pkl",
    "@class_names": cfg.TARGETASSIGNER["@classes"].copy(),
    "prep": {
        "@training": False,
        "@augment_dict": None,
        "@filter_label_dict": dict(),
        "@feature_map_size": None # TBD in dataloader_builder.py
    },
    "prep_infos": {
        "@valid_range": cfg.TASK["valid_range"],
        "@target_classes": ["Car", "Pedestrian", "Cyclist", "Van"]
    }
}

cfg.TESTDATA = {
    "dataset": "kitti", # carla
    "training": False,
    "batch_size": 1,
    "num_workers": 5,
    "feature_map_size": [1, 160, 132],
    "@root_path": "/usr/app/data/KITTI/training",
    "@info_path": "/usr/app/data/KITTI/KITTI_infos_val.pkl",
    "@class_names": cfg.TARGETASSIGNER["@classes"].copy(),
    "prep": {
        "@training": False,
        "@augment_dict": None,
        "@filter_label_dict": dict(),
        "@feature_map_size": None # TBD in dataloader_builder.py
    },
    "prep_infos": {
        "@valid_range": cfg.TASK["valid_range"],
        "@target_classes": ["Cyclist", "Van", "Truck"]
    }
}

cfg.NETWORK = {
    "@classes_target": ["Car", "Pedestrian", "Cyclist", "Van", "Truck"],
    "@classes_source": None,
    "@model_resume_dict": {
        "ckpt_path": "saved_weights/20200919-expkitti2+3-ewc-saved_weights/20200919-expkitti2+3-ewc/IncDetMain-TBDstepsTBD.tckpt",
        "num_classes": 5,
        "num_anchor_per_loc": 10,
        "partially_load_params": [],
        "ignore_params": [],
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
    class2anchor_range = {
        "Car": [itm if i not in [2, 5] else -0.6
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "Pedestrian": [itm if i not in [2, 5] else  -0.6
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "Cyclist": [itm if i not in [2, 5] else -0.6
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "Van": [itm if i not in [2, 5] else -1.41
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "Truck": [itm if i not in [2, 5] else -1.6
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "Tram": [itm if i not in [2, 5] else -1.2
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
        "Person_sitting": [itm if i not in [2, 5] else -1.5
        for i, itm in enumerate(cfg_.TASK["valid_range"])],
    }
    class2anchor_size = {
        "Car": [1.6, 3.9, 1.56],
        "Pedestrian": [0.6, 0.8, 1.73],
        "Cyclist": [0.6, 1.76, 1.73],
        "Van": [1.87103749, 5.02808195, 2.20964255],
        "Truck": [2.60938525, 9.20477459, 3.36219262],
        "Tram": [2.36035714, 15.55767857,  3.529375],
        "Person_sitting": [0.54357143, 1.06392857, 1.27928571]
    }
    class2anchor_match_th = {
        "Car": 0.6,
        "Pedestrian": 0.35,
        "Cyclist": 0.35,
        "Van": 0.6,
        "Truck": 0.6,
        "Tram": 0.6,
        "Person_sitting": 0.35
    }
    class2anchor_unmatch_th = {
        "Car": 0.45,
        "Pedestrian": 0.2,
        "Cyclist": 0.2,
        "Van": 0.45,
        "Truck": 0.45,
        "Tram": 0.45,
        "Person_sitting": 0.2
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
    assert cfg_.TARGETASSIGNER["class_settings_Car"]["AnchorGenerator"]["@anchor_ranges"][2] == -0.6
    assert cfg_.TARGETASSIGNER["class_settings_Car"]["AnchorGenerator"]["@anchor_ranges"][-1] == -0.6
    return True