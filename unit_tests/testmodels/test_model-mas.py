'''
 File Created: Mon Oct 05 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''

import torch
import unittest
import numpy as np
from copy import deepcopy
from incdet3.models.model import Network
from torch.nn.parameter import Parameter
from det3.ops import read_pkl, write_pkl
torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def build_dataloader():
    from incdet3.builders.dataloader_builder import build
    from incdet3.builders import voxelizer_builder, target_assigner_builder
    VOXELIZER_cfg = {
        "type": "VoxelizerV1",
        "@voxel_size": [0.05, 0.05, 0.1],
        "@point_cloud_range": [0, -32, -3, 52.8, 32.0, 1],
        "@max_num_points": 5,
        "@max_voxels": 20000
    }
    TARGETASSIGNER_cfg = {
        "type": "TaskAssignerV1",
        "@classes": ["Car", "Pedestrian", "Cyclist"],
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
                "@anchor_ranges": [0, -32, 0, 52.8, 32.0, 0], # TBD in modify_cfg(cfg)
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
                "@anchor_ranges": [0, -32, 0, 52.8, 32.0, 0], # TBD in modify_cfg(cfg)
                "@sizes": [0.6, 0.8, 1.73], # wlh
                "@rotations": [0, 1.57],
                "@match_threshold": 0.6,
                "@unmatch_threshold": 0.45,
            },
            "SimilarityCalculator": {
                "type": "NearestIoUSimilarity"
            }
        },
        "class_settings_cyclist": {
            "AnchorGenerator": {
                "type": "AnchorGeneratorBEV",
                "@class_name": "Cyclist",
                "@anchor_ranges": [0, -32, 0, 52.8, 32.0, 0], # TBD in modify_cfg(cfg)
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
    TRAINDATA_cfg = {
        "dataset": "kitti", # carla
        "training": False, # set this to false to avoid shuffle
        "batch_size": 1,
        "num_workers": 1,
        "@root_path": "unit_tests/data/test_kittidata",
        "@info_path": "unit_tests/data/test_kittidata/KITTI_infos_train.pkl",
        "@class_names": ["Car", "Pedestrian", "Cyclist"],
        "prep": {
            "@training": True, # set this to True to return targets
            "@augment_dict": None,
            "@filter_label_dict":
            {
                "keep_classes": ["Car", "Pedestrian", "Cyclist"],
                "min_num_pts": -1,
                "label_range": [0, -32, -3, 52.8, 32.0, 1],
                # [min_x, min_y, min_z, max_x, max_y, max_z] FIMU
            },
            "@feature_map_size": [1, 200, 176] # TBD
        }
    }
    data_cfg = TRAINDATA_cfg
    voxelizer = voxelizer_builder.build(VOXELIZER_cfg)
    target_assigner = target_assigner_builder.build(TARGETASSIGNER_cfg)
    dataloader = build(data_cfg,
        ext_dict={
            "voxelizer": voxelizer,
            "target_assigner": target_assigner,
            "feature_map_size": [1, 200, 176]
        })
    return dataloader

def build_network():
    network_cfg_template =  {
        "VoxelEncoder": {
            "name": "SimpleVoxel",
            "@num_input_features": 4,
        },
        "MiddleLayer":{
            "name": "SpMiddleFHD",
            "@use_norm": True,
            "@num_input_features": 4,
            "@output_shape": [1, 41, 1600, 1408, 16], #TBD
            "downsample_factor": 8
        },
        "RPN":{
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
            "@box_code_size": 7, # TBD
            "@num_direction_bins": 2,
        },
    }
    name_template = "IncDetTest"
    rpn_name = "ResNetRPN"
    network_cfg = deepcopy(network_cfg_template)
    network_cfg["RPN"]["name"] = rpn_name
    network_cfg["RPN"]["@num_class"] = 3
    network_cfg["RPN"]["@num_anchor_per_loc"] = 6
    params = {
        "classes_target": ["class1", "class2", "class3"],
        "classes_source": ["class1", "class2"],
        "model_resume_dict": {
            "ckpt_path": "unit_tests/data/train_class2-23200.tckpt",
            "num_classes": 2,
            "num_anchor_per_loc": 4,
            "partially_load_params": [
                "rpn.conv_cls.weight", "rpn.conv_cls.bias",
                "rpn.conv_box.weight", "rpn.conv_box.bias",
                "rpn.conv_dir_cls.weight", "rpn.conv_dir_cls.bias",
            ]
        },
        "sub_model_resume_dict": {
            "ckpt_path": "unit_tests/data/train_class2-23200.tckpt",
            "num_classes": 2,
            "num_anchor_per_loc": 4,
            "partially_load_params": []
        },
        "voxel_encoder_dict": network_cfg["VoxelEncoder"],
        "middle_layer_dict": network_cfg["MiddleLayer"],
        "rpn_dict": network_cfg["RPN"],
        "training_mode": "lwf",
        "is_training": True,
        "pos_cls_weight": 1.0,
        "neg_cls_weight": 1.0,
        "l2sp_alpha_coef": 2.0,
        "weight_decay_coef": 0.01,
        "delta_coef": 4.0,
        "ewc_coef": 2.0,
        "ewc_weights_path": "unit_tests/data/test_model-ewc-ewc_weights.pkl",
        "distillation_loss_cls_coef": 1.0,
        "distillation_loss_reg_coef": 1.0,
        "num_biased_select": 2,
        "loss_dict": {
            "ClassificationLoss":{
                "name": "SigmoidFocalClassificationLoss",
                "@alpha": 0.25,
                "@gamma": 2.0,
            },
            "LocalizationLoss":{
                "name": "WeightedSmoothL1LocalizationLoss",
                "@sigma": 3.0,
                "@code_weights": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "@codewise": True,
            },
            "DirectionLoss":{
                "name": "WeightedSoftmaxClassificationLoss",
            },
            "DistillationClassificationLoss":{
                "name": "WeightedSmoothL1LocalizationLoss",
                "@sigma": 1.3,
                "@code_weights": [4.0] * 2,
                "@codewise": True,
            },
            "DistillationRegressionLoss":{
                "name": "WeightedSmoothL1LocalizationLoss",
                "@sigma": 3.0,
                "@code_weights": [3.0] * 7,
                "@codewise": True,
            },
        },
        "hook_layers": [],
        "distillation_mode": ["ewc"],
        "bool_reuse_anchor_for_cls": False,
        "bool_biased_select_with_submodel": False
    }
    network = Network(**params).cuda()
    return network


def build_network_class4():
    network_cfg_template =  {
        "VoxelEncoder": {
            "name": "SimpleVoxel",
            "@num_input_features": 4,
        },
        "MiddleLayer":{
            "name": "SpMiddleFHD",
            "@use_norm": True,
            "@num_input_features": 4,
            "@output_shape": [1, 41, 1600, 1408, 16], #TBD
            "downsample_factor": 8
        },
        "RPN":{
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
            "@box_code_size": 7, # TBD
            "@num_direction_bins": 2,
        },
    }
    name_template = "IncDetTest"
    rpn_name = "ResNetRPN"
    network_cfg = deepcopy(network_cfg_template)
    network_cfg["RPN"]["name"] = rpn_name
    network_cfg["RPN"]["@num_class"] = 4
    network_cfg["RPN"]["@num_anchor_per_loc"] = 8
    params = {
        "classes_target": ["class1", "class2", "class3", "class4"],
        "classes_source": ["class1", "class2", "class3", "class4"],
        "model_resume_dict": {
            "ckpt_path": "unit_tests/data/train_class4-23200.tckpt",
            "num_classes": 4,
            "num_anchor_per_loc": 8,
            "partially_load_params": []
        },
        "sub_model_resume_dict": {
            "ckpt_path": "unit_tests/data/train_class4-23200.tckpt",
            "num_classes": 4,
            "num_anchor_per_loc": 8,
            "partially_load_params": []
        },
        "voxel_encoder_dict": network_cfg["VoxelEncoder"],
        "middle_layer_dict": network_cfg["MiddleLayer"],
        "rpn_dict": network_cfg["RPN"],
        "training_mode": "lwf",
        "is_training": True,
        "pos_cls_weight": 1.0,
        "neg_cls_weight": 1.0,
        "l2sp_alpha_coef": 2.0,
        "weight_decay_coef": 0.01,
        "delta_coef": 4.0,
        "ewc_coef": 2.0,
        "ewc_weights_path": "unit_tests/data/test_model-ewc-ewc_weights.pkl",
        "distillation_loss_cls_coef": 1.0,
        "distillation_loss_reg_coef": 1.0,
        "num_biased_select": 2,
        "loss_dict": {
            "ClassificationLoss":{
                "name": "SigmoidFocalClassificationLoss",
                "@alpha": 0.25,
                "@gamma": 2.0,
            },
            "LocalizationLoss":{
                "name": "WeightedSmoothL1LocalizationLoss",
                "@sigma": 3.0,
                "@code_weights": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "@codewise": True,
            },
            "DirectionLoss":{
                "name": "WeightedSoftmaxClassificationLoss",
            },
            "DistillationClassificationLoss":{
                "name": "WeightedSmoothL1LocalizationLoss",
                "@sigma": 1.3,
                "@code_weights": [4.0] * 2,
                "@codewise": True,
            },
            "DistillationRegressionLoss":{
                "name": "WeightedSmoothL1LocalizationLoss",
                "@sigma": 3.0,
                "@code_weights": [3.0] * 7,
                "@codewise": True,
            },
        },
        "hook_layers": [],
        "distillation_mode": ["ewc"],
        "bool_reuse_anchor_for_cls": False,
        "bool_biased_select_with_submodel": False
    }
    network = Network(**params).cuda()
    return network

class TestModel(torch.nn.Module):
    def __init__(self):
        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        super(TestModel, self).__init__()
        self.w = Parameter(torch.randn([2,3], requires_grad=True).float())
        self.b = Parameter(torch.randn([3,], requires_grad=True).float())
        self.act_fun = lambda x: x**2
    def forward(self, x):
        return self.act_fun(torch.matmul(x, self.w) + self.b)

class Test_compute_mas_weights(unittest.TestCase):
    def test_compute_omega_cls_term(self):
        from incdet3.models.mas_func import _compute_omega_cls_term, _compute_omega_reg_term
        x = torch.randn([5, 2]).cuda()
        model = TestModel().cuda()
        cls_preds = model(x)
        cls_term = _compute_omega_cls_term(cls_preds, model)
        reg_term = _compute_omega_reg_term(cls_preds, model)
        # # compute gt
        cls_preds = model(x)
        weight_w, weight_b = 0, 0
        model.zero_grad()
        target = 0
        for i in range(x.shape[0]):
            target += sum([cls_preds[i:i+1, j]**2 for j in range(3)])
        d_target_w = torch.autograd.grad(outputs=target, inputs=model.w, retain_graph=True, only_inputs=True)[0]
        d_target_b = torch.autograd.grad(outputs=target, inputs=model.b, retain_graph=True, only_inputs=True)[0]
        weight_w += d_target_w **2
        weight_b += d_target_b **2
        weight_w /= x.shape[0]
        weight_b /= x.shape[0]
        self.assertTrue(torch.allclose(cls_term["w"], weight_w))
        self.assertTrue(torch.allclose(cls_term["b"], weight_b))

    def test_compute_omega_reg_term(self):
        from incdet3.models.mas_func import _compute_omega_cls_term, _compute_omega_reg_term
        x = torch.randn([5, 2]).cuda()
        model = TestModel().cuda()
        reg_preds = model(x)
        cls_term = _compute_omega_cls_term(reg_preds, model)
        reg_term = _compute_omega_reg_term(reg_preds, model)
        # # compute gt
        reg_preds = model(x)
        weight_w, weight_b = 0, 0
        model.zero_grad()
        target = 0
        for i in range(x.shape[0]):
            target += sum([reg_preds[i:i+1, j]**2 for j in range(3)])
        d_target_w = torch.autograd.grad(outputs=target, inputs=model.w, retain_graph=True, only_inputs=True)[0]
        d_target_b = torch.autograd.grad(outputs=target, inputs=model.b, retain_graph=True, only_inputs=True)[0]
        weight_w += d_target_w **2
        weight_b += d_target_b **2
        weight_w /= x.shape[0]
        weight_b /= x.shape[0]
        self.assertTrue(torch.allclose(cls_term["w"], weight_w))
        self.assertTrue(torch.allclose(cls_term["b"], weight_b))

    def test_compute_mas_weights(self):
        network = build_network().cuda()
        dataloader = build_dataloader()
        num_of_datasamples = len(dataloader)
        num_of_anchorsamples = 15
        anchor_sample_strategy = "all"
        reg_coef = 0.1
        oldtask_omega_paths = []
        oldtask_omega_weights = []
        newtask_omega_weight = 1.0
        est_mas_dict = network.compute_mas_weights(
            dataloader,
            num_of_datasamples,
            num_of_anchorsamples,
            anchor_sample_strategy,
            reg_coef,
            oldtask_omega_paths,
            oldtask_omega_weights,
            newtask_omega_weight)
        for name, param in est_mas_dict["omega"].items():
            self.assertTrue(torch.all(param == est_mas_dict["new_omega"][name]))
            self.assertTrue(torch.all(param == est_mas_dict["new_clsterm"][name]
                + est_mas_dict["new_regterm"][name] * reg_coef))
        # write_pkl({name: param.cpu().numpy() for name, param in est_mas_dict["omega"].items()}, "unit_tests/data/test_model-mas-omega-class2-npy.pkl")
        gt_omega = read_pkl("unit_tests/data/test_model-mas-omega.pkl")
        for name, param in est_mas_dict["omega"].items():
            self.assertTrue(torch.allclose(param, gt_omega[name]))
        
    def test_compute_mas_weights_with_oldomegas(self):
        network = build_network_class4().cuda()
        dataloader = build_dataloader()
        num_of_datasamples = len(dataloader)
        num_of_anchorsamples = 15
        anchor_sample_strategy = "all"
        reg_coef = 0.1
        oldtask_omega_paths = ["unit_tests/data/test_model-mas-omega-class3-npy.pkl"]
        oldtask_omega_weights = [0.5]
        newtask_omega_weight = 0.9
        est_mas_dict = network.compute_mas_weights(
            dataloader,
            num_of_datasamples,
            num_of_anchorsamples,
            anchor_sample_strategy,
            reg_coef,
            oldtask_omega_paths,
            oldtask_omega_weights,
            newtask_omega_weight)
        for name, param in est_mas_dict["omega"].items():
            self.assertTrue(torch.all(est_mas_dict["new_omega"][name] ==
                est_mas_dict["new_clsterm"][name]
                + est_mas_dict["new_regterm"][name] * reg_coef))
        from incdet3.models.ewc_func import parse_numclasses_numanchorperloc, expand_old_weights
        gt_omega = read_pkl("unit_tests/data/test_model-mas-omega-class3-npy.pkl")
        num_new_classes, num_new_anchor_per_loc = parse_numclasses_numanchorperloc(est_mas_dict["new_omega"])
        num_old_classes, num_old_anchor_per_loc = parse_numclasses_numanchorperloc(gt_omega)
        for name, param in est_mas_dict["omega"].items():
            oldparam = expand_old_weights(name, torch.from_numpy(gt_omega[name]).cuda(),
                num_new_classes, num_new_anchor_per_loc,
                num_old_classes, num_old_anchor_per_loc)
            self.assertTrue(torch.allclose(param,
                oldtask_omega_weights[0] * oldparam +
                newtask_omega_weight * est_mas_dict["new_omega"][name]))

if __name__ == "__main__":
    unittest.main()
