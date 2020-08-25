'''
 File Created: Sat Jun 20 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''

import torch
import unittest
import torch.nn as nn
import numpy as np
from incdet3.models.model import Network
from det3.utils.utils import load_pickle
from incdet3.utils.utils import bcolors
from copy import deepcopy

class Test_compute_loss(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_compute_loss, self).__init__(*args, **kwargs)
        network_cfg_template =  {
            "VoxelEncoder": {
                "name": "SimpleVoxel",
                "@num_input_features": 3,
            },
            "MiddleLayer":{
                "name": "SpMiddleFHD",
                "@use_norm": True,
                "@num_input_features": 3,
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
        self.params = {
            "classes_target": ["class1", "class2", "class3"],
            "classes_source": ["class1", "class2"],
            "model_resume_dict": {
                "ckpt_path": "./unit_tests/data/test_build_model_and_init_weight_ResNetRPN_3.tckpt",
                "num_classes": 3,
                "num_anchor_per_loc": 6,
                "partially_load_params": []
            },
            "sub_model_resume_dict": {
                "ckpt_path": "./unit_tests/data/test_build_model_and_init_weight_ResNetRPN_2.tckpt",
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
            "hook_layers": ["rpn.deblocks.0.2"],
            "distillation_mode": ["l2sp", "delta", "distillation_loss"],
            "bool_reuse_anchor_for_cls": False,
            "bool_biased_select_with_submodel": False
        }
        self.network = Network(**self.params).cuda()
        self.data = load_pickle("./unit_tests/data/test_build_model_and_init_data.pkl")

    def test_distillation_loss_weights3(self):
        '''
        network._distillation_loss_sampling_strategy = "biased"
        '''
        network = self.network
        data = self.data
        network._distillation_loss_sampling_strategy = "biased"
        network._num_biased_select = 4
        preds_dict = network._network_forward(network._model,
            data["voxels"],
            data["num_points"],
            data["coordinates"],
            data["anchors"].shape[0])
        preds_dict_sub = network._network_forward(network._sub_model,
            data["voxels"],
            data["num_points"],
            data["coordinates"],
            data["anchors"].shape[0])
        preds_dict_sub["cls_preds"][:, :, 0, 0, 0] += 1000
        org_shape = preds_dict["cls_preds"].shape
        loss = network._compute_distillation_loss(preds_dict, preds_dict_sub)
        gt_weights = torch.LongTensor(
            [[0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 2, 0, 0],
            [0, 3, 0, 0]]).cuda()
        self.assertTrue(torch.all(loss["weights"].reshape(1, 4, *org_shape[2:-1]).nonzero() == gt_weights))

        preds_dict = network._network_forward(network._model,
            data["voxels"],
            data["num_points"],
            data["coordinates"],
            data["anchors"].shape[0])
        preds_dict_sub = network._network_forward(network._sub_model,
            data["voxels"],
            data["num_points"],
            data["coordinates"],
            data["anchors"].shape[0])
        preds_dict["cls_preds"][:, :, 0, 0, 0] += 1000
        org_shape = preds_dict["cls_preds"].shape
        loss = network._compute_distillation_loss(preds_dict, preds_dict_sub)
        gt_weights = torch.LongTensor(
            [[0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 2, 0, 0],
            [0, 3, 0, 0]]).cuda()
        self.assertFalse(torch.all(loss["weights"].reshape(1, 4, *org_shape[2:-1]).nonzero() == gt_weights))

    def test_distillation_loss_weights4(self):
        '''
        network._distillation_loss_sampling_strategy = "unbiased"
        '''
        network = self.network
        data = self.data
        network._distillation_loss_sampling_strategy = "unbiased"
        network._num_biased_select = 4
        preds_dict = network._network_forward(network._model,
            data["voxels"],
            data["num_points"],
            data["coordinates"],
            data["anchors"].shape[0])
        preds_dict_sub = network._network_forward(network._sub_model,
            data["voxels"],
            data["num_points"],
            data["coordinates"],
            data["anchors"].shape[0])
        preds_dict_sub["cls_preds"][:, :, 0, 0, 0] += 1000
        org_shape = preds_dict["cls_preds"].shape
        loss = network._compute_distillation_loss(preds_dict, preds_dict_sub)
        weight1 = loss["weights"].reshape(1, 4, *org_shape[2:-1]).nonzero()
        loss = network._compute_distillation_loss(preds_dict, preds_dict_sub)
        weight2 = loss["weights"].reshape(1, 4, *org_shape[2:-1]).nonzero()
        self.assertFalse(torch.all(weight1 == weight2))

    def test_distillation_loss_weights5(self):
        '''
        network._distillation_loss_sampling_strategy = "all"
        '''
        network = self.network
        data = self.data
        network._distillation_loss_sampling_strategy = "all"
        network._num_biased_select = 4
        preds_dict = network._network_forward(network._model,
            data["voxels"],
            data["num_points"],
            data["coordinates"],
            data["anchors"].shape[0])
        preds_dict_sub = network._network_forward(network._sub_model,
            data["voxels"],
            data["num_points"],
            data["coordinates"],
            data["anchors"].shape[0])
        preds_dict_sub["cls_preds"][:, :, 0, 0, 0] += 1000
        org_shape = preds_dict["cls_preds"].shape
        loss = network._compute_distillation_loss(preds_dict, preds_dict_sub)
        weight = loss["weights"]
        self.assertTrue(weight.sum() == weight.shape[1])

    def test_distillation_loss_weights6(self):
        '''
        network._distillation_loss_sampling_strategy = "threshold"
        '''
        network = self.network
        data = self.data
        network._distillation_loss_sampling_strategy = "threshold"
        network._num_biased_select = 4
        preds_dict = network._network_forward(network._model,
            data["voxels"],
            data["num_points"],
            data["coordinates"],
            data["anchors"].shape[0])
        preds_dict_sub = network._network_forward(network._sub_model,
            data["voxels"],
            data["num_points"],
            data["coordinates"],
            data["anchors"].shape[0])
        preds_dict_sub["cls_preds"][:, :, 0, 0, 0] += 1000
        org_shape = preds_dict["cls_preds"].shape
        loss = network._compute_distillation_loss(preds_dict, preds_dict_sub)
        weight = loss["weights"]
        score = preds_dict_sub["cls_preds"].max(dim=-1)[0].flatten()
        score[score<=0.1] = 0
        score[score>0.1] = 1
        self.assertTrue(score.sum() == weight.sum())
        self.assertTrue(score.sum() != weight.shape[1])

if __name__ == "__main__":
    unittest.main()