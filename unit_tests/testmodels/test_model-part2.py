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
        network_cfg = network_cfg_template.copy()
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
            "bool_oldclass_use_newanchor_for_cls": False,
            "bool_biased_select_with_submodel": False
        }
        self.network = Network(**self.params).cuda()
        self.data = load_pickle("./unit_tests/data/test_build_model_and_init_data.pkl")

    def test_classification_loss(self):
        '''
        compare with hand-computed result
        '''
        # @est: [batch_size, num_anchors_per_loc, H, W, num_classes] logits
        # @gt: [batch_size, num_anchors, 1] int32
        # @weights: [batch_size, num_anchors]
        batch_size = 2
        num_anchors_per_loc = 6
        H = 2
        W = 2
        num_anchors = num_anchors_per_loc * W * H
        num_classes = 3
        alpha = 0.25
        gamma = 2.0
        # create est
        est = torch.randn([batch_size, num_anchors_per_loc, H, W, num_classes]).float().cuda()
        # create gt
        gt = torch.randint(low=0, high=num_classes, size=[batch_size, num_anchors, 1]).long().cuda()
        # create weight
        weights = torch.randn([batch_size, num_anchors]).float().cuda()
        # compute loss
        loss = self.network._compute_classification_loss(est, gt, weights)
        # compare with hand-computed results
        hc_logits = est.reshape(batch_size, -1, num_classes)
        hc_est = torch.sigmoid(hc_logits)
        hc_gt = gt.reshape(batch_size, -1)
        hc_gt = nn.functional.one_hot(hc_gt, num_classes=num_classes+1).float()[..., 1:]
        hc_weights = weights.reshape(batch_size, -1, 1).repeat([1,1,3])
        # compute_entropy
        hc_xe_term = (-1 * hc_gt * torch.log(hc_est) -
            (1-hc_gt) * torch.log(1 - hc_est))
        hc_alpha_weight_factor = hc_gt * alpha + (1-hc_gt) * (1-alpha)
        hc_pt = hc_est * hc_gt + (1-hc_est) * (1-hc_gt)
        hc_modulating_factor = torch.pow(1.0 - hc_pt, gamma)
        hc_loss = hc_xe_term * hc_alpha_weight_factor * hc_modulating_factor * hc_weights
        self.assertTrue(torch.allclose(hc_loss, loss))

    def test_location_loss(self):
        '''
        @est: [batch_size, num_anchors_per_loc, H, W, 7]
        @gt: [batch_size, num_anchors, 7]
        @weights: [batch_size, num_anchors]
        '''
        batch_size = 2
        num_anchors_per_loc = 6
        H = 2
        W = 3
        box_code_size = 7
        num_anchors = num_anchors_per_loc * W * H
        sigmma = 3
        code_weights = self.params["loss_dict"]["LocalizationLoss"]["@code_weights"]
        # create est
        est = torch.randn([batch_size, num_anchors_per_loc, H, W, box_code_size]).float().cuda() + 3
        # create gt
        gt = torch.randn([batch_size, num_anchors, box_code_size]).float().cuda()
        # create weight
        weights = torch.randn([batch_size, num_anchors]).float().cuda() * 0.5
        # compute loss
        loss = self.network._compute_location_loss(est, gt, weights)
        hc_est = est.view(batch_size, -1, box_code_size)
        hc_gt = gt
        rot_est = hc_est[..., -1:].clone()
        rot_gt = hc_gt[..., -1:].clone()
        hc_est[..., -1:] = torch.sin(rot_est) * torch.cos(rot_gt)
        hc_gt[..., -1:] = torch.cos(rot_est) * torch.sin(rot_gt)
        hc_weights = weights.unsqueeze(-1).repeat([1,1,box_code_size])
        hc_code_weights = torch.FloatTensor(code_weights).cuda().repeat(batch_size, num_anchors, 1)
        # diff
        hc_diff = torch.abs(hc_est - hc_gt)
        hc_diff *= hc_code_weights
        hc_diff_lt_1 = torch.le(hc_diff, 1 / sigmma ** 2 ).type_as(hc_diff)
        # smooth_l1_loss_term
        hc_smooth_l1_loss_term = (hc_diff_lt_1 * 0.5 * torch.pow(hc_diff * sigmma, 2)
            +  (1. - hc_diff_lt_1) * (hc_diff - 0.5 / (sigmma ** 2)))
        hc_loss = hc_smooth_l1_loss_term * hc_weights
        self.assertTrue(torch.allclose(hc_loss, loss))

    def test_direction_loss(self):
        '''
        @est: [batch_size, num_anchors_per_loc, H, W, 2] logits
        @gt: [batch_size, num_anchors, 1] int32
        @weights: [batch_size, num_anchors] float32
        '''
        batch_size = 2
        num_anchors_per_loc = 6
        H = 2
        W = 3
        box_code_size = 7
        num_anchors = num_anchors_per_loc * W * H
        # create est
        est = torch.randn([batch_size, num_anchors_per_loc, H, W, 2]).float().cuda()
        # create gt
        gt = torch.randint(low=0, high=2, size=[batch_size, num_anchors, 1]).long().cuda()
        gt = gt.reshape(batch_size, -1)
        gt = nn.functional.one_hot(gt, num_classes=2).float()
        # create weight
        weights = torch.randn([batch_size, num_anchors]).float().cuda() * 0.5
        # compute loss
        loss = self.network._compute_direction_loss(est, gt, weights)
        hc_est = est.clone()
        hc_est = torch.nn.functional.softmax(hc_est, dim=-1)
        hc_est = hc_est.view(batch_size, -1, 2)
        hc_gt = gt
        hc_xe_term = (-1 * hc_gt * torch.log(hc_est) -
            (1-hc_gt) * torch.log(1 - hc_est)).mean(-1)
        hc_weights = weights
        hc_loss = hc_xe_term * hc_weights
        self.assertTrue(torch.allclose(hc_loss, loss))

    def test_l2sp_loss(self):
        loss_alpha = self.network._compute_l2sp_loss()
        loss_beta = self.network._compute_l2_loss()
        loss = loss_alpha + loss_beta
        hc_loss = 0
        hc_loss_beta = 0
        alpha = 2.0
        beta = 0.01
        num_new_classes = len(self.params["classes_target"])
        num_old_classes = len(self.params["classes_source"])
        num_new_anchor_per_loc = 2 * num_new_classes
        num_old_anchor_per_loc = 2 * num_old_classes
        head_name = ["rpn.conv_cls", "rpn.conv_box", "rpn.conv_dir_cls"]
        submodle_sd = self.network._sub_model.state_dict()
        for name, param in self.network._model.named_parameters():
            is_head = any([name.startswith(head_name_) for head_name_ in head_name])
            if not is_head:
                hc_loss += 0.5 * torch.norm(param-submodle_sd[name].detach(), 2) ** 2
            else:
                compute_param_shape = param.shape
                if name.startswith("rpn.conv_cls"):
                    compute_param = param.reshape(num_new_anchor_per_loc, num_new_classes, *compute_param_shape[1:])
                    compute_oldparam = (compute_param[:num_old_anchor_per_loc, :num_old_classes, ...]
                        .reshape(-1, *compute_param_shape[1:]).contiguous())
                    # new classes
                    compute_newparam = (compute_param[:, num_old_classes:, ...]
                        .reshape(-1, *compute_param_shape[1:]).contiguous())
                    # old classes with new anchors
                    compute_newparam_ = (compute_param[num_old_anchor_per_loc:, :num_old_classes, ...]
                        .reshape(-1, *compute_param_shape[1:]).contiguous())
                    compute_newparam = torch.cat([compute_newparam, compute_newparam_], dim=0)
                elif name.startswith("rpn.conv_box"):
                    compute_param = param.reshape(num_new_anchor_per_loc, 7, *compute_param_shape[1:])
                    compute_oldparam = (compute_param[:num_old_anchor_per_loc, ...]
                        .reshape(-1, *compute_param_shape[1:]).contiguous())
                    compute_newparam = (compute_param[num_old_anchor_per_loc:, ...]
                        .reshape(-1, *compute_param_shape[1:]).contiguous())
                elif name.startswith("rpn.conv_dir_cls"):
                    compute_param = param.reshape(num_new_anchor_per_loc, 2, *compute_param_shape[1:])
                    compute_oldparam = (compute_param[:num_old_anchor_per_loc, ...]
                        .reshape(-1, *compute_param_shape[1:]).contiguous())
                    compute_newparam = (compute_param[num_old_anchor_per_loc:, ...]
                        .reshape(-1, *compute_param_shape[1:]).contiguous())
                else:
                    raise NotImplementedError
                hc_loss += 0.5 * torch.norm(compute_oldparam-submodle_sd[name].detach(), 2) ** 2
                hc_loss_beta += 0.5 * torch.norm(compute_newparam, 2) ** 2
        hc_loss = alpha * hc_loss + beta * hc_loss_beta
        self.assertTrue(torch.all(hc_loss == loss))

    def test_delta_loss_woatten(self):
        network = self.network
        data = self.data
        delta_coef = self.params["delta_coef"]
        network._hook_features_model.clear()
        network._hook_features_submodel.clear()
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
        loss = network._compute_delta_loss(None)
        hc_feat = network._hook_features_model
        hc_feat_sub = network._hook_features_submodel
        hc_loss = 0
        for feat_, feat_sub_ in zip(hc_feat, hc_feat_sub):
            assert feat_.shape == feat_sub_.shape
            hc_loss += 0.5 * torch.norm(feat_ - feat_sub_.detach()) ** 2
        hc_loss *= delta_coef
        self.assertTrue(torch.allclose(hc_loss, loss))
        network._hook_features_model.clear()
        network._hook_features_submodel.clear()

    def test_delta_loss_atten(self):
        print(bcolors.WARNING+ "test_delta_loss_atten() TBD" + bcolors.ENDC)

    def test_distillation_loss_weights1(self):
        '''
        network._bool_biased_select_with_submodel = True
        '''
        network = self.network
        data = self.data
        network._num_biased_select = 4
        network._bool_biased_select_with_submodel = True
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

    def test_distillation_loss_weights2(self):
        '''
        network._bool_biased_select_with_submodel = False
        '''
        network = self.network
        data = self.data
        network._num_biased_select = 4
        network._bool_biased_select_with_submodel = False
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
        self.assertFalse(torch.all(loss["weights"].reshape(1, 4, *org_shape[2:-1]).nonzero() == gt_weights))

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
        self.assertTrue(torch.all(loss["weights"].reshape(1, 4, *org_shape[2:-1]).nonzero() == gt_weights))

    def test_distillation_loss_cls(self):
        '''
        network._bool_biased_select_with_submodel = True
        '''
        network = self.network
        data = self.data
        network._num_biased_select = 4
        network._bool_biased_select_with_submodel = True
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
        preds_dict_sub["cls_preds"][:, :, 0, 0, 0] = 300
        preds_dict["cls_preds"][:, :, 0, 0, 0] = 0
        org_shape = preds_dict["cls_preds"].shape
        loss = network._compute_distillation_loss(preds_dict, preds_dict_sub)
        weights = loss["weights"]
        preds_dict_sub["cls_preds"][:, :, 0, 0, 0] = 300
        preds_dict["cls_preds"][:, :, 0, 0, 0] = 0

        code_weights = self.params["loss_dict"]["DistillationClassificationLoss"]["@code_weights"]
        sigmma = self.params["loss_dict"]["DistillationClassificationLoss"]["@sigma"]
        num_old_classes = network._num_old_classes
        num_new_classes = network._num_new_classes
        num_old_anchor_per_loc = network._num_old_anchor_per_loc
        num_new_anchor_per_loc = network._num_new_anchor_per_loc
        batch_size = preds_dict["cls_preds"].shape[0]
        hc_est = preds_dict["cls_preds"][:, :num_old_anchor_per_loc, ..., :num_old_classes].view(batch_size, -1, num_old_classes)

        num_anchors = hc_est.shape[1]
        hc_gt = preds_dict_sub["cls_preds"].view(batch_size, -1, num_old_classes)
        hc_weights = weights.unsqueeze(-1).repeat([1,1,num_old_classes])
        hc_code_weights = torch.FloatTensor(code_weights).cuda().repeat(batch_size, num_anchors, 1)
        # diff
        hc_diff = torch.abs(hc_est - hc_gt)
        hc_diff *= hc_code_weights
        hc_diff_lt_1 = torch.le(hc_diff, 1 / sigmma ** 2 ).type_as(hc_diff)
        # smooth_l1_loss_term
        hc_smooth_l1_loss_term = (hc_diff_lt_1 * 0.5 * torch.pow(hc_diff * sigmma, 2)
            +  (1. - hc_diff_lt_1) * (hc_diff - 0.5 / (sigmma ** 2)))
        hc_loss = hc_smooth_l1_loss_term * hc_weights
        self.assertTrue(torch.all(hc_loss == loss["loss_distillation_loss_cls"]))

    def test_distillation_loss_reg(self):
        '''
        network._bool_biased_select_with_submodel = True
        '''
        network = self.network
        data = self.data
        network._num_biased_select = 4
        network._bool_biased_select_with_submodel = True
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
        preds_dict_sub["cls_preds"][:, :, 0, 0, 0] = 300
        preds_dict["cls_preds"][:, :, 0, 0, 0] = 0
        preds_dict_sub["box_preds"][:, :, 0, 0, :] = 666
        preds_dict["box_preds"][:, :, 0, 0, :] = 0
        org_shape = preds_dict["cls_preds"].shape
        loss = network._compute_distillation_loss(preds_dict, preds_dict_sub)
        weights = loss["weights"]
        preds_dict_sub["cls_preds"][:, :, 0, 0, 0] = 300
        preds_dict["cls_preds"][:, :, 0, 0, 0] = 0
        preds_dict_sub["box_preds"][:, :, 0, 0, :] = 666
        preds_dict["box_preds"][:, :, 0, 0, :] = 0

        code_weights = self.params["loss_dict"]["DistillationRegressionLoss"]["@code_weights"]
        sigmma = self.params["loss_dict"]["DistillationRegressionLoss"]["@sigma"]
        num_old_classes = network._num_old_classes
        num_new_classes = network._num_new_classes
        num_old_anchor_per_loc = network._num_old_anchor_per_loc
        num_new_anchor_per_loc = network._num_new_anchor_per_loc
        batch_size = preds_dict["box_preds"].shape[0]
        hc_est = preds_dict["box_preds"][:, :num_old_anchor_per_loc, ...].view(batch_size, -1, 7)

        num_anchors = hc_est.shape[1]
        hc_gt = preds_dict_sub["box_preds"].view(batch_size, -1, 7)
        hc_weights = weights.unsqueeze(-1).repeat([1,1,7])
        hc_code_weights = torch.FloatTensor(code_weights).cuda().repeat(batch_size, num_anchors, 1)
        # diff
        hc_diff = torch.abs(hc_est - hc_gt)
        hc_diff *= hc_code_weights
        hc_diff_lt_1 = torch.le(hc_diff, 1 / sigmma ** 2 ).type_as(hc_diff)
        # smooth_l1_loss_term
        hc_smooth_l1_loss_term = (hc_diff_lt_1 * 0.5 * torch.pow(hc_diff * sigmma, 2)
            +  (1. - hc_diff_lt_1) * (hc_diff - 0.5 / (sigmma ** 2)))
        hc_loss = hc_smooth_l1_loss_term * hc_weights
        self.assertTrue(torch.all(hc_loss == loss["loss_distillation_loss_reg"]))

class Test_compute_l2_loss(unittest.TestCase):
    def test_l2_loss_train_from_scratch(self):
        '''
        test case: train_from_scratch
        '''
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
        network_cfg = network_cfg_template.copy()
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
            "training_mode": "train_from_scratch",
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
            "bool_oldclass_use_newanchor_for_cls": False,
            "bool_biased_select_with_submodel": False
        }
        self.network = Network(**self.params).cuda()
        self.data = load_pickle("./unit_tests/data/test_build_model_and_init_data.pkl")
        loss = self.network._compute_l2_loss()
        hc_loss = 0
        hc_weight_decay_coef = 0.01
        for name, param in self.network._model.named_parameters():
            hc_loss += 0.5 * torch.norm(param) ** 2
        hc_loss = hc_loss * hc_weight_decay_coef
        self.assertTrue(torch.all(hc_loss == loss))

    def test_l2_loss_has_l2sp(self):
        '''
        test case: not train_from_scratch (has_l2sp)
        '''
        '''
        test case: train_from_scratch
        '''
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
        network_cfg = network_cfg_template.copy()
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
            "training_mode": "feature_extraction",
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
            "bool_oldclass_use_newanchor_for_cls": False,
            "bool_biased_select_with_submodel": False
        }
        self.network = Network(**self.params).cuda()
        self.data = load_pickle("./unit_tests/data/test_build_model_and_init_data.pkl")
        loss = self.network._compute_l2_loss()
        hc_loss = 0
        hc_weight_decay_coef = 0.01
        num_old_classes = self.network._num_old_classes
        num_old_anchor_per_loc = self.network._num_old_anchor_per_loc
        num_new_classes = self.network._num_new_classes
        num_new_anchor_per_loc = self.network._num_new_anchor_per_loc
        for name, param in self.network._model.named_parameters():
            compute_param_shape = param.shape
            if name.startswith("rpn.conv_cls"):
                compute_param = param.reshape(num_new_anchor_per_loc, num_new_classes, *compute_param_shape[1:])
                compute_oldparam = (compute_param[:num_old_anchor_per_loc, :num_old_classes, ...]
                    .reshape(-1, *compute_param_shape[1:]).contiguous())
                # new classes
                compute_newparam = (compute_param[:, num_old_classes:, ...]
                    .reshape(-1, *compute_param_shape[1:]).contiguous())
                # old classes with new anchors
                compute_newparam_ = (compute_param[num_old_anchor_per_loc:, :num_old_classes, ...]
                    .reshape(-1, *compute_param_shape[1:]).contiguous())
                compute_newparam = torch.cat([compute_newparam, compute_newparam_], dim=0)
            elif name.startswith("rpn.conv_box"):
                compute_param = param.reshape(num_new_anchor_per_loc, 7, *compute_param_shape[1:])
                compute_oldparam = (compute_param[:num_old_anchor_per_loc, ...]
                    .reshape(-1, *compute_param_shape[1:]).contiguous())
                compute_newparam = (compute_param[num_old_anchor_per_loc:, ...]
                    .reshape(-1, *compute_param_shape[1:]).contiguous())
            elif name.startswith("rpn.conv_dir_cls"):
                compute_param = param.reshape(num_new_anchor_per_loc, 2, *compute_param_shape[1:])
                compute_oldparam = (compute_param[:num_old_anchor_per_loc, ...]
                    .reshape(-1, *compute_param_shape[1:]).contiguous())
                compute_newparam = (compute_param[num_old_anchor_per_loc:, ...]
                    .reshape(-1, *compute_param_shape[1:]).contiguous())
            else:
                continue
            hc_loss += 0.5 * torch.norm(compute_newparam, 2) ** 2
        hc_loss = hc_loss * hc_weight_decay_coef
        self.assertTrue(torch.all(hc_loss == loss))


    def test_l2_loss_not_have_l2sp(self):
        '''
        test case: not train_from_scratch (has_l2sp)
        '''
        '''
        test case: train_from_scratch
        '''
        for training_scheme in ["feature_extraction", "joint_training", "lwf", "fine_tuning"]:
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
            network_cfg = network_cfg_template.copy()
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
                "training_mode": training_scheme,
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
                "distillation_mode": ["delta", "distillation_loss"],
                "bool_oldclass_use_newanchor_for_cls": False,
                "bool_biased_select_with_submodel": False
            }
            self.network = Network(**self.params).cuda()
            self.data = load_pickle("./unit_tests/data/test_build_model_and_init_data.pkl")
            loss = self.network._compute_l2_loss()
            hc_loss = 0
            hc_weight_decay_coef = 0.01
            num_old_classes = self.network._num_old_classes
            num_old_anchor_per_loc = self.network._num_old_anchor_per_loc
            num_new_classes = self.network._num_new_classes
            num_new_anchor_per_loc = self.network._num_new_anchor_per_loc
            for name, param in self.network._model.named_parameters():
                compute_param_shape = param.shape
                if name.startswith("rpn.conv_cls"):
                    compute_param = param.reshape(num_new_anchor_per_loc, num_new_classes, *compute_param_shape[1:])
                    compute_oldparam = (compute_param[:num_old_anchor_per_loc, :num_old_classes, ...]
                        .reshape(-1, *compute_param_shape[1:]).contiguous())
                    # new classes
                    compute_newparam = (compute_param[:, num_old_classes:, ...]
                        .reshape(-1, *compute_param_shape[1:]).contiguous())
                    # old classes with new anchors
                    compute_newparam_ = (compute_param[num_old_anchor_per_loc:, :num_old_classes, ...]
                        .reshape(-1, *compute_param_shape[1:]).contiguous())
                    compute_newparam = torch.cat([compute_newparam, compute_newparam_], dim=0)
                elif name.startswith("rpn.conv_box"):
                    compute_param = param.reshape(num_new_anchor_per_loc, 7, *compute_param_shape[1:])
                    compute_oldparam = (compute_param[:num_old_anchor_per_loc, ...]
                        .reshape(-1, *compute_param_shape[1:]).contiguous())
                    compute_newparam = (compute_param[num_old_anchor_per_loc:, ...]
                        .reshape(-1, *compute_param_shape[1:]).contiguous())
                elif name.startswith("rpn.conv_dir_cls"):
                    compute_param = param.reshape(num_new_anchor_per_loc, 2, *compute_param_shape[1:])
                    compute_oldparam = (compute_param[:num_old_anchor_per_loc, ...]
                        .reshape(-1, *compute_param_shape[1:]).contiguous())
                    compute_newparam = (compute_param[num_old_anchor_per_loc:, ...]
                        .reshape(-1, *compute_param_shape[1:]).contiguous())
                else:
                    compute_newparam = param if param.requires_grad else torch.zeros(0)
                hc_loss += 0.5 * torch.norm(compute_newparam, 2) ** 2
            hc_loss = hc_loss * hc_weight_decay_coef
            self.assertTrue(torch.all(hc_loss == loss))

if __name__ == "__main__":
    unittest.main()