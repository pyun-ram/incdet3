'''
 File Created: Sat Sep 19 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''
import os
import torch
import unittest
import numpy as np
from copy import deepcopy
from incdet3.models.model import Network
from torch.nn.parameter import Parameter
from incdet3.models.ewc_func import compute_FIM_v2, _init_ewc_weights, _cycle_next
from incdet3.builders.dataloader_builder import example_convert_to_torch
from det3.ops import write_pkl, read_pkl

torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class TestModel(torch.nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.w = Parameter(torch.randn([2,3], requires_grad=True).float())
        self.b = Parameter(torch.randn([3,], requires_grad=True).float())
        self.act_fun = lambda x: x**2
    def forward(self, x):
        return self.act_fun(torch.matmul(x, self.w) + self.b)
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
        "ewc_weights_path": "unit_tests/data/test_model-ewc-ewc_weights_v1.pkl",
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

class Test_compute_ewc_weights_v2(unittest.TestCase):
    def test_compute_FIM_v2(self):
        x = torch.randn([5, 2])
        model = TestModel()
        output = model(x)
        cls_target = torch.nn.functional.one_hot(torch.randint(high=3, size=(5,))).float()
        reg_target = torch.randn([5, 3])
        cls_loss_ftor = torch.nn.MSELoss(reduction="none")
        loss_cls = cls_loss_ftor(output, cls_target)
        loss_reg = reg_target - output
        loss_det = loss_cls.sum() + loss_reg.sum()
        loss_det.backward(retain_graph=True)
        gt = {}
        for name, param in model.named_parameters():
            gt[name] = param.grad**2

        est = compute_FIM_v2(loss_det, model)
        # any param of model should not have grad
        for param in model.parameters():
            self.assertTrue(param.grad.sum() == 0)
        for name, param in gt.items():
            self.assertTrue(torch.all(param == est[name]))

    def test_compute_ewc_weights_v2_newtaskFIM(self):
        network = build_network().cuda()
        dataloader = build_dataloader()
        num_of_datasamples = len(dataloader)
        est_FIM_dict = network.compute_ewc_weights_v2(dataloader,
            num_of_datasamples,
            oldtask_FIM_paths=[],
            oldtask_FIM_weights=[],
            newtask_FIM_weight=1.0)
        # newtaskFIM should be equal to FIM
        for name, param in network._model.named_parameters():
            self.assertTrue(torch.all(est_FIM_dict["newtask_FIM"][name] == est_FIM_dict["FIM"][name]))
        # compute gt
        newtask_FIM_list = []
        dataloader_itr = dataloader.__iter__()
        batch_size = dataloader.batch_size
        for data in dataloader:
            data = example_convert_to_torch(data,
                dtype=torch.float32, device=torch.device("cuda:0"))
            loss = network.forward(data)
            loss_det = loss["loss_cls"] + loss["loss_reg"]
            network._model.zero_grad()
            loss_det.backward()
            tmp_FIM = {}
            for name, param in network._model.named_parameters():
                tmp_FIM[name] = (param.grad **2 if param.grad is not None 
                    else torch.zeros(1).float().cuda())
            newtask_FIM_list.append(tmp_FIM)
        gt_FIM = newtask_FIM_list[0]
        for i in range(1, len(newtask_FIM_list)):
            for name, param in gt_FIM.items():
                gt_FIM[name] += newtask_FIM_list[i][name]
        for name, param in gt_FIM.items():
            gt_FIM[name] /= len(newtask_FIM_list)
            self.assertTrue(torch.allclose(gt_FIM[name],
                est_FIM_dict["FIM"][name], atol=1e-8, rtol=1e-4))
            
    def test_compute_ewc_weights_v2_oldandnewtasksFIM(self):
        network = build_network().cuda()
        dataloader = build_dataloader()
        num_of_datasamples = len(dataloader)
        est_FIM_dict = network.compute_ewc_weights_v2(dataloader,
            num_of_datasamples,
            oldtask_FIM_paths=["unit_tests/data/test_model-ewcv2-newtaskFIM.pkl"],
            oldtask_FIM_weights=[0.2],
            newtask_FIM_weight=1.2)
        # newtaskFIM should not be equal to FIM
        flag = True
        for name, param in network._model.named_parameters():
            flag = torch.all(est_FIM_dict["newtask_FIM"][name] == est_FIM_dict["FIM"][name])
            if not flag:
                break
        self.assertFalse(flag)
        # compute gt
        newtask_FIM_list = []
        dataloader_itr = dataloader.__iter__()
        batch_size = dataloader.batch_size
        for data in dataloader:
            data = example_convert_to_torch(data,
                dtype=torch.float32, device=torch.device("cuda:0"))
            loss = network.forward(data)
            loss_det = loss["loss_cls"] + loss["loss_reg"]
            network._model.zero_grad()
            loss_det.backward()
            tmp_FIM = {}
            for name, param in network._model.named_parameters():
                tmp_FIM[name] = (param.grad **2 if param.grad is not None 
                    else torch.zeros(1).float().cuda())
            newtask_FIM_list.append(tmp_FIM)
        gt_FIM = newtask_FIM_list[0]
        for i in range(1, len(newtask_FIM_list)):
            for name, param in gt_FIM.items():
                gt_FIM[name] += newtask_FIM_list[i][name]
        for name, pram in gt_FIM.items():
            gt_FIM[name] /= len(newtask_FIM_list)
        old_FIM = read_pkl("unit_tests/data/test_model-ewcv2-newtaskFIM.pkl")
        for name, param in gt_FIM.items():
            # self.assertTrue(torch.allclose(gt_FIM[name], est_FIM_dict["newtask_FIM"][name], atol=1e-8, rtol=1e-4))
            # print(name, est_FIM_dict["FIM"][name].sum(), (torch.from_numpy(old_FIM[name]*0.2).float().cuda() + gt_FIM[name]*1.2).sum())
            self.assertTrue(torch.allclose(est_FIM_dict["FIM"][name],
                torch.from_numpy(old_FIM[name]*0.2).float().cuda() + gt_FIM[name]*1.2,
                atol=1e-8, rtol=1e-4))

    def test_ewc_measure_distance_l2(self):
        from incdet3.models.ewc_func import ewc_measure_distance
        diff = torch.randn(9).reshape(3,3).cuda()
        loss_type = "l2"
        beta = 1
        weights = torch.randn(9).reshape(3,3).cuda()
        dist = ewc_measure_distance(diff, loss_type, beta, weights)
        self.assertTrue(torch.all(dist == 0.5*weights*diff**2))

    def test_ewc_measure_distance_huber(self):
        from incdet3.models.ewc_func import ewc_measure_distance
        diff = torch.arange(9).reshape(3,3).cuda()
        loss_type = "huber"
        beta = 5
        weights = torch.arange(9).reshape(3,3).cuda()
        dist = ewc_measure_distance(diff, "huber", beta, weights)
        self.assertTrue(torch.all(dist == torch.FloatTensor([0.0, 0.5, 4.0, 13.5, 27.5, 37.5, 47.5, 57.5, 67.5]).reshape(3,3).cuda()))

if __name__ == "__main__":
    unittest.main()