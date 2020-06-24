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

class Test_build_model_and_init(unittest.TestCase):
    '''
    @related data
    test_build_model_and_init_data.pkl: data
    test_build_model_and_init_weight_{rpn_name}_{num_classes}.tckpt: weights
        (empty num_classes if only one class)
    test_build_model_and_init_output_{rpn_name}_{num_classes}.pkl: network outputs
        (empty num_classes if only one class)
    @test cases:
    single class, rpnv2&resnet, no resume
    single class, rpnv2&resnet, resume single class
    multi classes (>=3), rpnv2&resnet, no resume
    multi classes (=3), rpnv2&resnet, resume class(0,1,2)
    multi classes (=3), rpnv2&resnet, resume class(0,1)
    multi classes (=3), rpnv2&resnet, resume class(0)
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

    def test_singleclass_noresume(self):
        '''
        single class, rpnv2&resnet, no resume
        '''
        torch.cuda.empty_cache()
        for rpn_name in ["RPNV2", "ResNetRPN"]:
            network_cfg = Test_build_model_and_init.network_cfg_template.copy()
            network_cfg["RPN"]["name"] = rpn_name
            network_cfg["RPN"]["@num_class"] = 1
            network_cfg["RPN"]["@num_anchor_per_loc"] = 2
            params = {
                "classes": ["class1"],
                "network_cfg": network_cfg,
                "resume_dict": None,
                "name": Test_build_model_and_init.name_template
            }
            model = Network._build_model_and_init(**params).cuda()
            self.assertTrue(all([itm in model.keys() for itm in ["vfe_layer", "middle_layer", "rpn"]]))
            self.assertTrue(model.rpn.__class__.__name__ == rpn_name)
            data = load_pickle("./unit_tests/data/test_build_model_and_init_data.pkl")
            voxels = data["voxels"]
            num_points = data["num_points"]
            coors = data["coordinates"]
            batch_size = data["anchors"].shape[0]
            voxel_features = model["vfe_layer"](voxels, num_points,coors)
            spatial_features = model["middle_layer"](voxel_features, coors, batch_size)
            preds_dict = model["rpn"](spatial_features)
            self.assertTrue(preds_dict["cls_preds"].shape[1]
                == network_cfg["RPN"]["@num_anchor_per_loc"])
            self.assertTrue(preds_dict["cls_preds"].shape[-1]
                == network_cfg["RPN"]["@num_class"])
            self.assertTrue(preds_dict["box_preds"].shape[1]
                == network_cfg["RPN"]["@num_anchor_per_loc"])
            self.assertTrue(preds_dict["box_preds"].shape[-1]
                == network_cfg["RPN"]["@box_code_size"])
            self.assertTrue(preds_dict["dir_cls_preds"].shape[1]
                == network_cfg["RPN"]["@num_anchor_per_loc"])
            self.assertTrue(preds_dict["dir_cls_preds"].shape[-1]
                == network_cfg["RPN"]["@num_direction_bins"])
            # Network.save_weight(model, save_dir=f"./unit_tests/data/{rpn_name}", itr=1)
            # from det3.utils.utils import save_pickle
            # save_pickle(preds_dict, f"./unit_tests/data/{rpn_name}/test_build_model_and_init_output.pkl")

    def test_singleclass_resume(self):
        '''
        single class, rpnv2&resnet, resume single class
        '''
        for rpn_name in ["RPNV2", "ResNetRPN"]:
            network_cfg = Test_build_model_and_init.network_cfg_template.copy()
            network_cfg["RPN"]["name"] = rpn_name
            network_cfg["RPN"]["@num_class"] = 1
            network_cfg["RPN"]["@num_anchor_per_loc"] = 2
            resume_dict = {
                "ckpt_path": f"unit_tests/data/test_build_model_and_init_weight_{rpn_name}.tckpt",
                "num_classes": 1,
                "num_anchor_per_loc": 2,
                "partially_load_params": []
            }
            params = {
                "classes": ["class1"],
                "network_cfg": network_cfg,
                "resume_dict": resume_dict,
                "name": Test_build_model_and_init.name_template
            }
            model = Network._build_model_and_init(**params).cuda()
            self.assertTrue(all([itm in model.keys() for itm in ["vfe_layer", "middle_layer", "rpn"]]))
            self.assertTrue(model.rpn.__class__.__name__ == rpn_name)
            data = load_pickle("./unit_tests/data/test_build_model_and_init_data.pkl")
            voxels = data["voxels"]
            num_points = data["num_points"]
            coors = data["coordinates"]
            batch_size = data["anchors"].shape[0]
            voxel_features = model["vfe_layer"](voxels, num_points,coors)
            spatial_features = model["middle_layer"](voxel_features, coors, batch_size)
            preds_dict = model["rpn"](spatial_features)
            self.assertTrue(preds_dict["cls_preds"].shape[1]
                == network_cfg["RPN"]["@num_anchor_per_loc"])
            self.assertTrue(preds_dict["cls_preds"].shape[-1]
                == network_cfg["RPN"]["@num_class"])
            self.assertTrue(preds_dict["box_preds"].shape[1]
                == network_cfg["RPN"]["@num_anchor_per_loc"])
            self.assertTrue(preds_dict["box_preds"].shape[-1]
                == network_cfg["RPN"]["@box_code_size"])
            self.assertTrue(preds_dict["dir_cls_preds"].shape[1]
                == network_cfg["RPN"]["@num_anchor_per_loc"])
            self.assertTrue(preds_dict["dir_cls_preds"].shape[-1]
                == network_cfg["RPN"]["@num_direction_bins"])
            gt = load_pickle(f"unit_tests/data/test_build_model_and_init_output_{rpn_name}.pkl")
            for k, v in preds_dict.items():
                self.assertTrue(torch.all(torch.eq(v, gt[k])))

    def test_multiclasses_noresume(self):
        '''
        multi class (>=3), rpnv2&resnet, no resume
        '''
        for rpn_name in ["RPNV2", "ResNetRPN"]:
            network_cfg = Test_build_model_and_init.network_cfg_template.copy()
            network_cfg["RPN"]["name"] = rpn_name
            network_cfg["RPN"]["@num_class"] = 2
            network_cfg["RPN"]["@num_anchor_per_loc"] = 4
            params = {
                "classes": ["class1", "class2"],
                "network_cfg": network_cfg,
                "resume_dict": None,
                "name": Test_build_model_and_init.name_template
            }
            model = Network._build_model_and_init(**params).cuda()
            self.assertTrue(all([itm in model.keys() for itm in ["vfe_layer", "middle_layer", "rpn"]]))
            self.assertTrue(model.rpn.__class__.__name__ == rpn_name)
            data = load_pickle("./unit_tests/data/test_build_model_and_init_data.pkl")
            voxels = data["voxels"]
            num_points = data["num_points"]
            coors = data["coordinates"]
            batch_size = data["anchors"].shape[0]
            voxel_features = model["vfe_layer"](voxels, num_points,coors)
            spatial_features = model["middle_layer"](voxel_features, coors, batch_size)
            preds_dict = model["rpn"](spatial_features)
            self.assertTrue(preds_dict["cls_preds"].shape[1]
                == network_cfg["RPN"]["@num_anchor_per_loc"])
            self.assertTrue(preds_dict["cls_preds"].shape[-1]
                == network_cfg["RPN"]["@num_class"])
            self.assertTrue(preds_dict["box_preds"].shape[1]
                == network_cfg["RPN"]["@num_anchor_per_loc"])
            self.assertTrue(preds_dict["box_preds"].shape[-1]
                == network_cfg["RPN"]["@box_code_size"])
            self.assertTrue(preds_dict["dir_cls_preds"].shape[1]
                == network_cfg["RPN"]["@num_anchor_per_loc"])
            self.assertTrue(preds_dict["dir_cls_preds"].shape[-1]
                == network_cfg["RPN"]["@num_direction_bins"])
            # Network.save_weight(model, save_dir=f"./unit_tests/data/{rpn_name}", itr=2)
            # from det3.utils.utils import save_pickle
            # save_pickle(preds_dict, f"./unit_tests/data/{rpn_name}/test_build_model_and_init_output.pkl")

    def test_multiclasses_resume1(self):
        '''
        multi class (=3), rpnv2&resnet, resume class(0,1,2)
        '''
        for rpn_name in ["RPNV2", "ResNetRPN"]:
            network_cfg = Test_build_model_and_init.network_cfg_template.copy()
            network_cfg["RPN"]["name"] = rpn_name
            network_cfg["RPN"]["@num_class"] = 3
            network_cfg["RPN"]["@num_anchor_per_loc"] = 6
            resume_dict = {
                "ckpt_path": f"unit_tests/data/test_build_model_and_init_weight_{rpn_name}_3.tckpt",
                "num_classes": 3,
                "num_anchor_per_loc": 6,
                "partially_load_params": []
            }
            params = {
                "classes": ["class1", "class2", "class3"],
                "network_cfg": network_cfg,
                "resume_dict": resume_dict,
                "name": Test_build_model_and_init.name_template
            }
            model = Network._build_model_and_init(**params).cuda()
            self.assertTrue(all([itm in model.keys() for itm in ["vfe_layer", "middle_layer", "rpn"]]))
            self.assertTrue(model.rpn.__class__.__name__ == rpn_name)
            data = load_pickle("./unit_tests/data/test_build_model_and_init_data.pkl")
            voxels = data["voxels"]
            num_points = data["num_points"]
            coors = data["coordinates"]
            batch_size = data["anchors"].shape[0]
            voxel_features = model["vfe_layer"](voxels, num_points,coors)
            spatial_features = model["middle_layer"](voxel_features, coors, batch_size)
            preds_dict = model["rpn"](spatial_features)
            self.assertTrue(preds_dict["cls_preds"].shape[1]
                == network_cfg["RPN"]["@num_anchor_per_loc"])
            self.assertTrue(preds_dict["cls_preds"].shape[-1]
                == network_cfg["RPN"]["@num_class"])
            self.assertTrue(preds_dict["box_preds"].shape[1]
                == network_cfg["RPN"]["@num_anchor_per_loc"])
            self.assertTrue(preds_dict["box_preds"].shape[-1]
                == network_cfg["RPN"]["@box_code_size"])
            self.assertTrue(preds_dict["dir_cls_preds"].shape[1]
                == network_cfg["RPN"]["@num_anchor_per_loc"])
            self.assertTrue(preds_dict["dir_cls_preds"].shape[-1]
                == network_cfg["RPN"]["@num_direction_bins"])
            gt = load_pickle(f"unit_tests/data/test_build_model_and_init_output_{rpn_name}_3.pkl")
            for k, v in preds_dict.items():
                self.assertTrue(torch.all(torch.eq(v, gt[k])))

    def test_multiclasses_resume2(self):
        '''
        multi class (=3), rpnv2&resnet, resume class(0,1)
        '''
        for rpn_name in ["RPNV2", "ResNetRPN"]:
            network_cfg = Test_build_model_and_init.network_cfg_template.copy()
            network_cfg["RPN"]["name"] = rpn_name
            network_cfg["RPN"]["@num_class"] = 3
            network_cfg["RPN"]["@num_anchor_per_loc"] = 6
            resume_dict = {
                "ckpt_path": f"unit_tests/data/test_build_model_and_init_weight_{rpn_name}_2.tckpt",
                "num_classes": 2,
                "num_anchor_per_loc": 4,
                "partially_load_params": [
                    "rpn.conv_cls.weight", "rpn.conv_cls.bias",
                    "rpn.conv_box.weight", "rpn.conv_box.bias",
                    "rpn.conv_dir_cls.weight", "rpn.conv_dir_cls.bias"]
            }
            params = {
                "classes": ["class1", "class2", "class3"],
                "network_cfg": network_cfg,
                "resume_dict": resume_dict,
                "name": Test_build_model_and_init.name_template
            }
            model = Network._build_model_and_init(**params).cuda()
            self.assertTrue(all([itm in model.keys() for itm in ["vfe_layer", "middle_layer", "rpn"]]))
            self.assertTrue(model.rpn.__class__.__name__ == rpn_name)
            data = load_pickle("./unit_tests/data/test_build_model_and_init_data.pkl")
            voxels = data["voxels"]
            num_points = data["num_points"]
            coors = data["coordinates"]
            batch_size = data["anchors"].shape[0]
            voxel_features = model["vfe_layer"](voxels, num_points,coors)
            spatial_features = model["middle_layer"](voxel_features, coors, batch_size)
            preds_dict = model["rpn"](spatial_features)
            self.assertTrue(preds_dict["cls_preds"].shape[1]
                == network_cfg["RPN"]["@num_anchor_per_loc"])
            self.assertTrue(preds_dict["cls_preds"].shape[-1]
                == network_cfg["RPN"]["@num_class"])
            self.assertTrue(preds_dict["box_preds"].shape[1]
                == network_cfg["RPN"]["@num_anchor_per_loc"])
            self.assertTrue(preds_dict["box_preds"].shape[-1]
                == network_cfg["RPN"]["@box_code_size"])
            self.assertTrue(preds_dict["dir_cls_preds"].shape[1]
                == network_cfg["RPN"]["@num_anchor_per_loc"])
            self.assertTrue(preds_dict["dir_cls_preds"].shape[-1]
                == network_cfg["RPN"]["@num_direction_bins"])
            gt = load_pickle(f"unit_tests/data/test_build_model_and_init_output_{rpn_name}_2.pkl")
            num_old_classes = resume_dict["num_classes"]
            num_old_anchor_per_loc = resume_dict["num_anchor_per_loc"]
            est = {}
            est["cls_preds"] = preds_dict["cls_preds"][:, :num_old_anchor_per_loc, ...,:num_old_classes]
            est["box_preds"] = preds_dict["box_preds"][:, :num_old_anchor_per_loc, ...]
            est["dir_cls_preds"] = preds_dict["dir_cls_preds"][:, :num_old_anchor_per_loc, ...]
            for k, v in est.items():
                self.assertTrue(torch.all(torch.eq(v, gt[k])))

    def test_multiclasses_resume3(self):
        '''
        multi class (=3), rpnv2&resnet, resume class(0)
        '''
        for rpn_name in ["RPNV2", "ResNetRPN"]:
            network_cfg = Test_build_model_and_init.network_cfg_template.copy()
            network_cfg["RPN"]["name"] = rpn_name
            network_cfg["RPN"]["@num_class"] = 3
            network_cfg["RPN"]["@num_anchor_per_loc"] = 6
            resume_dict = {
                "ckpt_path": f"unit_tests/data/test_build_model_and_init_weight_{rpn_name}.tckpt",
                "num_classes": 1,
                "num_anchor_per_loc": 2,
                "partially_load_params": [
                    "rpn.conv_cls.weight", "rpn.conv_cls.bias",
                    "rpn.conv_box.weight", "rpn.conv_box.bias",
                    "rpn.conv_dir_cls.weight", "rpn.conv_dir_cls.bias"]
            }
            params = {
                "classes": ["class1", "class2", "class3"],
                "network_cfg": network_cfg,
                "resume_dict": resume_dict,
                "name": Test_build_model_and_init.name_template
            }
            model = Network._build_model_and_init(**params).cuda()
            self.assertTrue(all([itm in model.keys() for itm in ["vfe_layer", "middle_layer", "rpn"]]))
            self.assertTrue(model.rpn.__class__.__name__ == rpn_name)
            data = load_pickle("./unit_tests/data/test_build_model_and_init_data.pkl")
            voxels = data["voxels"]
            num_points = data["num_points"]
            coors = data["coordinates"]
            batch_size = data["anchors"].shape[0]
            voxel_features = model["vfe_layer"](voxels, num_points,coors)
            spatial_features = model["middle_layer"](voxel_features, coors, batch_size)
            preds_dict = model["rpn"](spatial_features)
            self.assertTrue(preds_dict["cls_preds"].shape[1]
                == network_cfg["RPN"]["@num_anchor_per_loc"])
            self.assertTrue(preds_dict["cls_preds"].shape[-1]
                == network_cfg["RPN"]["@num_class"])
            self.assertTrue(preds_dict["box_preds"].shape[1]
                == network_cfg["RPN"]["@num_anchor_per_loc"])
            self.assertTrue(preds_dict["box_preds"].shape[-1]
                == network_cfg["RPN"]["@box_code_size"])
            self.assertTrue(preds_dict["dir_cls_preds"].shape[1]
                == network_cfg["RPN"]["@num_anchor_per_loc"])
            self.assertTrue(preds_dict["dir_cls_preds"].shape[-1]
                == network_cfg["RPN"]["@num_direction_bins"])
            gt = load_pickle(f"unit_tests/data/test_build_model_and_init_output_{rpn_name}.pkl")
            num_old_classes = resume_dict["num_classes"]
            num_old_anchor_per_loc = resume_dict["num_anchor_per_loc"]
            est = {}
            est["cls_preds"] = preds_dict["cls_preds"][:, :num_old_anchor_per_loc, ...,:num_old_classes]
            est["box_preds"] = preds_dict["box_preds"][:, :num_old_anchor_per_loc, ...]
            est["dir_cls_preds"] = preds_dict["dir_cls_preds"][:, :num_old_anchor_per_loc, ...]
            for k, v in est.items():
                self.assertTrue(torch.all(torch.eq(v, gt[k])))

class Test_freeze_model_and_detach_variables(unittest.TestCase):
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
    def test_train_from_scratch(self):
        '''
        training scheme: train_from_scratch
        '''
        for rpn_name in ["RPNV2", "ResNetRPN"]:
            network_cfg = Test_freeze_model_and_detach_variables.network_cfg_template.copy()
            network_cfg["RPN"]["name"] = rpn_name
            network_cfg["RPN"]["@num_class"] = 2
            network_cfg["RPN"]["@num_anchor_per_loc"] = 4
            params = {
                "classes_target": ["class1", "class2"],
                "classes_source": None,
                "model_resume_dict": None,
                "sub_model_resume_dict": None,
                "voxel_encoder_dict": network_cfg["VoxelEncoder"],
                "middle_layer_dict": network_cfg["MiddleLayer"],
                "rpn_dict": network_cfg["RPN"],
                "training_mode": "train_from_scratch",
                "is_training": True,
                "bool_oldclass_use_newanchor_for_cls": False,
            }
            network = Network(**params).cuda()
            for name, param in network.named_parameters():
                self.assertTrue(name, param.requires_grad)
            data = load_pickle("./unit_tests/data/test_build_model_and_init_data.pkl")
            voxels = data["voxels"]
            num_points = data["num_points"]
            coors = data["coordinates"]
            batch_size = data["anchors"].shape[0]

            preds_dict = network._network_forward(network._model,
                voxels,
                num_points,
                coors,
                batch_size)
            network._detach_variables(preds_dict)
            loss = 0
            target = {
                "cls_preds": torch.randn(1,4,200,176,2).float().cuda(),
                "box_preds": torch.randn(1,4,200,176,7).float().cuda(),
                "dir_cls_preds": torch.randn(1,4,200,176,2).float().cuda()
            }
            criterion = nn.MSELoss()
            for k, v in preds_dict.items():
                loss += criterion(v, target[k])
            loss.backward()
            # check grad
            check_grad_not_have_zero = lambda param: len((param.grad == 0).nonzero()) == 0
            self.assertTrue(check_grad_not_have_zero(network._model.rpn.conv_cls.weight))
            self.assertTrue(check_grad_not_have_zero(network._model.rpn.conv_cls.bias))
            self.assertTrue(check_grad_not_have_zero(network._model.rpn.conv_box.weight))
            self.assertTrue(check_grad_not_have_zero(network._model.rpn.conv_box.bias))
            self.assertTrue(check_grad_not_have_zero(network._model.rpn.conv_dir_cls.weight))
            self.assertTrue(check_grad_not_have_zero(network._model.rpn.conv_dir_cls.bias))

    def test_joint_training(self):
        '''
        training scheme: joint_training
        '''
        for rpn_name in ["RPNV2", "ResNetRPN"]:
            network_cfg = Test_freeze_model_and_detach_variables.network_cfg_template.copy()
            network_cfg["RPN"]["name"] = rpn_name
            network_cfg["RPN"]["@num_class"] = 2
            network_cfg["RPN"]["@num_anchor_per_loc"] = 4
            params = {
                "classes_target": ["class1", "class2"],
                "classes_source": None,
                "model_resume_dict": None,
                "sub_model_resume_dict": None,
                "voxel_encoder_dict": network_cfg["VoxelEncoder"],
                "middle_layer_dict": network_cfg["MiddleLayer"],
                "rpn_dict": network_cfg["RPN"],
                "training_mode": "joint_training",
                "is_training": True,
                "bool_oldclass_use_newanchor_for_cls": False,
            }
            network = Network(**params).cuda()
            for name, param in network.named_parameters():
                self.assertTrue(name, param.requires_grad)
            data = load_pickle("./unit_tests/data/test_build_model_and_init_data.pkl")
            voxels = data["voxels"]
            num_points = data["num_points"]
            coors = data["coordinates"]
            batch_size = data["anchors"].shape[0]

            preds_dict = network._network_forward(network._model,
                voxels,
                num_points,
                coors,
                batch_size)
            network._detach_variables(preds_dict)
            loss = 0
            target = {
                "cls_preds": torch.randn(1,4,200,176,2).float().cuda(),
                "box_preds": torch.randn(1,4,200,176,7).float().cuda(),
                "dir_cls_preds": torch.randn(1,4,200,176,2).float().cuda()
            }
            criterion = nn.MSELoss()
            for k, v in preds_dict.items():
                loss += criterion(v, target[k])
            loss.backward()
            # check grad
            check_grad_not_have_zero = lambda param: len((param.grad == 0).nonzero()) == 0
            self.assertTrue(check_grad_not_have_zero(network._model.rpn.conv_cls.weight))
            self.assertTrue(check_grad_not_have_zero(network._model.rpn.conv_cls.bias))
            self.assertTrue(check_grad_not_have_zero(network._model.rpn.conv_box.weight))
            self.assertTrue(check_grad_not_have_zero(network._model.rpn.conv_box.bias))
            self.assertTrue(check_grad_not_have_zero(network._model.rpn.conv_dir_cls.weight))
            self.assertTrue(check_grad_not_have_zero(network._model.rpn.conv_dir_cls.bias))

    def test_lwf(self):
        '''
        training scheme: lwf
        '''
        for rpn_name in ["RPNV2", "ResNetRPN"]:
            network_cfg = Test_freeze_model_and_detach_variables.network_cfg_template.copy()
            network_cfg["RPN"]["name"] = rpn_name
            network_cfg["RPN"]["@num_class"] = 2
            network_cfg["RPN"]["@num_anchor_per_loc"] = 4
            params = {
                "classes_target": ["class1", "class2"],
                "classes_source": ["class1"],
                "model_resume_dict": None,
                "sub_model_resume_dict": None,
                "voxel_encoder_dict": network_cfg["VoxelEncoder"],
                "middle_layer_dict": network_cfg["MiddleLayer"],
                "rpn_dict": network_cfg["RPN"],
                "training_mode": "lwf",
                "is_training": True,
                "bool_oldclass_use_newanchor_for_cls": False,
            }
            network = Network(**params).cuda()
            for name, param in network.named_parameters():
                self.assertTrue(name, param.requires_grad)
            data = load_pickle("./unit_tests/data/test_build_model_and_init_data.pkl")
            voxels = data["voxels"]
            num_points = data["num_points"]
            coors = data["coordinates"]
            batch_size = data["anchors"].shape[0]

            preds_dict = network._network_forward(network._model,
                voxels,
                num_points,
                coors,
                batch_size)
            network._detach_variables(preds_dict)
            preds_dict_sub = network._network_forward(network._sub_model,
                voxels,
                num_points,
                coors,
                batch_size)
            network._detach_variables(preds_dict_sub)
            loss = 0
            target = {
                "cls_preds": torch.randn(1,4,200,176,2).float().cuda(),
                "box_preds": torch.randn(1,4,200,176,7).float().cuda(),
                "dir_cls_preds": torch.randn(1,4,200,176,2).float().cuda()
            }
            target_sub = {
                "cls_preds": torch.randn(1,2,200,176,1).float().cuda(),
                "box_preds": torch.randn(1,2,200,176,7).float().cuda(),
                "dir_cls_preds": torch.randn(1,2,200,176,2).float().cuda()
            }
            criterion = nn.MSELoss()
            for k, v in preds_dict.items():
                loss += criterion(v, target[k])
            for k, v in preds_dict_sub.items():
                loss += criterion(v, target_sub[k])
            loss.backward()
            # check grad
            # return true if do not have zero
            check_grad_not_have_zero = lambda param: len((param.grad == 0).nonzero()) == 0
            check_grad_is_none = lambda param: param.grad is None
            self.assertTrue(check_grad_not_have_zero(network._model.rpn.conv_cls.weight))
            self.assertTrue(check_grad_not_have_zero(network._model.rpn.conv_cls.bias))
            self.assertTrue(check_grad_not_have_zero(network._model.rpn.conv_box.weight))
            self.assertTrue(check_grad_not_have_zero(network._model.rpn.conv_box.bias))
            self.assertTrue(check_grad_not_have_zero(network._model.rpn.conv_dir_cls.weight))
            self.assertTrue(check_grad_not_have_zero(network._model.rpn.conv_dir_cls.bias))

            self.assertTrue(check_grad_is_none(network._sub_model.rpn.conv_cls.weight))
            self.assertTrue(check_grad_is_none(network._sub_model.rpn.conv_cls.bias))
            self.assertTrue(check_grad_is_none(network._sub_model.rpn.conv_box.weight))
            self.assertTrue(check_grad_is_none(network._sub_model.rpn.conv_box.bias))
            self.assertTrue(check_grad_is_none(network._sub_model.rpn.conv_dir_cls.weight))
            self.assertTrue(check_grad_is_none(network._sub_model.rpn.conv_dir_cls.bias))

    def test_fine_tuning_new2_old1(self):
        '''
        training scheme: fine_tuning
        num_new_class = 2
        num_old_class = 1
        '''
        for rpn_name in ["RPNV2", "ResNetRPN"]:
            for bool_oldclass_use_newanchor_for_cls in [True, False]:
                network_cfg = Test_freeze_model_and_detach_variables.network_cfg_template.copy()
                network_cfg["RPN"]["name"] = rpn_name
                network_cfg["RPN"]["@num_class"] = 2
                network_cfg["RPN"]["@num_anchor_per_loc"] = 4
                params = {
                    "classes_target": ["class1", "class2"],
                    "classes_source": None,
                    "model_resume_dict": {
                        "num_classes": 1,
                        "num_anchor_per_loc": 2,
                        "ckpt_path": None,
                        "partially_load_params": []
                    },
                    "sub_model_resume_dict": None,
                    "voxel_encoder_dict": network_cfg["VoxelEncoder"],
                    "middle_layer_dict": network_cfg["MiddleLayer"],
                    "rpn_dict": network_cfg["RPN"],
                    "training_mode": "fine_tuning",
                    "is_training": True,
                    "bool_oldclass_use_newanchor_for_cls": bool_oldclass_use_newanchor_for_cls,
                }
                network = Network(**params).cuda()
                for name, param in network.named_parameters():
                    self.assertTrue(name, param.requires_grad)
                data = load_pickle("./unit_tests/data/test_build_model_and_init_data.pkl")
                voxels = data["voxels"]
                num_points = data["num_points"]
                coors = data["coordinates"]
                batch_size = data["anchors"].shape[0]

                preds_dict = network._network_forward(network._model,
                    voxels,
                    num_points,
                    coors,
                    batch_size)
                network._detach_variables(preds_dict)
                loss = 0
                target = {
                    "cls_preds": torch.randn(1,4,200,176,2).float().cuda(),
                    "box_preds": torch.randn(1,4,200,176,7).float().cuda(),
                    "dir_cls_preds": torch.randn(1,4,200,176,2).float().cuda()
                }
                criterion = nn.MSELoss()
                for k, v in preds_dict.items():
                    loss += criterion(v, target[k])
                loss.backward()
                # check grad
                check_grad_not_have_zero = lambda param: len((param.grad == 0).nonzero()) == 0
                print_zero_grad = lambda param: (param.grad == 0).nonzero()
                conv_cls_idx = [0, 2] if bool_oldclass_use_newanchor_for_cls else [0,2,4,6]
                # conv_cls
                tmp = print_zero_grad(network._model.rpn.conv_cls.bias)
                tmp = tmp == torch.Tensor(conv_cls_idx).reshape(-1,1).cuda()
                self.assertTrue(torch.all(tmp))

                tmp = print_zero_grad(network._model.rpn.conv_cls.weight)
                y, x = np.meshgrid([i for i in range(128)], conv_cls_idx)
                gt = np.asarray([[x_, y_] for x_, y_ in zip(x.flatten(), y.flatten())])
                gt = gt.reshape(x.shape[0]*x.shape[1], -1)
                gt = np.concatenate([gt, np.zeros_like(gt)], axis=1)
                gt = torch.from_numpy(gt).cuda()
                tmp = tmp == gt
                self.assertTrue(torch.all(tmp))

                # conv_box
                tmp = print_zero_grad(network._model.rpn.conv_box.bias)
                tmp = tmp == torch.Tensor(range(14)).reshape(-1,1).cuda()
                self.assertTrue(torch.all(tmp))

                tmp = print_zero_grad(network._model.rpn.conv_box.weight)
                y, x = np.meshgrid([i for i in range(128)], range(14))
                gt = np.asarray([[x_, y_] for x_, y_ in zip(x.flatten(), y.flatten())])
                gt = gt.reshape(x.shape[0]*x.shape[1], -1)
                gt = np.concatenate([gt, np.zeros_like(gt)], axis=1)
                gt = torch.from_numpy(gt).cuda()
                tmp = tmp == gt
                self.assertTrue(torch.all(tmp))

                # conv_dir_cls
                tmp = print_zero_grad(network._model.rpn.conv_dir_cls.bias)
                tmp = tmp == torch.Tensor(range(4)).reshape(-1,1).cuda()
                self.assertTrue(torch.all(tmp))

                tmp = print_zero_grad(network._model.rpn.conv_dir_cls.weight)
                y, x = np.meshgrid([i for i in range(128)], range(4))
                gt = np.asarray([[x_, y_] for x_, y_ in zip(x.flatten(), y.flatten())])
                gt = gt.reshape(x.shape[0]*x.shape[1], -1)
                gt = np.concatenate([gt, np.zeros_like(gt)], axis=1)
                gt = torch.from_numpy(gt).cuda()
                tmp = tmp == gt
                self.assertTrue(torch.all(tmp))

                self.assertTrue(check_grad_not_have_zero(network._model.rpn.deblocks[0][0].weight))

    def test_fine_tuning_new3_old2(self):
        '''
        training scheme: fine_tuning
        num_new_class = 3
        num_old_class = 2
        '''
        for rpn_name in ["RPNV2", "ResNetRPN"]:
            for bool_oldclass_use_newanchor_for_cls in [True, False]:
                network_cfg = Test_freeze_model_and_detach_variables.network_cfg_template.copy()
                network_cfg["RPN"]["name"] = rpn_name
                network_cfg["RPN"]["@num_class"] = 3
                network_cfg["RPN"]["@num_anchor_per_loc"] = 6
                params = {
                    "classes_target": ["class1", "class2", "class3"],
                    "classes_source": None,
                    "model_resume_dict": {
                        "num_classes": 2,
                        "num_anchor_per_loc": 4,
                        "ckpt_path": None,
                        "partially_load_params": []
                    },
                    "sub_model_resume_dict": None,
                    "voxel_encoder_dict": network_cfg["VoxelEncoder"],
                    "middle_layer_dict": network_cfg["MiddleLayer"],
                    "rpn_dict": network_cfg["RPN"],
                    "training_mode": "fine_tuning",
                    "is_training": True,
                    "bool_oldclass_use_newanchor_for_cls": bool_oldclass_use_newanchor_for_cls,
                }
                network = Network(**params).cuda()
                for name, param in network.named_parameters():
                    self.assertTrue(name, param.requires_grad)
                data = load_pickle("./unit_tests/data/test_build_model_and_init_data.pkl")
                voxels = data["voxels"]
                num_points = data["num_points"]
                coors = data["coordinates"]
                batch_size = data["anchors"].shape[0]

                preds_dict = network._network_forward(network._model,
                    voxels,
                    num_points,
                    coors,
                    batch_size)
                network._detach_variables(preds_dict)
                loss = 0

                target = {}
                for k, v in preds_dict.items():
                    target[k] = torch.randn_like(v).float().cuda()
                criterion = nn.MSELoss()
                for k, v in preds_dict.items():
                    loss += criterion(v, target[k])
                loss.backward()
                # check grad
                check_grad_not_have_zero = lambda param: len((param.grad == 0).nonzero()) == 0
                print_zero_grad = lambda param: (param.grad == 0).nonzero()
                conv_cls_idx = [0,1,3,4,6,7,9,10] if bool_oldclass_use_newanchor_for_cls else [0,1,3,4,6,7,9,10,12,13,15,16]
                # conv_cls
                tmp = print_zero_grad(network._model.rpn.conv_cls.bias)
                tmp = tmp == torch.Tensor(conv_cls_idx).reshape(-1,1).cuda()
                self.assertTrue(torch.all(tmp))

                tmp = print_zero_grad(network._model.rpn.conv_cls.weight)
                y, x = np.meshgrid([i for i in range(128)], conv_cls_idx)
                gt = np.asarray([[x_, y_] for x_, y_ in zip(x.flatten(), y.flatten())])
                gt = gt.reshape(x.shape[0]*x.shape[1], -1)
                gt = np.concatenate([gt, np.zeros_like(gt)], axis=1)
                gt = torch.from_numpy(gt).cuda()
                tmp = tmp == gt
                self.assertTrue(torch.all(tmp))

                # conv_box
                tmp = print_zero_grad(network._model.rpn.conv_box.bias)
                tmp = tmp == torch.Tensor(range(7*4)).reshape(-1,1).cuda()
                self.assertTrue(torch.all(tmp))

                tmp = print_zero_grad(network._model.rpn.conv_box.weight)
                y, x = np.meshgrid([i for i in range(128)], range(7*4))
                gt = np.asarray([[x_, y_] for x_, y_ in zip(x.flatten(), y.flatten())])
                gt = gt.reshape(x.shape[0]*x.shape[1], -1)
                gt = np.concatenate([gt, np.zeros_like(gt)], axis=1)
                gt = torch.from_numpy(gt).cuda()
                tmp = tmp == gt
                self.assertTrue(torch.all(tmp))

                # conv_dir_cls
                tmp = print_zero_grad(network._model.rpn.conv_dir_cls.bias)
                tmp = tmp == torch.Tensor(range(2*4)).reshape(-1,1).cuda()
                self.assertTrue(torch.all(tmp))

                tmp = print_zero_grad(network._model.rpn.conv_dir_cls.weight)
                y, x = np.meshgrid([i for i in range(128)], range(2*4))
                gt = np.asarray([[x_, y_] for x_, y_ in zip(x.flatten(), y.flatten())])
                gt = gt.reshape(x.shape[0]*x.shape[1], -1)
                gt = np.concatenate([gt, np.zeros_like(gt)], axis=1)
                gt = torch.from_numpy(gt).cuda()
                tmp = tmp == gt
                self.assertTrue(torch.all(tmp))

                self.assertTrue(check_grad_not_have_zero(network._model.rpn.deblocks[0][0].weight))

    def test_feature_extraction_new2_old1(self):
        '''
        training scheme: feature_extraction
        '''
        '''
        training scheme: feature_extraction
        num_new_class = 2
        num_old_class = 1
        '''
        for rpn_name in ["RPNV2", "ResNetRPN"]:
            for bool_oldclass_use_newanchor_for_cls in [True, False]:
                network_cfg = Test_freeze_model_and_detach_variables.network_cfg_template.copy()
                network_cfg["RPN"]["name"] = rpn_name
                network_cfg["RPN"]["@num_class"] = 2
                network_cfg["RPN"]["@num_anchor_per_loc"] = 4
                params = {
                    "classes_target": ["class1", "class2"],
                    "classes_source": None,
                    "model_resume_dict": {
                        "num_classes": 1,
                        "num_anchor_per_loc": 2,
                        "ckpt_path": None,
                        "partially_load_params": []
                    },
                    "sub_model_resume_dict": None,
                    "voxel_encoder_dict": network_cfg["VoxelEncoder"],
                    "middle_layer_dict": network_cfg["MiddleLayer"],
                    "rpn_dict": network_cfg["RPN"],
                    "training_mode": "feature_extraction",
                    "is_training": True,
                    "bool_oldclass_use_newanchor_for_cls": bool_oldclass_use_newanchor_for_cls,
                }
                network = Network(**params).cuda()
                for name, param in network.named_parameters():
                    self.assertTrue(name, param.requires_grad)
                data = load_pickle("./unit_tests/data/test_build_model_and_init_data.pkl")
                voxels = data["voxels"]
                num_points = data["num_points"]
                coors = data["coordinates"]
                batch_size = data["anchors"].shape[0]

                preds_dict = network._network_forward(network._model,
                    voxels,
                    num_points,
                    coors,
                    batch_size)
                network._detach_variables(preds_dict)
                loss = 0

                target = {}
                for k, v in preds_dict.items():
                    target[k] = torch.randn_like(v).float().cuda()
                criterion = nn.MSELoss()
                for k, v in preds_dict.items():
                    loss += criterion(v, target[k])
                loss.backward()
                # check grad
                check_grad_not_have_zero = lambda param: len((param.grad == 0).nonzero()) == 0 if param.grad is not None else False
                print_zero_grad = lambda param: (param.grad == 0).nonzero()
                conv_cls_idx = [0,2] if bool_oldclass_use_newanchor_for_cls else [0,2,4,6]
                # conv_cls
                tmp = print_zero_grad(network._model.rpn.conv_cls.bias)
                tmp = tmp == torch.Tensor(conv_cls_idx).reshape(-1,1).cuda()
                self.assertTrue(torch.all(tmp))

                tmp = print_zero_grad(network._model.rpn.conv_cls.weight)
                y, x = np.meshgrid([i for i in range(128)], conv_cls_idx)
                gt = np.asarray([[x_, y_] for x_, y_ in zip(x.flatten(), y.flatten())])
                gt = gt.reshape(x.shape[0]*x.shape[1], -1)
                gt = np.concatenate([gt, np.zeros_like(gt)], axis=1)
                gt = torch.from_numpy(gt).cuda()
                tmp = tmp == gt
                self.assertTrue(torch.all(tmp))

                # conv_box
                tmp = print_zero_grad(network._model.rpn.conv_box.bias)
                tmp = tmp == torch.Tensor(range(7*2)).reshape(-1,1).cuda()
                self.assertTrue(torch.all(tmp))

                tmp = print_zero_grad(network._model.rpn.conv_box.weight)
                y, x = np.meshgrid([i for i in range(128)], range(7*2))
                gt = np.asarray([[x_, y_] for x_, y_ in zip(x.flatten(), y.flatten())])
                gt = gt.reshape(x.shape[0]*x.shape[1], -1)
                gt = np.concatenate([gt, np.zeros_like(gt)], axis=1)
                gt = torch.from_numpy(gt).cuda()
                tmp = tmp == gt
                self.assertTrue(torch.all(tmp))

                # conv_dir_cls
                tmp = print_zero_grad(network._model.rpn.conv_dir_cls.bias)
                tmp = tmp == torch.Tensor(range(2*2)).reshape(-1,1).cuda()
                self.assertTrue(torch.all(tmp))

                tmp = print_zero_grad(network._model.rpn.conv_dir_cls.weight)
                y, x = np.meshgrid([i for i in range(128)], range(2*2))
                gt = np.asarray([[x_, y_] for x_, y_ in zip(x.flatten(), y.flatten())])
                gt = gt.reshape(x.shape[0]*x.shape[1], -1)
                gt = np.concatenate([gt, np.zeros_like(gt)], axis=1)
                gt = torch.from_numpy(gt).cuda()
                tmp = tmp == gt
                self.assertTrue(torch.all(tmp))

                self.assertFalse(check_grad_not_have_zero(network._model.rpn.deblocks[0][0].weight))

    def test_feature_extraction_new3_old2(self):
        '''
        training scheme: feature_extraction
        '''
        '''
        training scheme: feature_extraction
        num_new_class = 3
        num_old_class = 2
        '''
        for rpn_name in ["RPNV2", "ResNetRPN"]:
            for bool_oldclass_use_newanchor_for_cls in [True, False]:
                network_cfg = Test_freeze_model_and_detach_variables.network_cfg_template.copy()
                network_cfg["RPN"]["name"] = rpn_name
                network_cfg["RPN"]["@num_class"] = 3
                network_cfg["RPN"]["@num_anchor_per_loc"] = 6
                params = {
                    "classes_target": ["class1", "class2", "class3"],
                    "classes_source": None,
                    "model_resume_dict": {
                        "num_classes": 2,
                        "num_anchor_per_loc": 4,
                        "ckpt_path": None,
                        "partially_load_params": []
                    },
                    "sub_model_resume_dict": None,
                    "voxel_encoder_dict": network_cfg["VoxelEncoder"],
                    "middle_layer_dict": network_cfg["MiddleLayer"],
                    "rpn_dict": network_cfg["RPN"],
                    "training_mode": "feature_extraction",
                    "is_training": True,
                    "bool_oldclass_use_newanchor_for_cls": bool_oldclass_use_newanchor_for_cls,
                }
                network = Network(**params).cuda()
                for name, param in network.named_parameters():
                    self.assertTrue(name, param.requires_grad)
                data = load_pickle("./unit_tests/data/test_build_model_and_init_data.pkl")
                voxels = data["voxels"]
                num_points = data["num_points"]
                coors = data["coordinates"]
                batch_size = data["anchors"].shape[0]

                preds_dict = network._network_forward(network._model,
                    voxels,
                    num_points,
                    coors,
                    batch_size)
                network._detach_variables(preds_dict)
                loss = 0

                target = {}
                for k, v in preds_dict.items():
                    target[k] = torch.randn_like(v).float().cuda()
                criterion = nn.MSELoss()
                for k, v in preds_dict.items():
                    loss += criterion(v, target[k])
                loss.backward()
                # check grad
                check_grad_not_have_zero = lambda param: len((param.grad == 0).nonzero()) == 0 if param.grad is not None else False
                print_zero_grad = lambda param: (param.grad == 0).nonzero()
                conv_cls_idx = [0,1,3,4,6,7,9,10] if bool_oldclass_use_newanchor_for_cls else [0,1,3,4,6,7,9,10,12,13,15,16]
                # conv_cls
                tmp = print_zero_grad(network._model.rpn.conv_cls.bias)
                tmp = tmp == torch.Tensor(conv_cls_idx).reshape(-1,1).cuda()
                self.assertTrue(torch.all(tmp))

                tmp = print_zero_grad(network._model.rpn.conv_cls.weight)
                y, x = np.meshgrid([i for i in range(128)], conv_cls_idx)
                gt = np.asarray([[x_, y_] for x_, y_ in zip(x.flatten(), y.flatten())])
                gt = gt.reshape(x.shape[0]*x.shape[1], -1)
                gt = np.concatenate([gt, np.zeros_like(gt)], axis=1)
                gt = torch.from_numpy(gt).cuda()
                tmp = tmp == gt
                self.assertTrue(torch.all(tmp))

                # conv_box
                tmp = print_zero_grad(network._model.rpn.conv_box.bias)
                tmp = tmp == torch.Tensor(range(7*4)).reshape(-1,1).cuda()
                self.assertTrue(torch.all(tmp))

                tmp = print_zero_grad(network._model.rpn.conv_box.weight)
                y, x = np.meshgrid([i for i in range(128)], range(7*4))
                gt = np.asarray([[x_, y_] for x_, y_ in zip(x.flatten(), y.flatten())])
                gt = gt.reshape(x.shape[0]*x.shape[1], -1)
                gt = np.concatenate([gt, np.zeros_like(gt)], axis=1)
                gt = torch.from_numpy(gt).cuda()
                tmp = tmp == gt
                self.assertTrue(torch.all(tmp))

                # conv_dir_cls
                tmp = print_zero_grad(network._model.rpn.conv_dir_cls.bias)
                tmp = tmp == torch.Tensor(range(2*4)).reshape(-1,1).cuda()
                self.assertTrue(torch.all(tmp))

                tmp = print_zero_grad(network._model.rpn.conv_dir_cls.weight)
                y, x = np.meshgrid([i for i in range(128)], range(2*4))
                gt = np.asarray([[x_, y_] for x_, y_ in zip(x.flatten(), y.flatten())])
                gt = gt.reshape(x.shape[0]*x.shape[1], -1)
                gt = np.concatenate([gt, np.zeros_like(gt)], axis=1)
                gt = torch.from_numpy(gt).cuda()
                tmp = tmp == gt
                self.assertTrue(torch.all(tmp))

                self.assertFalse(check_grad_not_have_zero(network._model.rpn.deblocks[0][0].weight))

class Test_register_hook(unittest.TestCase):
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
    def test_register_model_hook(self):
        for rpn_name in ["RPNV2", "ResNetRPN"]:
            network_cfg = Test_register_hook.network_cfg_template.copy()
            network_cfg["RPN"]["name"] = rpn_name
            network_cfg["RPN"]["@num_class"] = 2
            network_cfg["RPN"]["@num_anchor_per_loc"] = 4
            params = {
                "classes_target": ["class1", "class2"],
                "classes_source": ["class1"],
                "model_resume_dict": None,
                "sub_model_resume_dict": None,
                "voxel_encoder_dict": network_cfg["VoxelEncoder"],
                "middle_layer_dict": network_cfg["MiddleLayer"],
                "rpn_dict": network_cfg["RPN"],
                "training_mode": "lwf",
                "hook_layers": ["rpn.blocks.0.4.conv2", "middle_layer.middle_conv.41"],
                "is_training": True,
                "bool_oldclass_use_newanchor_for_cls": False,
            }
            network = Network(**params).cuda()
            data = load_pickle("./unit_tests/data/test_build_model_and_init_data.pkl")
            voxels = data["voxels"]
            num_points = data["num_points"]
            coors = data["coordinates"]
            batch_size = data["anchors"].shape[0]

            preds_dict = network._network_forward(network._model,
                voxels,
                num_points,
                coors,
                batch_size)
            preds_dict_sub = network._network_forward(network._sub_model,
                voxels,
                num_points,
                coors,
                batch_size)
            if rpn_name == "ResNetRPN":
                self.assertTrue(list(network._hook_features_model[0].shape) == [17860, 64])
                self.assertTrue(list(network._hook_features_model[1].shape) == [1, 128, 200, 176])
                self.assertTrue(list(network._hook_features_submodel[0].shape) == [17860, 64])
                self.assertTrue(list(network._hook_features_submodel[1].shape) == [1, 128, 200, 176])
            elif rpn_name == "RPNV2":
                self.assertTrue(list(network._hook_features_model[0].shape) == [17860, 64])
                self.assertTrue(list(network._hook_features_submodel[0].shape) == [17860, 64])
            else:
                raise NotImplementedError

class Test_predict(unittest.TestCase):
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
    def test_predict(self):
        for rpn_name in ["RPNV2", "ResNetRPN"]:
            network_cfg = Test_predict.network_cfg_template.copy()
            network_cfg["RPN"]["name"] = rpn_name
            network_cfg["RPN"]["@num_class"] = 2
            network_cfg["RPN"]["@num_anchor_per_loc"] = 4
            params = {
                "classes_target": ["Car", "Pedestrian"],
                "classes_source": None,
                "model_resume_dict": None,
                "sub_model_resume_dict": None,
                "voxel_encoder_dict": network_cfg["VoxelEncoder"],
                "middle_layer_dict": network_cfg["MiddleLayer"],
                "rpn_dict": network_cfg["RPN"],
                "training_mode": "train_from_scratch",
                "hook_layers": [],
                "is_training": True,
                "bool_oldclass_use_newanchor_for_cls": False,
                "post_center_range" : [-35.2, -40, -1.5, 35.2, 40, 2.6],
                "nms_score_thresholds" : [0.6] ,
                "nms_pre_max_sizes" : [1000],
                "nms_post_max_sizes" : [100],
                "nms_iou_thresholds" : [0.3],
            }
            network = Network(**params).cuda()
            from det3.ops import read_pkl
            from incdet3.builders.dataloader_builder import example_convert_to_torch
            example = read_pkl("unit_tests/data/test_model.pkl")
            example = example_convert_to_torch(example,
                dtype=torch.float32,
                device=torch.device("cuda:0"))
            preds_dict = read_pkl("unit_tests/data/test_model_est.pkl")
            box_coder = preds_dict["box_coder"]
            pred_dict = preds_dict["pred_dict"]
            pred_dict["dir_cls_preds"] = torch.zeros_like(pred_dict["box_preds"][..., :2])
            pred_dict["dir_cls_preds"][..., 0] = 1
            predictions_dict = network.predict(example, pred_dict, {"box_coder": box_coder})
            predictions_dict = predictions_dict[0]
            label = example["metadata"][0]["label"]
            from det3.dataloader.carladata import CarlaObj, CarlaLabel
            label_gt = CarlaLabel()
            label_est = CarlaLabel()
            for obj_str in label.split("\n"):
                try:
                    obj = CarlaObj(obj_str)
                except:
                    pass
                if -35.2 < obj.x < 35.2 and -40 < obj.y < 40:
                    label_gt.add_obj(obj)
            for box3d_lidar, label_preds, score in zip(
                    predictions_dict["box3d_lidar"],
                    predictions_dict["label_preds"],
                    predictions_dict["scores"]):
                obj = CarlaObj()
                obj.type = "Car" if label_preds == 0 else "Pedestrian"
                obj.x, obj.y, obj.z, obj.w, obj.l, obj.h, obj.ry = box3d_lidar.cpu().numpy().flatten()
                obj.truncated = 0
                obj.occluded = 0
                obj.alpha = 0
                obj.bbox_l = 0
                obj.bbox_t = 0
                obj.bbox_r = 0
                obj.bbox_b = 0
                obj.score = score
                # if -35.2 < obj.x < 35.2 and -40 < obj.y < 40:
                label_est.add_obj(obj)
            self.assertTrue(label_gt.equal(label_est, acc_cls=["Car", "Pedestrian"], rtol=1e-2))

if __name__ == "__main__":
    unittest.main()