'''
 File Created: Sat Jun 20 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''

import torch
import unittest
from incdet3.models.model import Network
from det3.utils.utils import load_pickle
import inspect

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

if __name__ == "__main__":
    unittest.main()