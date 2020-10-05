'''
 File Created: Thu Sep 03 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''
import torch
import unittest
import numpy as np
from copy import deepcopy
from incdet3.models.model import Network
from torch.nn.parameter import Parameter
torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

class Test_compute_ewc_weights_v1(unittest.TestCase):
    def test_compute_ewc_weights_v1(self):
        return
        from incdet3.models.ewc_func import (_init_ewc_weights,
            _cycle_next, _update_ewc_term, _compute_accum_grad_v1,
            _compute_FIM_cls2term_v1, _compute_FIM_reg2term_v1,
            _compute_FIM_clsregterm_v1, _update_ewc_weights_v1)
        from incdet3.builders.dataloader_builder import example_convert_to_torch
        from tqdm import tqdm
        from det3.ops import write_pkl
        state = np.random.get_state()
        torch_state_cpu = torch.Generator().get_state()
        torch_state_gpu = torch.Generator(device="cuda:0").get_state()
        network = build_network().cuda()
        dataloader = build_dataloader()
        num_of_datasamples = len(dataloader)
        debug_mode = False
        reg2_coef=0.1
        clsreg_coef=0.1
        est_ewc_weights = network.compute_ewc_weights_v1(dataloader,
            num_of_datasamples,
            reg2_coef=reg2_coef,
            clsreg_coef=clsreg_coef,
            debug_mode=debug_mode)
        
        # compute gt
        gt_ewc_weights = _init_ewc_weights(network._model)
        network.eval()
        for i, data in tqdm(enumerate(dataloader)):
            data = example_convert_to_torch(data,
                dtype=torch.float32, device=torch.device("cuda:0"))
            voxels = data["voxels"]
            num_points = data["num_points"]
            coors = data["coordinates"]
            batch_anchors = data["anchors"]
            preds_dict = network._network_forward(network._model, voxels, num_points, coors, 1)
            box_preds = preds_dict["box_preds"]
            cls_preds = preds_dict["cls_preds"]
            labels = data['labels']
            reg_targets = data['reg_targets']
            importance = data['importance']
            weights = Network._prepare_loss_weights(
                labels,
                pos_cls_weight=network._pos_cls_weight,
                neg_cls_weight=network._neg_cls_weight,
                loss_norm_type=network._loss_norm_type,
                importance=importance,
                use_direction_classifier=True,
                dtype=box_preds.dtype)
            cls_targets = labels * weights["cared"].type_as(labels)
            cls_targets = cls_targets.unsqueeze(-1)
            loss_cls = network._compute_classification_loss(
                est=cls_preds,
                gt=cls_targets,
                weights=weights["cls_weights"]*importance)*network._cls_loss_weight
            loss_reg = network._compute_location_loss(
                est=box_preds,
                gt=reg_targets,
                weights=weights["reg_weights"]*importance)*network._loc_loss_weight
            accum_grad_dict = _compute_accum_grad_v1(loss_cls, loss_reg, network._model)
            cls2_term = _compute_FIM_cls2term_v1(accum_grad_dict["cls_grad"])
            reg2_term = _compute_FIM_reg2term_v1(accum_grad_dict["reg_grad"])
            clsreg_term = _compute_FIM_clsregterm_v1(
                accum_grad_dict["cls_grad"],
                accum_grad_dict["reg_grad"])
            gt_ewc_weights = _update_ewc_weights_v1(gt_ewc_weights,
                cls2_term, reg2_term, clsreg_term,
                reg2_coef, clsreg_coef, i)
        for name, param in gt_ewc_weights.items():
            self.assertTrue(torch.allclose(param, est_ewc_weights[name]))
        np.random.set_state(state)
        torch.Generator().set_state(torch_state_cpu)
        torch.Generator(device="cuda:0").set_state(torch_state_gpu)
        # write_pkl({k: v.cpu().numpy() for k, v in gt_ewc_weights.items()},
        #     "unit_tests/data/test_model-ewc-ewc_weights_v1.pkl")

    def test_compute_ewc_weights_v1_debug(self):
        return
        state = np.random.get_state()
        torch_state_cpu = torch.Generator().get_state()
        torch_state_gpu = torch.Generator(device="cuda:0").get_state()
        network = build_network().cuda()
        dataloader = build_dataloader()
        num_of_datasamples = len(dataloader)
        debug_mode = True
        reg2_coef=0.1
        clsreg_coef=0.01
        est_ewc_weights_dict = network.compute_ewc_weights_v1(dataloader,
            num_of_datasamples,
            reg2_coef=reg2_coef,
            clsreg_coef=clsreg_coef,
            debug_mode=debug_mode)
        gt_ewc_weights = network.compute_ewc_weights_v1(dataloader,
            num_of_datasamples,
            reg2_coef=reg2_coef,
            clsreg_coef=clsreg_coef,
            debug_mode=False)
        for name, param in gt_ewc_weights.items():
            self.assertTrue(torch.allclose(param, est_ewc_weights_dict["ewc_weights"][name]))
        for name, param in gt_ewc_weights.items():
            self.assertTrue(torch.allclose(param, est_ewc_weights_dict["cls2_term"][name]
                + est_ewc_weights_dict["reg2_term"][name] * reg2_coef
                + est_ewc_weights_dict["clsreg_term"][name] * clsreg_coef))
        from det3.ops import write_pkl, read_pkl
        import subprocess
        import os
        write_pkl({k: v.cpu().numpy() for k, v in est_ewc_weights_dict["cls2_term"].items()}, f"ewc_cls2term-tmp.pkl")
        write_pkl({k: v.cpu().numpy() for k, v in est_ewc_weights_dict["reg2_term"].items()}, f"ewc_reg2term-tmp.pkl")
        write_pkl({k: v.cpu().numpy() for k, v in est_ewc_weights_dict["clsreg_term"].items()}, f"ewc_clsregterm-tmp.pkl")
        cmd = "python tools/impose_ewc-reg2coef-clsregcoef.py "
        cmd += "--cls2term-path ewc_cls2term-tmp.pkl "
        cmd += "--reg2term-path ewc_reg2term-tmp.pkl "
        cmd += "--clsregterm-path ewc_clsregterm-tmp.pkl "
        cmd += f"--reg2coef {reg2_coef} "
        cmd += f"--clsregcoef {clsreg_coef} "
        cmd += "--output-path ewc_weights-tmp.pkl"
        subprocess.check_output(cmd, shell=True)
        est_ewc_weights = read_pkl("ewc_weights-tmp.pkl")
        for name, _ in est_ewc_weights.items():
            self.assertTrue(np.allclose(est_ewc_weights[name], gt_ewc_weights[name].cpu().numpy(), atol=1e-08, rtol=1e-05))
        os.remove("ewc_cls2term-tmp.pkl")
        os.remove("ewc_reg2term-tmp.pkl")
        os.remove("ewc_clsregterm-tmp.pkl")
        os.remove("ewc_weights-tmp.pkl")
        np.random.set_state(state)
        torch.Generator().set_state(torch_state_cpu)
        torch.Generator(device="cuda:0").set_state(torch_state_gpu)

    def test_compute_accum_grad_v1(self):
        from incdet3.models.ewc_func import _compute_accum_grad_v1
        x = torch.randn([5, 2])
        model = TestModel()
        output = model(x)
        cls_target = torch.nn.functional.one_hot(torch.randint(high=3, size=(5,))).float()
        reg_target = torch.randn([5, 3])
        cls_loss_ftor = torch.nn.MSELoss(reduction="none")
        loss_cls = cls_loss_ftor(output, cls_target)
        loss_reg = reg_target - output
        accum_grad_dict = _compute_accum_grad_v1(loss_cls, loss_reg, model)
        # compute gt:
        for name, param in model.named_parameters():
            self.assertTrue(param.grad.sum() == 0)
        gt_grad_cls = {}
        loss_cls.sum().backward(retain_graph=True)
        for name, param in model.named_parameters():
            gt_grad_cls[name] = param.grad.clone()
        model.zero_grad()
        gt_grad_reg = {}
        loss_reg.sum().backward(retain_graph=True)
        for name, param in model.named_parameters():
            gt_grad_reg[name] = param.grad.clone()
        # compare gt and est:
        for name, param in gt_grad_cls.items():
            self.assertTrue(torch.all(param == accum_grad_dict["cls_grad"][name]))
        for name, param in gt_grad_reg.items():
            self.assertTrue(torch.all(param == accum_grad_dict["reg_grad"][name]))

    def test_compute_FIM_cls2term_v1(self):
        from det3.ops import read_pkl
        from incdet3.models.ewc_func import _compute_FIM_cls2term_v1
        accum_grad_cls = read_pkl("unit_tests/data/test_model-ewcv1_accum_grad_dict.pkl")["cls_grad"]
        cls2term_v1 = _compute_FIM_cls2term_v1(accum_grad_cls)
        for name, param in cls2term_v1.items():
            self.assertTrue(torch.all(param == accum_grad_cls[name] * accum_grad_cls[name]))

    def test_compute_FIM_reg2term_v1(self):
        from det3.ops import read_pkl
        from incdet3.models.ewc_func import _compute_FIM_reg2term_v1
        accum_grad_reg = read_pkl("unit_tests/data/test_model-ewcv1_accum_grad_dict.pkl")["reg_grad"]
        reg2term_v1 = _compute_FIM_reg2term_v1(accum_grad_reg)
        for name, param in reg2term_v1.items():
            self.assertTrue(torch.all(param == accum_grad_reg[name] * accum_grad_reg[name]))

    def test_compute_FIM_clsregterm_v1(self):
        from det3.ops import read_pkl
        from incdet3.models.ewc_func import _compute_FIM_clsregterm_v1
        accum_grad_cls = read_pkl("unit_tests/data/test_model-ewcv1_accum_grad_dict.pkl")["cls_grad"]
        accum_grad_reg = read_pkl("unit_tests/data/test_model-ewcv1_accum_grad_dict.pkl")["reg_grad"]
        clsregterm_v1 = _compute_FIM_clsregterm_v1(accum_grad_cls, accum_grad_reg)
        for name, param in clsregterm_v1.items():
            self.assertTrue(torch.all(param == accum_grad_cls[name] * accum_grad_reg[name]))
        regclsterm_v1 = _compute_FIM_clsregterm_v1(accum_grad_reg, accum_grad_cls)
        for name, param in regclsterm_v1.items():
            self.assertTrue(torch.all(param == clsregterm_v1[name]))

    def test_update_ewc_weights_v1(self):
        from det3.ops import read_pkl
        from incdet3.models.ewc_func import (_compute_FIM_clsregterm_v1,
            _compute_FIM_cls2term_v1,
            _compute_FIM_reg2term_v1,
            _init_ewc_weights, _update_ewc_weights_v1)
        reg2_coef = 0.1
        clsreg_coef = 0.2
        accum_grad_cls = read_pkl("unit_tests/data/test_model-ewcv1_accum_grad_dict.pkl")["cls_grad"]
        accum_grad_reg = read_pkl("unit_tests/data/test_model-ewcv1_accum_grad_dict.pkl")["reg_grad"]
        cls2term_v1 = _compute_FIM_cls2term_v1(accum_grad_cls)
        reg2term_v1 = _compute_FIM_reg2term_v1(accum_grad_reg)
        clsregterm_v1 = _compute_FIM_clsregterm_v1(accum_grad_cls, accum_grad_reg)
        network = TestModel()
        ewc_weights = _init_ewc_weights(network)
        ewc_weights = _update_ewc_weights_v1(ewc_weights, cls2term_v1, reg2term_v1, clsregterm_v1,
            reg2_coef, clsreg_coef, accum_idx=0)
        ewc_weights = _update_ewc_weights_v1(ewc_weights, cls2term_v1, reg2term_v1, clsregterm_v1,
            reg2_coef, clsreg_coef, accum_idx=1)
        ewc_weights = _update_ewc_weights_v1(ewc_weights, cls2term_v1, reg2term_v1, clsregterm_v1,
            reg2_coef, clsreg_coef, accum_idx=2)
        for name, param in ewc_weights.items():
            self.assertTrue(torch.allclose(param,cls2term_v1[name] + reg2term_v1[name] * reg2_coef + clsregterm_v1[name] * clsreg_coef))

    def test_compute_ewc_loss(self):
        from torch.optim import Adam
        network = build_network().cuda()
        optimizer = Adam(lr=1e-5, params=[param for param in network.parameters() if param.requires_grad])
        for i in range(100):
            optimizer.zero_grad()
            loss = network._compute_l2_loss()
            loss.backward()
            optimizer.step()
        for name, weight in network._ewc_weights.items():
            network._ewc_weights[name] = torch.ones_like(weight)
        loss_ewc = network._compute_ewc_loss()
        loss_l2sp = network._compute_l2sp_loss()
        self.assertTrue(torch.allclose(loss_ewc, loss_l2sp))

if __name__ == "__main__":
    unittest.main()