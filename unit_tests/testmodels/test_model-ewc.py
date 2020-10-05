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

class Test_compute_ewc_weights(unittest.TestCase):
    def test_init_ewc_weights(self):
        network = build_network()
        from incdet3.models.ewc_func import _init_ewc_weights
        init_ewc_weights = _init_ewc_weights
        ewc_weights = init_ewc_weights(network._model)
        for k, v in ewc_weights.items():
            self.assertTrue(float(v.sum()) == 0)
            self.assertTrue(v.shape == network._model.state_dict()[k].shape)

    # test sampling ewc
    def test_sampling_ewc_all(self):
        from incdet3.models.ewc_func import _sampling_ewc
        sample_strategy = "all"
        num_of_samples = None
        batch_size = 4
        num_of_anchors = 10
        num_of_classes = 8
        size_of_box = 7
        cls_preds = torch.zeros([batch_size, num_of_anchors, num_of_classes]).float().cuda()
        box_preds = torch.zeros([batch_size, num_of_anchors, size_of_box]).float().cuda()
        selected_cls, selected_box = _sampling_ewc(cls_preds,
            box_preds,
            sample_strategy,
            num_of_samples)
        self.assertTrue(selected_cls.shape ==
            torch.Size([batch_size*num_of_anchors, num_of_classes]))
        self.assertTrue(selected_box.shape ==
            torch.Size([batch_size*num_of_anchors, 7]))

    def test_sampling_ewc_biased(self):
        from incdet3.models.ewc_func import _sampling_ewc
        sample_strategy = "biased"
        num_of_samples = 4
        batch_size = 4
        num_of_anchors = 10
        num_of_classes = 8
        size_of_box = 7
        cls_preds = torch.zeros([batch_size, num_of_anchors, num_of_classes]).float().cuda()
        cls_gt = []
        indices = []
        for batch in range(batch_size):
            index = torch.randint(low=0, high=num_of_anchors, size=(1,), device="cuda:0")
            cls_preds[batch, index, :] += index.float() * 0.1 + 0.1
            cls_gt.append(cls_preds[batch, index, :].clone())
            indices.append(index)
        box_preds = torch.zeros([batch_size, num_of_anchors, size_of_box]).float().cuda()
        box_gt = []
        for batch in range(batch_size):
            index = indices[batch]
            box_preds[batch, index, :] += index.float() * 0.1 + 1
            box_gt.append(box_preds[batch, index, :].clone())
        selected_cls, selected_box = _sampling_ewc(cls_preds,
            box_preds,
            sample_strategy,
            num_of_samples)
        self.assertTrue(selected_cls.shape ==
            torch.Size([num_of_samples, num_of_classes]))
        self.assertTrue(selected_box.shape ==
            torch.Size([num_of_samples, 7]))
        for cls_gt_, box_gt_ in zip(cls_gt, box_gt):
            flag = False
            for selected_cls_, selected_box_ in zip(selected_cls, selected_box):
                flag = (flag or
                    (torch.all(cls_gt_ == selected_cls_)
                     and torch.all(box_gt_ == selected_box_)))
            self.assertTrue(flag)

    def test_sampling_ewc_unbiased(self):
        from incdet3.models.ewc_func import _sampling_ewc
        sample_strategy = "unbiased"
        num_of_samples = 5
        batch_size = 4
        num_of_anchors = 10
        num_of_classes = 8
        size_of_box = 7
        cls_preds = torch.zeros([batch_size, num_of_anchors, num_of_classes]).float().cuda()
        box_preds = torch.zeros([batch_size, num_of_anchors, size_of_box]).float().cuda()
        selected_cls, selected_box = _sampling_ewc(cls_preds,
            box_preds,
            sample_strategy,
            num_of_samples)
        self.assertTrue(selected_cls.shape ==
            torch.Size([num_of_samples, num_of_classes]))
        self.assertTrue(selected_box.shape ==
            torch.Size([num_of_samples, 7]))

# test compute_FIM_cls_term
    def test_compute_FIM_cls_term(self):
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
        from incdet3.models.ewc_func import _compute_FIM_cls_term, _compute_FIM_reg_term
        x = torch.randn([5, 2])
        model = TestModel()
        cls_preds = model(x)
        reg_term = _compute_FIM_reg_term(cls_preds, model, sigma_prior=0.01)
        cls_term = _compute_FIM_cls_term(cls_preds, model)
        # # compute gt
        weight_w, weight_b = 0, 0
        for i in range(x.shape[0]):
            logits = cls_preds[i:i+1, :]
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log(probs)
            d_logy0_w = torch.autograd.grad(outputs=log_probs[0, 0], inputs=model.w, retain_graph=True, only_inputs=True)[0]
            d_logy0_b = torch.autograd.grad(outputs=log_probs[0, 0], inputs=model.b, retain_graph=True, only_inputs=True)[0]
            d_logy1_w = torch.autograd.grad(outputs=log_probs[0, 1], inputs=model.w, retain_graph=True, only_inputs=True)[0]
            d_logy1_b = torch.autograd.grad(outputs=log_probs[0, 1], inputs=model.b, retain_graph=True, only_inputs=True)[0]
            d_logy2_w = torch.autograd.grad(outputs=log_probs[0, 2], inputs=model.w, retain_graph=True, only_inputs=True)[0]
            d_logy2_b = torch.autograd.grad(outputs=log_probs[0, 2], inputs=model.b, retain_graph=True, only_inputs=True)[0]
            weight_w += d_logy0_w**2 * probs[0, 0] + d_logy1_w**2 * probs[0, 1] +  d_logy2_w**2 * probs[0, 2]
            weight_b += d_logy0_b**2 * probs[0, 0] + d_logy1_b**2 * probs[0, 1] +  d_logy2_b**2 * probs[0, 2]
        weight_w /= x.shape[0]
        weight_b /= x.shape[0]
        self.assertTrue(torch.allclose(cls_term["w"], weight_w))
        self.assertTrue(torch.allclose(cls_term["b"], weight_b))

# test compute_FIM_reg_term
    def test_compute_FIM_reg_term(self):
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
        from incdet3.models.ewc_func import _compute_FIM_cls_term, _compute_FIM_reg_term
        sigma_prior = 1
        x = torch.randn([5, 2])
        model = TestModel()
        reg_preds = model(x)
        cls_term = _compute_FIM_cls_term(reg_preds, model)
        reg_term = _compute_FIM_reg_term(reg_preds, model, sigma_prior=sigma_prior)
        # # compute gt
        weight_w = 0
        weight_b = 0
        for i in range(x.shape[0]):
            reg_output = reg_preds[i:i+1]
            d_logy0_w = torch.autograd.grad(outputs=reg_output[0, 0], inputs=model.w, retain_graph=True, only_inputs=True)[0]
            d_logy0_b = torch.autograd.grad(outputs=reg_output[0, 0], inputs=model.b, retain_graph=True, only_inputs=True)[0]
            d_logy1_w = torch.autograd.grad(outputs=reg_output[0, 1], inputs=model.w, retain_graph=True, only_inputs=True)[0]
            d_logy1_b = torch.autograd.grad(outputs=reg_output[0, 1], inputs=model.b, retain_graph=True, only_inputs=True)[0]
            d_logy2_w = torch.autograd.grad(outputs=reg_output[0, 2], inputs=model.w, retain_graph=True, only_inputs=True)[0]
            d_logy2_b = torch.autograd.grad(outputs=reg_output[0, 2], inputs=model.b, retain_graph=True, only_inputs=True)[0]
            weight_w += (d_logy0_w**2 + d_logy1_w**2 +  d_logy2_w**2) / (sigma_prior**2)
            weight_b += (d_logy0_b**2 + d_logy1_b**2 +  d_logy2_b**2) / (sigma_prior**2)
        weight_w = weight_w / x.shape[0]
        weight_b = weight_b / x.shape[0]
        self.assertTrue(torch.all(reg_term["w"] == weight_w))
        self.assertTrue(torch.all(reg_term["b"] == weight_b))

# test update_ewc_weights
    def test_update_ewc_weights(self):
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
        from incdet3.models.ewc_func import (_compute_FIM_reg_term,
            _compute_FIM_cls_term, _update_ewc_weights,
            _init_ewc_weights)
        sigma_prior = 1
        batch_x = torch.randn([3, 5, 2])
        model = TestModel()
        ewc_weights = _init_ewc_weights(model)
        cls_term_list, reg_term_list = [], []
        for i, x in enumerate(batch_x):
            reg_preds = model(x)
            cls_term = _compute_FIM_cls_term(reg_preds, model)
            reg_term = _compute_FIM_reg_term(reg_preds, model, sigma_prior=sigma_prior)
            ewc_weights = _update_ewc_weights(ewc_weights, cls_term, reg_term, i)
            cls_term_list.append(cls_term)
            reg_term_list.append(reg_term)
        gt_ewc_weights = _init_ewc_weights(model)
        for cls_term, reg_term in zip(cls_term_list, reg_term_list):
            for name, _ in gt_ewc_weights.items():
                gt_ewc_weights[name] += cls_term[name] + reg_term[name]
        for name, param in gt_ewc_weights.items():
            gt_ewc_weights[name] = param / batch_x.shape[0]
        for name, param in gt_ewc_weights.items():
            self.assertTrue(torch.allclose(param, ewc_weights[name]))


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

    def test_compute_ewc_weights(self):
        return
        state = np.random.get_state()
        torch_state_cpu = torch.Generator().get_state()
        torch_state_gpu = torch.Generator(device="cuda:0").get_state()
        network = build_network().cuda()
        dataloader = build_dataloader()
        num_of_datasamples = len(dataloader)
        num_of_anchorsamples = 2
        anchor_sample_strategy = "biased"
        debug_mode = False
        from det3.ops import read_pkl, write_pkl
        from incdet3.models.ewc_func import (_compute_FIM_reg_term,
            _compute_FIM_cls_term, _update_ewc_weights,
            _init_ewc_weights, _sampling_ewc)
        from incdet3.builders.dataloader_builder import example_convert_to_torch
        for reg_sigma_prior in [1, 0.1]:
            est_ewc_weights = network.compute_ewc_weights(dataloader,
                num_of_datasamples,
                num_of_anchorsamples,
                anchor_sample_strategy,
                reg_sigma_prior,
                debug_mode)
            gt_ewc_weights = read_pkl(f"unit_tests/results/test_model-ewc_compute_ewc_weights-{reg_sigma_prior}.pkl")
            for name, _ in est_ewc_weights.items():
                self.assertTrue(torch.allclose(est_ewc_weights[name], gt_ewc_weights[name], atol=1e-08, rtol=1e-05))
        np.random.set_state(state)
        torch.Generator().set_state(torch_state_cpu)
        torch.Generator(device="cuda:0").set_state(torch_state_gpu)

    def test_compute_ewc_weights_debug(self):
        return
        state = np.random.get_state()
        torch_state_cpu = torch.Generator().get_state()
        torch_state_gpu = torch.Generator(device="cuda:0").get_state()
        network = build_network().cuda()
        dataloader = build_dataloader()
        num_of_datasamples = len(dataloader)
        num_of_anchorsamples = 2
        anchor_sample_strategy = "biased"
        debug_mode = True
        reg_sigma_prior = 1
        from det3.ops import read_pkl, write_pkl
        import subprocess
        import os
        cls_term, reg_term, est_ewc_weights = network.compute_ewc_weights(dataloader,
            num_of_datasamples,
            num_of_anchorsamples,
            anchor_sample_strategy,
            reg_sigma_prior,
            debug_mode)
        gt_ewc_weights1 = network.compute_ewc_weights(dataloader,
            num_of_datasamples,
            num_of_anchorsamples,
            anchor_sample_strategy,
            reg_sigma_prior=1,
            debug_mode=False)
        gt_ewc_weights01 = network.compute_ewc_weights(dataloader,
            num_of_datasamples,
            num_of_anchorsamples,
            anchor_sample_strategy,
            reg_sigma_prior=0.1,
            debug_mode=False)
        for name, _ in est_ewc_weights.items():
            self.assertTrue(torch.allclose(est_ewc_weights[name], gt_ewc_weights1[name], atol=1e-08, rtol=1e-05))
        flag = False
        for name, _ in est_ewc_weights.items():
            if not torch.allclose(est_ewc_weights[name], gt_ewc_weights01[name], atol=1e-08, rtol=1e-05):
                flag = True
        self.assertTrue(flag)
        write_pkl({k: v.cpu().numpy() for k, v in cls_term.items()}, f"ewc_clsterm-tmp.pkl")
        write_pkl({k: v.cpu().numpy() for k, v in reg_term.items()}, f"ewc_regterm-tmp.pkl")
        cmd = "python tools/impose_ewc-regsigmaprior.py "
        cmd += "--cls-term-path ewc_clsterm-tmp.pkl "
        cmd += "--reg-term-path ewc_regterm-tmp.pkl "
        cmd += "--reg-sigma-prior 0.1 "
        cmd += "--output-path ewc_weights-tmp-0.1.pkl"
        subprocess.check_output(cmd, shell=True)
        est_ewc_weights01 = read_pkl("ewc_weights-tmp-0.1.pkl")
        for name, _ in est_ewc_weights01.items():
            self.assertTrue(np.allclose(est_ewc_weights01[name], gt_ewc_weights01[name].cpu().numpy(), atol=1e-08, rtol=1e-05))
        os.remove("ewc_clsterm-tmp.pkl")
        os.remove("ewc_regterm-tmp.pkl")
        os.remove("ewc_weights-tmp-0.1.pkl")
        np.random.set_state(state)
        torch.Generator().set_state(torch_state_cpu)
        torch.Generator(device="cuda:0").set_state(torch_state_gpu)

if __name__ == "__main__":
    unittest.main()