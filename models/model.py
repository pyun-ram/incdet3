'''
network needs to handle the following training schemes (by set train/val & require_grad):
"feature extraction", "fine-tuning", "joint training", "lwf"
network needs to handle different distillation schemes
"l2sp" (loss), "delta" (loss + hook), "distillation loss" (loss + hook), "ewc" (loss)
To be compatible with "delta", we need to change the rpn part to resnet.
'''
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from det3.utils.log_tool import Logger
from det3.utils.utils import is_param, proc_param
from det3.methods.second.models.voxel_encoder import get_vfe_class
from det3.methods.second.models.middle import get_middle_class
from det3.methods.second.core.model_manager import save_models
from det3.methods.second.models.losses import get_loss_class
from det3.methods.second.core.second import add_sin_difference, get_direction_target
from det3.methods.second.ops.torch_ops import rotate_nms
from det3.ops import read_pkl
from incdet3.models.rpn import get_rpn_class
from incdet3.utils.utils import bcolors
from incdet3.builders.dataloader_builder import example_convert_to_torch
from incdet3.models.ewc_func import (_init_ewc_weights, _sampling_ewc,
    _compute_FIM_cls_term, _compute_FIM_reg_term, _update_ewc_weights,
    _cycle_next, _update_ewc_term, compute_FIM_v2, update_ewc_weights_v2,
    ewc_measure_distance, parse_numclasses_numanchorperloc, expand_old_weights)
from incdet3.models.mas_func import (_init_mas_weights, _sampling_mas,
    _compute_omega_cls_term, _compute_omega_reg_term, _compute_omega,
    _update_mas_term)

class Network(nn.Module):
    HEAD_NEAMES = ["rpn.conv_cls", "rpn.conv_box", "rpn.conv_dir_cls"]
    def __init__(self,
        classes_target,
        classes_source,
        model_resume_dict,
        sub_model_resume_dict,
        voxel_encoder_dict,
        middle_layer_dict,
        rpn_dict,
        training_mode,
        is_training,
        weight_decay_coef=0.01,
        cls_loss_weight=1.0,
        loc_loss_weight=1.0,
        dir_loss_weight=1.0,
        pos_cls_weight=1.0,
        neg_cls_weight=1.0,
        l2sp_alpha_coef=1.0,
        # l2sp_beta_coef=1.0,
        delta_coef=1.0,
        distillation_loss_cls_coef=1.0,
        distillation_loss_reg_coef=1.0,
        num_biased_select=64,
        threshold_delta_fgmask=0.3,
        loss_dict={},
        hook_layers=[],
        distillation_mode=[],
        bool_reuse_anchor_for_cls=True,
        bool_biased_select_with_submodel=False, # decorated
        distillation_loss_sampling_strategy="biased",
        bool_oldclassoldanchor_predicts_only=False,
        bool_delta_use_mask=False,
        post_center_range=[],
        nms_score_thresholds=[],
        nms_pre_max_sizes=[],
        nms_post_max_sizes=[],
        nms_iou_thresholds=[],
        box_coder=None,
        ewc_weights_path=None,
        ewc_coef=1.0,
        ewc_loss_type="l2",
        ewc_huberloss_beta=1e-6,
        ):
        super().__init__()
        self._model = None
        self._model_resume_dict = model_resume_dict
        self._sub_model = None
        self._sub_model_resume_dict = sub_model_resume_dict
        # _classes_target is the inference classes of _model
        self._classes_target = classes_target
        # _classes_source is the inference classes of _sub_model
        self._classes_source = classes_source
        # train-from-scratch, feature-extraction, fine-tuning, joint-training, lwf
        self._training_mode = training_mode
        # distillation scheme: [l2sp, delta, distillation-loss]
        self._distillation_mode = distillation_mode
        self._is_training = is_training
        self._hook_layers = hook_layers
        self._hook_features_model = []
        self._hook_features_submodel = []
        self.register_buffer("global_step", torch.LongTensor(1).zero_())
        self._box_coder = box_coder

        self._cls_loss_weight = cls_loss_weight
        self._loc_loss_weight = loc_loss_weight
        self._dir_loss_weight = dir_loss_weight
        self._pos_cls_weight = pos_cls_weight
        self._neg_cls_weight = neg_cls_weight
        self._loss_norm_type = "NormByNumPositives"
        self._weight_decay_coef = weight_decay_coef
        self._l2sp_alpha_coef = l2sp_alpha_coef
        # self._l2sp_beta_coef = l2sp_beta_coef
        self._delta_coef = delta_coef
        self._distillation_loss_cls_coef = distillation_loss_cls_coef
        self._distillation_loss_reg_coef = distillation_loss_reg_coef
        self._ewc_coef = ewc_coef
        self._num_biased_select = num_biased_select
        self._threshold_delta_fgmask = threshold_delta_fgmask
        self._distillation_loss_sampling_strategy = distillation_loss_sampling_strategy
        Logger.log_txt(f"Network._bool_biased_select_with_submodel has been decorated."
                      +f"We do sampling according to submodel outputs.")
        self._bool_oldclassoldanchor_predicts_only = bool_oldclassoldanchor_predicts_only
        self._post_center_range = post_center_range
        self._nms_score_thresholds = nms_score_thresholds
        self._nms_pre_max_sizes = nms_pre_max_sizes
        self._nms_post_max_sizes = nms_post_max_sizes
        self._nms_iou_thresholds = nms_iou_thresholds

        network_cfg = {
            "VoxelEncoder": voxel_encoder_dict,
            "MiddleLayer": middle_layer_dict,
            "RPN": rpn_dict,
        }
        if len(loss_dict) != 0:
            param = {proc_param(k):v
                for k, v in loss_dict["ClassificationLoss"].items() if is_param(k)}
            self._cls_loss_ftor = get_loss_class(loss_dict["ClassificationLoss"]["name"])(**param)
            param = {proc_param(k):v
                for k, v in loss_dict["LocalizationLoss"].items() if is_param(k)}
            self._loc_loss_ftor = get_loss_class(loss_dict["LocalizationLoss"]["name"])(**param)
            param = {proc_param(k):v
                for k, v in loss_dict["DirectionLoss"].items() if is_param(k)}
            self._dir_cls_loss_ftor = get_loss_class(loss_dict["DirectionLoss"]["name"])(**param)
            if "distillation_loss" in self._distillation_mode:
                param = {proc_param(k):v
                    for k, v in loss_dict["DistillationClassificationLoss"].items() if is_param(k)}
                self._distillation_loss_cls_ftor = get_loss_class(loss_dict["DistillationClassificationLoss"]["name"])(**param)
                param = {proc_param(k):v
                    for k, v in loss_dict["DistillationRegressionLoss"].items() if is_param(k)}
                self._distillation_loss_reg_ftor = get_loss_class(loss_dict["DistillationRegressionLoss"]["name"])(**param)

        self._bool_reuse_anchor_for_cls = bool_reuse_anchor_for_cls
        if self._training_mode in ["feature_extraction", "fine_tuning", "joint_training"]:
            self._num_old_classes = self._model_resume_dict["num_classes"]
            self._num_old_anchor_per_loc = self._model_resume_dict["num_anchor_per_loc"]
            self._num_new_classes = len(self._classes_target)
            self._num_new_anchor_per_loc = 2 * self._num_new_classes
        elif self._training_mode in ["train_from_scratch"]:
            self._num_old_classes = 0
            self._num_old_anchor_per_loc = 0
            self._num_new_classes = len(self._classes_target)
            self._num_new_anchor_per_loc = 2 * self._num_new_classes
        elif self._training_mode == "lwf":
            self._num_old_classes = len(self._classes_source)
            self._num_old_anchor_per_loc = 2 * self._num_old_classes
            self._num_new_classes = len(self._classes_target)
            self._num_new_anchor_per_loc = 2 * self._num_new_classes
        else:
            raise NotImplementedError

        self._channel_weights = None
        self._bool_delta_use_mask = False
        if "delta" in self._distillation_mode:
            self._channel_weights = None # TBD
            self._bool_delta_use_mask = bool_delta_use_mask

        if "ewc" in self._distillation_mode:
            self._ewc_weights = self._load_ewc_weights(ewc_weights_path)
            self._ewc_loss_type = ewc_loss_type
            self._ewc_huberloss_beta = ewc_huberloss_beta

        self._model = Network._build_model_and_init(
            classes=self._classes_target,
            network_cfg=network_cfg,
            resume_dict=self._model_resume_dict,
            name="IncDetMain")
        if self._model_resume_dict is not None:
            try:
                resume_step = self._model_resume_dict["ckpt_path"].split("/")[-1]
                resume_step = resume_step.split(".")[0]
                resume_step = int(resume_step.split("-")[-1])
                self.global_step += resume_step
            except:
                Logger.log_txt("Failed to parse global_step from ckpt_path. Will train from step 0.")
        if len(self._distillation_mode) > 0:
            self._sub_model = self._build_model_and_init(
                classes=self._classes_source,
                network_cfg=network_cfg,
                resume_dict=self._sub_model_resume_dict,
                name="IncDetSub")
            self._register_model_hook()
            self._register_submodel_hook()
        else:
            self._sub_model = None
        self._freeze_model()

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])

    def update_global_step(self):
        self.global_step += 1

    def _register_model_hook(self):
        def _model_hook_func(module, input, output):
            '''
            hook function for model
            '''
            self._hook_features_model.append(output)
        for name, layer in self._model.named_modules():
            if name in self._hook_layers:
                layer.register_forward_hook(_model_hook_func)

    def _register_submodel_hook(self):
        def _submodel_hook_func(module, input, output):
            '''
            hook function for sub_model
            '''
            self._hook_features_submodel.append(output)
        for name, layer in self._sub_model.named_modules():
            if name in self._hook_layers:
                layer.register_forward_hook(_submodel_hook_func)

    def _detach_variables(self, preds_dict):
        '''
        @training_mode
        @preds_dict: {
                box_preds: ...
                cls_preds: ...
                dir_cls_preds: ...
            }
        '''
        if self._training_mode in ["train_from_scratch", "joint_training", "lwf"]:
            return
        elif self._training_mode in ["feature_extraction", "fine_tuning"]:
            num_old_classes = self._num_old_classes
            num_old_anchor_per_loc = self._num_old_anchor_per_loc
            old_classes = [i for i in range(self._num_old_classes)]
            new_classes = [i for i in range(self._num_new_classes)]
            new_classes_to_learn = [itm for itm in new_classes if itm not in old_classes]
            tmp_var = preds_dict["cls_preds"]
            tmp_ts = preds_dict["cls_preds"].detach()
            if not self._bool_reuse_anchor_for_cls:
                for cls in new_classes_to_learn:
                    anchor_list = [2 * cls, 2 * cls + 1]
                    tmp_ts[:, [anchor_list], ..., [cls]] = tmp_var[:, [anchor_list], ..., [cls]]
                preds_dict["cls_preds"] = tmp_ts
            preds_dict["cls_preds"][:, :num_old_anchor_per_loc, ..., :num_old_classes] = \
                preds_dict["cls_preds"][:, :num_old_anchor_per_loc, ..., :num_old_classes].detach()
            preds_dict["box_preds"][:, :num_old_anchor_per_loc, ...] = \
                preds_dict["box_preds"][:, :num_old_anchor_per_loc, ...].detach()
            preds_dict["dir_cls_preds"][:, :num_old_anchor_per_loc, ...] = \
                preds_dict["dir_cls_preds"][:, :num_old_anchor_per_loc, ...].detach()
        else:
            raise NotImplementedError

    def _freeze_model(self):
        '''
        freeze model according to training_mode. (modify require_grad)
        @ model: nn.ModuleDict {"vfe_layer", "middle_layer", "rpn"}
        @ training_mode: str
        "train_from_scratch",
        "feature_extraction", "fine_tuning","joint_training", "lwf"
        Note1: This function will not affect the behaviors of batch norms.
        Therefore, you need to set the train() or eval() to get suitable batch norm manners.
        Note2: To achieve the specific training mode ("finetune", "feature_extraction"),
        this function should work with specific forward() to detach the correspondent tensors.
        Note3: This function is specifically tailored for SECOND.
        '''
        training_mode = self._training_mode
        if training_mode in ["train_from_scratch", "joint_training", "fine_tuning"]:
            for name, param in self._model.named_parameters():
                param.requires_grad = True
        elif training_mode == "lwf":
            for name, param in self._model.named_parameters():
                param.requires_grad = True
            for name, param in self._sub_model.named_parameters():
                param.requires_grad = False
        elif training_mode == "feature_extraction":
            for name, param in self._model.named_parameters():
                if "rpn.conv_cls" in name:
                    param.requires_grad = False
                elif "rpn.conv_box" in name:
                    param.requires_grad = False
                elif "rpn.conv_dir_cls" in name:
                    param.requires_grad = False
                elif "rpn.featext" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            raise NotImplementedError

    def _network_forward(self, model, voxels, num_points, coors, batch_size):
        """this function is used for subclass.
        @preds_dict: {
                box_preds: ...
                cls_preds: ...
                dir_cls_preds: ...
            }
        """
        voxel_features = model.vfe_layer(voxels, num_points, coors)
        spatial_features = model.middle_layer(voxel_features, coors, batch_size)
        preds_dict = model.rpn(spatial_features)
        return preds_dict

    def forward(self, data):
        voxels = data["voxels"]
        num_points = data["num_points"]
        coors = data["coordinates"]
        batch_anchors = data["anchors"]
        batch_size = batch_anchors.shape[0]
        preds_dict = self._network_forward(self._model, voxels, num_points, coors, batch_size)
        
        if not self._bool_reuse_anchor_for_cls:
            new_classes = [i for i in range(self._num_new_classes)]
            cls_preds = torch.zeros_like(preds_dict["cls_preds"])-10
            for cls in new_classes:
                anchor_list = [2 * cls, 2 * cls + 1]
                cls_preds[:, [anchor_list], ..., [cls]] = preds_dict["cls_preds"][:, [anchor_list], ..., [cls]]
            preds_dict["cls_preds"] = cls_preds
        box_preds = preds_dict["box_preds"].view(batch_size, -1, 7)
        err_msg = f"num_anchors={batch_anchors.shape[1]}, but num_output={box_preds.shape[1]}. please check size"
        assert batch_anchors.shape[1] == box_preds.shape[1], err_msg
        if self.training:
            self._detach_variables(preds_dict)
            return self.loss(data, preds_dict, channel_weights=self._channel_weights)
        else:
            with torch.no_grad():
                res = self.predict(data, preds_dict,
                    ext_dict={"box_coder":self._box_coder})
            return res

    @staticmethod
    def _build_model_and_init(
        classes,
        network_cfg,
        name,
        resume_dict):
        '''
        -> nn.ModuleDict{
            "vfe_layer": nn.Module,
            "middle_layer": nn.Module,
            "RPN": nn.Module
        }
        '''
        # build vfe layer
        vfe_cfg = network_cfg["VoxelEncoder"].copy()
        vfe_name = vfe_cfg["name"]
        if vfe_name == "SimpleVoxel":
            param = {proc_param(k): v
                for k, v in vfe_cfg.items() if is_param(k)}
            vfe_layer = get_vfe_class(vfe_name)(**param)
        else:
            raise NotImplementedError
        # build middle layer
        ml_cfg = network_cfg["MiddleLayer"].copy()
        ml_name = ml_cfg["name"]
        if ml_name == "SpMiddleFHD":
            param = {proc_param(k): v
                for k, v in ml_cfg.items() if is_param(k)}
            middle_layer = get_middle_class(ml_name)(**param)
        else:
            raise NotImplementedError
        # build rpn
        rpn_cfg = network_cfg["RPN"].copy()
        rpn_name = rpn_cfg["name"]
        rpn_cfg["@num_class"] = len(classes)
        rpn_cfg["@num_anchor_per_loc"] =  len(classes) * 2 # two anchors for each class
        if rpn_name in ["RPNV2", "ResNetRPN", "ResNetRPNFeatExt"]:
            param = {proc_param(k): v
                for k, v in rpn_cfg.items() if is_param(k)}
            rpn = get_rpn_class(rpn_name)(**param)
        else:
            raise NotImplementedError
        model = nn.ModuleDict({
            "vfe_layer": vfe_layer,
            "middle_layer": middle_layer,
            "rpn": rpn
        })
        model.name = name
        # after initilization, both middle_layer and rpn have been initialized with kaiming norm.
        Network.load_weights(model, resume_dict)
        return model

    @staticmethod
    def load_weights(
        model,
        resume_dict):
        '''
        load weights (support partially load weights for specific layers)
        @model: nn.Module
        @resume_dict: dict
        e.g.
        resume_dict = {
        "ckpt_path": "VoxelNet-23427.tckpt",
        "num_classes": 4,
        "num_anchor_per_loc": 8,
        "partially_load_params": [
            "rpn.conv_cls.weight", "rpn.conv_cls.bias",
            "rpn.conv_box.weight", "rpn.conv_box.bias",
            "rpn.conv_dir_cls.weight", "rpn.conv_dir_cls.bias"]
        # "partially_load_params": [] if direct load
        }
        Note: This function is specifically for SECOND.
        If you want to tailor it for other models,
        you should carefully change the _create_param_from_old_sd()
        '''
        def _create_param_from_old_sd(
            old_param,
            new_param,
            old_num_classes,
            old_num_anchor_per_loc,
            new_num_classes,
            new_num_anchor_per_loc,
            key):
            '''
            This function is specifically for SECOND.
            '''
            new_shape = new_param.shape
            old_shape = old_param.shape
            if "rpn.conv_cls" in key:
                new_tmp = new_param.reshape(new_num_anchor_per_loc, new_num_classes, *new_shape[1:])
                old_tmp = old_param.reshape(old_num_anchor_per_loc, old_num_classes, *old_shape[1:])
                new_tmp[:old_num_anchor_per_loc, :old_num_classes, ...] = \
                    old_tmp[:old_num_anchor_per_loc, :old_num_classes, ...]
            elif "rpn.conv_box" in key:
                new_tmp = new_param.reshape(new_num_anchor_per_loc, 7, *new_shape[1:])
                old_tmp = old_param.reshape(old_num_anchor_per_loc, 7, *old_shape[1:])
                new_tmp[:old_num_anchor_per_loc, ...] = \
                    old_tmp[:old_num_anchor_per_loc, ...]
            elif "rpn.conv_dir_cls"in key:
                new_tmp = new_param.reshape(new_num_anchor_per_loc, 2, *new_shape[1:])
                old_tmp = old_param.reshape(old_num_anchor_per_loc, 2, *old_shape[1:])
                new_tmp[:old_num_anchor_per_loc, ...] = \
                    old_tmp[:old_num_anchor_per_loc, ...]
            else:
                raise NotImplementedError
            new_tmp = new_tmp.reshape(new_shape).contiguous()
            return new_tmp
        if resume_dict is None:
            return
        ckpt_path = resume_dict["ckpt_path"]
        old_num_classes = resume_dict["num_classes"]
        old_num_anchor_per_loc = resume_dict["num_anchor_per_loc"]
        partially_load_params = resume_dict["partially_load_params"]
        ignore_params = (resume_dict["ignore_params"]
            if "ignore_params" in resume_dict.keys() else [])
        parse_cls_conv_layer = {
            2: {"num_classes": 1, "num_anchor_per_loc": 2},
            8: {"num_classes": 2, "num_anchor_per_loc": 4},
            18: {"num_classes": 3, "num_anchor_per_loc": 6},
            32: {"num_classes": 4, "num_anchor_per_loc": 8},
            50: {"num_classes": 5, "num_anchor_per_loc": 10},
        }
        parse_cls_conv_layer = {
            **parse_cls_conv_layer,
            **{2*i*i: {"num_classes": i, "num_anchor_per_loc": 2*i}
               for i in range(6, 11)}}
        if ckpt_path is None:
            return
        if not Path(ckpt_path).is_file():
            raise ValueError("checkpoint {} not exist.".format(ckpt_path))
        Logger.log_txt("Restoring parameters from {}".format(ckpt_path))
        old_sd = torch.load(ckpt_path)
        new_sd = model.state_dict()
        num_conv_cls_bias = new_sd["rpn.conv_cls.bias"].shape[0]
        new_num_classes = parse_cls_conv_layer[num_conv_cls_bias]["num_classes"]
        new_num_anchor_per_loc = parse_cls_conv_layer[num_conv_cls_bias]["num_anchor_per_loc"]
        old_sd_change = {}
        if "KITTIRESUME.pkl" in ckpt_path:
            for k, v in old_sd.items():
                if "middle_feature_extractor" in k:
                    new_k = "middle_layer"
                    new_k = [new_k] + k.split(".")[1:]
                    new_k = ".".join(new_k)
                else:
                    new_k = k
                old_sd_change[new_k] = v
            old_sd = old_sd_change
        for key in new_sd.keys():
            if key in ignore_params:
                new_sd[key] = new_sd[key]
                Logger.log_txt("Ignore loading {}.".format(key))
                continue
            if key not in partially_load_params:
                new_sd[key] = old_sd[key]
            else:
                Logger.log_txt("Partilly loaded {}.".format(key))
                new_sd[key] = _create_param_from_old_sd(
                    old_param=old_sd[key],
                    new_param=new_sd[key],
                    old_num_classes=old_num_classes,
                    old_num_anchor_per_loc=old_num_anchor_per_loc,
                    new_num_classes=new_num_classes,
                    new_num_anchor_per_loc=new_num_anchor_per_loc,
                    key=key)
        missing_keys, unexpected_keys = model.load_state_dict(new_sd, strict=False)
        Logger.log_txt("Missing Keys {}.".format(missing_keys))
        Logger.log_txt("Unexpected Keys from {}.".format(unexpected_keys))

    def _load_ewc_weights(self, path):
        '''
        load ewc_weights from path
        @path: pkl path
        -> ewc_weights: {name: torch.FloatTensor.cuda}
        '''
        ewc_weights = read_pkl(path)
        ewc_weights_ts = {}
        for name, param in ewc_weights.items():
            if isinstance(param, np.ndarray):
                ewc_weights_ts[name] = torch.from_numpy(param).cuda()
            elif isinstance(param, np.float64):
                ewc_weights_ts[name] = torch.FloatTensor([param]).cuda()
            else:
                print(f"Failed to load ewc weights: {name} {type(param)}")
                raise RuntimeError
        print(f"loading ewc weights from {path}")
        return ewc_weights_ts

    @staticmethod
    def save_weight(model, save_dir, itr):
        save_models(save_dir, [model],
            itr, max_to_keep=float('inf'))
        Logger.log_txt("Saving parameters to {}".format(save_dir))

    def loss(self,
        example,
        preds_dict,
        channel_weights=None):
        box_preds = preds_dict["box_preds"]
        cls_preds = preds_dict["cls_preds"]
        batch_size_dev = cls_preds.shape[0]
        labels = example['labels']
        reg_targets = example['reg_targets']
        importance = example['importance']
        loss_dict = dict()

        weights = Network._prepare_loss_weights(
            labels,
            pos_cls_weight=self._pos_cls_weight,
            neg_cls_weight=self._neg_cls_weight,
            loss_norm_type=self._loss_norm_type,
            importance=importance,
            use_direction_classifier=True,
            dtype=box_preds.dtype)
        cls_targets = labels * weights["cared"].type_as(labels)
        cls_targets = cls_targets.unsqueeze(-1)
        loss_dict["loss_cls"] = self._compute_classification_loss(
            est=cls_preds,
            gt=cls_targets,
            weights=weights["cls_weights"]*importance)*self._cls_loss_weight
        loss_dict["loss_reg"] = self._compute_location_loss(
            est=box_preds,
            gt=reg_targets,
            weights=weights["reg_weights"]*importance)*self._loc_loss_weight
        dir_targets = get_direction_target(
            example["anchors"],
            reg_targets,
            dir_offset=0,
            num_bins=2)
        dir_logits = preds_dict["dir_cls_preds"].view(batch_size_dev, -1, 2)
        loss_dict["loss_dir_cls"] = self._compute_direction_loss(
            est=dir_logits,
            gt=dir_targets,
            weights=weights["dir_weights"]*importance)*self._dir_loss_weight
        loss_dict["loss_l2"] = self._compute_l2_loss()
        if "l2sp" in self._distillation_mode:
            loss_dict["loss_l2sp"] = self._compute_l2sp_loss()
        if "ewc" in self._distillation_mode:
            loss_dict["loss_ewc"] = self._compute_ewc_loss()
        if "delta" in self._distillation_mode:
            preds_dict_sub = self._network_forward(self._sub_model,
                example["voxels"],
                example["num_points"],
                example["coordinates"],
                example["anchors"].shape[0])
            delta_fg_mask = (Network._delta_create_fg_mask(
                preds_dict_sub["cls_preds"],
                score_threshold=self._threshold_delta_fgmask,
                num_feat_channel=128)
                if self._bool_delta_use_mask else None)
            loss_dict["loss_delta"] = self._compute_delta_loss(channel_weights, delta_fg_mask)
            self._hook_features_model.clear()
            self._hook_features_submodel.clear()
        if "distillation_loss" in self._distillation_mode:
            if "delta" not in self._distillation_mode:
                preds_dict_sub = self._network_forward(self._sub_model,
                    example["voxels"],
                    example["num_points"],
                    example["coordinates"],
                    example["anchors"].shape[0])
            loss_dl = self._compute_distillation_loss(preds_dict, preds_dict_sub)
            loss_dict["loss_distillation_loss_cls"] = loss_dl["loss_distillation_loss_cls"]
            loss_dict["loss_distillation_loss_reg"] = loss_dl["loss_distillation_loss_reg"]
        for k, v in loss_dict.items():
            if k in ["loss_l2", "loss_l2sp", "loss_ewc"]:
                loss_dict[k] = v.sum()
            else:
                loss_dict[k] = v.sum() / batch_size_dev
        loss_dict["loss_total"] = sum(loss_dict.values())
        return loss_dict

    def _compute_l2_loss(self):
        loss = 0
        # handle train_from_scratch since it does not have
        # num_old_classes and num_old_anchor_per_loc
        if self._training_mode == "train_from_scratch":
            for name, param in self._model.named_parameters():
                loss += 0.5 * torch.norm(param, 2) ** 2
            return loss * self._weight_decay_coef
        num_old_classes = self._num_old_classes
        num_old_anchor_per_loc = self._num_old_anchor_per_loc
        num_new_classes = self._num_new_classes
        num_new_anchor_per_loc = self._num_new_anchor_per_loc
        # handle feature_extraction since it have extended layers
        if self._training_mode == "feature_extraction":
            for name, param in self._model.named_parameters():
                compute_param_shape = param.shape
                if not param.requires_grad:
                    continue
                elif name.startswith("rpn.featext_conv_cls"):
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
                    loss += 0.5 * torch.norm(compute_newparam, 2) ** 2
                elif name.startswith("rpn.featext_conv_box"):
                    compute_param = param.reshape(num_new_anchor_per_loc, 7, *compute_param_shape[1:])
                    compute_oldparam = (compute_param[:num_old_anchor_per_loc, ...]
                        .reshape(-1, *compute_param_shape[1:]).contiguous())
                    compute_newparam = (compute_param[num_old_anchor_per_loc:, ...]
                        .reshape(-1, *compute_param_shape[1:]).contiguous())
                    loss += 0.5 * torch.norm(compute_newparam, 2) ** 2
                elif name.startswith("rpn.featext_conv_dir_cls"):
                    compute_param = param.reshape(num_new_anchor_per_loc, 2, *compute_param_shape[1:])
                    compute_oldparam = (compute_param[:num_old_anchor_per_loc, ...]
                        .reshape(-1, *compute_param_shape[1:]).contiguous())
                    compute_newparam = (compute_param[num_old_anchor_per_loc:, ...]
                        .reshape(-1, *compute_param_shape[1:]).contiguous())
                    loss += 0.5 * torch.norm(compute_newparam, 2) ** 2
                else:
                    loss += 0.5 * torch.norm(param, 2) ** 2
            return loss * self._weight_decay_coef
        # handle joint_training, fine_tuning, lwf
        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                continue
            is_head = any([name.startswith(itm) for itm in Network.HEAD_NEAMES])
            compute_param_shape = param.shape
            if not is_head and "l2sp" not in self._distillation_mode:
                loss += 0.5 * torch.norm(param, 2) ** 2
            elif not is_head and "l2sp" in self._distillation_mode:
                # l2sp will compute the weight decay according to old weights
                loss += 0
            elif is_head and "l2sp" not in self._distillation_mode and self._training_mode == "joint_training":
                loss += 0.5 * torch.norm(param, 2) ** 2
            elif name.startswith("rpn.conv_cls"):
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
                loss += 0.5 * torch.norm(compute_newparam, 2) ** 2
            elif name.startswith("rpn.conv_box"):
                compute_param = param.reshape(num_new_anchor_per_loc, 7, *compute_param_shape[1:])
                compute_oldparam = (compute_param[:num_old_anchor_per_loc, ...]
                    .reshape(-1, *compute_param_shape[1:]).contiguous())
                compute_newparam = (compute_param[num_old_anchor_per_loc:, ...]
                    .reshape(-1, *compute_param_shape[1:]).contiguous())
                loss += 0.5 * torch.norm(compute_newparam, 2) ** 2
            elif name.startswith("rpn.conv_dir_cls"):
                compute_param = param.reshape(num_new_anchor_per_loc, 2, *compute_param_shape[1:])
                compute_oldparam = (compute_param[:num_old_anchor_per_loc, ...]
                    .reshape(-1, *compute_param_shape[1:]).contiguous())
                compute_newparam = (compute_param[num_old_anchor_per_loc:, ...]
                    .reshape(-1, *compute_param_shape[1:]).contiguous())
                loss += 0.5 * torch.norm(compute_newparam, 2) ** 2
            else:
                raise NotImplementedError
        return loss * self._weight_decay_coef

    @staticmethod
    def _prepare_loss_weights(labels,
        pos_cls_weight,
        neg_cls_weight,
        loss_norm_type,
        importance,
        use_direction_classifier,
        dtype=torch.float32):
        '''
        prepare weights for each anchor according to pos_cls_weight, neg_cls_weight, loss_norm_type
        @pos_cls_weight: float
        @neg_cls_weight: float
        @loss_norm_type: str "NormByNumPositives"
        @importance: [batch_size, #anchor] torch.FloatTensor (all 1.0)
        -> weights = {
            "cls_weights", weights for classification
            "reg_weights", weights for bbox regression
            "dir_weights", weights for dir classification
            "cares", labels >= 0
        }
        '''
        cared = labels >= 0
        # cared: [N, num_anchors]
        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = negatives.type(dtype) * neg_cls_weight
        cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
        reg_weights = positives.type(dtype)
        cls_weights *= importance
        reg_weights *= importance
        if loss_norm_type == "NormByNumPositives":  # for focal loss
            pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        else:
            raise NotImplementedError
        if use_direction_classifier:
            dir_weights = positives.type(dtype) * importance
            dir_normalizer = dir_weights.sum(-1, keepdim=True)
            dir_weights /= torch.clamp(dir_normalizer, min=1.0)
        weights = {
            "cls_weights": cls_weights,
            "reg_weights": reg_weights,
            "dir_weights": dir_weights,
            "cared": cared
        }
        return weights

    def _compute_classification_loss(
            self,
            est,
            gt,
            weights):
        '''
        @est: [batch_size, num_anchors_per_loc, H, W, num_classes] logits
        @gt: [batch_size, num_anchors, 1] int32
        @weights: [batch_size, num_anchors] float32
        '''
        batch_size = int(est.shape[0])
        num_class = len(self._classes_target)
        est_cls = est.view(batch_size, -1, num_class)
        gt_cls = gt.squeeze(-1).long()
        one_hot_targets = nn.functional.one_hot(gt_cls, num_classes=num_class+1).float()
        one_hot_targets = one_hot_targets[..., 1:]
        return self._cls_loss_ftor(est_cls, one_hot_targets, weights=weights)

    def _compute_location_loss(self,
        est,
        gt,
        weights):
        '''
        @est: [batch_size, num_anchors_per_loc, H, W, 7] float32
        @gt: [batch_size, num_anchors, 7] float32
        @weights: [batch_size, num_anchors] float32
        '''
        batch_size = int(est.shape[0])
        est_box = est.view(batch_size, -1, 7)
        gt_box = gt
        est_box, gt_box = add_sin_difference(
            est_box, gt_box,
            est_box[..., 6:7], gt_box[..., 6:7], factor=1.0)
        return self._loc_loss_ftor(est_box, gt_box, weights=weights)

    def _compute_direction_loss(self,
        est,
        gt,
        weights):
        '''
        @est: [batch_size, num_anchors_per_loc, H, W, 2] logits
        @gt: [batch_size, num_anchors, 1] int32
        @weights: [batch_size, num_anchors] float32
        '''
        batch_size = int(est.shape[0])
        est_dir = est.view(batch_size, -1, 2)
        return self._dir_cls_loss_ftor(est_dir, gt, weights=weights)

    def _compute_ewc_loss(self):
        loss_alpha = 0
        assert self._training_mode not in ["train_from_scratch", "feature_extraction"]
        # loss_beta = 0
        sub_model_weights = self._sub_model.state_dict()
        num_old_classes = self._num_old_classes
        num_old_anchor_per_loc = self._num_old_anchor_per_loc
        num_new_classes = self._num_new_classes
        num_new_anchor_per_loc = self._num_new_anchor_per_loc
        for name, param in self._model.named_parameters():
            is_head = any([name.startswith(headname) for headname in Network.HEAD_NEAMES])
            if not is_head:
                diff = param - sub_model_weights[name].detach()
                diff = ewc_measure_distance(
                    diff=diff,
                    loss_type=self._ewc_loss_type,
                    beta=self._ewc_huberloss_beta,
                    weights=self._ewc_weights[name])
                loss_alpha += diff.sum()
            else:
                compute_param_shape = param.shape
                if name.startswith("rpn.conv_cls"):
                    compute_param = param.reshape(num_new_anchor_per_loc,
                        num_new_classes, *compute_param_shape[1:])
                    compute_oldparam = compute_param[:num_old_anchor_per_loc,
                        :num_old_classes, ...].reshape(-1, *compute_param_shape[1:]).contiguous()
                elif name.startswith("rpn.conv_box"):
                    compute_param = param.reshape(num_new_anchor_per_loc,
                        7, *compute_param_shape[1:])
                    compute_oldparam = compute_param[:num_old_anchor_per_loc,
                        ...].reshape(-1, *compute_param_shape[1:]).contiguous()
                elif name.startswith("rpn.conv_dir_cls"):
                    compute_param = param.reshape(num_new_anchor_per_loc,
                        2, *compute_param_shape[1:])
                    compute_oldparam = compute_param[:num_old_anchor_per_loc,
                        ...].reshape(-1, *compute_param_shape[1:]).contiguous()
                else:
                    raise NotImplementedError
                diff = compute_oldparam - sub_model_weights[name].detach()
                diff = ewc_measure_distance(
                    diff=diff,
                    loss_type=self._ewc_loss_type,
                    beta=self._ewc_huberloss_beta,
                    weights=self._ewc_weights[name])
                loss_alpha += diff.sum()
        return self._ewc_coef * loss_alpha

    def _compute_l2sp_loss(self):
        loss_alpha = 0
        assert self._training_mode not in ["train_from_scratch", "feature_extraction"]
        # loss_beta = 0
        sub_model_weights = self._sub_model.state_dict()
        num_old_classes = self._num_old_classes
        num_old_anchor_per_loc = self._num_old_anchor_per_loc
        num_new_classes = self._num_new_classes
        num_new_anchor_per_loc = self._num_new_anchor_per_loc
        for name, param in self._model.named_parameters():
            is_head = any([name.startswith(headname) for headname in Network.HEAD_NEAMES])
            if not is_head:
                loss_alpha += 0.5 * torch.norm(param - sub_model_weights[name].detach()) ** 2
            else:
                compute_param_shape = param.shape
                if name.startswith("rpn.conv_cls"):
                    compute_param = param.reshape(num_new_anchor_per_loc,
                        num_new_classes, *compute_param_shape[1:])
                    compute_oldparam = compute_param[:num_old_anchor_per_loc,
                        :num_old_classes, ...].reshape(-1, *compute_param_shape[1:]).contiguous()
                elif name.startswith("rpn.conv_box"):
                    compute_param = param.reshape(num_new_anchor_per_loc,
                        7, *compute_param_shape[1:])
                    compute_oldparam = compute_param[:num_old_anchor_per_loc,
                        ...].reshape(-1, *compute_param_shape[1:]).contiguous()
                elif name.startswith("rpn.conv_dir_cls"):
                    compute_param = param.reshape(num_new_anchor_per_loc,
                        2, *compute_param_shape[1:])
                    compute_oldparam = compute_param[:num_old_anchor_per_loc,
                        ...].reshape(-1, *compute_param_shape[1:]).contiguous()
                else:
                    raise NotImplementedError
                loss_alpha += 0.5 * torch.norm(compute_oldparam - sub_model_weights[name].detach()) ** 2
                # loss_beta += 0.5 * torch.norm(compute_newparam) ** 2
        # return self._l2sp_alpha_coef * loss_alpha + self._l2sp_beta_coef * loss_beta
        return self._l2sp_alpha_coef * loss_alpha

    def _compute_delta_loss(self,
        channel_weights,
        mask):
        def flatten_outputs(fea):
            return torch.reshape(fea, (fea.shape[0], fea.shape[1], fea.shape[2] * fea.shape[3]))
        fea_loss = 0
        if channel_weights is None and mask is None:
            for fm_model, fm_submodel in zip(self._hook_features_model, self._hook_features_submodel):
                b, c, h, w = fm_model.shape
                fea_loss += 0.5 * (torch.norm(fm_model - fm_submodel.detach()) ** 2)
        elif channel_weights is None and mask is not None:
            for fm_model, fm_submodel in zip(self._hook_features_model, self._hook_features_submodel):
                b, c, h, w = fm_model.shape
                fea_loss += 0.5 * (torch.norm(fm_model*mask - fm_submodel.detach()*mask) ** 2)
        elif channel_weights is not None and mask is None:
            for i, (fm_model, fm_submodel) in enumerate(zip(self._hook_features_model, self._hook_features_submodel)):
                b, c, h, w = fm_model.shape
                fm_model = flatten_outputs(fm_model)
                fm_submodel = flatten_outputs(fm_submodel)
                div_norm = h * w
                distance = torch.norm(fm_model - fm_submodel.detach(), 2, 2)
                distance = c * torch.mul(channel_weights[i], distance ** 2) / (h * w)
                fea_loss += 0.5 * torch.sum(distance)
            Logger.log_txt(bcolors.WARNING +
                "Network._compute_delta_loss() with channel_weights needs unit-test." +
                bcolors.ENDC)
        else:
            raise NotImplementedError
        return self._delta_coef * fea_loss

    @staticmethod
    def _delta_create_fg_mask(cls_preds,
        score_threshold=0.5,
        num_feat_channel=128):
        batch_size = cls_preds.shape[0]
        cls_preds_shape = cls_preds.shape
        mask = torch.zeros(batch_size, 1, cls_preds_shape[2], cls_preds_shape[3],
            dtype=torch.float32, device=torch.device("cuda:0"))
        for i in range(batch_size):
            fg_score, _ = torch.max(cls_preds[i, ...], dim=0)
            fg_score, _ = torch.max(fg_score, dim=-1)
            mask[i, 0, ...] = (torch.sigmoid(fg_score) > score_threshold).squeeze(-1).float()
        mask = mask.repeat([1, num_feat_channel, 1, 1]).detach()
        return mask

    def _compute_distillation_loss(self, preds_dict, preds_dict_sub):
        def _compute_weights(cls_preds,
            num_select=32,
            sampling_strategy="biased"):
            '''
            @cls_preds: [batch_size, num_anchor, H, W, num_classes] logits
            -> weights [batch_size, num_anchor]
            '''
            batch_size = cls_preds.shape[0]
            num_new_classes = cls_preds.shape[-1]
            fg_score = cls_preds.view(batch_size, -1, num_new_classes)
            num_anchor = fg_score.shape[1]
            weights = torch.zeros(batch_size, num_anchor,
                dtype=cls_preds.dtype,
                device=torch.device("cuda:0"))
            if sampling_strategy == "biased":
                for i in range(batch_size):
                    fg_score_, _ = torch.max(fg_score[i, ...], dim=-1)
                    tmp, indices = torch.topk(fg_score_, num_select)
                    weights[i, indices] = 1
            elif sampling_strategy == "unbiased":
                for i in range(batch_size):
                    fg_score_, _ = torch.max(fg_score[i, ...], dim=-1)
                    indices = torch.randperm(
                        num_anchor,
                        device=torch.device("cuda:0"),
                        requires_grad=False)
                    indices = indices[:num_select]
                    weights[i, indices] = 1
            elif sampling_strategy == "all":
                weights = torch.ones(batch_size, num_anchor,
                dtype=cls_preds.dtype,
                device=torch.device("cuda:0"))
            elif sampling_strategy == "threshold":
                threshold = 0.1
                for i in range(batch_size):
                    fg_score_, _ = torch.max(fg_score[i, ...], dim=-1)
                    indices = torch.nonzero(fg_score > threshold)
                    weights[i, indices] = 1
            else:
                raise NotImplementedError
            return weights

        batch_size = preds_dict["cls_preds"].shape[0]
        num_new_classes = self._num_new_classes
        num_new_anchor_per_loc = self._num_new_anchor_per_loc
        num_old_classes = self._num_old_classes
        num_old_anchor_per_loc = self._num_old_anchor_per_loc
        cls_output = (preds_dict["cls_preds"]
            [:, :num_old_anchor_per_loc, ..., :num_old_classes]
            .reshape(batch_size, -1, num_old_classes))
        cls_target = (preds_dict_sub["cls_preds"].detach()
            .reshape(batch_size, -1, num_old_classes))
        assert (self._distillation_loss_sampling_strategy in ["biased",
            "unbiased", "threshold", "all"],
            f"{self._distillation_loss_sampling_strategy} is not supported.")
        weights = _compute_weights(cls_target,
            sampling_strategy=self._distillation_loss_sampling_strategy,
            num_select=self._num_biased_select)
        batch_size = preds_dict["cls_preds"].shape[0]
        cls_loss = self._distillation_loss_cls_ftor(
            cls_output, cls_target, weights=weights)
        reg_output = (preds_dict["box_preds"]
            [:, :num_old_anchor_per_loc, ...]
            .reshape(batch_size, -1, 7))
        reg_target = (preds_dict_sub["box_preds"].detach()
            .reshape(batch_size, -1, 7))
        reg_loss = self._distillation_loss_reg_ftor(
            reg_output, reg_target, weights=weights)
        return {
            "weights": weights,
            "loss_distillation_loss_cls": self._distillation_loss_cls_coef * cls_loss,
            "loss_distillation_loss_reg": self._distillation_loss_reg_coef * reg_loss,
        }

    def predict(self, example, preds_dict, ext_dict=None):
        """start with v1.6.0, this function don't contain any kitti-specific code.
        Returns:
            predict: list of pred_dict.
            pred_dict: {
                box3d_lidar: [N, 7] 3d box.
                scores: [N]
                label_preds: [N]
                metadata: meta-data which contains dataset-specific information.
                    for kitti, it contains image idx (label idx), 
                    for nuscenes, sample_token is saved in it.
            }
        source: github.com/traveller59/second.pytorch
        """
        def limit_period_torch(val, offset=0.5, period=np.pi):
            return val - torch.floor(val / period + offset) * period
        self._hook_features_model.clear()
        self._hook_features_submodel.clear()
        batch_size = example['anchors'].shape[0]
        if "metadata" not in example or len(example["metadata"]) == 0:
            meta_list = [None] * batch_size
        else:
            meta_list = example["metadata"]
        batch_anchors = example["anchors"].view(batch_size, -1,
                                                example["anchors"].shape[-1])
        if "anchors_mask" not in example:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)

        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        box_coder = ext_dict["box_coder"]
        num_class_with_bg = len(self._classes_target)
        batch_dir_preds = preds_dict["dir_cls_preds"]
        if self._bool_oldclassoldanchor_predicts_only:
            num_old_anchor_per_loc = self._num_old_anchor_per_loc
            num_old_classes = self._num_old_classes
            # deactivate new classes
            batch_cls_preds[..., num_old_classes:] = -100
            # deactivate new anchors of old classes
            batch_cls_preds[:, num_old_anchor_per_loc:, ..., :num_old_classes] = -100

        batch_cls_preds = batch_cls_preds.view(batch_size, -1, num_class_with_bg)
        batch_box_preds = batch_box_preds.view(batch_size, -1, box_coder.code_size)
        batch_box_preds = box_coder.decode(batch_box_preds, batch_anchors)
        batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)

        predictions_dicts = []
        post_center_range = None
        if len(self._post_center_range) > 0:
            post_center_range = torch.tensor(
                self._post_center_range,
                dtype=batch_box_preds.dtype,
                device=batch_box_preds.device).float()
        for box_preds, cls_preds, dir_preds, a_mask, meta in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds,
                batch_anchors_mask, meta_list):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
                dir_preds = dir_preds[a_mask]
            box_preds = box_preds.float()
            cls_preds = cls_preds.float()
            dir_labels = torch.max(dir_preds, dim=-1)[1]
            total_scores = torch.sigmoid(cls_preds)
            nms_func = rotate_nms
            # get highest score per prediction, than apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = total_scores.squeeze(-1)
                top_labels = torch.zeros(
                    total_scores.shape[0],
                    device=total_scores.device,
                    dtype=torch.long)
            else:
                top_scores, top_labels = torch.max(
                    total_scores, dim=-1)
            if self._nms_score_thresholds[0] > 0.0:
                top_scores_keep = top_scores >= self._nms_score_thresholds[0]
                top_scores = top_scores.masked_select(top_scores_keep)
            if top_scores.shape[0] != 0:
                if self._nms_score_thresholds[0] > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    dir_labels = dir_labels[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                # the nms in 3d detection just remove overlap boxes.
                selected = nms_func(
                    boxes_for_nms,
                    top_scores,
                    pre_max_size=self._nms_pre_max_sizes[0],
                    post_max_size=self._nms_post_max_sizes[0],
                    iou_threshold=self._nms_iou_thresholds[0],
                )
            else:
                selected = []
            # if selected is not None:
            selected_boxes = box_preds[selected]
            selected_dir_labels = dir_labels[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]
            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                dir_labels = selected_dir_labels
                period = (2 * np.pi / 2)
                dir_rot = limit_period_torch(box_preds[..., 6], 1, period)
                box_preds[..., 6] = dir_rot + period * dir_labels.to(box_preds.dtype)
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = {
                        "box3d_lidar": final_box_preds[mask],
                        "scores": final_scores[mask],
                        "label_preds": label_preds[mask],
                        "metadata": meta,
                    }
                else:
                    predictions_dict = {
                        "box3d_lidar": final_box_preds,
                        "scores": final_scores,
                        "label_preds": label_preds,
                        "metadata": meta,
                    }
            else:
                dtype = batch_box_preds.dtype
                device = batch_box_preds.device
                predictions_dict = {
                    "box3d_lidar": torch.zeros([0, box_preds.shape[-1]],
                                dtype=dtype, device=device),
                    "scores": torch.zeros([0], dtype=dtype, device=device),
                    "label_preds": torch.zeros([0], dtype=top_labels.dtype, device=device),
                    "metadata": meta,
                }
            predictions_dicts.append(predictions_dict)
        return predictions_dicts

    def compute_ewc_weights(self,
        dataloader,
        num_of_datasamples,
        num_of_anchorsamples,
        anchor_sample_strategy,
        reg_sigma_prior=0.1,
        debug_mode=False):
        '''
        compute ewc weights after training
        @dataloader: torch.Dataloader (shuffled)
        @num_of_datasamples: int
        @num_of_anchorsamples: int
        @anchor_sample_strategy: str "biased", "unbiased", "all"
        @reg_sigma_prior: float
        @debug_mode: if True, return the cls_term and reg_term.
            Please setup the reg_sigma_prior=1 if debug_mode=True.
        -> ewc_weights {name: param (torch.FloatTensor.cuda)}
        '''
        assert False, "This function has been discarded."
        ## compute FIM
        ewc_weights = _init_ewc_weights(self._model)
        if debug_mode:
            assert reg_sigma_prior == 1, "Please set the reg_sigma_prior=1 in the debug mode."
            final_cls_term = _init_ewc_weights(self._model)
            final_reg_term = _init_ewc_weights(self._model)
        dataloader_itr = dataloader.__iter__()
        batch_size = dataloader.batch_size
        self.eval()
        for i in tqdm(range(num_of_datasamples // batch_size)):
            data, dataloader_itr = _cycle_next(dataloader, dataloader_itr)
            data = example_convert_to_torch(data,
                dtype=torch.float32, device=torch.device("cuda:0"))
            voxels = data["voxels"]
            num_points = data["num_points"]
            coors = data["coordinates"]
            batch_anchors = data["anchors"]
            preds_dict = self._network_forward(self._model, voxels, num_points, coors, batch_size)
            num_of_classes = preds_dict["cls_preds"].shape[-1]
            size_of_reg_encode = preds_dict["box_preds"].shape[-1]
            cls_preds = preds_dict["cls_preds"].reshape(batch_size, -1, num_of_classes)
            box_preds = preds_dict["box_preds"].reshape(batch_size, -1, size_of_reg_encode)
            selected_cls, selected_box = _sampling_ewc(cls_preds, box_preds,
                sample_strategy=anchor_sample_strategy,
                num_of_samples=num_of_anchorsamples)
            cls_term = _compute_FIM_cls_term(selected_cls, self._model)
            reg_term = _compute_FIM_reg_term(selected_box, self._model, sigma_prior=reg_sigma_prior)
            ewc_weights = _update_ewc_weights(ewc_weights, cls_term, reg_term, i)
            if debug_mode:
                final_cls_term = _update_ewc_term(final_cls_term, cls_term, i)
                final_reg_term = _update_ewc_term(final_reg_term, reg_term, i)
        if debug_mode:
            return final_cls_term, final_reg_term, ewc_weights
        else:
            return ewc_weights

    def compute_ewc_weights_v1(self,
        dataloader,
        num_of_datasamples,
        reg2_coef=0.1,
        clsreg_coef=0.1,
        debug_mode=False):
        '''
        compute ewc weights after training
        @dataloader: torch.Dataloader (shuffled)
        @num_of_datasamples: int
        @reg2_coef: float, the coefficient of regression gradient square term
        @clsreg_coef: float, the coefficient of regression-classification gradient cross product term
        @debug_mode: if True, return the cls2_term, reg2_term and clsreg_term.
        -> ewc_weights {name: param (torch.FloatTensor.cuda)}
        '''
        assert False, "This function has been discarded."
        ## compute FIM
        ewc_weights = _init_ewc_weights(self._model)
        final_cls2_term = _init_ewc_weights(self._model)
        final_reg2_term = _init_ewc_weights(self._model)
        final_clsreg_term = _init_ewc_weights(self._model)
        dataloader_itr = dataloader.__iter__()
        batch_size = dataloader.batch_size
        # freeze the batch normalization running mean & var
        self.eval()
        for i in tqdm(range(num_of_datasamples // batch_size)):
            # compute loss
            data, dataloader_itr = _cycle_next(dataloader, dataloader_itr)
            data = example_convert_to_torch(data,
                dtype=torch.float32, device=torch.device("cuda:0"))
            voxels = data["voxels"]
            num_points = data["num_points"]
            coors = data["coordinates"]
            batch_anchors = data["anchors"]
            preds_dict = self._network_forward(self._model, voxels, num_points, coors, batch_size)
            box_preds = preds_dict["box_preds"]
            cls_preds = preds_dict["cls_preds"]
            labels = data['labels']
            reg_targets = data['reg_targets']
            importance = data['importance']
            weights = Network._prepare_loss_weights(
                labels,
                pos_cls_weight=self._pos_cls_weight,
                neg_cls_weight=self._neg_cls_weight,
                loss_norm_type=self._loss_norm_type,
                importance=importance,
                use_direction_classifier=True,
                dtype=box_preds.dtype)
            cls_targets = labels * weights["cared"].type_as(labels)
            cls_targets = cls_targets.unsqueeze(-1)
            loss_cls = self._compute_classification_loss(
                est=cls_preds,
                gt=cls_targets,
                weights=weights["cls_weights"]*importance)*self._cls_loss_weight
            loss_reg = self._compute_location_loss(
                est=box_preds,
                gt=reg_targets,
                weights=weights["reg_weights"]*importance)*self._loc_loss_weight
            # compute gradient
            accum_grad_dict = _compute_accum_grad_v1(loss_cls, loss_reg, self._model)
            cls2_term = _compute_FIM_cls2term_v1(accum_grad_dict["cls_grad"])
            reg2_term = _compute_FIM_reg2term_v1(accum_grad_dict["reg_grad"])
            clsreg_term = _compute_FIM_clsregterm_v1(
                accum_grad_dict["cls_grad"],
                accum_grad_dict["reg_grad"])
            ewc_weights = _update_ewc_weights_v1(ewc_weights,
                cls2_term, reg2_term, clsreg_term,
                reg2_coef, clsreg_coef, i)
            final_cls2_term = _update_ewc_term(final_cls2_term, cls2_term, i)
            final_reg2_term = _update_ewc_term(final_reg2_term, reg2_term, i)
            final_clsreg_term = _update_ewc_term(final_clsreg_term, clsreg_term, i)
        if debug_mode:
            return {
                "cls2_term": final_cls2_term,
                "reg2_term": final_reg2_term,
                "clsreg_term": final_clsreg_term,
                "ewc_weights": ewc_weights}
        else:
            return ewc_weights

    def compute_ewc_weights_v2(self,
        dataloader,
        num_of_datasamples,
        oldtask_FIM_paths,
        oldtask_FIM_weights,
        newtask_FIM_weight):
        '''
        compute ewc weights after training
        @dataloader: torch.Dataloader (shuffled)
        @num_of_datasamples: int
        @oldtask_FIMs: List
        @oldtask_FIM_weights: List
        @newtask_FIM_wegiht: float
        -> newtask_FIM, FIM {name: param (torch.FloatTensor.cuda)}
        '''
        newtask_FIM = _init_ewc_weights(self._model)
        dataloader_itr = dataloader.__iter__()
        batch_size = dataloader.batch_size
        # freeze the batch normalization running mean & var
        for i in tqdm(range(num_of_datasamples // batch_size)):
            # compute loss
            data, dataloader_itr = _cycle_next(dataloader, dataloader_itr)
            data = example_convert_to_torch(data,
                dtype=torch.float32, device=torch.device("cuda:0"))
            loss = self.forward(data)
            loss_cls_sum = loss["loss_cls"]
            loss_reg_sum = loss["loss_reg"]
            loss_detection_sum = loss_cls_sum + loss_reg_sum
            tmp_FIM = compute_FIM_v2(loss_detection_sum, self._model)
            newtask_FIM = update_ewc_weights_v2(newtask_FIM, tmp_FIM, i)

        # load oldtask_FIMs
        oldtask_FIMs = []
        for oldtask_FIM_path in oldtask_FIM_paths:
            Logger.log_txt(f"Loading from {oldtask_FIM_path}")
            oldtask_FIM = read_pkl(oldtask_FIM_path)
            oldtask_FIM = {name: torch.from_numpy(param).cuda()
                for name, param in oldtask_FIM.items()}
            oldtask_FIMs.append(oldtask_FIM)
        # merge oldtask_FIMs and newtask_FIM:
        FIM = _init_ewc_weights(self._model)
        # parse num_new_classes, num_new_anchor_per_loc
        num_new_classes, num_new_anchor_per_loc = parse_numclasses_numanchorperloc(FIM)
        print(f"num_new_classes: {num_new_classes}")
        print(f"num_new_anchor_per_loc: {num_new_anchor_per_loc}")
        for oldtask_FIM, oldtask_FIM_weight in zip(oldtask_FIMs, oldtask_FIM_weights):
            # parse num_old_classes, num_old_anchor_per_loc
            num_old_classes, num_old_anchor_per_loc = parse_numclasses_numanchorperloc(oldtask_FIM)
            print(f"num_old_classes: {num_old_classes}")
            print(f"num_old_anchor_per_loc: {num_old_anchor_per_loc}")
            for name, param in oldtask_FIM.items():
                param = expand_old_weights(name, param,
                num_new_classes, num_new_anchor_per_loc,
                num_old_classes, num_old_anchor_per_loc)
                FIM[name] = FIM[name] + param * oldtask_FIM_weight
        for name, param in newtask_FIM.items():
            FIM[name] += param * newtask_FIM_weight
        return {
            "newtask_FIM": newtask_FIM,
            "FIM": FIM
        }

    def compute_mas_weights(
        self,
        dataloader,
        num_of_datasamples,
        num_of_anchorsamples,
        anchor_sample_strategy,
        reg_coef,
        oldtask_omega_paths,
        oldtask_omega_weights,
        newtask_omega_weight):
        '''
        compute mas weights after training
        @dataloader: torch.Dataloader (shuffled)
        @num_of_datasamples: int
        @num_of_anchorsamples: int
        @anchor_sample_strategy: str "biased", "unbiased", "all"
        @reg_coef: float
        @oldtask_omega_paths: List
        @oldtask_omega_weights: List
        @newtask_omega_weight: float
        -> mas_omega {name: param (torch.FloatTensor.cuda)}
        -> mas_newomega {name: param (torch.FloatTensor.cuda)}
        -> mas_clsterm {name: param (torch.FloatTensor.cuda)}
        -> mas_regterm {name: param (torch.FloatTensor.cuda)} (without reg_coef weighted)
        '''
        ## compute FIM
        final_cls_term = _init_mas_weights(self._model)
        final_reg_term = _init_mas_weights(self._model)
        dataloader_itr = dataloader.__iter__()
        batch_size = dataloader.batch_size
        self.eval()
        for i in tqdm(range(num_of_datasamples // batch_size)):
            data, dataloader_itr = _cycle_next(dataloader, dataloader_itr)
            data = example_convert_to_torch(data,
                dtype=torch.float32, device=torch.device("cuda:0"))
            voxels = data["voxels"]
            num_points = data["num_points"]
            coors = data["coordinates"]
            batch_anchors = data["anchors"]
            preds_dict = self._network_forward(self._model, voxels, num_points, coors, batch_size)
            num_of_classes = preds_dict["cls_preds"].shape[-1]
            size_of_reg_encode = preds_dict["box_preds"].shape[-1]
            cls_preds = preds_dict["cls_preds"].reshape(batch_size, -1, num_of_classes)
            box_preds = preds_dict["box_preds"].reshape(batch_size, -1, size_of_reg_encode)
            selected_cls, selected_box = _sampling_mas(cls_preds, box_preds,
                sample_strategy=anchor_sample_strategy,
                num_of_samples=num_of_anchorsamples)

            cls_term = _compute_omega_cls_term(selected_cls, self._model)
            reg_term = _compute_omega_reg_term(selected_box, self._model)
            final_cls_term = _update_mas_term(final_cls_term, cls_term, i)
            final_reg_term = _update_mas_term(final_reg_term, reg_term, i)
        newomega = _compute_omega(final_cls_term, final_reg_term, reg_coef)
        # load oldtask_omegas
        oldtask_omegas = []
        for oldtask_omega_path in oldtask_omega_paths:
            Logger.log_txt(f"Loading from {oldtask_omega_path}")
            oldtask_omega = read_pkl(oldtask_omega_path)
            oldtask_omega = {name: torch.from_numpy(param).cuda()
                for name, param in oldtask_omega.items()}
            oldtask_omegas.append(oldtask_omega)
        # merge oldtask_FIMs and newtask_FIM:
        omega = _init_mas_weights(self._model)
        # parse num_new_classes, num_new_anchor_per_loc
        num_new_classes, num_new_anchor_per_loc = parse_numclasses_numanchorperloc(omega)
        print(f"num_new_classes: {num_new_classes}")
        print(f"num_new_anchor_per_loc: {num_new_anchor_per_loc}")
        for oldtask_omega, oldtask_omega_weight in zip(oldtask_omegas, oldtask_omega_weights):
            # parse num_old_classes, num_old_anchor_per_loc
            num_old_classes, num_old_anchor_per_loc = parse_numclasses_numanchorperloc(oldtask_omega)
            print(f"num_old_classes: {num_old_classes}")
            print(f"num_old_anchor_per_loc: {num_old_anchor_per_loc}")
            for name, param in oldtask_omega.items():
                param = expand_old_weights(name, param,
                num_new_classes, num_new_anchor_per_loc,
                num_old_classes, num_old_anchor_per_loc)
                omega[name] = omega[name] + param * oldtask_omega_weight
        for name, param in newomega.items():
            omega[name] += param * newtask_omega_weight
        return {
            "new_omega": newomega,
            "new_clsterm": final_cls_term,
            "new_regterm": final_reg_term,
            "omega": omega,
        }

    def train(self):
        super(Network, self).train(True)
        if self._training_mode == "train_from_scratch":
            self._model.train()
            assert self._sub_model is None
        elif self._training_mode in ["joint_training", "fine_tuning", "lwf"]:
            # freeze bn and no dropout
            self._model.eval()
            if self._sub_model is not None:
                self._sub_model.eval()
        elif self._training_mode in ["feature_extraction"]:
            # freeze bn and no dropout except for new layers
            for name, layer in self._model.named_modules():
                if "rpn.featext" not in name:
                    layer.eval()
                else:
                    layer.train()
            if self._sub_model is not None:
                self._sub_model.eval()
        else:
            raise NotImplementedError
    
    def eval(self):
        super(Network, self).train(False)
        self._model.eval()
        if self._sub_model is not None:
            self._sub_model.eval()

if __name__ == "__main__":
    from incdet3.data.carladataset import CarlaDataset
    from incdet3.configs.dev_cfg import cfg
    from incdet3.builders import voxelizer_builder, target_assigner_builder
    from tqdm import tqdm
    data_cfg = cfg.VALDATA
    voxelizer = voxelizer_builder.build(cfg.VOXELIZER)
    target_assigner = target_assigner_builder.build(cfg.TARGETASSIGNER)
    cfg.NETWORK["@middle_layer_dict"]["@output_shape"] = [1] + voxelizer.grid_size[::-1].tolist() + [16]
    # cfg.NETWORK["@rpn_dict"]["@num_anchor_per_loc"] = target_assigner.num_anchors_per_location
    cfg.NETWORK["@rpn_dict"]["@box_code_size"] = target_assigner.box_coder.code_size
    param = {proc_param(k): v
        for k, v in cfg.NETWORK.items() if is_param(k)}
    network = Network(**param)