'''
network needs to handle the following training schemes (by set train/val & require_grad):
"feature extraction", "fine-tuning", "joint training", "lwf"
network needs to handle different distillation schemes
"l2sp" (loss), "delta" (loss + hook), "distillation loss" (loss + hook)
To be compatible with "delta", we need to change the rpn part to resnet.
'''
import torch
import torch.nn as nn
from pathlib import Path
from det3.utils.log_tool import Logger
from det3.utils.utils import is_param, proc_param
from det3.methods.second.models.voxel_encoder import get_vfe_class
from det3.methods.second.models.middle import get_middle_class
from det3.methods.second.core.model_manager import save_models
from incdet3.models.rpn import get_rpn_class
from incdet3.utils.utils import bcolors

class Network(nn.Module):
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
        bool_oldclass_use_newanchor_for_cls=False
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
        self._distillation_mode = []
        self._is_training = is_training
        self._hook_layers = []
        self._hook_features_model = {}
        self._hook_features_submodel = {}

        network_cfg = {
            "VoxelEncoder": voxel_encoder_dict,
            "MiddleLayer": middle_layer_dict,
            "RPN": rpn_dict
        }
        if self._training_mode in ["feature_extraction", "fine_tuning"]:
            self._num_old_classes = self._model_resume_dict["num_classes"]
            self._num_old_anchor_per_loc = self._model_resume_dict["num_anchor_per_loc"]
            self._bool_oldclass_use_newanchor_for_cls = bool_oldclass_use_newanchor_for_cls

        self._model = Network._build_model_and_init(
            classes=self._classes_target,
            network_cfg=network_cfg,
            resume_dict=self._model_resume_dict,
            name="IncDetMain")
        if self._training_mode == "lwf":
            self._sub_model = self._build_model_and_init(
                classes=self._classes_source,
                network_cfg=network_cfg,
                resume_dict=self._sub_model_resume_dict,
                name="IncDetSub")
        else:
            self._sub_model = None
        self._freeze_model()
        # self._register_model_hook()
        # self._register_submodel_hook()

        self.register_buffer("global_step", torch.LongTensor(1).zero_())
        # raise NotImplementedError

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
            bool_oldclass_use_newanchor_for_cls = self._bool_oldclass_use_newanchor_for_cls
            if bool_oldclass_use_newanchor_for_cls:
                preds_dict["cls_preds"][:, :num_old_anchor_per_loc, ..., :num_old_classes] = \
                    preds_dict["cls_preds"][:, :num_old_anchor_per_loc, ..., :num_old_classes].detach()
            else:
                preds_dict["cls_preds"][..., :num_old_classes] = \
                    preds_dict["cls_preds"][..., :num_old_classes].detach()
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
                    param.requires_grad = True
                elif "rpn.conv_box" in name:
                    param.requires_grad = True
                elif "rpn.conv_dir_cls" in name:
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
        preds_dict = self.network_forward(voxels, num_points, coors, batch_size)
        box_preds = preds_dict["box_preds"].view(batch_size, -1, 7)
        err_msg = f"num_anchors={batch_anchors.shape[1]}, but num_output={box_preds.shape[1]}. please check size"
        assert batch_anchors.shape[1] == box_preds.shape[1], err_msg
        if self.training:
            self._detach_variables(preds_dict)
            return self.loss(example, preds_dict)
        else:
            with torch.no_grad():
                res = self.predict(example, preds_dict)
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
        if rpn_name in ["RPNV2", "ResNetRPN"]:
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
        parse_cls_conv_layer = {
            2: {"num_classes": 1, "num_anchor_per_loc": 2},
            8: {"num_classes": 2, "num_anchor_per_loc": 4},
            18: {"num_classes": 3, "num_anchor_per_loc": 6},
            32: {"num_classes": 4, "num_anchor_per_loc": 8},
            50: {"num_classes": 5, "num_anchor_per_loc": 10},
        }
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
        for key in new_sd.keys():
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

    @staticmethod
    def save_weight(model, save_dir, itr):
        save_models(save_dir, [model],
            itr, max_to_keep=float('inf'))
        Logger.log_txt("Saving parameters to {}".format(save_dir))

    def forward(self, data_dict):
        preds_dict = self._network_forward(self._model, data_dict)
        if self.training:
            return self.loss(data_dict, preds_dict)
        else:
            with torch.no_grad():
                return self.predict(data_dict, preds_dict)

    def loss(self, est, gt,
        channel_weight=None):
        loss_dict = dict()
        loss_dict["loss_cls"] = self._compute_classification_loss(est, gt)
        loss_dict["loss_reg"] = self._compute_regression_loss(est, gt)
        if "l2sp" in self._distillation_mode:
            # call model weights in the _compute_l2sp_loss()
            loss_dict["loss_l2sp"] = self._compute_l2sp_loss()
        if "delta" in self._distillation_mode:
            # call hook features in the _compute_l2sp_loss()
            loss_dict["loss_delta"] = self._compute_delta_loss(channel_weight)
        if "distillation-loss" in self._distillation_mode:
            # call hook features in the _compute_l2sp_loss()
            loss_dict["loss_distillation-loss"] = self._compute_distillation_loss()
        loss_dict["loss_total"] = sum(loss_dict.values())
        self._hook_features_model.clear()
        self._hook_features_submodel.clear()
        return loss_dict

    def predict():
        raise NotImplementedError

    def train(self):
        if self._training_mode == "train-from-scratch":
            self._model.train()
            assert self._sub_model is None
        elif self._training_mode in ["feature-extraction", "joint-training", "lwf"]:
            # freeze bn and no dropout
            self._model.eval()
            self._sub_model.eval()
        else:
            raise NotImplementedError
    
    def eval(self):
        self._model.eval()
        self._sub_model.eval()

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])

    def update_global_step(self):
        self.global_step += 1

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