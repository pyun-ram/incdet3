'''
network needs to handle the following training schemes (by set train/val & require_grad):
"feature extraction", "fine-tuning", "joint training", "lwf"
network needs to handle different distillation schemes
"l2sp" (loss), "delta" (loss + hook), "distillation loss" (loss + hook)
To be compatible with "delta", we need to change the rpn part to resnet.
'''

class Network(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self._model = None
        self._model_resume = None
        self._sub_model = None
        self._sub_model_resume = None
        # _classes_target is the inference classes of _model
        self._classes_target = []
        # _classes_source is the inference classes of _sub_model
        self._classes_source = []
        # train-from-scratch, feature-extraction, fine-tuning, joint-training, lwf
        self._training_mode = None
        # distillation scheme: [l2sp, delta, distillation-loss]
        self._distillation_mode = []
        self._is_training = None
        self._hook_layers = []
        self._hook_features_model = {}
        self._hook_features_submodel = {}

        self._model = self._build_model_and_init(classes=self._classes_target,
            resume=self._model_resume)
        self._sub_model = self._build_model_and_init(classes=self._classes_source,
            resume=self._sub_model_resume)
        self._freeze_model(self._model, self._training_mode)
        self._freeze_model(self._sub_model, self._training_mode)
        self._register_model_hook()
        self._register_submodel_hook()

        self.register_buffer("global_step", torch.LongTensor(1).zero_())
        raise NotImplementedError


    def forward(self, data_dict):
        preds_dict = self._network_forward(self._model, data_dict)
        if self.training:
            return self.loss(data_dict, preds_dict)
        else:
            with torch.no_grad():
                return self.predict(data_dict, preds_dict)

    def _network_forward(self, model, x):
        return model(x)

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
