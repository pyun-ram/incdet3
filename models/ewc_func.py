'''
 File Created: Thu Sep 03 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''
import torch
def _init_ewc_weights(model_sd):
    '''
    init ewc_weights from model state_dict
    @model_sd: nn.Module.state_dict()
    -> ewc_weights: dict {name: torch.FloatTensor.cuda (zeros)}
    '''
    ewc_weights = {}
    for name, param in model_sd.items():
        ewc_weights[name] = torch.zeros_like(param)
    return ewc_weights

def _sampling_ewc(
    cls_preds,
    box_preds,
    sample_strategy="all",
    num_of_samples=None):
    '''
    sampling from cls_preds and box_preds according to sample_strategy
    @cls_preds: torch.FloatTensor.cuda
        [batch_size, num_of_anchors, num_of_classes]
    @box_preds: torch.FloatTensor.cuda
        [batch_size, num_of_anchors, 7]
    @sample_strategy: str: "all", "biased", "unbiased"
    @num_of_samples: None if not applicable
    -> selected_cls: torch.FloatTensor.cuda
        [num_of_samples, num_of_classes]
    -> selected_box: torch.FloatTensor.cuda
        [num_of_samples, 7]
    '''
    num_of_classes = cls_preds.shape[-1]
    size_of_reg_encode = box_preds.shape[-1]
    batch_size = cls_preds.shape[0]
    if sample_strategy == "all":
        all_cls = cls_preds.reshape(-1, num_of_classes)
        all_box = box_preds.reshape(-1, size_of_reg_encode)
        assert all_cls.shape[0] == all_box.shape[0]
        selected_cls, selected_box = all_cls, all_box
    elif sample_strategy == "biased":
        assert num_of_samples >= batch_size
        selected_cls = []
        selected_box = []
        for i in range(batch_size):
            fg_score, _ = cls_preds[i, ...].max(dim=-1)
            if i == batch_size -1:
                tmp, indices = torch.topk(fg_score, num_of_samples - num_of_samples // batch_size * i)
            else:
                tmp, indices = torch.topk(fg_score, num_of_samples//batch_size)
            selected_cls.append(cls_preds[i, indices, :])
            selected_box.append(box_preds[i, indices, :])
        selected_cls = torch.cat(selected_cls, dim=0)
        selected_box = torch.cat(selected_box, dim=0)
    elif sample_strategy == "unbiased":
        all_cls = cls_preds.reshape(-1, num_of_classes)
        all_box = box_preds.reshape(-1, size_of_reg_encode)
        assert all_cls.shape[0] == all_box.shape[0]
        indices = torch.randperm(
            all_cls.shape[0],
            device=torch.device("cuda:0"),
            requires_grad=False)
        indices = indices[:num_of_samples]
        selected_cls = all_cls[indices, :]
        selected_box = all_box[indices, :]
    else:
        raise NotImplementedError
    return selected_cls, selected_box

def _compute_FIM_cls_term(cls_preds, model):
    '''
    compute the FIM classification term
    @cls_preds: torch.FloatTensor.cuda
        [num_of_anchors, num_of_classes]
    @model: nn.Module
    -> cls_term:dict {name: torch.FloatTensor.cuda}
    '''
    cls_term = _init_ewc_weights(model.state_dict())
    num_anchors = cls_preds.shape[0]
    num_cls = cls_preds.shape[1]
    for logit in cls_preds:
        prob = torch.softmax(logit, dim=-1)
        for cls in range(num_cls):
            model.zero_grad()
            log_prob = torch.log(prob[cls])
            log_prob.backward(retain_graph=True)
            for name, param in model.named_parameters():
                grad = param.grad if param.grad is not None else 0
                cls_term[name] += (grad **2 * prob[cls].detach()).detach()
    for name, _ in model.named_parameters():
        cls_term[name] /= num_anchors
    return cls_term

def _compute_FIM_reg_term(reg_preds, model, sigma_prior=0.1):
    '''
    compute the FIM regression term
    @reg_preds: torch.FloatTensor.cuda
        [num_of_anchors, 7]
    @model: nn.Module
    -> reg_term:dict {name: torch.FloatTensor.cuda}
    '''
    reg_term = _init_ewc_weights(model.state_dict())
    num_anchors = reg_preds.shape[0]
    for reg_output in reg_preds:
        for reg_output_ in reg_output:
            model.zero_grad()
            reg_output_.backward(retain_graph=True)
            for name, param in model.named_parameters():
                grad = param.grad if param.grad is not None else torch.zeros(1,
                    device=torch.device("cuda:0"), requires_grad=False)
                reg_term[name] += (grad**2).detach() / (sigma_prior**2)
    for name, _ in model.named_parameters():
        reg_term[name] /= num_anchors
        # print(name, float(reg_term[name].mean()))
    return reg_term

def _update_ewc_weights(old_ewc_weights, cls_term, reg_term, accum_idx):
    '''
    update ewc_weights by accumulating cls_term+reg_term into old_ewc_weights.
    @old_ewc_weights, cls_term, reg_term:
        dict{name: param}
    @accum_idx: int
    -> dict {name: torch.FloatTensor.cuda}
    '''
    new_ewc_weights = {}
    for name, _ in cls_term.items():
        new_ewc_weights[name] = cls_term[name] + reg_term[name]
    if accum_idx == 0:
        ewc_weights = new_ewc_weights
    elif accum_idx > 0:
        ewc_weights = {}
        for name, _ in cls_term.items():
            ewc_weights[name] = old_ewc_weights[name] * accum_idx + new_ewc_weights[name]
            ewc_weights[name] /= accum_idx+1
    else:
        raise RuntimeError
    return ewc_weights

def _cycle_next(dataloader, dataloader_itr):
    try:
        data = dataloader_itr.__next__()
        return data, dataloader_itr
    except StopIteration:
        newdataloader_itr = dataloader.__iter__()
        data = newdataloader_itr.__next__()
        return data, newdataloader_itr