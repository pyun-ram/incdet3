'''
 File Created: Thu Sep 03 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''
import torch
from tqdm import tqdm

def _init_ewc_weights(model):
    '''
    init ewc_weights from model
    @model: nn.Module
    -> ewc_weights: dict {name: torch.FloatTensor.cuda (zeros)}
    '''
    ewc_weights = {}
    for name, param in model.named_parameters():
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
    cls_term = _init_ewc_weights(model)
    num_anchors = cls_preds.shape[0]
    num_cls = cls_preds.shape[1]
    for logit in tqdm(cls_preds):
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
    reg_term = _init_ewc_weights(model)
    num_anchors = reg_preds.shape[0]
    for reg_output in tqdm(reg_preds):
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
        dict{name: torch.FloatTensor.cuda}
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

def _update_ewc_term(old_term, new_term, accum_idx):
    '''
    update ewc_term by accumulating new_term into old_term.
    @old_term, new_term:
        dict{name: torch.FloatTensor.cuda}
    @accum_idx: int
    -> dict {name: torch.FloatTensor.cuda}
    '''
    if accum_idx == 0:
        term = new_term
    elif accum_idx > 0:
        term = {}
        for name, _ in old_term.items():
            term[name] = old_term[name] * accum_idx + new_term[name]
            term[name] /= accum_idx+1
    else:
        raise RuntimeError
    return term

def _cycle_next(dataloader, dataloader_itr):
    try:
        data = dataloader_itr.__next__()
        return data, dataloader_itr
    except StopIteration:
        newdataloader_itr = dataloader.__iter__()
        data = newdataloader_itr.__next__()
        return data, newdataloader_itr

def _compute_accum_grad_v1(loss_cls, loss_reg, model):
    '''
    compute the accum_grad_cls and accum_grad_reg
    @loss_cls: [batch_size, num_of_anchors,1]
    @loss_reg: [batch_size, num_of_anchors,1]
    @model: nn.Module
    -> accum_grad {
        'cls_grad': accum_grad_cls, (sum of the batch gradients)
        'reg_grad': accum_grad_reg, (sum of the batch gradients)
    }
    '''
    # cls_grad
    model.zero_grad()
    loss_cls.sum().backward(retain_graph=True)
    accum_grad_cls = {}
    for name, param in model.named_parameters():
        accum_grad_cls[name] = (param.grad.clone()
            if param.grad is not None else
            torch.zeros(1, device=torch.device("cuda:0"),
            requires_grad=False))
    # reg_grad
    model.zero_grad()
    loss_reg.sum().backward(retain_graph=True)
    accum_grad_reg = {}
    for name, param in model.named_parameters():
        accum_grad_reg[name] = (param.grad.clone()
            if param.grad is not None else
            torch.zeros(1, device=torch.device("cuda:0"),
            requires_grad=False))
    model.zero_grad()
    return {
        'cls_grad': accum_grad_cls,
        'reg_grad': accum_grad_reg,
    }

def _compute_FIM_cls2term_v1(accum_grad_cls):
    '''
    compute the classification gradient square term.
    @accum_grad_cls: {name: param}
    -> cls2_term: {name: param}
    '''
    cls2_term = {}
    for name, param in accum_grad_cls.items():
        cls2_term[name] = param **2
    return cls2_term

def _compute_FIM_reg2term_v1(accum_grad_reg):
    '''
    compute the regression gradient square term.
    @accum_grad_reg: {name: param}
    -> reg2_term: {name: param}
    '''
    reg2_term = {}
    for name, param in accum_grad_reg.items():
        reg2_term[name] = param **2
    return reg2_term

def _compute_FIM_clsregterm_v1(accum_grad_cls, accum_grad_reg):
    '''
    compute the classification gradient and regression gradient production term.
    @accum_grad_reg: {name: param}
    -> clsreg_term: {name: param}
    '''
    clsreg_term = {}
    for name, _ in accum_grad_reg.items():
        # element-wise production
        clsreg_term[name] = torch.mul(accum_grad_cls[name], accum_grad_reg[name])
    return clsreg_term

def _update_ewc_weights_v1(old_ewc_weights,
    cls2_term, reg2_term, clsreg_term,
    reg2_coef, clsreg_coef, accum_idx):
    '''
    update old_ewc_weights by accumulating
        cls2_term + reg2_coef x reg2_term + clsreg_coef x clsreg_term
        into old_ewc_weights.
    @cls2_term, reg2_term, clsreg_term:
        dict{name: torch.FloatTensor.cuda}
    @reg2_coef, clsreg_coef:
        float
    @accum_idx: int
    -> dict {name: torch.FloatTensor.cuda}
    '''
    new_ewc_weights = {}
    for name, param in old_ewc_weights.items():
        new_ewc_weights[name] = (cls2_term[name]
            + reg2_coef * reg2_term[name]
            + clsreg_coef * clsreg_term[name])
    if accum_idx == 0:
        term = new_ewc_weights
    elif accum_idx > 0:
        term = {}
        for name, _ in old_ewc_weights.items():
            term[name] = old_ewc_weights[name] * accum_idx + new_ewc_weights[name]
            term[name] /= accum_idx+1
    else:
        raise RuntimeError
    return term

def compute_FIM_v2(loss_det, model):
    '''
    compute the FIM according to with FIM = d/d\theta loss_det^2
    @loss_det: detection loss with summation reduction
    @model: torch.nn.Module
    -> FIM {name: param (torch.FloatTensor.cuda)}
    '''
    model.zero_grad()
    loss_det.backward()
    FIM = {}
    for name, param in model.named_parameters():
        FIM[name] = (param.grad.clone() ** 2
            if param.grad is not None else
            torch.zeros(1, device=torch.device("cuda:0"),
            requires_grad=False))
    model.zero_grad()
    return FIM

def update_ewc_weights_v2(oldFIM, newFIM, accum_idx):
    '''
    accum newFIM into oldFIM according to accum_idx.
    '''
    return _update_ewc_term(oldFIM, newFIM, accum_idx)

def ewc_measure_distance(diff, loss_type, beta, weights):
    '''
    measure the distance of diff with L2-norm or huber loss.
    The huber loss part is from: "Incdet: In Defense of Elastic
    Weight Consolidation for Incrmental Object Detection", TNNLS2020.
    @diff: torch.FloatTensor
    @loss_tyep: str "l2" or "huber"
    @beta: float
    @weights: torch.FloatTensor with the same shape as diff
    -> dist: torch.FloatTensor with the same shape as diff
    '''
    if loss_type == "l2":
        return 0.5 * weights * diff ** 2
    elif loss_type == "huber":
        cond_lhs = weights ** 0.5 * torch.abs(diff)
        mask_l2 = cond_lhs <= beta
        l2_diff = 0.5 * weights * diff ** 2
        huber_diff = beta * (weights ** 0.5 * torch.abs(diff) - beta * 0.5)
        return (mask_l2.float() * l2_diff +
            torch.logical_not(mask_l2).float() * huber_diff)
    else:
        raise NotImplementedError

def parse_numclasses_numanchorperloc(FIM: dict)->(int, int):
    '''
    parse num_classes and num_anchor_per_loc from FIM
    @FIM: fisher information matrix (dict)
        {name: param}
    -> num_classes, num_anchor_per_loc
    '''
    parse_cls_conv_layer = {
        2*i*i: {"num_classes": i, "num_anchor_per_loc": 2*i}
        for i in range(1, 11)}
    num_conv_cls_bias = FIM["rpn.conv_cls.bias"].shape[0]
    num_classes = parse_cls_conv_layer[num_conv_cls_bias]["num_classes"]
    num_anchor_per_loc = parse_cls_conv_layer[num_conv_cls_bias]["num_anchor_per_loc"]
    return num_classes, num_anchor_per_loc

def expand_old_weights(
    name:str,
    param:torch.FloatTensor,
    num_new_classes:int,
    num_new_anchor_per_loc:int,
    num_old_classes:int,
    num_old_anchor_per_loc:int) -> torch.FloatTensor:
    '''
    expand param if it is the head of SECOND
    -> expanded_param: torch.FloatTensor
    '''
    param_shape = param.shape
    if "rpn.conv_cls" in name:
        expanded_param = torch.zeros([
            num_new_anchor_per_loc,
            num_new_classes,
            *param_shape[1:]], dtype=param.dtype, device=param.device)
        old_tmp = param.reshape(
            num_old_anchor_per_loc,
            num_old_classes,
            *param_shape[1:])
        expanded_param[:num_old_anchor_per_loc, :num_old_classes, ...] = \
            old_tmp[:num_old_anchor_per_loc, :num_old_classes, ...]
    elif "rpn.conv_box" in name:
        expanded_param = torch.zeros([
            num_new_anchor_per_loc,
            7, *param_shape[1:]], dtype=param.dtype, device=param.device)
        old_tmp = param.reshape(num_old_anchor_per_loc, 7, *param_shape[1:])
        expanded_param[:num_old_anchor_per_loc, ...] = \
            old_tmp[:num_old_anchor_per_loc, ...]
    elif "rpn.conv_dir_cls" in name:
        expanded_param = torch.zeros([
            num_new_anchor_per_loc,
            2, *param_shape[1:]], dtype=param.dtype, device=param.device)
        if torch.all(param == 0):
            return expanded_param.reshape(-1, *param_shape[1:]).contiguous()
        old_tmp = param.reshape(num_old_anchor_per_loc, 2, *param_shape[1:])
        expanded_param[:num_old_anchor_per_loc, ...] = \
            old_tmp[:num_old_anchor_per_loc, ...]
    else:
        return param
    return expanded_param.reshape(-1, *param_shape[1:]).contiguous()