'''
 File Created: Sat Oct 03 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''
import torch
from torch import nn
from tqdm import tqdm
from incdet3.models.ewc_func import _init_ewc_weights, _sampling_ewc, _update_ewc_term

def _init_mas_weights(model):
    '''
    init mas_weights from model
    @model: nn.Module
    -> mas_weights: dict {name: torch.FloatTensor.cuda (zeros)}
    '''
    return _init_ewc_weights(model)

def _sampling_mas(
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
    return _sampling_ewc(
        cls_preds,
        box_preds,
        sample_strategy,
        num_of_samples)

def _compute_omega_cls_term(cls_preds, model):
    '''
    compute the omega classification term
    @cls_preds: torch.FloatTensor.cuda
        [num_of_anchors, num_of_classes]
    @model: nn.Module
    -> cls_term:dict {name: torch.FloatTensor.cuda}
    '''
    cls_term = _init_mas_weights(model)
    num_anchors = cls_preds.shape[0]
    cls_targets = torch.zeros(cls_preds.size()).cuda()
    criterion = nn.MSELoss(reduction='none')
    loss = criterion(cls_preds, cls_targets).sum()
    model.zero_grad()
    loss.backward(retain_graph=True)
    for name, param in model.named_parameters():
        grad = (param.grad
            if param.grad is not None
            else torch.zeros(1, device=torch.device("cuda:0"),
            requires_grad=False))
        cls_term[name] += (grad ** 2).detach()
    for name, _ in model.named_parameters():
        cls_term[name] /= num_anchors
    return cls_term

def _compute_omega_reg_term(reg_preds, model):
    '''
    compute the omega regression term
    @reg_preds: torch.FloatTensor.cuda
        [num_of_anchors, 7]
    @model: nn.Module
    -> reg_term:dict {name: torch.FloatTensor.cuda}
    Note: This function has to be run after _compute_omega_cls_term,
    because the first calling need setting retain_graph=True in backward().
    '''
    reg_term = _init_mas_weights(model)
    num_anchors = reg_preds.shape[0]
    reg_targets = torch.zeros(reg_preds.size()).cuda()
    criterion = nn.MSELoss(reduction='none')
    loss = criterion(reg_preds, reg_targets).sum()
    model.zero_grad()
    loss.backward()
    for name, param in model.named_parameters():
        grad = (param.grad
            if param.grad is not None
            else torch.zeros(1, device=torch.device("cuda:0"),
            requires_grad=False))
        reg_term[name] += (grad ** 2).detach()
    for name, _ in model.named_parameters():
        reg_term[name] /= num_anchors
    return reg_term

def _update_mas_term(old_term, new_term, accum_idx):
    '''
    update mas_term by accumulating new_term into old_term.
    @old_term, new_term:
        dict{name: torch.FloatTensor.cuda}
    @accum_idx: int
    -> dict {name: torch.FloatTensor.cuda}
    '''
    return _update_ewc_term(old_term, new_term, accum_idx)

def _compute_omega(cls_term, reg_term, reg_coef):
    '''
    compute omega by combine cls_term and reg_term weighted by reg_coef
    @cls_term, reg_term:
        dict{name: torch.FloatTensor.cuda}
    @reg_coef:
        float
    -> dict {name: torch.FloatTensor.cuda}
    '''
    omega = {}
    for name, _ in cls_term.items():
        omega[name] = (cls_term[name] + reg_coef * reg_term[name])
    return omega
