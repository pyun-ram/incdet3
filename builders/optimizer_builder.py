import torch
import torch.nn as nn
from functools import partial
from det3.methods.second.core.optimizer import OptimWrapper
from det3.methods.second.utils.torch_utils import OneCycle
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam

def children(m: nn.Module):
    "Get children of `m`."
    return list(m.children())

def num_children(m: nn.Module) -> int:
    "Get number of children modules in `m`."
    return len(children(m))

flatten_model = lambda m: sum(map(flatten_model,m.children()),[]) if num_children(m) else [m]
get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

def build_optimizer(net, optimizer_cfg):
    '''
    @net: torch.nn.module
    @optimizer_cfg: dict
    {
        "type": "ADAMOptimizer",
        "amsgrad": False,
        "init_lr": 1e-3, # it will be changed by OneCycle LRscheduler
        "weight_decay": 0.01,
        "fixed_weight_decay": True,
    },
    {
        "type": "adam",
        "init_lr": 5e-3,
        "weight_decay": 0.01,
    },
    '''
    class_name = optimizer_cfg["type"]
    if class_name == "ADAMOptimizer":
        optimizer_func = partial(
            torch.optim.Adam, betas=(0.9, 0.99), amsgrad=optimizer_cfg["amsgrad"])
        optimizer = OptimWrapper.create(
        optimizer_func,
        optimizer_cfg["init_lr"],
        get_layer_groups(net),
        wd=optimizer_cfg["weight_decay"],
        true_wd=optimizer_cfg["fixed_weight_decay"],
        bn_wd=True)
        optimizer.name = class_name
    elif class_name == "adam":
        optimizer = Adam(filter(lambda p: p.requires_grad, net.parameters()),
            lr=optimizer_cfg["init_lr"], weight_decay=optimizer_cfg["weight_decay"],
            betas=(0.95,0.99),eps=1e-08, amsgrad=False)
    else:
        raise NotImplementedError
    return optimizer

def build_lr_scheduler(optimizer, lr_scheduler_cfg):
    '''
    @optimizer: torch.optim.optimizer
    @lr_scheduler_cfg: dict
    {
        "type": "OneCycle",
        "total_step": 40k
        "lr_max": 2.25e-3,
        "moms": [0.95, 0.85],
        "div_factor": 10.0,
        "pct_start": 0.4,
    }
    {
        "type": "StepLR",
        "step_size": 40k * 0.8,
        "gamma": 0.1
    }
    '''
    class_name = lr_scheduler_cfg["type"]
    if class_name == "OneCycle":
        lr_scheduler = OneCycle(optimizer,
            lr_scheduler_cfg["total_step"], 
            lr_scheduler_cfg["lr_max"],
            list(lr_scheduler_cfg["moms"]),
            lr_scheduler_cfg["div_factor"],
            lr_scheduler_cfg["pct_start"])
    elif class_name == "StepLR":
        lr_scheduler = StepLR(optimizer,
            step_size=lr_scheduler_cfg["step_size"],
            gamma=lr_scheduler_cfg["gamma"])
    else:
        raise NotImplementedError
    return lr_scheduler

def build(net,
          optimizer_cfg,
          lr_scheduler_cfg,
          start_iter=0):
    optimizer = build_optimizer(net, optimizer_cfg)
    lr_scheduler = build_lr_scheduler(optimizer, lr_scheduler_cfg)
    lr_scheduler.last_epoch = start_iter-1
    return optimizer, lr_scheduler

if __name__ == "__main__":
    import torchvision.models as models
    from incdet3.configs.dev_cfg import cfg, modify_cfg
    net = models.resnet18()
    modify_cfg(cfg)
    optimizer, lr_scheduler = build(net, cfg.TRAIN["optimizer_dict"], cfg.TRAIN["lr_scheduler_dict"])
    print(optimizer)
    print(dir(optimizer))
    print(lr_scheduler)
    print(dir(lr_scheduler))