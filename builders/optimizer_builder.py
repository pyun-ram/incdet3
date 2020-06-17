'''
optimizer and lr_scheduler can inherit from second
'''

def build(optimizer_cfg, lr_scheduler_cfg):
    raise NotImplementedError
    return optimizer, lr_scheduler