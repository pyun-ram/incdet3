'''
This script runs the compute attention value (delta), training, testing.
support "feature extraction", "fine-tuning", "joint training", "lwf"
old task: source domain P_D(C), old classes C
new task: target domain P_D'(C'), new classes C' \ C (C' \supseteq C)
merged task: merge two dataset P_D o P_D', all classes C'
feature extraction: load pretrained model, only new fc is trainable with new task gt from random init.
fine-tuning: load pretrained model, backbone is trainable with new task gt from pretrained model;
    new fc is trainable with new task gt from new task gt
joint training: load pretrained model, all parameters are trainable with merged task from pretrained model
lwf: load pretrained model, all parameters are trainable with new task from pretrained model
    under the distillation of old model
'''
g_log_dir, g_save_dir = None, None
g_since = None
#####################################################
def setup_cores(cfg, mode):
    cores = {
        "dataloader_train": None,
        "dataloader_val": None,
        "dataloader_test": None,
        "model": None,
        "optimizer": None,
        "lr_scheduler": None
    }
    if mode == "train":
    raise NotImplementedError
    return cores

def get_data(dataloader,
    mode,
    dataloader_itr=None)
    data = {
        "data": None,
        "dataloader_itr": None
    }
    if mode == "train":
        # cycle_next
    else:
        raise NotImplementedError
    return data

def train_one_iter(model,
    data,
    optimizer,
    lr_scheduler,
    num_iter):
    # setup model train/eval
    # zero gradient
    # compute loss
    # backward loss
    # step optimizer
    # step lr_scheduler
    info = {
        "losses_dict": None,
        "lr": None,
        "num_iter": None
    }
    raise NotImplementedError
    return info

def val_one_epoch(model, dataloader):
    # setup model train/eval
    # detections
    # losses_dict
    # for data in dataloader:
    ## res = model(data)
    ## detections.append(res['detection'])
    ## losses.append(res['loss'])
    # aps = dataloader.evaluate(detections)
    # losses_dict = average_losses(losses_dict)
    # num_iter = model.get_global_iter()
    info = {
        "losses_dict": None,
        "aps": None,
        "num_iter": None
    }
    raise NotImplementedError
    return info

def test_one_epoch(model, dataloader):
    info = val_one_epoch(model, dataloader)
    raise NotImplementedError
    return info

#####################################################
def compute_delta_weights(cfg):
    raise NotImplementedError

def train(cfg):
    global g_log_dir, g_save_dir
    cores = setup_cores(cfg, mode="train")
    model = cores["model"]
    dataloader_train = cores["dataloader_train"]
    dataloader_val = cores["dataloader_val"]
    optimizer = cores["optimizer"]
    lr_scheduler = cores["lr_scheduler"]

    max_iter = cfg.TRAIN["train_iter"]
    num_log_iter = cfg.TRAIN["num_log_iter"]
    num_val_iter = cfg.TRAIN["num_val_iter"]
    num_save_iter = cfg.TRAIN["num_save_iter"]
    dataitr_train = dataloader_train.__iter__()
    while model.get_global_step() < max_iter:
        model.update_global_step()
        data, dataitr_train = get_data(
            dataloader_train, dataitr_train,
            mode="train")
        train_info = train_one_iter(model, data, optimizer, lr_scheduler, model.get_global_step())
        if model.get_global_step() % num_save_iter == 0:
            model.save_weight(model._model, g_save_dir, model.get_global_step())
        if model.get_global_step() % num_log_iter == 0:
            log_train_info(train_info, model.get_global_step())
        if model.get_global_step() % num_val_iter == 0:
            val_info = val_one_epoch(model, dataloader_val)
            log_val_info(val_info, model.get_global_step())

def test(cfg):
    global g_log_dir, g_save_dir
    cores = setup_cores(cfg, mode="test")
    model = cores["model"]
    dataloader_test = cores["dataloader_test"]
    test_info = test_one_epoch(model, dataloader_test)
    log_test_info(test_info, model.get_global_step())

if __name__ == "__main__":
    # parse arg: tag, cfg-path, mode
    # setup dirs
    # setup g_since
    # handle different mode