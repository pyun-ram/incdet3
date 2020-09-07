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
import os
import argparse
import time
import torch
import numpy as np
from tqdm import tqdm
from apex import amp
from datetime import datetime
from shutil import copy as fcopy
from det3.ops import write_pkl
from det3.utils.utils import proc_param, is_param
from det3.utils.log_tool import Logger
from det3.utils.import_tool import load_module
from det3.dataloader.carladata import CarlaData, CarlaObj
from det3.dataloader.kittidata import KittiData, KittiObj
from det3.visualizer.vis import BEVImage
from incdet3.models.model import Network
from incdet3.builders.voxelizer_builder import build as build_voxelizer
from incdet3.builders.target_assigner_builder import build as build_target_assigner
from incdet3.builders.dataloader_builder import build as build_dataloader, example_convert_to_torch
from incdet3.builders.optimizer_builder import build as build_optimizer_and_lr_scheduler
from incdet3.utils.utils import nusc_cls2color

g_log_dir, g_save_dir = None, None
g_since = None
g_use_fp16 = None

def setup_cores(cfg, mode):
    global g_use_fp16
    if mode == "train":
        # build dataloader_train
        voxelizer = build_voxelizer(cfg.VOXELIZER)
        target_assigner = build_target_assigner(cfg.TARGETASSIGNER)
        dataloader_train = build_dataloader(
            data_cfg=cfg.TRAINDATA,
            ext_dict={
                "voxelizer": voxelizer,
                "target_assigner": target_assigner,
                "feature_map_size": cfg.TRAINDATA["feature_map_size"]})
        # build dataloader_val
        dataloader_val = build_dataloader(
            data_cfg=cfg.VALDATA,
            ext_dict={
                "voxelizer": voxelizer,
                "target_assigner": target_assigner,
                "feature_map_size": cfg.VALDATA["feature_map_size"]})
        # build dataloader_test
        dataloader_test = None
        # build model
        param = cfg.NETWORK
        param["@middle_layer_dict"]["@output_shape"] = [1] + voxelizer.grid_size[::-1].tolist() + [16]
        param["@is_training"] = True
        param["@box_coder"] = target_assigner.box_coder
        param = {proc_param(k): v
            for k, v in param.items() if is_param(k)}
        network = Network(**param).cuda()
        # build optimizer & lr_scheduler
        optimizer, lr_scheduler = build_optimizer_and_lr_scheduler(
            net=network,
            optimizer_cfg=cfg.TRAIN["optimizer_dict"],
            lr_scheduler_cfg=cfg.TRAIN["lr_scheduler_dict"],
            start_iter=network.get_global_step())
        # handle fp16 training
        use_fp16 = cfg.TASK["use_fp16"] if "use_fp16" in cfg.TASK.keys() else False
        if use_fp16:
            network, optimizer = amp.initialize(network, optimizer, opt_level="O2")
        g_use_fp16 = use_fp16
    elif mode == "test":
        # build dataloader_train
        voxelizer = build_voxelizer(cfg.VOXELIZER)
        target_assigner = build_target_assigner(cfg.TARGETASSIGNER)
        dataloader_train = None
        # build dataloader_val
        dataloader_val = None
        # build dataloader_test
        dataloader_test = build_dataloader(
            data_cfg=cfg.TESTDATA,
            ext_dict={
                "voxelizer": voxelizer,
                "target_assigner": target_assigner,
                "feature_map_size": cfg.TESTDATA["feature_map_size"]})
        # build model
        param = cfg.NETWORK
        param["@is_training"] = False
        param["@box_coder"] = target_assigner.box_coder
        param["@middle_layer_dict"]["@output_shape"] = [1] + voxelizer.grid_size[::-1].tolist() + [16]
        param = {proc_param(k): v
            for k, v in param.items() if is_param(k)}
        network = Network(**param).cuda()
        # build optimizer & lr_scheduler
        optimizer, lr_scheduler = None, None
    elif mode == "compute_ewc_weights":
        voxelizer = build_voxelizer(cfg.VOXELIZER)
        target_assigner = build_target_assigner(cfg.TARGETASSIGNER)
        dataloader_train = build_dataloader(
            data_cfg=cfg.TRAINDATA,
            ext_dict={
                "voxelizer": voxelizer,
                "target_assigner": target_assigner,
                "feature_map_size": cfg.TRAINDATA["feature_map_size"]})
        dataloader_val, dataloader_test = None, None
        # build model
        param = cfg.NETWORK
        param["@middle_layer_dict"]["@output_shape"] = [1] + voxelizer.grid_size[::-1].tolist() + [16]
        param["@is_training"] = True
        param["@box_coder"] = target_assigner.box_coder
        param = {proc_param(k): v
            for k, v in param.items() if is_param(k)}
        network = Network(**param).cuda()
        # build optimizer & lr_scheduler
        optimizer, lr_scheduler = None, None
    else:
        raise NotImplementedError
    cores = {
        "dataloader_train": dataloader_train,
        "dataloader_val": dataloader_val,
        "dataloader_test": dataloader_test,
        "model": network,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler
    }
    return cores

def get_data(dataloader,
    mode,
    dataloader_itr=None):
    def _cycle_next(dataloader, dataloader_itr):
        try:
            data = dataloader_itr.__next__()
            return data, dataloader_itr
        except StopIteration:
            newdataloader_itr = dataloader.__iter__()
            data = newdataloader_itr.__next__()
            return data, newdataloader_itr
    if mode == "train":
        data, dataloader_itr = _cycle_next(dataloader, dataloader_itr)
        data = example_convert_to_torch(data,
            dtype=torch.float32, device=torch.device("cuda:0"))
    else:
        raise NotImplementedError
    data_dict = {
        "data": data,
        "dataloader_itr": dataloader_itr
    }
    return data_dict

def train_one_iter(model,
    data,
    optimizer,
    lr_scheduler,
    num_iter):
    global g_use_fp16
    model.train()
    optimizer.zero_grad()
    loss_dict = model(data)
    loss = loss_dict["loss_total"].mean()
    if g_use_fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    lr_scheduler.step()
    info = {
        "losses_dict": {k: v.mean() for k, v in loss_dict.items()},
        "lr": lr_scheduler.get_lr(),
        "num_iter": model.get_global_step()
    }
    return info

def log_train_info(info, itr):
    losses_dict = info["losses_dict"]
    s = f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] Iter {itr}: "
    s += "".join([f"{k}: {v:.2f} " for k, v in losses_dict.items()])
    Logger().log_txt(s)
    for k, v in losses_dict.items():
        Logger().log_tsbd_scalor(f"train/{k}", v, itr)
    Logger().log_tsbd_scalor("train/lr", info["lr"], itr)

def val_one_epoch(model, dataloader):
    model.eval()
    detections = []
    for data in tqdm(dataloader):
        data = example_convert_to_torch(data)
        detection = model(data)
        detections.append(detection[0])
    dataset_type = str(type(dataloader.dataset))
    if 'NuScenesDataset' in dataset_type:
        # to avoid the file conflict situation
        # if train multiple cases on a single machine simultaneously.
        eval_res = dataloader.dataset.evaluation(detections, output_dir=g_log_dir)
    elif 'KittiDataset' in dataset_type:
        label_dir = os.path.join(dataloader.dataset._root_path, "label_2")
        eval_res = dataloader.dataset.evaluation(
            detections,
            output_dir=g_log_dir,
            label_dir=label_dir)
    else:
        eval_res = dataloader.dataset.evaluation(detections)
    info = {
        "eval_res": eval_res,
        "num_iter": model.get_global_step(),
        "detections": detections
    }
    return info

def log_val_info(info, itr, vis_param_dict=None):
    num_iter = info["num_iter"]
    detections = info["detections"]
    eval_res = info["eval_res"]
    Logger().log_metrics(eval_res["detail"], num_iter)
    for itm in detections:
        for k, v in itm.items():
            if k in ["metadata", "measurement_forward_time_ms"]:
                itm[k] = v
            elif v is not None:
                itm[k] = v.cpu().numpy() if v.is_cuda else v
    log_val_dir = os.path.join(g_log_dir, str(num_iter))
    os.makedirs(log_val_dir, exist_ok=True)
    write_pkl(detections, os.path.join(log_val_dir, "val_detections.pkl"))
    write_pkl(eval_res, os.path.join(log_val_dir, "val_eval_res.pkl"))
    # only kitti dataset evaluation output "result"
    # carla dataset evaluation output "results"
    if "result" in eval_res.keys():
        Logger().log_txt(str(eval_res['result']))
        num_vis = 1 if len(detections) < 10 else 10
        interval = int(len(detections)/ 10)
        for idx in [i*interval for i in range(num_vis)]:
            detection = detections[idx]
            tag = detection['metadata']['tag']
            vis_img = vis_fn_kitti(idx=tag,
                detection=detection,
                data_dir=vis_param_dict["data_dir"],
                x_range=vis_param_dict["x_range"],
                y_range=vis_param_dict["y_range"],
                grid_size=vis_param_dict["grid_size"])
            Logger().log_tsbd_img("val/detections", vis_img.data, num_iter)
        return

    if "carla" in eval_res['results'].keys():
        Logger().log_txt(str(eval_res['results']['carla']))
        num_vis = 1 if len(detections) < 10 else 10
        interval = int(len(detections)/ 10)
        for idx in [i*interval for i in range(num_vis)]:
            detection = detections[idx]
            tag = detection['metadata']['tag']
            vis_img = vis_fn(idx=tag,
                detection=detection,
                data_dir=vis_param_dict["data_dir"],
                x_range=vis_param_dict["x_range"],
                y_range=vis_param_dict["y_range"],
                grid_size=vis_param_dict["grid_size"])
            Logger().log_tsbd_img("val/detections", vis_img.data, num_iter)
    elif "nusc" in eval_res['results'].keys():
        Logger().log_txt(str(eval_res['results']['nusc']))
        num_vis = 1 if len(detections) < 10 else 10
        interval = int(len(detections)/ 10)
        for idx in [i*interval for i in range(num_vis)]:
            detection = detections[idx]
            vis_img = vis_fn_nusc(idx=idx,
                detection=detection,
                dataset=vis_param_dict["dataset"],
                x_range=vis_param_dict["x_range"],
                y_range=vis_param_dict["y_range"],
                grid_size=vis_param_dict["grid_size"])
            Logger().log_tsbd_img("val/detections", vis_img.data, num_iter)
    else:
        raise NotImplementedError

def vis_fn_kitti(data_dir,
    idx,
    detection,
    x_range=(-35.2, 35.2),
    y_range=(-40, 40),
    grid_size=(0.1, 0.1)):
    itm = detection
    output_dict = {
        "calib": True,
        "image": False,
        "label": True,
        "velodyne": True
    }
    calib, _, label, pc = KittiData(data_dir,
        idx,output_dict=output_dict).read_data()
    bevimg = BEVImage(x_range, y_range, grid_size)
    bevimg.from_lidar(pc)
    for obj in label.data:
        bevimg.draw_box(obj, calib, bool_gt=True)
    box3d_lidar = itm["box3d_lidar"]
    score = itm["scores"]
    for box3d_lidar_, score_ in zip(box3d_lidar, score):
        x, y, z, w, l, h, ry = box3d_lidar_
        obj = KittiObj()
        bcenter_Flidar = np.array([x, y, z]).reshape(1, -1)
        bcenter_Fcam = calib.lidar2leftcam(bcenter_Flidar)
        obj.x, obj.y, obj.z = bcenter_Fcam.flatten()
        obj.w, obj.l, obj.h = w, l, h
        obj.ry = ry
        bevimg.draw_box(obj, calib, bool_gt=False, width=2)
    return bevimg

def vis_fn_nusc(dataset,
    idx,
    detection,
    x_range=(-35.2, 35.2),
    y_range=(-40, 40),
    grid_size=(0.1, 0.1)):
    itm = detection
    input_dict = dataset.get_sensor_data(idx)
    input_dict = dataset._to_carlaloader(input_dict)
    pc = input_dict["lidar"]["points"]["velo_top"]
    label = input_dict["imu"]["label"]
    calib = input_dict["calib"]
    bevimg = BEVImage(x_range, y_range, grid_size)
    bevimg.from_lidar(pc)
    for obj in label.data:
        bevimg.draw_box(obj, calib, bool_gt=True, width=3, text=obj.type)
    box3d_lidar = itm["box3d_lidar"]
    score = itm["scores"]
    label_preds = itm["label_preds"]
    for box3d_lidar_, label_pred_ in zip(box3d_lidar, label_preds):
        label_pred_name = dataset._class_names[label_pred_]
        if label_pred_name in nusc_cls2color.keys():
            color = nusc_cls2color[label_pred_name]
        else:
            color = nusc_cls2color["default"]
        x, y, z, w, l, h, ry = box3d_lidar_
        obj = CarlaObj()
        obj.x, obj.y, obj.z = x, y, z
        obj.w, obj.l, obj.h = w, l, h
        obj.ry = ry
        bevimg.draw_box(obj, calib, bool_gt=False, width=2, c=color)
    return bevimg

def vis_fn(data_dir,
    idx,
    detection,
    x_range=(-35.2, 35.2),
    y_range=(-40, 40),
    grid_size=(0.1, 0.1)):
    itm = detection
    lidar = "velo_top"
    pc_dict, label, calib = CarlaData(data_dir, idx).read_data()
    pc = calib.lidar2imu(pc_dict[lidar][:, :3], key=f"Tr_imu_to_{lidar}")
    bevimg = BEVImage(x_range, y_range, grid_size)
    bevimg.from_lidar(pc)
    for obj in label.data:
        bevimg.draw_box(obj, calib, bool_gt=True)
    box3d_lidar = itm["box3d_lidar"]
    score = itm["scores"]
    for box3d_lidar_, score_ in zip(box3d_lidar, score):
        x, y, z, w, l, h, ry = box3d_lidar_
        obj = CarlaObj()
        obj.x, obj.y, obj.z = x, y, z
        obj.w, obj.l, obj.h = w, l, h
        obj.ry = ry
        bevimg.draw_box(obj, calib, bool_gt=False, width=2)
    return bevimg

def test_one_epoch(model, dataloader):
    info = val_one_epoch(model, dataloader)
    return info

def log_test_info(info, log_dir):
    detections = info["detections"]
    eval_res = info["eval_res"]
    for itm in detections:
        for k, v in itm.items():
            if k in ["metadata", "measurement_forward_time_ms"]:
                itm[k] = v
            elif v is not None:
                itm[k] = v.cpu().numpy() if v.is_cuda else v
    log_val_dir = g_log_dir
    os.makedirs(log_val_dir, exist_ok=True)
    write_pkl(detections, os.path.join(log_val_dir, "test_detections.pkl"))
    write_pkl(eval_res, os.path.join(log_val_dir, "test_results.pkl"))
    if "result" in eval_res.keys():
        Logger().log_txt(str(eval_res['result']))
        return
    if "carla" in eval_res["results"].keys():
        Logger().log_txt(str(eval_res['results']['carla']))
    elif "nusc" in eval_res["results"].keys():
        Logger().log_txt(str(eval_res['results']['nusc']))
    else:
        raise NotImplementedError

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
    iter_elapsed = 0
    while model.get_global_step() < max_iter:
        iter_elapsed += 1
        model.update_global_step()
        data_dict = get_data(dataloader_train, mode="train", dataloader_itr=dataitr_train)
        data = data_dict["data"]
        dataitr_train = data_dict["dataloader_itr"]
        train_info = train_one_iter(model, data, optimizer, lr_scheduler, model.get_global_step())
        if model.get_global_step() % num_save_iter == 0 or model.get_global_step() >= max_iter:
            Network.save_weight(model._model, g_save_dir, model.get_global_step())
            if model._sub_model is not None:
                Network.save_weight(model._sub_model, g_save_dir, model.get_global_step())
        if model.get_global_step() % num_log_iter == 0 or model.get_global_step() >= max_iter:
            log_train_info(train_info, model.get_global_step())
            time_elapsed = time.time() - g_since
            ert = (time_elapsed / iter_elapsed * (max_iter - model.get_global_step()))
            print(f"Estimated time remaining: {int(ert / 60):d} min {int(ert % 60):d} s")
        if model.get_global_step() % num_val_iter == 0 or model.get_global_step() >= max_iter:
            val_info = val_one_epoch(model, dataloader_val)
            log_val_info(val_info, model.get_global_step(),
                vis_param_dict={
                    "data_dir": cfg.VALDATA["@root_path"],
                    "x_range": (cfg.TASK["valid_range"][0], cfg.TASK["valid_range"][3]),
                    "y_range": (cfg.TASK["valid_range"][1], cfg.TASK["valid_range"][4]),
                    "grid_size": (0.1, 0.1),
                    "dataset": dataloader_val.dataset
                })
    Logger.log_txt("Training DONE!")

def test(cfg):
    global g_log_dir, g_save_dir
    cores = setup_cores(cfg, mode="test")
    model = cores["model"]
    dataloader_test = cores["dataloader_test"]
    test_info = test_one_epoch(model, dataloader_test)
    log_test_info(test_info, model.get_global_step())

def compute_ewc_weights(cfg):
    global g_log_dir, g_save_dir
    cores = setup_cores(cfg, mode="compute_ewc_weights")
    model = cores["model"]
    dataloader_train = cores["dataloader_train"]
    if "@num_of_datasamples" in cfg.EWC.keys():
        num_of_datasamples = cfg.EWC["@num_of_datasamples"]
    else:
        num_of_datasamples = len(dataloader_train.dataset)
    params = {proc_param(k): v
              for k, v in cfg.EWC.items() if is_param(k)}
    params["num_of_datasamples"] = num_of_datasamples
    params["dataloader"] = dataloader_train
    if "@debug_mode" not in cfg.EWC.keys() or not cfg.EWC["@debug_mode"]:
        ewc_weights = model.compute_ewc_weights(**params)
        write_pkl({k: v.cpu().numpy() for k, v in ewc_weights.items()},
            os.path.join(g_save_dir, f"ewc_weights-{model.get_global_step()}.pkl"))
    else:
        cls_term, reg_term, ewc_weights = model.compute_ewc_weights(**params)
        write_pkl({k: v.cpu().numpy() for k, v in cls_term.items()},
            os.path.join(g_save_dir, f"ewc_clsterm-{model.get_global_step()}.pkl"))
        write_pkl({k: v.cpu().numpy() for k, v in reg_term.items()},
            os.path.join(g_save_dir, f"ewc_regterm-{model.get_global_step()}.pkl"))
        write_pkl({k: v.cpu().numpy() for k, v in ewc_weights.items()},
            os.path.join(g_save_dir, f"ewc_weights-{model.get_global_step()}.pkl"))

def setup_dir_and_logger(tag):
    global g_log_dir, g_save_dir
    root_dir = "./"
    g_log_dir = os.path.join(root_dir, f"logs/{tag}")
    g_save_dir = os.path.join(root_dir, f"saved_weights/{tag}")
    os.makedirs(g_save_dir, exist_ok=True)
    os.makedirs(g_log_dir, exist_ok=True)
    logger = Logger()
    logger.global_dir = g_log_dir

def load_config_file(cfg_path,
                     log_dir=None,
                     backup=True) -> dict:
    assert os.path.isfile(cfg_path)
    if backup:
        bkup_path = os.path.join(log_dir, f"config-{datetime.fromtimestamp(time.time())}.py")
        fcopy(cfg_path, bkup_path)
        cfg = load_module(bkup_path, "cfg")
        check_cfg = load_module(bkup_path, "check_cfg")
        modify_cfg = load_module(bkup_path, "modify_cfg")
    else:
        cfg = load_module(cfg_path, "cfg")
        check_cfg = load_module(cfg_path, "check_cfg")
        modify_cfg = load_module(cfg_path, "modify_cfg")
    modify_cfg(cfg)
    assert check_cfg(cfg)
    return cfg

if __name__ == "__main__":
    # parse arg: tag, cfg-path, mode
    parser = argparse.ArgumentParser(description="Incremental 3D Detector")
    parser.add_argument('--tag',
                        type=str, metavar='TAG',
                        help='tag', default=None)
    parser.add_argument('--cfg-path',
                        type=str, metavar='CFG',
                        help='config file path')
    parser.add_argument('--mode',
        choices = ['train', 'test', 'compute_channel_weights', 'compute_ewc_weights'],
        default = 'test')
    args = parser.parse_args()
    cfg_path = args.cfg_path
    tag = args.tag if args.tag is not None else f"IncDet3-{time.time():.2f}"
    # setup dirs
    setup_dir_and_logger(tag)
    cfg = load_config_file(cfg_path, log_dir=g_log_dir, backup=True)
    # setup g_since
    g_since = time.time()
    # handle different mode
    if args.mode == "train":
        train(cfg)
    elif args.mode == "test":
        test(cfg)
    elif args.mode == "compute_channel_weights":
        raise NotImplementedError
    elif args.mode == "compute_ewc_weights":
        compute_ewc_weights(cfg)
    else:
        raise NotImplementedError
