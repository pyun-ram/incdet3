'''
File Created: Wednesday, 30th October 2019 2:51:52 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
Note: Create Data for 3D object detection. It creates info files and db files for CARLA and WAYMO dataset.
Script: python3 tools/create_data.py \
            --dataset carla \
            --data-dir /usr/app/data/CARLA
        python3 tools/create_data.py \
            --dataset waymo \
            --data-dir /usr/app/data/WAYMO/
Data organization: 
    CARLA/ (WAYMO/)
        training/
            calib/
            label_imu/
            velo_xxx/
            image_2/ (image_front/)
        testing/
            calib/
            label_imu/ (optional)
            velo_xxx/
            image_2/ (image_front/)
        split_index/
            train.txt
            val.txt
            test.txt
            dev.txt (optional)
        (CARLA_infos_train.pkl) / (WAYMO_infos_train.pkl)
        (CARLA_infos_val.pkl) / (WAYMO_infos_val.pkl)
        (CARLA_infos_test.pkl) / (WAYMO_infos_test.pkl)
        (gt_database/) / (gt_database/)
        (CARLA_dbinfos_train.pkl) / (WAYMO_dbinfos_train.pkl)
'''
import argparse
import os
import random
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Manager, Pool
from det3.utils.utils import get_idx_list, read_pc_from_npy, save_pickle, write_pc_to_file
from det3.dataloader.carladata import CarlaCalib, CarlaLabel, CarlaData
from det3.dataloader.waymodata import WaymoCalib, WaymoLabel, WaymoData

def create_info_file_carla_wk_fn(idx):
    global g_data_dir, g_infos
    root_dir = g_data_dir
    infos = g_infos
    info = dict()
    tag = idx
    pc_list = [itm for itm in os.listdir(root_dir) if itm.split('_')[0] == 'velo']
    pc_paths = {itm: os.path.join(root_dir, itm, idx+'.npy') for itm in pc_list}
    img_path = os.path.join(root_dir, "image_2", idx+'.png')
    calib = CarlaCalib(os.path.join(root_dir, "calib", idx+".txt")).read_calib_file()
    label = CarlaLabel(os.path.join(root_dir, "label_imu", idx+".txt")).read_label_file()
    info["tag"] = tag
    info["pc_paths"] = pc_paths
    info["img_path"] = img_path
    info["calib"] = calib
    info["label"] = label
    infos.append(info)

def create_info_file_waymo_wk_fn(idx):
    global g_data_dir, g_infos
    root_dir = g_data_dir
    infos = g_infos
    info = dict()
    tag = idx
    pc_list = [itm for itm in os.listdir(root_dir) if itm.split('_')[0] == 'velo']
    pc_paths = {itm: os.path.join(root_dir, itm, idx+'.npy') for itm in pc_list}
    img_path = os.path.join(root_dir, "image_front", idx+'.png')
    calib = WaymoCalib(os.path.join(root_dir, "calib", idx+".txt")).read_calib_file()
    label = WaymoLabel(os.path.join(root_dir, "label_imu", idx+".txt")).read_label_file()
    info["tag"] = tag
    info["pc_paths"] = pc_paths
    info["img_path"] = img_path
    info["calib"] = calib
    info["label"] = label
    infos.append(info)

def create_info_file(root_dir:str, idx_path:str, save_path:str, dataset: str):
    '''
    Create CARLA_infos_xxx.pkl
    [<info>,]
    info: {
        tag: str (e.g. '000000'),
        pc_paths: {
            velo_top: abs_path,
            ...
        }
        img_path: str,
        calib: CarlaCalib/WaymoCalib,
        label: CarlaLabel/WaymoLabel,
    }
    '''
    global g_data_dir, g_infos
    root_dir = Path(root_dir)
    idx_list = get_idx_list(idx_path)
    idx_list.sort()
    infos = Manager().list()
    g_data_dir = root_dir
    g_infos = infos
    with Pool(8) as p:
        if dataset == "carla":
            r = list(tqdm(p.imap(create_info_file_carla_wk_fn, idx_list), total=len(idx_list)))
        elif dataset == "waymo":
            r = list(tqdm(p.imap(create_info_file_waymo_wk_fn, idx_list), total=len(idx_list)))
    infos = list(infos)
    save_pickle(infos, save_path)
    print(f"Created {save_path}: {len(infos)} samples")

def get_classes(label_dir:str, idx_list:list, dataset:str):
    '''
    Get all the classes by going through all the label files.
    '''
    res = []
    for idx in idx_list:
        if dataset == "carla":
            label = CarlaLabel(label_path=f"{label_dir}/{idx}.txt").read_label_file()
        elif dataset == "waymo":
            label = WaymoLabel(label_path=f"{label_dir}/{idx}.txt").read_label_file()
        else:
            raise NotImplementedError
        for cls in label.bboxes_name:
            if cls not in res:
                res.append(cls)
    return res

def create_db_file_carla_wk_fn(idx):
    import numpy as np
    global g_data_dir, g_dbinfos
    root_dir = g_data_dir
    tag = idx
    pc_list = [itm for itm in os.listdir(root_dir) if itm.split('_')[0] == 'velo']
    pc_dict_Flidar, label, calib = CarlaData(root_dir=str(root_dir),
                                             idx=tag,
                                             output_dict=None).read_data()
    for i, obj in enumerate(label.data):
        name = obj.type
        gt_idx = i
        gtpc_paths = {itm: str(root_dir.parent/"gt_database"/f"{tag}_{itm}_{name}_{gt_idx}.bin")
                      for itm in pc_list}
        nums_point_in_gt = dict()
        box3d_imu = obj
        for velo in pc_list:
            pc_Flidar = pc_dict_Flidar[velo]
            pc_FIMU = calib.lidar2imu(pc_Flidar[:, :3], key=f'Tr_imu_to_{velo}')
            gtpc_idx = obj.get_pts_idx(pc_FIMU[:, :3], calib)
            gtpc = pc_FIMU[gtpc_idx, :]
            nums_point_in_gt[velo] = gtpc.shape[0]
            write_pc_to_file(gtpc.astype(np.float32), gtpc_paths[velo])
        obj_info = {
            "name": name,
            "gtpc_paths": gtpc_paths,
            "tag": tag,
            "gt_idx": gt_idx,
            "box3d_imu": box3d_imu,
            "nums_points_in_gt": nums_point_in_gt,
            "calib": calib
            }
        g_dbinfos.append(obj_info)

def create_db_file_waymo_wk_fn(idx):
    import numpy as np
    global g_data_dir, g_dbinfos
    root_dir = g_data_dir
    tag = idx
    pc_list = [itm for itm in os.listdir(root_dir) if itm.split('_')[0] == 'velo']
    pc_dict_Flidar, label, calib = WaymoData(root_dir=str(root_dir),
                                             idx=tag,
                                             output_dict=None).read_data()
    for i, obj in enumerate(label.data):
        name = obj.type
        gt_idx = i
        gtpc_paths = {itm: str(root_dir.parent/"gt_database"/f"{tag}_{itm}_{name}_{gt_idx}.bin")
                      for itm in pc_list}
        nums_point_in_gt = dict()
        box3d_imu = obj
        for velo in pc_list:
            pc_Flidar = pc_dict_Flidar[velo]
            pc_FIMU = calib.lidar2imu(pc_Flidar, key=f'Tr_imu_to_{velo}')
            gtpc_idx = obj.get_pts_idx(pc_FIMU[:, :3], calib)
            gtpc = pc_FIMU[gtpc_idx, :]
            nums_point_in_gt[velo] = gtpc.shape[0]
            write_pc_to_file(gtpc.astype(np.float32), gtpc_paths[velo])
        obj_info = {
            "name": name,
            "gtpc_paths": gtpc_paths,
            "tag": tag,
            "gt_idx": gt_idx,
            "box3d_imu": box3d_imu,
            "nums_points_in_gt": nums_point_in_gt,
            "calib": calib
            }
        g_dbinfos.append(obj_info)

def create_db_file(root_dir:str, idx_path:str, save_dir:str, dataset:str):
    '''
    Create CARLA_dbinfos_xxx.pkl and save gt_pc into gt_database/
    dbinfo: {
        "Car": [<car_dbinfo>, ],
        "Pedestrian": [<ped_dbinfo>, ],
        ...
    }
    car_dbinfo:{
        name: str,
        gtpc_paths: {
            velo_top: abs_path of saved gtpc (IMU Frame),
            ...
        }
        tag: str,
        gt_idx: int, # no. of obj
        box3d_imu: CarlaObj,
        num_points_in_gt: int,
        calib: CarlaCalib
    }
    '''
    global g_data_dir, g_dbinfos
    root_dir = Path(root_dir)
    save_dir = Path(save_dir)
    (save_dir/"gt_database").mkdir(exist_ok=False)
    idx_list = get_idx_list(idx_path)
    idx_list.sort()
    # get classes
    cls_list = get_classes(root_dir/"label_imu",
                           idx_list,
                           dataset=dataset)
    dbinfo = {itm: [] for itm in cls_list}
    g_data_dir = root_dir
    g_dbinfos = Manager().list()

    if dataset == "carla":
        with Pool(8) as p:
            r = list(tqdm(p.imap(create_db_file_carla_wk_fn, idx_list), total=len(idx_list)))
        for info in g_dbinfos:
            dbinfo[info["name"]].append(info)
        save_pickle(dbinfo, str(save_dir/"CARLA_dbinfos_train.pkl"))
        print("CARLA_dbinfos_train.pkl saved.")
    elif dataset == "waymo":
        with Pool(8) as p:
            r = list(tqdm(p.imap(create_db_file_waymo_wk_fn, idx_list), total=len(idx_list)))
        for info in g_dbinfos:
            dbinfo[info["name"]].append(info)
        save_pickle(dbinfo, str(save_dir/"WAYMO_dbinfos_train.pkl"))
        print("WAYMO_dbinfos_train.pkl saved.")
    else:
        raise NotImplementedError
    for k, v in dbinfo.items():
        print(f"{k}: {len(v)}")

def main(args):
    global g_data_dir, g_infos, g_dbinfos
    dataset = args.dataset.lower()
    data_dir = Path(args.data_dir)
    assert dataset in ["carla", "waymo"], f"Sorry the {dataset} cannot be hundled."
    assert data_dir.exists()

    if dataset == "carla":
        create_info_file(root_dir=str(data_dir/"training"),
                         idx_path=str(data_dir/"split_index"/"dev.txt"),
                         save_path=str(data_dir/"CARLA_infos_dev.pkl"),
                         dataset=dataset)
        create_info_file(root_dir=str(data_dir/"training"),
                         idx_path=str(data_dir/"split_index"/"train.txt"),
                         save_path=str(data_dir/"CARLA_infos_train.pkl"),
                         dataset=dataset)
        create_info_file(root_dir=str(data_dir/"training"),
                         idx_path=str(data_dir/"split_index"/"val.txt"),
                         save_path=str(data_dir/"CARLA_infos_val.pkl"),
                         dataset=dataset)
        create_info_file(root_dir=str(data_dir/"training"),
                         idx_path=str(data_dir/"split_index"/"test.txt"),
                         save_path=str(data_dir/"CARLA_infos_test.pkl"),
                         dataset=dataset)
        # create_db_file(root_dir=str(data_dir/"training"),
        #                      idx_path=str(data_dir/"split_index"/"train.txt"),
        #                      save_dir=str(data_dir),
        #                      dataset=dataset)
    elif dataset == "waymo":
        create_info_file(root_dir=str(data_dir/"training"),
                         idx_path=str(data_dir/"split_index"/"dev.txt"),
                         save_path=str(data_dir/"WAYMO_infos_dev.pkl"),
                         dataset=dataset)
        create_info_file(root_dir=str(data_dir/"training"),
                         idx_path=str(data_dir/"split_index"/"train.txt"),
                         save_path=str(data_dir/"WAYMO_infos_train.pkl"),
                         dataset=dataset)
        create_info_file(root_dir=str(data_dir/"training"),
                         idx_path=str(data_dir/"split_index"/"val.txt"),
                         save_path=str(data_dir/"WAYMO_infos_val.pkl"),
                         dataset=dataset)
        create_info_file(root_dir=str(data_dir/"testing"),
                         idx_path=str(data_dir/"split_index"/"test.txt"),
                         save_path=str(data_dir/"WAYMO_infos_test.pkl"),
                         dataset=dataset)
        create_db_file(root_dir=str(data_dir/"training"),
                             idx_path=str(data_dir/"split_index"/"train.txt"),
                             save_dir=str(data_dir),
                             dataset=dataset)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Data for CARLA & WAYMO dataset.')
    parser.add_argument('--dataset', type=str, help='carla or waymo')
    parser.add_argument('--data-dir', type=str, help='root dir of data')
    args = parser.parse_args()
    random.seed(123)
    main(args)
