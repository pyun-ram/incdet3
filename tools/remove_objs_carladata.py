'''
 File Created: Wed Jul 15 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
 python3 tools/remove_objs_carladata.py \
    --data-dir /usr/app/data/CARLA-P2-TASK2/training/ \
    --remove-classes Car,Cyclist
'''
import os
import argparse
import numpy as np
from tqdm import tqdm
from typing import List
from multiprocessing import Pool
from det3.ops import write_npy, write_txt
from det3.dataloader.carladata import CarlaData, CarlaLabel

def remove_pts_from_pc(pc, mask_list:List):
    '''
    @pc: np.ndarry (N, >=3)
    @mask_list: [np.ndarray (type: bool, shape:(pc.shape[0],))]
    each element is a mask (True: pt to remove).
    '''
    mask = np.zeros(pc.shape[0]).astype(np.bool)
    for other in mask_list:
        mask = np.logical_or(mask, other)
    mask = np.logical_not(mask)
    return pc[mask, :].copy()

def remove_objs_from_label(label, idx_list:List):
    '''
    @label: CarlaLabel
    @idx_list: [int]
    each element indicate the index of the object to remove.
    '''
    res_label = CarlaLabel()
    for i, obj in enumerate(label.data):
        if i in idx_list:
            continue
        res_label.add_obj(obj)
    return res_label

def work_fn(tag):
    pc, label, calib = CarlaData(g_data_dir, tag).read_data()
    pc_lidar = pc["velo_top"].copy()
    pc_imu = calib.lidar2imu(pc["velo_top"], key="Tr_imu_to_velo_top")
    pts_idx_list = []
    obj_list = []
    for i, obj in enumerate(label.data):
        if obj.type not in g_remove_classes:
            continue
        obj.l += 0.5
        obj.w += 0.5
        obj.h += 0.5
        pts_idx = obj.get_pts_idx(pc_imu, calib)
        pts_idx_list.append(pts_idx)
        obj_list.append(i)
    reduced_pc_lidar = remove_pts_from_pc(pc_lidar, pts_idx_list)
    reduced_label = remove_objs_from_label(label, obj_list)
    write_npy(reduced_pc_lidar, os.path.join(g_data_dir, "reduced_velo_top", f"{tag}.npy"))
    write_txt([str(reduced_label)], os.path.join(g_data_dir, "reduced_label_imu", f"{tag}.txt"))

g_data_dir = None
g_remove_classes = []

def main(data_dir, remove_classes):
    global g_data_dir, g_remove_classes
    g_data_dir = data_dir
    g_remove_classes = remove_classes
    tag_list = [itm.split(".")[0]
        for itm in os.listdir(os.path.join(data_dir, "label_imu"))]
    with Pool(8) as p:
        r = list(tqdm(p.imap(work_fn, tag_list), total=len(tag_list)))

if __name__ == "__main__":
    # argparser: data_dir, remove_classes
    parser = argparse.ArgumentParser(description='Post-process the CARLA data '+
        '(removing specific classes in label and point clouds).')
    parser.add_argument('--data-dir', type=str, help='root dir of data')
    parser.add_argument('--remove-classes', type=str, help='remove classes, e.g. Car,Pedestrian')
    args = parser.parse_args()
    data_dir = args.data_dir
    remove_classes = args.remove_classes.split(',')
    os.makedirs(os.path.join(data_dir, "reduced_velo_top"), exist_ok=False)
    os.makedirs(os.path.join(data_dir, "reduced_label_imu"), exist_ok=False)
    assert len(remove_classes) > 0
    assert all([itm in ["Car", "Pedestrian", "Cyclist"] for itm in remove_classes])
    main(data_dir, remove_classes)