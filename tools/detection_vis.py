'''
File Created: Sunday, 24th March 2019 8:08:05 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
Usage: python3 tools/detection_vis.py \
    --data-dir /usr/app/data/CARLA_MULTI/training \
    --info-path /usr/app/data/CARLA_MULTI/CARLA_infos_train.pkl \
    --det-path logs/incdet-fine-tuning-l2sp/2900/val_detections.pkl \
    --lidar velo_top \
    --dataset carla
        python3 tools/detections_vis.py \
    --data-dir /usr/app/data/WAYMO/training \
    --info-path /usr/app/data/WAYMO/WAYMO_infos_dev.pkl \
    --det-path logs/MLOD-MLOD-DEV/test_detections.pkl \
    --lidar merge \
    --dataset waymo
        python3 tools/detections_vis.py \
    --data-dir /usr/app/data/LYFT/training \
    --info-path /usr/app/data/LYFT/CARLA_infos_test.pkl \
    --det-path logs/MLOD-LYFTCAR-PRE-A/test_detections.pkl \
    --lidar velo_top \
    --dataset carla
        python3 tools/detection_vis.py \
    --data-dir /usr/app/data/KITTI/training \
    --info-path /usr/app/data/KITTI/KITTI_infos_val_cv3-tuning0.pkl \
    --det-path logs/20200907-tune-ewcsigma-logs/20200905-tune-ewcsigma-kitti2+3-kdewc-cv0-0.1/29450/val_detections.pkl \
    --lidar velo_top \
    --dataset kitti
'''
import argparse
import os
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
import numpy as np
from det3.utils.utils import get_idx_list, load_pickle
from det3.visualizer.vis import BEVImage
from det3.dataloader.carladata import CarlaData, CarlaObj
from det3.dataloader.waymodata import WaymoData, WaymoObj
from det3.dataloader.kittidata import KittiData, KittiObj

detections = None
dataset = None
data_dir = None
lidar = None
detections_path = None

def vis_fn(i):
    global detections, dataset, data_dir, lidar, detections_path
    idx, itm = detections[i]
    if dataset == "carla":
        pc_dict, label, calib = CarlaData(data_dir, idx).read_data()
    elif dataset == "waymo":
        pc_dict, label, calib = WaymoData(data_dir, idx).read_data()
    elif dataset == "kitti":
        output_dict = {
            "calib": True,
            "image": False,
            "label": True,
            "velodyne": True
        }
        calib, image, label, pc = KittiData(data_dir, idx, output_dict).read_data()
    else:
        raise NotImplementedError
    if dataset != "kitti":
        if lidar == "merge":
            pc = np.vstack([calib.lidar2imu(v, key="Tr_imu_to_{}".format(k))
                for k, v in pc_dict.items()])
        elif lidar in ["velo_top", "velo_left", "velo_right"]:
            pc = calib.lidar2imu(pc_dict[lidar][:, :3], key=f"Tr_imu_to_{lidar}")
        else:
            raise NotImplementedError
        bevimg = BEVImage(x_range=(-35.2, 35.2), y_range=(-40, 40), grid_size=(0.1, 0.1))
    else:
        bevimg = BEVImage(x_range=(0, 52.8), y_range=(-40, 40), grid_size=(0.1, 0.1))
    bevimg.from_lidar(pc)
    for obj in label.data:
        bevimg.draw_box(obj, calib, bool_gt=True)
    box3d_lidar = itm["box3d_lidar"]
    score = itm["scores"]
    for box3d_lidar_, score_ in zip(box3d_lidar, score):
        x, y, z, w, l, h, ry = box3d_lidar_
        if dataset == "carla":
            obj = CarlaObj()
            obj.x, obj.y, obj.z = x, y, z
            obj.w, obj.l, obj.h = w, l, h
            obj.ry = ry
        elif dataset == "waymo":
            obj = WaymoObj()
            obj.x, obj.y, obj.z = x, y, z
            obj.w, obj.l, obj.h = w, l, h
            obj.ry = ry
        elif dataset == "kitti":
            obj = KittiObj()
            bcenter_Flidar = np.array([x, y, z]).reshape(1,-1)
            bcenter_Fcam = calib.lidar2leftcam(bcenter_Flidar)
            obj.x, obj.y, obj.z = bcenter_Fcam.flatten()
            obj.w, obj.l, obj.h = w, l, h
            obj.ry = ry
        else:
            raise NotImplementedError
        bevimg.draw_box(obj, calib, bool_gt=False, width=2)
    bevimg.save(path=Path(detections_path).parent/f"{idx}.png")

def main(args):
    '''
    visualize data
    '''
    global detections, dataset, data_dir, lidar, detections_path
    # param
    detections_path = args.det_path
    data_dir = args.data_dir
    lidar = args.lidar
    dataset = args.dataset
    # get_idx
    idx_list = [itm["tag"] for itm in load_pickle(args.info_path)]
    # load detections
    detections = load_pickle(detections_path)
    idx_list = idx_list[:len(detections)]
    detections = [(idx, itm) for idx, itm in zip(idx_list, detections)]
    # match detection with tag
    # visualization
    with Pool(8) as p:
        r = list(tqdm(p.imap(vis_fn, range(len(detections))), total=len(detections)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visulize Detection')
    parser.add_argument('--data-dir',
                        type=str, metavar='DATA PATH',
                        help='dataset dir')
    parser.add_argument('--det-path',
                        type=str, metavar='DETS PATH',
                        help='detections.pkl path')
    parser.add_argument('--info-path',
                        type=str, metavar='INFO PATH',
                        help='info.pkl path')
    parser.add_argument('--lidar',
                        type=str, metavar='LIDAR',
                        help='velo_top, velo_left, velo_right, merge')
    parser.add_argument('--dataset',
                        type=str, metavar='DATASET',
                        help='carla or waymo')
    args = parser.parse_args()
    main(args)