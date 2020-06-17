'''
File Created: Sunday, 24th March 2019 8:08:05 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab
Usage: python3 tools/data_vis.py \
    --data-dir /usr/app/data/CARLA/training \
    --idx-file /usr/app/data/CARLA/split_index/dev.txt \
    --dataset CARLA \
    --output-dir /usr/app/vis/
        python3 tools/data_vis.py \
    --data-dir /usr/app/data/WAYMO/training \
    --idx-file /usr/app/data/WAYMO/split_index/train.txt \
    --dataset WAYMO \
    --output-dir /usr/app/vis/waymo-train/
'''
import argparse
import os
import numpy as np
from PIL import Image
from det3.utils.utils import get_idx_list
from det3.visualizer.vis import BEVImage
from multiprocessing import Pool
from tqdm import tqdm

def vis_fn(idx):
    global dataset, data_dir, output_dir
    if dataset == "KITTI":
        from det3.dataloader.kittidata import KittiData
        calib, img, label, pc = KittiData(data_dir, idx).read_data()
    elif dataset == "CARLA":
        from det3.dataloader.carladata import CarlaData
        pc_dict, label, calib = CarlaData(data_dir, idx).read_data()
        pc = np.vstack([calib.lidar2imu(v, key="Tr_imu_to_{}".format(k)) for k, v in pc_dict.items()])
    elif dataset == "WAYMO":
        from det3.dataloader.waymodata import WaymoData
        pc_dict, label, calib = WaymoData(data_dir, idx).read_data()
        pc = np.vstack([calib.lidar2imu(v, key="Tr_imu_to_{}".format(k)) for k, v in pc_dict.items()])
    bevimg = BEVImage(x_range=(-50, 50), y_range=(-50, 50), grid_size=(0.05, 0.05))
    bevimg.from_lidar(pc, scale=1)
    for obj in label.read_label_file().data:
        bevimg.draw_box(obj, calib, bool_gt=True)
    bevimg_img = Image.fromarray(bevimg.data)
    bevimg_img.save(os.path.join(output_dir, idx+'.png'))

def main():
    '''
    visualize data
    '''
    parser = argparse.ArgumentParser(description='Visulize Dataset')
    parser.add_argument('--data-dir',
                        type=str, metavar='INPUT PATH',
                        help='dataset dir')
    parser.add_argument('--idx-file',
                        type=str, metavar='INDEX FILE PATH',
                        help='the txt file containing the indeces of the smapled data')
    parser.add_argument('--output-dir',
                        type=str, metavar='OUTPUT PATH',
                        help='output dir')
    parser.add_argument('--dataset',
                        type=str, metavar='DATASET',
                        help='KITTI' or 'CARLA')
    args = parser.parse_args()
    global dataset, data_dir, output_dir
    data_dir = args.data_dir
    idx_path = args.idx_file
    output_dir = args.output_dir
    dataset = args.dataset.upper()
    idx_list = get_idx_list(idx_path)
    with Pool(8) as p:
        r = list(tqdm(p.imap(vis_fn, idx_list), total=len(idx_list)))

if __name__ == '__main__':
    main()