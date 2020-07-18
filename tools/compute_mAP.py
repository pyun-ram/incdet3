'''
 File Created: Mon Jul 06 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
 python3 tools/compute_mAP.py \
    --log-dir logs/July14-expcarlamore/July14-expcarlamore-lwf-cv0 \
    --val-pkl-path /usr/app/data/CARLA-TOWN01CARPEDCYC/CARLA_infos_val0.pkl \
    --valid-range -35.2 -40 -1.5 35.2 40 2.6 \
    --valid-classes Car,Pedestrian,Cyclist

'''
import argparse
from glob import glob
from incdet3.utils import filt_label_by_range
from det3.ops import read_pkl
import os

g_config_dict = {
    "Car": "3d@0.50",
    "Pedestrian": "3d@0.25",
    "Cyclist": "3d@0.25"
}

def main(log_dir, val_pkl_path, valid_range, valid_classes):
    # compute num of cars and num of pedes
    acc_dict = {itm: 0 for itm in valid_classes}
    ## load val pkl
    val_pkl = read_pkl(val_pkl_path)
    for itm in val_pkl:
        label = itm['label']
        label_ = filt_label_by_range(label, valid_range)
        ## accumulate
        for obj in label_.data:
            acc_dict[obj.type] += 1
    print(acc_dict)
    # get all eval_pkl_list
    global g_config_dict
    eval_pkl_list = glob(os.path.join(log_dir, '*00'))
    eval_pkl_list = [os.path.join(itm, 'val_eval_res.pkl') for itm in eval_pkl_list]
    max_eval_pkl = (None, 0)
    for eval_pkl_path in eval_pkl_list:
        res_pkl = read_pkl(eval_pkl_path)
        res_pkl = res_pkl['detail']['eval.carla']['carla']
        calc_map_dict = {itm: None for itm in valid_classes}
        for cls in valid_classes:
            eval_attrib = g_config_dict[cls]
            calc_map_dict[cls] = res_pkl[cls][eval_attrib]
        # output results
        map_val = 0
        for cls in valid_classes:
            mean_ap = sum(calc_map_dict[cls]) / len(calc_map_dict[cls])
            cls_norm = acc_dict[cls] / sum([v for k, v in acc_dict.items()])
            map_val += mean_ap * cls_norm
        if map_val > max_eval_pkl[1]:
            max_eval_pkl = (eval_pkl_path, map_val)
        print(eval_pkl_path, f"{map_val:.2f}")
    print("Max:", f"{max_eval_pkl}")



if __name__ == "__main__":
    # argparse: log_dir, val_pkl_path, valid_range, valid_classes
    parser = argparse.ArgumentParser(description='Compute mAP from evaluation pkls.')
    parser.add_argument('--log-dir', type=str, help='')
    parser.add_argument('--val-pkl-path', type=str, help='pkl path of val data pkl')
    # parser.add_argument('--valid-range', type=str, help='xmin,ymin,zmin,xmax,ymax,zmax')
    parser.add_argument("--valid-range", nargs=6, metavar=('xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax'),
                            help="", type=float,
                            default=None)
    parser.add_argument('--valid-classes', type=str, help='e.g. Car,Pedestrian')
    args = parser.parse_args()
    log_dir = args.log_dir
    val_pkl_path = args.val_pkl_path
    # valid_range = tuple(*(args.valid_range.split(",")))
    valid_range = args.valid_range
    valid_classes = args.valid_classes.split(",")
    # assert len(valid_range) == 6
    # for i in range(3):
    #     assert valid_range[i] < valid_range[i+3]
    print(log_dir, val_pkl_path, valid_range, valid_classes)
    main(log_dir, val_pkl_path, valid_range, valid_classes)