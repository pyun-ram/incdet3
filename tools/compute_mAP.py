'''
 File Created: Mon Jul 06 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
 python3 tools/compute_mAP.py \
    --log-dir logs/July14-expcarlamore/July14-expcarlamore-lwf-cv0 \
    --val-pkl-path /usr/app/data/CARLA-TOWN01CARPEDCYC/CARLA_infos_val0.pkl \
    --valid-range -35.2 -40 -1.5 35.2 40 2.6 \
    --valid-classes Car,Pedestrian,Cyclist \
    --dataset carla
 python3 tools/compute_mAP.py \
    --log-dir logs/20200813-expkitti2+3/20200813-expkitti2+3-train_class2 \
    --val-pkl-path /usr/app/data/KITTI/KITTI_infos_val.pkl \
    --valid-range 0 -32.0 -3 52.8 32.0 1 \
    --valid-classes Car,Pedestrian \
    --dataset kitti

'''
import argparse
from glob import glob
from incdet3.utils import filt_label_by_range
from det3.ops import read_pkl
import os

g_config_dict = {
    "Car": "3d@0.50",
    "Pedestrian": "3d@0.25",
    "Cyclist": "3d@0.25",
    "Van": "3d@0.50",
    "Truck": "3d@0.50",
}

def main(log_dir, val_pkl_path, valid_range, valid_classes, dataset):
    # compute num of cars and num of pedes
    acc_dict = {itm: 0 for itm in valid_classes}
    ## load val pkl
    val_pkl = read_pkl(val_pkl_path)
    for itm in val_pkl:
        label = itm['label']
        calib = itm['calib'] if dataset == "kitti" else None
        # compute weights by the number of data samples
        # instead of computing weights by the number of instances
        label_ = filt_label_by_range(label, valid_range, calib)
        if len(label_) == 0:
            continue
        has_classes = [obj.type for obj in label_.data]
        has_classes = list(set(has_classes))
        for cls in has_classes:
            if cls in valid_classes:
                acc_dict[cls] += 1

    print(acc_dict)
    # get all eval_pkl_list
    global g_config_dict
    eval_pkl_list = glob(os.path.join(log_dir, '[0-9]*'))
    eval_pkl_list = [os.path.join(itm, 'val_eval_res.pkl') for itm in eval_pkl_list]
    max_eval_pkl = (None, 0)
    for eval_pkl_path in eval_pkl_list:
        res_pkl = read_pkl(eval_pkl_path)
        res_pkl = (res_pkl['detail']
            if dataset == "kitti" else res_pkl['detail']['eval.carla']['carla'])
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
    parser.add_argument('--dataset', type=str, help='e.g. carla or kitti')
    args = parser.parse_args()
    log_dir = args.log_dir
    val_pkl_path = args.val_pkl_path
    # valid_range = tuple(*(args.valid_range.split(",")))
    valid_range = args.valid_range
    valid_classes = args.valid_classes.split(",")
    dataset = args.dataset
    assert dataset in ["kitti", "carla"], f"{dataset} is not supported"
    # assert len(valid_range) == 6
    # for i in range(3):
    #     assert valid_range[i] < valid_range[i+3]
    print(log_dir, val_pkl_path, valid_range, valid_classes, dataset)
    main(log_dir, val_pkl_path, valid_range, valid_classes, dataset)