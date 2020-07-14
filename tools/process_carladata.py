'''
This script post-processes the collected Carla data.
1) Rearrange the collected Carla data
python3 tools/process_carladata.py --data-dir /usr/app/data
'''
import argparse
import os
from tqdm import tqdm
from shutil import move
from copy import deepcopy
from det3.ops import write_txt
from det3.dataloader.carladata import CarlaLabel

def create_move_dict(source_dir, target_dir, start_tag:int):
    move_dict = {} # "source_path": "target_path"
    tags_source = os.listdir(os.path.join(source_dir, "label_imu"))
    tags_source = [int(itm.split(".")[0]) for itm in tags_source]
    tags_source.sort()
    tag_pointer = start_tag + len(tags_source)
    # calib
    for tag_source in tags_source:
        source_path = os.path.join(source_dir, "calib", f"{tag_source:06d}.txt")
        target_path = os.path.join(target_dir, "calib", f"{tag_source+start_tag:06d}.txt")
        move_dict[source_path] = target_path
    # label_imu
    for tag_source in tags_source:
        source_path = os.path.join(source_dir, "label_imu", f"{tag_source:06d}.txt")
        target_path = os.path.join(target_dir, "label_imu", f"{tag_source+start_tag:06d}.txt")
        move_dict[source_path] = target_path
    # velo_top
    for tag_source in tags_source:
        source_path = os.path.join(source_dir, "velo_top", f"{tag_source:06d}.npy")
        target_path = os.path.join(target_dir, "velo_top", f"{tag_source+start_tag:06d}.npy")
        move_dict[source_path] = target_path
    # image_2
    for tag_source in tags_source:
        source_path = os.path.join(source_dir, "image_2", f"{tag_source:06d}.png")
        target_path = os.path.join(target_dir, "image_2", f"{tag_source+start_tag:06d}.png")
        move_dict[source_path] = target_path
    target_idx_list = []
    for tag_source in tags_source:
        target_idx_list.append(f"{tag_source+start_tag:06d}")
    return move_dict, tag_pointer, target_idx_list

def sort_fn(dir_name):
    mode = dir_name.split("_")[0]
    num = int(dir_name.split("_")[1])*0.1
    if mode == "train":
        return 100.0+num
    if mode == "val":
        return 200.0+num
    if mode == "test":
        return 300.0+num

def find_cyclists(label):
    '''
    cyclist: w < 1.5 or l < 2 or h < 1
    ->idx list
    '''
    idx_list = []
    for idx, obj in enumerate(label.data):
        if obj.type == "Pedestrian":
            continue
        if obj.w < 1.5 or obj.l < 2 or obj.h < 1:
            idx_list.append(idx)
    return idx_list

def change_objs(label, cyc_idx_list):
    '''
    -> label (copy)
    '''
    res_label = CarlaLabel()
    for i, obj in enumerate(label.data):
        if i in cyc_idx_list:
            obj.type = "Cyclist"
        res_label.add_obj(obj)
    return res_label

def save_label(label, path):
    with open(path, "w") as f:
        f.writelines(str(label))

def main(data_dir, with_cyc=False):
    # get all dirs
    dir_names = os.listdir(data_dir)
    # create move_dict
    move_dict = {}
    start_tag = 0
    idx_lists = {"train": [], "val": [], "test": []}
    dir_names.sort(key=sort_fn)
    for dir_name in dir_names:
        newmove_dict, start_tag, idx_list = create_move_dict(
            source_dir=os.path.join(data_dir, dir_name),
            target_dir=os.path.join(data_dir, "CARLA"),
            start_tag=start_tag
        )
        mode=dir_name.split("_")[0]
        idx_lists[mode] += idx_list
        move_dict = {**move_dict, **newmove_dict}
    # create dir
    os.makedirs(os.path.join(data_dir, "CARLA"))
    os.makedirs(os.path.join(data_dir, "CARLA", "calib"))
    os.makedirs(os.path.join(data_dir, "CARLA", "label_imu"))
    os.makedirs(os.path.join(data_dir, "CARLA", "velo_top"))
    os.makedirs(os.path.join(data_dir, "CARLA", "image_2"))
    # move
    for source_path, target_path in move_dict.items():
        os.symlink(source_path, target_path)
        # move(source_path, target_path)
    write_txt(idx_lists["train"],
        os.path.join(data_dir, "CARLA", "train.txt"))
    write_txt(idx_lists["val"],
        os.path.join(data_dir, "CARLA", "val.txt"))
    write_txt(idx_lists["test"],
        os.path.join(data_dir, "CARLA", "test.txt"))

    if with_cyc:
        print("Will create Cyclist labels.")
        label_dir = os.path.join(data_dir, "CARLA", "label_imu")
        proclabel_save_dir = os.path.join(data_dir, "CARLA", "label_imu_proc")
        os.makedirs(proclabel_save_dir, exist_ok=False)
        tag_list = os.listdir(label_dir)
        for tag in tqdm(tag_list):
            label = CarlaLabel(os.path.join(label_dir, tag)).read_label_file()
            cyc_idx = find_cyclists(label)
            label = change_objs(label, cyc_idx)
            save_label(label, os.path.join(proclabel_save_dir, tag))
        print("Successfully create the Cyclist labels.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--data-dir',
                        type=str, metavar='Data Dir',
                        help='data dir', default=None)
    parser.add_argument('--with-cyc',
                        action='store_true',
                        help='enable cyclists')
    args = parser.parse_args()
    data_dir = args.data_dir
    with_cyc = args.with_cyc
    main(data_dir, with_cyc)