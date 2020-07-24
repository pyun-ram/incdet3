'''
Create CARLA-P1+ from CARLA-P1
'''
import os
from pathlib import Path
from shutil import copyfile
from det3.ops import read_pkl
from incdet3.utils import filt_label_by_cls, filt_label_by_range

def filter_info_file_and_save(data_dir,
    valid_range,
    task_name,
    info_file_name):
    source_path = Path(data_dir, f"CARLA-P1-{task_name}", info_file_name)
    target_path = Path(data_dir, f"CARLA-P1+-{task_name}", info_file_name)
    ## get train_info
    src_info = read_pkl(source_path)
    target_info = []
    ## filter train_info
    if task_name == "TASK2":
        target_class = ["Pedestrian"]
    elif task_name == "TASK3":
        target_class = ["Cyclist"]
    else:
        raise NotImplementedError
    for itm in src_info:
        label = info["label"]

    ## save train_info

def setup_dirs_and_create_soft_links(data_dir):
    os.makedirs(Path(data_dir)/"CARLA-P1+-TASK2")
    os.makedirs(Path(data_dir)/"CARLA-P1+-TASK3")

    os.symlink(Path(data_dir)/"CARLA-P1-TASK1",
               Path(data_dir)/"CARLA-P1+-TASK1")
    os.symlink(Path(data_dir)/"CARLA-P1-TASK2"/"training",
               Path(data_dir)/"CARLA-P1+-TASK2"/"training")
    os.symlink(Path(data_dir)/"CARLA-P1-TASK3"/"training",
               Path(data_dir)/"CARLA-P1+-TASK3"/"training")

    task_name_list = ["TASK2", "TASK3"]
    info_file_name_list = ["CARLA_infos_test.pkl",
        "CARLA_infos_val0.pkl", "CARLA_infos_val1.pkl"
        "CARLA_infos_val2.pkl", "CARLA_infos_val3.pkl",
        "CARLA_infos_val4.pkl"]
    for task_name in task_name_list:
        for info_file_name in info_file_name_list:
            copyfile(Path(data_dir)/f"CARLA-P1-{task_name}"/info_file_name,
                     Path(data_dir)/f"CARLA-P1+-{task_name}"/info_file_name)

def main(data_dir, valid_range):
    # check task1, task2, task3 dir existence
    # make dirs and create soft links
    setup_dirs_and_create_soft_links(data_dir)
    task_name_list = ["TASK2", "TASK3"]
    info_file_name_list = ["CARLA_infos_test.pkl",
        "CARLA_infos_train0.pkl", "CARLA_infos_train1.pkl"
        "CARLA_infos_train2.pkl", "CARLA_infos_train3.pkl",
        "CARLA_infos_train4.pkl"]
    for task_name in task_name_list:
        for info_file_name in info_file_name_list:
            filter_info_file_and_save(
                data_dir,
                valid_range,
                task_name,
                info_file_name)

if __name__ == "__main__":
    # argparser: data_dir (CARLA-P1 dir), valid_range
    main(data_dir)