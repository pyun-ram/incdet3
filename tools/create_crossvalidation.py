'''
 File Created: Tue Jul 07 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
 python3 tools/create_crossvalidation.py \
     --num-of-folds 5 \
     --data-dir /usr/app/data/CARLA-TOWN02CARPED \
     --mode ordered
'''
import argparse
import os
from det3.ops import write_txt, read_txt
from shutil import copyfile, rmtree, copytree
from random import shuffle
from pathlib import Path

def main(num_of_folds, data_dir, split_dir, train_val_split, mode):
    root_dir = str(Path(data_dir).parent)
    if mode == "random":
    # shuffle train_val_split
        shuffle(train_val_split)
    elif mode == "ordered":
    # pass
        pass
    else:
        raise NotImplementedError
    assert num_of_folds > 1
    num_of_one_part = int(len(train_val_split) / num_of_folds)
    for i in range(num_of_folds):
        val_start = i * num_of_one_part
        val_end = (i+1) * num_of_one_part
        val_split = train_val_split[val_start:val_end]
        train_split = [itm for itm in train_val_split if itm not in val_split]
        # # backup split_dir
        bkupsplit_dir = str(Path(split_dir).parent/"split_dir_bkup")
        copytree(split_dir, bkupsplit_dir)
        # # create new split_dir
        write_txt(train_split, os.path.join(split_dir, "train.txt"))
        write_txt(val_split, os.path.join(split_dir, "val.txt"))
        # # system call
        cmd = "python3 tools/create_data.py " +\
            "--dataset carla " +\
            f"--data-dir {str(Path(data_dir).parent)}"
        os.system(cmd)
        # # rename
        os.rename(os.path.join(root_dir, "CARLA_infos_train.pkl"),
                  os.path.join(root_dir, f"CARLA_infos_train{i}.pkl"))
        os.rename(os.path.join(root_dir, "CARLA_infos_val.pkl"),
                  os.path.join(root_dir, f"CARLA_infos_val{i}.pkl"))
        rmtree(split_dir)
        copytree(bkupsplit_dir, split_dir)
        rmtree(bkupsplit_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(None)
    parser.add_argument("--num-of-folds", type=int, help="num of fold in CV (=1)")
    parser.add_argument("--data-dir", type=str, help="data dir")
    parser.add_argument("--mode", type=str, help="random or ordered")
    args = parser.parse_args()
    num_of_folds = args.num_of_folds
    data_dir = os.path.join(args.data_dir, "training")
    split_idx_dir = os.path.join(args.data_dir, "split_index")
    train_split_path = os.path.join(split_idx_dir, "train.txt")
    val_split_path = os.path.join(split_idx_dir, "val.txt")
    test_split_path = os.path.join(split_idx_dir, "test.txt")
    mode = args.mode # random, ordered
    train_val_split = read_txt(train_split_path) + read_txt(val_split_path)
    main(num_of_folds, data_dir, split_idx_dir, train_val_split, mode)