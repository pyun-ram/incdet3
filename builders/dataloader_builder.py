'''
Dataloader can inherit from mlod
'''
import torch
import numpy as np
from functools import partial
from det3.utils.utils import is_param, proc_param
from incdet3.data.carlapreproc import anchor_creating as anchor_creating_carla
from incdet3.data.carlapreproc import prep_pointcloud as prep_func_carla
from incdet3.data.carladataset import CarlaDataset

def create_anchor_cache(target_assigner,
    feature_map_size,
    dataset_name):
    if dataset_name == "carla":
        return anchor_creating_carla(target_assigner,
            feature_map_size,
            anchor_cache=None)
    else:
        raise NotImplementedError

def build_prep_func(voxelizer,
                    target_assigner,
                    anchor_cache,
                    prep_cfg,
                    dataset_name):
    params = {proc_param(k): v
        for k, v in prep_cfg.items() if is_param(k)}
    params["voxelizer"] = voxelizer
    params["target_assigner"] = target_assigner
    params["anchor_cache"] = anchor_cache
    if dataset_name == "carla":
        prep_cfg = partial(prep_func_carla, **params)
    else:
        raise NotImplementedError
    return prep_cfg

def build_dataset(
    data_cfg,
    prep_func,
    dataset_name):
    params = {proc_param(k): v
        for k, v in data_cfg.items() if is_param(k)}
    params["prep_func"] = prep_func
    if dataset_name == "carla":
        dataset = CarlaDataset(**params)
    else:
        raise NotImplementedError
    return dataset

def _worker_init_fn(worker_id):
    np.random.seed(123 + worker_id)
    print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])


def merge_batch(batch_list):
    from collections import defaultdict
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    for key, elems in example_merged.items():
        if key in [
                'voxels', 'num_points',
        ]:
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        else:
            try:
                ret[key] = np.stack(elems, axis=0)
            except:
                print(f"{key} is not included in the dataloader.")
    return ret

def build(data_cfg, ext_dict: dict):
    dataset_name = data_cfg["dataset"]
    # get voxelizer
    voxelizer = ext_dict["voxelizer"]
    # get target assigner
    target_assigner = ext_dict["target_assigner"]
    # create anchor_cache
    feature_map_size = ext_dict["feature_map_size"]
    anchor_cache = create_anchor_cache(target_assigner,
        feature_map_size,
        dataset_name)
    # create prep
    prep_cfg = data_cfg["prep"]
    prep_cfg["@feature_map_size"] = feature_map_size
    prep_func = build_prep_func(
        prep_cfg=prep_cfg,
        voxelizer=voxelizer,
        target_assigner=target_assigner,
        anchor_cache=anchor_cache,
        dataset_name=dataset_name)
    # build dataset
    dataset = build_dataset(
        data_cfg=data_cfg,
        prep_func=prep_func,
        dataset_name=dataset_name)
    # create dataloader
    training = data_cfg["training"]
    if training:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=data_cfg["batch_size"],
            shuffle=True,
            num_workers=data_cfg["num_workers"],
            pin_memory=True,
            collate_fn=merge_batch,
            worker_init_fn=_worker_init_fn,
            drop_last=False)
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=data_cfg["batch_size"],
            shuffle=False,
            num_workers=data_cfg["num_workers"],
            pin_memory=True,
            collate_fn=merge_batch,
            worker_init_fn=_worker_init_fn,
            drop_last=False)
    return dataloader

if __name__ == "__main__":
    from incdet3.data.carladataset import CarlaDataset
    from incdet3.configs.dev_cfg import cfg
    from incdet3.builders import voxelizer_builder, target_assigner_builder
    from tqdm import tqdm
    data_cfg = cfg.VALDATA
    voxelizer = voxelizer_builder.build(cfg.VOXELIZER)
    target_assigner = target_assigner_builder.build(cfg.TARGETASSIGNER)
    dataloader = build(data_cfg,
        ext_dict={
            "voxelizer": voxelizer,
            "target_assigner": target_assigner,
            "feature_map_size": [1, 200, 176]
        })
    for data in dataloader:
        print(data.keys())
