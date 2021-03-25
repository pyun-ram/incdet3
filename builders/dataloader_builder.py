'''
Dataloader can inherit from mlod
'''
import torch
import numpy as np
from functools import partial
from det3.utils.utils import is_param, proc_param
from incdet3.data.carlapreproc import anchor_creating as anchor_creating_carla
from incdet3.data.carlapreproc import prep_pointcloud as prep_func_carla
from incdet3.data.carlapreproc import prep_info as prep_info_func_carla
from incdet3.data.carladataset import CarlaDataset
from incdet3.data.kittipreproc import anchor_creating as anchor_creating_kitti
from incdet3.data.kittipreproc import prep_pointcloud as prep_func_kitti
from incdet3.data.kittipreproc import prep_info as prep_info_func_kitti
from incdet3.data.kittidataset import KittiDataset
from incdet3.data.nuscenes_dataset import NuScenesDataset
# from incdet3.data.nusckitti_dataset import NuscenesKittiDataset

def create_anchor_cache(target_assigner,
    feature_map_size,
    dataset_name):
    if dataset_name == "carla":
        return anchor_creating_carla(target_assigner,
            feature_map_size,
            anchor_cache=None)
    elif dataset_name == "nusc":
        return anchor_creating_carla(target_assigner,
            feature_map_size,
            anchor_cache=None)
    elif dataset_name in ["kitti", "nusc-kitti"]:
        return anchor_creating_kitti(target_assigner,
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
        prep_func = partial(prep_func_carla, **params)
    elif dataset_name == "nusc":
        prep_func = partial(prep_func_carla, **params)
    elif dataset_name in ["kitti", "nusc-kitti"]:
        prep_func = partial(prep_func_kitti, **params)
    else:
        raise NotImplementedError
    return prep_func

def build_prep_info_func(prep_info_cfg, dataset_name):
    params = {proc_param(k): v
        for k, v in prep_info_cfg.items() if is_param(k)}
    if dataset_name == "carla":
        prep_info_func = partial(prep_info_func_carla, **params)
    elif dataset_name == "nusc":
        prep_info_func = partial(prep_info_func_carla, **params)
    elif dataset_name in ["kitti", 'nusc-kitti']:
        prep_info_func = partial(prep_info_func_kitti, **params)
    else:
        raise NotImplementedError
    return prep_info_func

def build_dataset(
    data_cfg,
    prep_func,
    prep_info_func,
    dataset_name):
    params = {proc_param(k): v
        for k, v in data_cfg.items() if is_param(k)}
    params["prep_func"] = prep_func
    if dataset_name == "carla":
        params["prep_info_func"] = prep_info_func
        dataset = CarlaDataset(**params)
    elif dataset_name == "nusc":
        dataset = NuScenesDataset(**params)
    elif dataset_name == "kitti":
        params["prep_info_func"] = prep_info_func
        dataset = KittiDataset(**params)
    elif dataset_name == "nusc-kitti":
        params["prep_info_func"] = prep_info_func
        dataset = NuscenesKittiDataset(**params)
    else:
        raise NotImplementedError
    return dataset

def _worker_init_fn(worker_id):
    np.random.seed(123 + worker_id)
    print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])

def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "importance",
    ]
    for k, v in example.items():
        if k in float_names:
            # slow when directly provide fp32 data with dtype=torch.half
            example_torch[k] = torch.tensor(
                v, dtype=torch.float32, device=device).to(dtype)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.int32, device=device)
        elif k == "num_voxels":
            example_torch[k] = torch.tensor(v)
        else:
            example_torch[k] = v
    return example_torch

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
    if "prep_infos" in data_cfg.keys():
        prep_info_func = build_prep_info_func(
            prep_info_cfg=data_cfg["prep_infos"],
            dataset_name=dataset_name)
    else:
        prep_info_func = lambda x: x
    # build dataset
    dataset = build_dataset(
        data_cfg=data_cfg,
        prep_func=prep_func,
        prep_info_func=prep_info_func,
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
    data_cfg = cfg.TRAINDATA
    voxelizer = voxelizer_builder.build(cfg.VOXELIZER)
    print(voxelizer.grid_size)
    target_assigner = target_assigner_builder.build(cfg.TARGETASSIGNER)
    dataloader = build(data_cfg,
        ext_dict={
            "voxelizer": voxelizer,
            "target_assigner": target_assigner,
            "feature_map_size": [1, 200, 176]
        })
    for data in dataloader:
        data = example_convert_to_torch(data)
        print(data.keys())
        print(type(data["voxels"]))
    # from det3.utils.utils import save_pickle
    # save_path = "./unit_tests/data/test_build_model_and_init.pkl"
    # save_pickle(data, save_path)
