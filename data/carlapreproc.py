import numpy as np
from det3.dataloader.carladata import Frame
from det3.dataloader.augmentor import CarlaAugmentor
from det3.methods.second.ops.ops import rbbox2d_to_near_bbox
from incdet3.utils import (filt_label_by_cls,
    filt_label_by_num_of_pts,
    filt_label_by_range, bcolors)
from det3.utils.log_tool import Logger

def get_pc_from_dict(pc_dict_FIMU: dict):
    '''
    convert pc_dict_FIMU to pc according to single_lidar and multi_lidar
    @pc_dict_FIMU: dict of pc
    ->pc: np.array or dict
    '''
    pc = pc_dict_FIMU["velo_top"]
    return pc

def db_sampling(pc,
                db_sampler,
                gt_calib,
                gt_label,
                training: bool,):
    '''
    sampling from db to augment label
    @pc: np.array
    @db_sampler: DBSampler
    @gt_calib: CarlaCalib
    @gt_label: CarlaLabel
    @training: bool
    ->label: [CarlaObj]
    ->calib: [None/CarlaCalib]
    ->pc: np.array
        None if the according obj is the original copy from gt.
    Note: this function provides an api for the exemplar improvement (as ICaRL).
    '''
    if not training or db_sampler is None:
        return  gt_label, [gt_calib] * len(gt_label.data), pc
    raise NotImplementedError

def augmenting(pc,
               calib,
               label,
               training: bool,
               augment_dict: dict):
    '''
    data augmentation
    @pc: np.array
    @calib: [CarlaCalib]
    @label: CarlaLabel
    @single_lidar: (None, str)
    @multi_lidar: (None, str)
    @training: bool
    @augment_dict: dict
    -> label: CarlaLabel
    -> pc: np.array
    '''
    if not training or augment_dict is None:
        return label, pc
    augmentor = CarlaAugmentor(**augment_dict)
    label, pc = augmentor.apply(label, pc, calib)
    return label, pc

def label_filtering(pc,
                    calib,
                    label,
                    filter_label_dict):
    '''
    @pc: np.array or dict
    @calib: CarlaCalib or list
    @label: CarlaLabel
    @filter_label_dict: dict
    e.g. {"keep_classes": ["Car", "Pedestrian"],
          "min_num_pts": 5,
          "label_range": [-50, -50, -1.5, 50, 50, 2.6],}
    @single_lidar: "velo_top", "velo_left", "velo_right", None
    @multi_lidar: None, "merge", "split", "split+merge"
    Note: This function has to be used with get_pc_from_dict
    '''
    if len(label) == 0:
        if label.data is None:
            label.data = []
        return label
    # by class
    if "keep_classes" in filter_label_dict.keys():
        keep_classes = filter_label_dict["keep_classes"]
        label = filt_label_by_cls(label, keep_classes)
    # by num of pts
    if "min_num_pts" in filter_label_dict.keys():
        min_num_pts = filter_label_dict["min_num_pts"]
        label = filt_label_by_num_of_pts(pc, calib, label, min_num_pts)
    # by range
    if "label_range" in filter_label_dict.keys():
        valid_range = filter_label_dict["label_range"]
        label = filt_label_by_range(label, valid_range=valid_range)
    return label

def voxelizing(pc,
               voxelizer):
    vox_res = voxelizer.generate(pc, voxelizer._max_voxels)
    return vox_res

def anchor_creating(target_assigner,
                    feature_map_size,
                    anchor_cache):
    if anchor_cache is not None:
        anchors = anchor_cache["anchors"]
        anchors_bv = anchor_cache["anchors_bv"]
        anchors_dict = anchor_cache["anchors_dict"]
        matched_thresholds = anchor_cache["matched_thresholds"]
        unmatched_thresholds = anchor_cache["unmatched_thresholds"]

    else:
        ret = target_assigner.generate_anchors(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, target_assigner.box_ndim])
        anchors_dict = target_assigner.generate_anchors_dict(feature_map_size)
        anchors_bv = rbbox2d_to_near_bbox(
            anchors[:, [0, 1, 3, 4, 6]])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
    anchor_res = {
        "anchors": anchors,
        "anchors_bv": anchors_bv,
        "anchors_dict": anchors_dict,
        "matched_thresholds": matched_thresholds,
        "unmatched_thresholds": unmatched_thresholds
    }
    return anchor_res

def target_assigning(target_assigner,
                     label,
                     calib,
                     anchor_res,
                     classes_to_exclude=[]):
    class_names = target_assigner.classes
    gt_dict = carlalabel2gt_dict(label,
        class_names,
        calib,
        classes_to_exclude=classes_to_exclude)
    targets_dict = target_assigner.assign(
        anchor_res["anchors"],
        anchor_res["anchors_dict"],
        gt_dict["gt_boxes"],
        anchors_mask=None,
        gt_classes=gt_dict["gt_classes"],
        gt_names=gt_dict["gt_names"],
        matched_thresholds=anchor_res["matched_thresholds"],
        unmatched_thresholds=anchor_res["unmatched_thresholds"],
        importance=gt_dict["gt_importance"])
    return targets_dict

def carlalabel2gt_dict(carlalabel,
    class_names,
    calib_list,
    classes_to_exclude=[]) -> dict:
    assert carlalabel.current_frame == Frame.IMU
    if not (carlalabel.data is None or carlalabel.data == []):
        gt_boxes = carlalabel.bboxes3d
        wlh = gt_boxes[:, [1, 2, 0]]
        ry = gt_boxes[:, -1:]
        xyz_FIMU = np.zeros_like(wlh)
        for i, (obj, calib) in enumerate(zip(carlalabel.data, calib_list)):
            bcenter_FIMU = np.array([obj.x, obj.y, obj.z]).reshape(1, 3)
            xyz_FIMU[i, :] = bcenter_FIMU
        gt_boxes_FIMU = np.concatenate([xyz_FIMU, wlh, ry], axis=1)
        gt_names = carlalabel.bboxes_name
        gt_classes = np.array(
            [class_names.index(n) + 1 if n not in classes_to_exclude else -1
            for n in gt_names],
            dtype=np.int32)
        gt_importance = np.ones([len(carlalabel.data)], dtype=gt_boxes.dtype)
        gt_importance[gt_classes == -1] = 0
    else:
        gt_boxes_FIMU = np.array([])
        gt_classes = []
        gt_names = []
        gt_importance = []
    gt_dict = {
        "gt_boxes": gt_boxes_FIMU,
        "gt_classes": gt_classes,
        "gt_names": gt_names,
        "gt_importance": gt_importance
    }
    return gt_dict

def prep_pointcloud(input_dict,
                    training: bool,
                    augment_dict,
                    filter_label_dict,
                    voxelizer,
                    anchor_cache,
                    feature_map_size,
                    target_assigner,
                    db_sampler=None,
                    classes_to_exclude=[]):
    '''
    input_dict:
        {
            "lidar":{
                "points": pc_dict,
            },
            "metadata":{
                "tag": tag,
            },
            "calib": calib,
            "cam":{
                "img_path": img_path,
            },
            "imu":{
                "label": label
            }
        }
    '''
    tag = input_dict["metadata"]["tag"]
    gt_calib = input_dict["calib"]
    gt_label = input_dict["imu"]["label"]
    pc_dict_Flidar = input_dict["lidar"]["points"]
    pc_dict_FIMU = {k: gt_calib.lidar2imu(v[:,:3], key=f'Tr_imu_to_{k}')
                    for k, v in pc_dict_Flidar.items()}
    pc = get_pc_from_dict(pc_dict_FIMU)

    label = label_filtering(pc=pc,
                            calib=gt_calib,
                            label=gt_label,
                            filter_label_dict=filter_label_dict)
    label, calib, pc = db_sampling(pc=pc,
                                   gt_calib=gt_calib,
                                   gt_label=label,
                                   training=training,
                                   db_sampler=None)
    label, pc = augmenting(pc=pc,
                           calib=calib,
                           label=label,
                           training=training,
                           augment_dict=augment_dict)
    label = label_filtering(pc=pc,
                            calib=calib,
                            label=label,
                            filter_label_dict=filter_label_dict)
    vox_res = voxelizing(pc=pc,
                         voxelizer=voxelizer)
    anchor_res = anchor_creating(target_assigner=target_assigner,
                                 feature_map_size=feature_map_size,
                                 anchor_cache=anchor_cache)
    example = {
        'tag': tag,
        'voxels': vox_res["voxels"],
        'num_points': vox_res["num_points_per_voxel"],
        'coordinates': vox_res["coordinates"],
        'num_voxels': np.array([vox_res["voxels"].shape[0]], dtype=np.int64),
        'anchors' : anchor_res["anchors"]
        }
    if not training:
        return example
    targets_dict = target_assigning(
        target_assigner=target_assigner,
        label=label,
        calib=calib,
        anchor_res=anchor_res,
        classes_to_exclude=classes_to_exclude)
    example.update({
        'labels': targets_dict['labels'],
        'reg_targets': targets_dict['bbox_targets'],
        'importance': targets_dict['importance']
        })
    return example

def prep_info(info_pkl, valid_range, target_classes):
    '''
    preserve the item in <info> if any instance of
    <target_classes> exists in <valid_range>.
    @info_pkl: List
    @valid_range: [xmin, ymin, zmin, xmax, ymax, zmax]
    @target_classes: [str]
    '''
    def print_info(info_pkl):
        print(f"info file has {len(info_pkl)} items.")
        cls_acc_dict = {}
        for itm in info_pkl:
            label = itm["label"]
            if label.data is None:
                continue
            for obj in label.data:
                if obj.type in cls_acc_dict.keys():
                    cls_acc_dict[obj.type] += 1
                else:
                    cls_acc_dict[obj.type] = 1
        for k, v in cls_acc_dict.items():
            print(f"{k}: {v} instances.")
    print("before preprocessing info file:")
    print_info(info_pkl)
    target_info = []
    for itm in info_pkl:
        label = itm["label"]
        if label.data is None:
            continue
        label_ = filt_label_by_range(label, valid_range)
        label_ = filt_label_by_cls(label_, target_classes)
        if len(label_) > 0:
            target_info.append(itm)
    print("after preprocessing info file:")
    print_info(target_info)
    return target_info

if __name__ == "__main__":
    from incdet3.data.carladataset import CarlaDataset
    from incdet3.configs.dev_cfg import cfg
    from incdet3.builders import voxelizer_builder, target_assigner_builder
    from functools import partial
    from tqdm import tqdm
    dataset_cfg = cfg.TRAINDATA
    root_path = dataset_cfg["@root_path"]
    info_path = dataset_cfg["@info_path"]
    class_names = dataset_cfg["@class_names"]
    dataset = CarlaDataset(root_path=root_path,
        info_path=info_path,
        class_names=class_names,
        prep_func=None)
    prep_cfg = dataset_cfg["prep"]
    training = prep_cfg["@training"]
    augment_dict = prep_cfg["@augment_dict"]
    filter_label_dict = prep_cfg["@filter_label_dict"]
    voxelizer = voxelizer_builder.build(cfg.VOXELIZER)
    target_assigner = target_assigner_builder.build(cfg.TARGETASSIGNER)
    anchor_cache = anchor_creating(target_assigner,
        feature_map_size=[1, 200, 176],
        anchor_cache=None)
    prep_func = partial(prep_pointcloud,
                    training=training,
                    augment_dict=augment_dict,
                    filter_label_dict=filter_label_dict,
                    voxelizer=voxelizer,
                    target_assigner=target_assigner,
                    anchor_cache=anchor_cache,
                    feature_map_size=None)
    for i in tqdm(range(100)):
        input_dict = dataset.get_sensor_data(i)
        example = prep_func(input_dict)
        print(example.keys())
        import sys
        sys.exit("DEBUG")