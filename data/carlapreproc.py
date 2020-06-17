import numpy as np
from det3.dataloader.augmentor import CarlaAugmentor

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
                training: bool,
                single_lidar: (None, str),
                multi_lidar: (None, str)):
    '''
    sampling from db to augment label
    @pc: np.array
    @db_sampler: DBSampler
    @gt_calib: CarlaCalib
    @gt_label: CarlaLabel
    @single_lidar: None or str
    @multi_lidar: None or str
    @training: bool
    ->label: [CarlaObj]
    ->calib: [None/CarlaCalib]
    ->pc: np.array
        None if the according obj is the original copy from gt.
    '''
    if not training or db_sampler is None:
        return  gt_label, [gt_calib] * len(gt_label.data), pc
    sample_res = db_sampler.sample(gt_label=gt_label,
                                   gt_calib=gt_calib)
    for i in range(sample_res["num_obj"]):
        obj = sample_res["res_label"].data[i]
        obj_calib = sample_res["calib_list"][i]
        objpc = sample_res["objpc_list"][i]
        is_gt = sample_res["gt_mask"][i]
        if is_gt:
            continue
        if single_lidar is not None:
            objpc = objpc[single_lidar]
            mask = obj.get_pts_idx(pc[:, :3], obj_calib)
            mask = np.logical_not(mask)
            pc = pc[mask, :]
            pc = np.vstack([pc, objpc])
            np.random.shuffle(pc)
        elif multi_lidar == "merge":
            objpc = np.vstack([v for k, v in objpc.items()])
            mask = obj.get_pts_idx(pc[:, :3], obj_calib)
            mask = np.logical_not(mask)
            pc = pc[mask, :]
            pc = np.vstack([pc, objpc])
            np.random.shuffle(pc)
        elif multi_lidar in ["split", "split+merge"]:
            for velo_ in pc.keys():
                objpc_ = (np.vstack([v for k, v in objpc.items()])
                    if velo_ == "merge" else objpc[velo_])
                pc_ = pc[velo_].copy()
                mask = obj.get_pts_idx(pc_[:, :3], obj_calib)
                mask = np.logical_not(mask)
                pc_ = pc_[mask, :]
                pc_ = np.vstack([pc_, objpc_])
                np.random.shuffle(pc_)
                pc[velo_] = pc_
        else:
            raise NotImplementedError
    calib = [itm if itm is not None else gt_calib
             for itm in sample_res["calib_list"]]
    label = sample_res["res_label"]
    return label, calib, pc

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
                    filter_label_dict,
                    single_lidar,
                    multi_lidar):
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
    from mlod.utils import (filt_label_by_cls,
        filt_label_by_num_of_pts,
        filt_label_by_range)
    # by class
    if "keep_classes" in filter_label_dict.keys():
        keep_classes = filter_label_dict["keep_classes"]
        label = filt_label_by_cls(label, keep_classes)
    # by num of pts
    if "min_num_pts" in filter_label_dict.keys():
        min_num_pts = filter_label_dict["min_num_pts"]
        if single_lidar is not None or multi_lidar == "merge":
            label = filt_label_by_num_of_pts(pc, calib, label, min_num_pts)
        elif multi_lidar == "split" or multi_lidar == "split+merge":
            label = filt_label_by_num_of_pts(pc["merge"], calib, label, min_num_pts)
        else:
            raise NotImplementedError
    # by range
    if "label_range" in filter_label_dict.keys():
        valid_range = filter_label_dict["label_range"]
        label = filt_label_by_range(label, valid_range=valid_range)
    return label

def voxelizing(pc,
               voxelizer,
               max_voxels,
               single_lidar,
               multi_lidar):
    if single_lidar is not None or multi_lidar == "merge":
        vox_res = voxelizer.generate(pc, max_voxels)
    else:
        vox_res = dict()
        for velo in pc.keys():
            vox_res[velo] = voxelizer.generate(pc[velo], max_voxels)
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
        from det3.methods.second.ops.ops import rbbox2d_to_near_bbox
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
                     anchor_res):
    class_names = target_assigner.classes
    gt_dict = carlalabel2gt_dict(label, class_names, calib)
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

def carlalabel2gt_dict(carlalabel, class_names, calib_list) -> dict:
    from det3.dataloader.carladata import Frame
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
            [class_names.index(n) + 1 for n in gt_names],
            dtype=np.int32)
        gt_importance = np.ones([len(carlalabel.data)], dtype=gt_boxes.dtype)
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
                    db_sampler,
                    augment_dict,
                    filter_label_dict,
                    voxelizer,
                    max_voxels,
                    anchor_cache,
                    feature_map_size,
                    target_assigner,
                    calib_uct_param=0):
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
    pc_dict_Flidar = input_dict["lidar"]["points"]
    gt_label = input_dict["imu"]["label"]
    pc_dict_FIMU = {k: gt_calib.lidar2imu(v[:,:3], key=f'Tr_imu_to_{k}')
                    for k, v in pc_dict_Flidar.items()}
    tag = input_dict["metadata"]["tag"]
    pc = get_pc_from_dict(pc_dict_FIMU, single_lidar, multi_lidar)

    # label, calib, pc = db_sampling(pc=pc,
    #                                gt_calib=gt_calib,
    #                                gt_label=gt_label,
    #                                single_lidar=single_lidar,
    #                                multi_lidar=multi_lidar,
    #                                training=training,
    #                                db_sampler=db_sampler)
    label, pc = augmenting(pc=pc,
                           calib=calib,
                           label=label,
                           training=training,
                           augment_dict=augment_dict)
    label = label_filtering(pc=pc,
                            calib=calib,
                            label=label,
                            filter_label_dict=filter_label_dict,
                            single_lidar=single_lidar,
                            multi_lidar=multi_lidar,)
    vox_res = voxelizing(pc=pc,
                         voxelizer=voxelizer,
                         max_voxels=max_voxels,
                         single_lidar=single_lidar,
                         multi_lidar=multi_lidar)
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
        anchor_res=anchor_res)
    example.update({
        'labels': targets_dict['labels'],
        'reg_targets': targets_dict['bbox_targets'],
        'importance': targets_dict['importance']
        })
    return example

if __name__ == "__main__":
    from functools import partial
    from tqdm import tqdm
    from det3.utils.utils import load_pickle
    from incdet3.data.carladataset import CarlaDataset
    from det3.methods.second.builder.target_assigner_builder import build_multiclass as build_target_assigner
    from incdet3.builders.voxelizer_builder import build as build_voxelizer
    from incdet3.builders.target_assigner_builder import build_box_coder
    def deg2rad(deg):
        return deg / 180 * np.pi
    root_path = "/usr/app/data/CARLA/training"
    info_path = "/usr/app/data/CARLA/CARLA_infos_dev.pkl"
    class_names = ["Car"]
    training = True
    augment_dict = {
        "p_rot": 0.25,
        "dry_range": [deg2rad(-45), deg2rad(45)],
        "p_tr": 0.25,
        "dx_range": [-1, 1],
        "dy_range": [-1, 1],
        "dz_range": [-0.1, 0.1],
        "p_flip": 0.25,
        "p_keep": 0.25
    }
    voxelizer_cfg = {
        "type": "VoxelizerV1",
        "@voxel_size": [0.05, 0.05, 0.1],
        "@point_cloud_range": [-35.2, -40, -1.5, 35.2, 40, 2.6],
        "@max_num_points": 5,
        "@max_voxels": 100000
    }
    target_assigner_cfg = {
    "type": "TaskAssignerV1",
    "sample_size": 512,
    "assign_per_class": True,
    "classes": ["Car"],
    "class_settings_car": {
        "AnchorGenerator": {
            "type": "AnchorGeneratorBEV",
            "class_name": "Car",
            "anchor_ranges": [-35.2, -40, 0, 35.2, 40, 0],
            "sizes": [1.6, 3.9, 1.56], # wlh
            "rotations": [0, 1.57],
            "match_threshold": 0.6,
            "unmatch_threshold": 0.45,
        },
        "SimilarityCalculator": {
            "type": "NearestIoUSimilarity"
        }
    },
    "feature_map_sizes": None,
    "positive_fraction": None,
    }
    box_coder_cfg = {
        "type": "BoxCoderV1",
        "custom_ndim": 0,
    }
    filter_label_dict = {"keep_classes": ["Car"],
                         "min_num_pts": 5,
                         "label_range": [-35.2, -40, -1.5, 35.2, 40, 2.6]
                        }
    # build dataset
    dataset = CarlaDataset(root_path=root_path,
        info_path=info_path,
        class_names=class_names)
    # build voxlizer
    voxelizer = build_voxelizer(det3_root_dir="/usr/app/det3",
        voxelizer_cfg=voxelizer_cfg)
    # build box_coder
    box_coder = build_box_coder(det3_root_dir="/usr/app/det3",
        box_coder_cfg=box_coder_cfg)
    # build target_assigner
    target_assigner = build_target_assigner(target_assigner_cfg=target_assigner_cfg,
        box_coder=box_coder)
    # setup parameters
    grid_size = voxelizer.grid_size
    out_size_factor = 2
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]
    from det3.methods.second.ops.ops import rbbox2d_to_near_bbox
    ret = target_assigner.generate_anchors(feature_map_size)
    class_names = target_assigner.classes
    anchors_dict = target_assigner.generate_anchors_dict(feature_map_size)
    anchors_list = []
    for k, v in anchors_dict.items():
        anchors_list.append(v["anchors"])
    anchors = np.concatenate(anchors_list, axis=0)
    anchors = anchors.reshape([-1, target_assigner.box_ndim])
    assert np.allclose(anchors, ret["anchors"].reshape(-1, target_assigner.box_ndim))
    matched_thresholds = ret["matched_thresholds"]
    unmatched_thresholds = ret["unmatched_thresholds"]
    anchors_bv = rbbox2d_to_near_bbox(
        anchors[:, [0, 1, 3, 4, 6]])
    anchor_cache = {
        "anchors": anchors,
        "anchors_bv": anchors_bv,
        "matched_thresholds": matched_thresholds,
        "unmatched_thresholds": unmatched_thresholds,
        "anchors_dict": anchors_dict,
    }
    prep_func = partial(prep_pointcloud,
                        training=True,
                        db_sampler=db_sampler,
                        augment_dict=augment_dict,
                        filter_label_dict=filter_label_dict,
                        voxelizer=voxelizer,
                        max_voxels=None,
                        anchor_cache=anchor_cache,
                        feature_map_size=feature_map_size,
                        target_assigner=target_assigner)
    for i in tqdm(range(10)):
        input_dict = dataset.get_sensor_data(i)
        example = prep_func(input_dict)
        print(example.keys())
