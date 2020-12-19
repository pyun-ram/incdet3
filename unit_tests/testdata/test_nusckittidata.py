'''
 File Created: Tue Aug 11 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''
import os
import unittest
import numpy as np
from det3.ops import write_npy, read_npy
from det3.dataloader.kittidata import KittiObj, KittiLabel, KittiCalib
from incdet3.data.nusckitti_dataset import NuscenesKittiDataset
from incdet3.builders import voxelizer_builder, target_assigner_builder
from incdet3.builders.dataloader_builder import build

class Test_nusckittidata_general(unittest.TestCase):
    VOXELIZER_cfg = {
        "type": "VoxelizerV1",
        "@voxel_size": [0.05, 0.05, 1],
        "@point_cloud_range": [-100, -100, -10, 100.0, 100.0, 10],
        "@max_num_points": 5,
        "@max_voxels": 20000
    }
    TARGETASSIGNER_cfg = {
        "type": "TaskAssignerV1",
        "@classes": ["car", "pedestrian"],
        "@feature_map_sizes": None,
        "@positive_fraction": None,
        "@sample_size": 512,
        "@assign_per_class": True,
        "box_coder": {
            "type": "BoxCoderV1",
            "@custom_ndim": 0
        },
        "class_settings_car": {
            "AnchorGenerator": {
                "type": "AnchorGeneratorBEV",
                "@class_name": "car",
                "@anchor_ranges": [-100, -100, 0, 100, 100, 0], # TBD in modify_cfg(cfg)
                "@sizes": [1.6, 3.9, 1.56], # wlh
                "@rotations": [0, 1.57],
                "@match_threshold": 0.6,
                "@unmatch_threshold": 0.45,
            },
            "SimilarityCalculator": {
                "type": "NearestIoUSimilarity"
            }
        },
        "class_settings_pedestrian": {
            "AnchorGenerator": {
                "type": "AnchorGeneratorBEV",
                "@class_name": "pedestrian",
                "@anchor_ranges": [-100, -100, 0, 100, 100, 0], # TBD in modify_cfg(cfg)
                "@sizes": [0.6, 0.8, 1.73], # wlh
                "@rotations": [0, 1.57],
                "@match_threshold": 0.6,
                "@unmatch_threshold": 0.45,
            },
            "SimilarityCalculator": {
                "type": "NearestIoUSimilarity"
            }
        },
    }
    TRAINDATA_cfg = {
        "dataset": "nusc-kitti", # carla
        "training": False, # set this to false to avoid shuffle
        "batch_size": 1,
        "num_workers": 1,
        "@root_path": "/usr/app/data/nusc-kitti/training",
        "@info_path": "/usr/app/data/nusc-kitti/KITTI_infos_train.pkl",
        "@class_names": ["car", "pedestrian"],
        "prep": {
            "@training": True, # set this to True to return targets
            "@augment_dict": None,
            "@filter_label_dict":
            {
                "keep_classes": ["car", "pedestrian"],
                "min_num_pts": -1,
                "label_range": [-100, -100, -10, 100, 100, 10],
                # [min_x, min_y, min_z, max_x, max_y, max_z] FIMU
            },
            "@feature_map_size": [1, 200, 176] # TBD
        }
    }
    def __init__(self, *args, **kwargs):
        super(Test_nusckittidata_general, self).__init__(*args, **kwargs)
        if not os.path.exists("/usr/app/data/nusc-kitti"):
            print("The data of nusc-kitti does not exist.")
            print("You can convert the nusc v1.0-mini to kitti format and put into /usr/app/data/nusc-kitti for this unit-test.")
            print("You can also download the data from https://pyun-data-hk.s3.ap-east-1.amazonaws.com/IncDet3/Data/20201219-nusc_kitti.zip")
            raise RuntimeError
        data_cfg = Test_nusckittidata_general.TRAINDATA_cfg
        voxelizer = voxelizer_builder.build(Test_nusckittidata_general.VOXELIZER_cfg)
        target_assigner = target_assigner_builder.build(Test_nusckittidata_general.TARGETASSIGNER_cfg)
        dataloader = build(data_cfg,
            ext_dict={
                "voxelizer": voxelizer,
                "target_assigner": target_assigner,
                "feature_map_size": [1, 200, 176]
            })
        self.dataloader = dataloader
        self.data = None
        self.box_coder = target_assigner.box_coder
        for i, data in enumerate(self.dataloader):
            if i == 1:
                self.data = data
                break
            else:
                continue
    def test_evaluation(self):
        import torch
        import torch.nn as nn
        from det3.methods.second.ops.torch_ops import rotate_nms
        from det3.dataloader.kittidata import KittiCalib
        detections = []
        for i, data in enumerate(self.dataloader):
            print(i)
            label = data["metadata"][0]["label"]
            tag = data["metadata"][0]["tag"]
            cls_pred = torch.from_numpy(data["labels"]).cuda().float()
            cls_pred *= (cls_pred >= 0).float()
            cls_pred = cls_pred.long()
            cls_pred = nn.functional.one_hot(cls_pred, num_classes=2+1)
            cls_pred = cls_pred[..., 1:]
            anchors = torch.from_numpy(data["anchors"]).cuda().float()
            box_pred = torch.from_numpy(data["reg_targets"]).cuda().float()
            box_pred = self.box_coder.decode(box_pred, anchors)
            for box_preds, cls_preds in zip(box_pred, cls_pred):
                box_preds = box_preds.float()
                cls_preds = cls_preds.float()
                total_scores = cls_preds
                nms_func = rotate_nms
                top_scores, top_labels = torch.max(
                    total_scores, dim=-1)
                top_scores_keep = top_scores >= 0.5
                top_scores = top_scores.masked_select(top_scores_keep)
                box_preds = box_preds[top_scores_keep]
                top_labels = top_labels[top_scores_keep]
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                selected = nms_func(
                    boxes_for_nms,
                    top_scores,
                    pre_max_size=1000,
                    post_max_size=1000,
                    iou_threshold=0.3,
                )
                selected_boxes = box_preds[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                final_box_preds = box_preds
                final_box_preds += 0.01
                final_scores = scores
                final_labels = label_preds
                predictions_dict = {
                    "box3d_lidar": final_box_preds,
                    "scores": final_scores,
                    "label_preds": final_labels,
                    "meta": data["metadata"]
                }
                detections.append(predictions_dict)
        val_ap_dict = self.dataloader.dataset.evaluation(detections=detections,
            label_dir="/usr/app/data/nusc-kitti/training/label_2",
            output_dir="/tmp/")
        self.assertTrue(val_ap_dict['detail']['car']["3d@0.50"][0] > 90)
        self.assertTrue(val_ap_dict['detail']['pedestrian']["3d@0.25"][0] > 90)
        print(val_ap_dict["result"])

if __name__ == "__main__":
    unittest.main()