'''
 File Created: Sat Jun 20 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''

import unittest
import numpy as np
from incdet3.utils import deg2rad
from det3.ops import read_npy, write_npy
from incdet3.data.carladataset import CarlaDataset
from incdet3.builders import voxelizer_builder, target_assigner_builder
from incdet3.builders.dataloader_builder import build


class Test_carladata_general(unittest.TestCase):
    VOXELIZER_cfg = {
        "type": "VoxelizerV1",
        "@voxel_size": [0.05, 0.05, 0.1],
        "@point_cloud_range": [-35.2, -40, -1.5, 35.2, 40, 2.6],
        "@max_num_points": 5,
        "@max_voxels": 100000
    }
    TARGETASSIGNER_cfg = {
        "type": "TaskAssignerV1",
        "@classes": ["Car", "Pedestrian"],
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
                "@class_name": "Car",
                "@anchor_ranges": [-35.2, -40,0, 35.2, 40, 0], # TBD in modify_cfg(cfg)
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
                "@class_name": "Pedestrian",
                "@anchor_ranges": [-35.2, -40, 0, 35.2, 40, 0], # TBD in modify_cfg(cfg)
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
        "dataset": "carla", # carla
        "training": False, # set this to false to avoid shuffle
        "batch_size": 1,
        "num_workers": 1,
        "@root_path": "unit_tests/data/test_carladata",
        "@info_path": "unit_tests/data/test_carladata/CARLA_infos_train.pkl",
        "@class_names": ["Car", "Pedestrian"],
        "prep": {
            "@training": True, # set this to True to return targets
            "@augment_dict": None,
            "@filter_label_dict":
            {
                "keep_classes": ["Car", "Pedestrian"],
                "min_num_pts": -1,
                "label_range": [-35.2, -40, -1.5, 35.2, 40, 10],
                # [min_x, min_y, min_z, max_x, max_y, max_z] FIMU
            },
            "@feature_map_size": [1, 200, 176] # TBD
        }
    }
    def __init__(self, *args, **kwargs):
        super(Test_carladata_general, self).__init__(*args, **kwargs)
        from incdet3.data.carladataset import CarlaDataset
        from incdet3.builders import voxelizer_builder, target_assigner_builder
        from incdet3.builders.dataloader_builder import build

        data_cfg = Test_carladata_general.TRAINDATA_cfg
        voxelizer = voxelizer_builder.build(Test_carladata_general.VOXELIZER_cfg)
        target_assigner = target_assigner_builder.build(Test_carladata_general.TARGETASSIGNER_cfg)
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

    def test_voxels(self):
        pc = self.data["voxels"].reshape(-1, 3)
        mask = pc.sum(-1) != 0
        pc = pc[mask]
        gt = read_npy("unit_tests/results/test_carladata_general.npy")
        self.assertTrue(np.array_equal(gt, pc))
        voxel_range = self.VOXELIZER_cfg["@point_cloud_range"]
        voxel_res = self.VOXELIZER_cfg["@voxel_size"]

        max_x_coord = int((voxel_range[3]-voxel_range[0]) / voxel_res[0])
        max_y_coord = int((voxel_range[4]-voxel_range[1]) / voxel_res[1])
        max_z_coord = int((voxel_range[5]-voxel_range[2]) / voxel_res[2])

        coord = self.data["coordinates"]
        max_zyx = coord.max(axis=0)[1:]
        self.assertTrue(max_zyx[0] <= max_z_coord)
        self.assertTrue(max_zyx[1] <= max_y_coord)
        self.assertTrue(max_zyx[2] <= max_x_coord)

    def test_anchors(self):
        data = self.data
        anchors = data["anchors"].reshape(1, 4, 200, 176, 7)
        self.assertTrue(np.allclose(anchors[:, :, 0, 0, :3],
            np.array([-35.2, -40, 0])))
        self.assertTrue(np.allclose(anchors[:, :, 0, 0, 3:],
            np.array([[1.6, 3.9, 1.56, 0],
                      [1.6, 3.9, 1.56, 1.57],
                      [0.6, 0.8, 1.73, 0],
                      [0.6, 0.8, 1.73, 1.57]
                     ])))
        self.assertTrue(np.allclose(anchors[:, :, 199, 175, :3],
            np.array([35.2, 40, 0])))
        self.assertTrue(np.allclose(anchors[:, :, 0, 0, 3:],
            np.array([[1.6, 3.9, 1.56, 0],
                      [1.6, 3.9, 1.56, 1.57],
                      [0.6, 0.8, 1.73, 0],
                      [0.6, 0.8, 1.73, 1.57]
                     ])))

    def test_targets(self):
        def limit_period_torch(val, offset=0.5, period=np.pi):
            return val - torch.floor(val / period + offset) * period
        import torch
        import torch.nn as nn
        from det3.methods.second.ops.torch_ops import rotate_nms
        data = self.data
        label = data["metadata"][0]["label"]
        cls_pred = torch.from_numpy(data["labels"]).cuda().float()
        cls_pred *= (cls_pred >= 0).float()
        cls_pred = cls_pred.long()
        cls_pred = nn.functional.one_hot(cls_pred, num_classes=2+1)
        cls_pred = cls_pred[..., 1:]
        anchors = torch.from_numpy(data["anchors"]).cuda().float()
        box_pred = torch.from_numpy(data["reg_targets"]).cuda().float()
        # pred_dict = {
        #     "cls_preds": cls_pred * 10,
        #     "box_preds": box_pred
        # }
        # box_coder = self.box_coder
        # from det3.ops import write_pkl
        # write_pkl({"pred_dict": pred_dict, "box_coder": box_coder}, "test_model_est.pkl")
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
            final_scores = scores
            final_labels = label_preds
            predictions_dict = {
                "box3d_lidar": final_box_preds,
                "scores": final_scores,
                "label_preds": label_preds,
            }
            from det3.dataloader.carladata import CarlaObj, CarlaLabel
            label_gt = CarlaLabel()
            label_est = CarlaLabel()
            for obj_str in label.split("\n"):
                try:
                    obj = CarlaObj(obj_str)
                except:
                    pass
                if -35.2 < obj.x < 35.2 and -40 < obj.y < 40:
                    label_gt.add_obj(obj)
            for box3d_lidar, label_preds, score in zip(
                    predictions_dict["box3d_lidar"],
                    predictions_dict["label_preds"],
                    predictions_dict["scores"]):
                obj = CarlaObj()
                obj.type = "Car" if label_preds == 0 else "Pedestrian"
                obj.x, obj.y, obj.z, obj.w, obj.l, obj.h, obj.ry = box3d_lidar.cpu().numpy().flatten()
                obj.truncated = 0
                obj.occluded = 0
                obj.alpha = 0
                obj.bbox_l = 0
                obj.bbox_t = 0
                obj.bbox_r = 0
                obj.bbox_b = 0
                label_est.add_obj(obj)
            self.assertTrue(label_gt.equal(label_est, acc_cls=["Car", "Pedestrian"], rtol=1e-2))

class Test_exclude_classes(unittest.TestCase):
    VOXELIZER_cfg = {
        "type": "VoxelizerV1",
        "@voxel_size": [0.05, 0.05, 0.1],
        "@point_cloud_range": [-35.2, -40, -1.5, 35.2, 40, 2.6],
        "@max_num_points": 5,
        "@max_voxels": 100000
    }
    TARGETASSIGNER_cfg = {
        "type": "TaskAssignerV1",
        "@classes": ["Car", "Pedestrian"],
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
                "@class_name": "Car",
                "@anchor_ranges": [-35.2, -40,0, 35.2, 40, 0], # TBD in modify_cfg(cfg)
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
                "@class_name": "Pedestrian",
                "@anchor_ranges": [-35.2, -40, 0, 35.2, 40, 0], # TBD in modify_cfg(cfg)
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
        "dataset": "carla", # carla
        "training": False, # set this to false to avoid shuffle
        "batch_size": 1,
        "num_workers": 1,
        "@root_path": "unit_tests/data/test_carladata",
        "@info_path": "unit_tests/data/test_carladata/CARLA_infos_train.pkl",
        "@class_names": ["Car", "Pedestrian"],
        "prep": {
            "@training": True, # set this to True to return targets
            "@augment_dict": None,
            "@filter_label_dict":
            {
                "keep_classes": ["Car", "Pedestrian"],
                "min_num_pts": -1,
                "label_range": [-35.2, -40, -1.5, 35.2, 40, 10],
                # [min_x, min_y, min_z, max_x, max_y, max_z] FIMU
            },
            "@feature_map_size": [1, 200, 176], # TBD
            "@classes_to_exclude": []
        }
    }
    def __init__(self, *args, **kwargs):
        super(Test_exclude_classes, self).__init__(*args, **kwargs)

    def test_exclude_classes1(self):
        '''
        not exclude
        '''
        data_cfg = Test_exclude_classes.TRAINDATA_cfg
        data_cfg["prep"]["@classes_to_exclude"] = []
        voxelizer = voxelizer_builder.build(Test_exclude_classes.VOXELIZER_cfg)
        target_assigner = target_assigner_builder.build(Test_exclude_classes.TARGETASSIGNER_cfg)
        dataloader = build(data_cfg,
            ext_dict={
                "voxelizer": voxelizer,
                "target_assigner": target_assigner,
                "feature_map_size": [1, 200, 176]
            })
        has1 = False
        has2 = False
        for data in dataloader:
            labels =data["labels"]
            labels1 = labels[labels == 1]
            labels2 = labels[labels == 2]
            if labels1.shape[0] > 0:
                has1 = True
            if labels2.shape[0] > 0:
                has2 = True
        self.assertTrue(has1)
        self.assertTrue(has2)

    def test_exclude_classes2(self):
        '''
        exclude Car
        '''
        data_cfg = Test_exclude_classes.TRAINDATA_cfg
        data_cfg["prep"]["@classes_to_exclude"] = ["Car"]
        voxelizer = voxelizer_builder.build(Test_exclude_classes.VOXELIZER_cfg)
        target_assigner = target_assigner_builder.build(Test_exclude_classes.TARGETASSIGNER_cfg)
        dataloader = build(data_cfg,
            ext_dict={
                "voxelizer": voxelizer,
                "target_assigner": target_assigner,
                "feature_map_size": [1, 200, 176]
            })
        has1 = False
        has2 = False
        for data in dataloader:
            labels =data["labels"]
            labels1 = labels[labels == 1]
            labels2 = labels[labels == 2]
            if labels1.shape[0] > 0:
                has1 = True
            if labels2.shape[0] > 0:
                has2 = True
        self.assertFalse(has1)
        self.assertTrue(has2)

    def test_exclude_classes3(self):
        '''
        exclude Pedestrian
        '''
        data_cfg = Test_exclude_classes.TRAINDATA_cfg
        data_cfg["prep"]["@classes_to_exclude"] = ["Pedestrian"]
        voxelizer = voxelizer_builder.build(Test_exclude_classes.VOXELIZER_cfg)
        target_assigner = target_assigner_builder.build(Test_exclude_classes.TARGETASSIGNER_cfg)
        dataloader = build(data_cfg,
            ext_dict={
                "voxelizer": voxelizer,
                "target_assigner": target_assigner,
                "feature_map_size": [1, 200, 176]
            })
        has1 = False
        has2 = False
        for data in dataloader:
            labels =data["labels"]
            labels1 = labels[labels == 1]
            labels2 = labels[labels == 2]
            if labels1.shape[0] > 0:
                has1 = True
            if labels2.shape[0] > 0:
                has2 = True
        self.assertTrue(has1)
        self.assertFalse(has2)

    def test_exclude_classes4(self):
        '''
        exclude Car Pedestrian
        '''
        data_cfg = Test_exclude_classes.TRAINDATA_cfg
        data_cfg["prep"]["@classes_to_exclude"] = ["Car", "Pedestrian"]
        voxelizer = voxelizer_builder.build(Test_exclude_classes.VOXELIZER_cfg)
        target_assigner = target_assigner_builder.build(Test_exclude_classes.TARGETASSIGNER_cfg)
        dataloader = build(data_cfg,
            ext_dict={
                "voxelizer": voxelizer,
                "target_assigner": target_assigner,
                "feature_map_size": [1, 200, 176]
            })
        has1 = False
        has2 = False
        for data in dataloader:
            labels =data["labels"]
            labels1 = labels[labels == 1]
            labels2 = labels[labels == 2]
            if labels1.shape[0] > 0:
                has1 = True
            if labels2.shape[0] > 0:
                has2 = True
        self.assertFalse(has1)
        self.assertFalse(has2)

if __name__ == "__main__":
    unittest.main()