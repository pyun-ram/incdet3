import os
import numpy as np
from typing import List, Dict, Tuple
from det3.methods.second.data import kitti_common as kitti
from second.utils.eval import get_official_eval_result_nusckitti
from .kittidataset import KittiDataset
from det3.ops import read_bin

class NuscenesKittiDataset(KittiDataset):
    def __init__(self,
                 root_path,
                 info_path,
                 class_names,
                 prep_func=None,
                 prep_info_func=lambda x:x):
        super(NuscenesKittiDataset, self).__init__(
            root_path,
            info_path,
            class_names,
            prep_func,
            prep_info_func
        )

    def get_sensor_data(self, query):
        idx = query
        info = self._kitti_infos[idx]
        calib = info["calib"]
        label = info["label"]
        tag = info["tag"]
        pc_reduced = read_bin(info["reduced_pc_path"], dtype=np.float32).reshape(-1, 5)
        res = {
            "lidar": {
                "points": pc_reduced,
            },
            "metadata": {
                "tag": tag
            },
            "calib": calib,
            "cam": {
                "label": label
            }
        }
        return res

    def evaluation(self, detections, label_dir, output_dir, x_range=None, y_range=None):
        tags = [itm["tag"] for itm in self._kitti_infos]
        calibs = [itm["calib"] for itm in self._kitti_infos]
        det_path = os.path.join(output_dir, "data")
        assert len(tags) == len(detections) == len(calibs)
        self.save_detections(detections, tags, calibs, det_path)
        assert len(detections) > 50
        dt_annos = kitti.get_label_annos_nusckitti(det_path)
        gt_path = os.path.join(label_dir)
        val_image_ids = os.listdir(det_path)
        val_image_ids = [itm.split(".")[0] for itm in val_image_ids]
        val_image_ids.sort()
        gt_annos = kitti.get_label_annos_nusckitti(gt_path, val_image_ids)
        if x_range is not None and y_range is not None:
            gt_annos = filter_annos_according_to_location(
                calibs,
                gt_annos,
                x_range=x_range,
                y_range=y_range)
        cls_to_idx = {"car": 0, "pedestrian": 1, "traffic_cone": 2, "truck": 3, "construction_vehicle": 4, "barrier": 5,
        "trailer": 6, "bus": 7, "motorcycle": 8, "bicycle": 9}
        current_classes = [cls_to_idx[itm] for itm in self._class_names]
        val_ap_dict = get_official_eval_result_nusckitti(gt_annos, dt_annos, current_classes)
        return val_ap_dict

def filter_annos_according_to_location(
    calibs: List,
    gt_annos: List[Dict],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float]):
    new_gt_annos = []
    for calib, gt_anno in zip(calibs, gt_annos):
        new_gt_anno = {k: None for k in gt_anno.keys()}
        save_list = []
        locs = gt_anno["location"]
        for i, loc in enumerate(locs):
            bcenter_Fcam = loc.reshape(1, -1)
            bcenter_Flidar = calib.leftcam2lidar(bcenter_Fcam)
            bcenterx_Flidar, bcentery_Flidar, _ = bcenter_Flidar.reshape(-1)
            if (x_range[0] <= bcenterx_Flidar <= x_range[1]) and (y_range[0] <= bcentery_Flidar <= y_range[1]):
                save_list.append(i)
        for k in new_gt_anno.keys():
            new_gt_anno[k] = gt_anno[k][save_list]
        new_gt_annos.append(new_gt_anno)
        print(f"filtering {len(save_list)}/{gt_anno['location'].shape[0]}")
    return new_gt_annos
