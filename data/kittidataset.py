import os
import shutil
import numpy as np
from det3.ops import read_bin, read_pkl
from det3.utils.utils import write_str_to_file
from det3.dataloader.kittidata import KittiLabel, KittiObj
from det3.methods.second.data import kitti_common as kitti
from second.utils.eval import get_coco_eval_result, get_official_eval_result

class KittiDataset:
    def __init__(self,
                 root_path,
                 info_path,
                 class_names,
                 prep_func=None,
                 prep_info_func=lambda x:x):
        self._kitti_infos = prep_info_func(read_pkl(info_path))
        self._root_path = root_path
        self._class_names = class_names
        self._prep_func = prep_func
    
    def __len__(self):
        return len(self._kitti_infos)

    @property
    def class_names(self):
        return self._class_names

    def __getitem__(self, idx):
        input_dict = self.get_sensor_data(idx)
        example = self._prep_func(input_dict)
        example["metadata"] = input_dict["metadata"]
        example["metadata"]["label"] = str(input_dict["cam"]["label"])
        return example

    def get_sensor_data(self, query):
        idx = query
        info = self._kitti_infos[idx]
        calib = info["calib"]
        label = info["label"]
        tag = info["tag"]
        pc_reduced = read_bin(info["reduced_pc_path"], dtype=np.float32).reshape(-1, 4)
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

    def evaluation(self, detections, label_dir, output_dir):
        tags = [itm["tag"] for itm in self._kitti_infos]
        calibs = [itm["calib"] for itm in self._kitti_infos]
        det_path = os.path.join(output_dir, "data")
        assert len(tags) == len(detections) == len(calibs)
        self.save_detections(detections, tags, calibs, det_path)
        assert len(detections) > 50
        dt_annos = kitti.get_label_annos(det_path)
        gt_path = os.path.join(label_dir)
        val_image_ids = os.listdir(det_path)
        val_image_ids = [int(itm.split(".")[0]) for itm in val_image_ids]
        val_image_ids.sort()
        gt_annos = kitti.get_label_annos(gt_path, val_image_ids)
        cls_to_idx = {"Car": 0, "Pedestrian": 1, "Cyclist": 2, "Van": 3}
        current_classes = [cls_to_idx[itm] for itm in self._class_names]
        val_ap_dict = get_official_eval_result(gt_annos, dt_annos, current_classes)
        return val_ap_dict

    def save_detections(self, detections, tags, calibs, save_dir):
        res_dir = save_dir
        if os.path.isdir(res_dir):
            shutil.rmtree(res_dir, ignore_errors=True)
        os.makedirs(res_dir)
        for det, tag, calib in zip(detections, tags, calibs):
            label = KittiLabel()
            label.current_frame = "Cam2"
            final_box_preds = det["box3d_lidar"].detach().cpu().numpy()
            label_preds = det["label_preds"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()
            for i in range(final_box_preds.shape[0]):
                obj_np = final_box_preds[i, :]
                bcenter_Flidar = obj_np[:3].reshape(1, 3)
                bcenter_Fcam = calib.lidar2leftcam(bcenter_Flidar)
                wlh = obj_np[3:6]
                ry = obj_np[-1]
                obj = KittiObj()
                obj.type = self._class_names[int(label_preds[i])]
                obj.score = scores[i]
                obj.x, obj.y, obj.z = bcenter_Fcam.flatten()
                obj.w, obj.l, obj.h = wlh.flatten()
                obj.ry = ry
                obj.from_corners(calib, obj.get_bbox3dcorners(), obj.type, obj.score)
                obj.truncated = 0
                obj.occluded = 0
                obj.alpha = -np.arctan2(-bcenter_Flidar[0, 1], bcenter_Flidar[0, 0]) + ry
                label.add_obj(obj)
            # save label
            write_str_to_file(str(label), os.path.join(res_dir, f"{tag}.txt"))
