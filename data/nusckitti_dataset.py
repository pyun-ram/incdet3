import os
from .kittidataset import KittiDataset
from det3.methods.second.data import kitti_common as kitti
from second.utils.eval import get_official_eval_result_nusckitti

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

    def evaluation(self, detections, label_dir, output_dir):
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
        cls_to_idx = {"car": 0, "pedestrian": 1, "traffic_cone": 2, "truck": 3, "construction_vehicle": 4, "barrier": 5,
        "trailer": 6, "bus": 7, "motorcycle": 8, "bicycle": 9}
        current_classes = [cls_to_idx[itm] for itm in self._class_names]
        val_ap_dict = get_official_eval_result_nusckitti(gt_annos, dt_annos, current_classes)
        return val_ap_dict
