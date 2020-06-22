'''
The dataset should support class selection.
The gt_database temporally is not needed, but will be added for including exemplars.
Otherpart can be inherited from mlod project.
'''
from det3.utils.utils import load_pickle
from det3.ops import read_npy
from incdet3.data.carlaeval import get_eval_result

class CarlaDataset:
    def __init__(self,
                 root_path,
                 info_path,
                 class_names,
                 prep_func=None):
        self._carla_infos = load_pickle(info_path)
        self._root_path = root_path
        self._class_names = class_names
        # TODO: Do we need the _classes_to_exclude?
        self._prep_func = prep_func

    def __len__(self):
        return len(self._carla_infos)

    def __getitem__(self, idx):
        input_dict = self.get_sensor_data(idx)
        example = self._prep_func(input_dict)
        example["metadata"] = input_dict["metadata"]
        example["metadata"]["label"] = str(input_dict["imu"]["label"])
        return example

    def get_sensor_data(self, query):
        idx = query
        info = self._carla_infos[idx]
        calib = info["calib"]
        label = info["label"]
        tag = info["tag"]
        img_path = info["img_path"]
        pc_dict = {velo: read_npy(pc_path) for velo, pc_path in info["pc_paths"].items()}
        res = {
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
        return res

    def evaluation(self, detections):
        tags = [itm["tag"] for itm in self._carla_infos]
        gts = [itm["label"] for itm in self._carla_infos]
        gt_annos = self._convert_carlalabel_to_annos(gts)
        dt_annos = self._convert_detection_to_annos(detections)
        for tag, dt_anno in zip(tags, dt_annos):
            assert tag == dt_anno["metadata"]["tag"]
        z_axis = 2 # [x, y, z] We use z as regular "z" axis
        z_center = 0.0 # CARLA IMU 3D box's center is [0.5, 0.5, 1]
        result_dict = get_eval_result(gt_annos[:len(dt_annos)],
            dt_annos,
            self.class_names,
            difficultys=[0, 1, 2],
            z_axis=z_axis,
            z_center=z_center)
        return {
            "results": {
                "carla": result_dict["result"],
            },
            "detail": {
                "eval.carla": {
                    "carla": result_dict["detail"],
                }
            },
        }

    def _convert_carlalabel_to_annos(self, gts):
        '''
        @gts: [CarlaLable]
        '''
        annos = []
        for label in gts:
            if len(label) != 0:
                final_box_preds = label.bboxes3d
                label_preds = label.bboxes_name
                xyz_FIMU = final_box_preds[:, 3:6]
                wlh = final_box_preds[:, [1, 2, 0]]
                ry = final_box_preds[:, 6]
            anno = {'name': [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'bbox': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': []}
            num_example = 0
            box3d_lidar = (final_box_preds if len(label) != 0
                else np.zeros([0, 7]))
            for j in range(box3d_lidar.shape[0]):
                anno["bbox"].append(np.zeros((1, 4)))
                anno["alpha"].append(0.0)
                anno["dimensions"].append(wlh[j, :])
                anno["location"].append(xyz_FIMU[j, :])
                anno["rotation_y"].append(ry[j])
                anno["name"].append(label_preds[j])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["score"].append(0)
                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                empty_anno = {'name': np.array([]),
                              'truncated': np.array([]),
                              'occluded': np.array([]),
                              'alpha': np.array([]),
                              'bbox': np.zeros([0, 4]),
                              'dimensions': np.zeros([0, 3]),
                              'location': np.zeros([0, 3]),
                              'rotation_y': np.array([]),
                              'score': np.array([])}
                annos.append(empty_anno)
            num_example = annos[-1]["name"].shape[0]
        return annos

    def _convert_detection_to_annos(self, detections):
        '''
        @detections: [detection], detection is the predict result of SECOND torch model
        {box3d_lidar: [N, 7] 3d box.
         scores: [N]
         label_preds: [N]
         metadata: meta-data which contains dataset-specific information.
             for carla, it contains tag.
        }
        '''
        class_names = self.class_names
        annos = []
        for det in detections:
            if isinstance(det["box3d_lidar"], np.ndarray):
                final_box_preds = det["box3d_lidar"] # xyz_Flidar, wlh, ry
                label_preds = det["label_preds"]
                scores = det["scores"]
            else:
                final_box_preds = det["box3d_lidar"].cpu().numpy() # xyz_Flidar, wlh, ry
                label_preds = det["label_preds"].cpu().numpy()
                scores = det["scores"].cpu().numpy()
            if final_box_preds.shape[0] != 0:
                # Note: we comment out the next line, since the det["box3d_lidar"] is bottom center
                # final_box_preds[:, 2] -= final_box_preds[:, 5] / 2
                # Note: we comment out the next line, since we want to evaluate under IMU frame.
                # box3d_camera = box_np_ops.box_lidar_to_camera(
                #     final_box_preds, rect, Trv2c)
                xyz_FIMU = final_box_preds[:, :3]
                wlh = final_box_preds[:, 3:6]
                ry = final_box_preds[:, 6]
                # Note: we comment out the next few lines, since we do not need bbox2d evaluation.
                # camera_box_origin = [0.5, 1.0, 0.5]
                # box_corners = box_np_ops.center_to_corner_box3d(
                #     locs, dims, angles, camera_box_origin, axis=1)
                # box_corners_in_image = box_np_ops.project_to_image(
                #     box_corners, P2)
                # # box_corners_in_image: [N, 8, 2]
                # minxy = np.min(box_corners_in_image, axis=1)
                # maxxy = np.max(box_corners_in_image, axis=1)
                # bbox = np.concatenate([minxy, maxxy], axis=1)
            anno = {'name': [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'bbox': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': []}
            num_example = 0
            box3d_lidar = final_box_preds
            for j in range(box3d_lidar.shape[0]):
                anno["bbox"].append(np.zeros((1, 4)))
                anno["alpha"].append(0.0)
                anno["dimensions"].append(wlh[j, :])
                anno["location"].append(xyz_FIMU[j, :])
                anno["rotation_y"].append(ry[j])
                anno["name"].append(class_names[int(label_preds[j])])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["score"].append(scores[j])
                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                empty_anno = {'name': np.array([]),
                              'truncated': np.array([]),
                              'occluded': np.array([]),
                              'alpha': np.array([]),
                              'bbox': np.zeros([0, 4]),
                              'dimensions': np.zeros([0, 3]),
                              'location': np.zeros([0, 3]),
                              'rotation_y': np.array([]),
                              'score': np.array([])}
                annos.append(empty_anno)
            num_example = annos[-1]["name"].shape[0]
            annos[-1]["metadata"] = det["metadata"]
        return annos
