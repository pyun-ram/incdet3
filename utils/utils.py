import numpy as np
from det3.dataloader.carladata import CarlaLabel
from det3.dataloader.waymodata import WaymoLabel
from det3.utils.utils import istype

def filt_label_by_cls(label, keep_classes):
    if istype(label, "CarlaLabel"):
        res = CarlaLabel()
    elif istype(label, "WaymoLabel"):
        res = WaymoLabel()
    else:
        raise NotImplementedError
    res.data = [] # It is really important,
    # otherwise the "for obj in label.data" of the next stage will raise a problem.
    for obj in label.data:
        if obj.type in keep_classes:
            res.add_obj(obj)
    res.current_frame = label.current_frame
    return res

def filt_label_by_num_of_pts(pc, calib, label, min_num_pts):
    '''
    @pc: np.ndarray
    '''
    if min_num_pts < 0:
        return label
    is_iter = isinstance(calib, list)
    if istype(label, "CarlaLabel"):
        res = CarlaLabel()
    elif istype(label, "WaymoLabel"):
        res = WaymoLabel()
    else:
        raise NotImplementedError
    res.data = [] # It is really important,
    # otherwise the "for obj in label.data" of the next stage will raise a problem.
    assert not isinstance(pc, dict)
    if not is_iter:
        for obj in label.data:
            num_pts = obj.get_pts_idx(pc, calib).sum()
            if num_pts > min_num_pts:
                res.add_obj(obj)
    else:
        for obj, calib_ in zip(label.data, calib):
            num_pts = obj.get_pts_idx(pc, calib_).sum()
            if num_pts > min_num_pts:
                res.add_obj(obj)
    res.current_frame = label.current_frame
    return res

def filt_label_by_range(label, valid_range):
    '''
    @label: CarlaLabel
    @valid_range: [min_x, min_y, min_z, max_x, max_y, max_z] FIMU
    '''
    min_x, min_y, min_z, max_x, max_y, max_z = valid_range
    if istype(label, "CarlaLabel"):
        res = CarlaLabel()
    elif istype(label, "WaymoLabel"):
        res = WaymoLabel()
    else:
        raise NotImplementedError
    res.data = [] # It is really important,
    # otherwise the "for obj in label.data" of the next stage will raise a problem.
    for obj in label.data:
        imu_pt = np.array([obj.x, obj.y+obj.h/2.0, obj.z]).reshape(1, 3)
        if (min_x <= imu_pt[0, 0] <= max_x and
            min_y <= imu_pt[0, 1] <= max_y and
            min_z <= imu_pt[0, 2] <= max_z):
            res.add_obj(obj)
    res.current_frame = label.current_frame
    return res

def deg2rad(deg):
    return deg / 180 * np.pi

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

nusc_cls2color = {
    "car": "yellow",
    "truck": "brown",
    "bus": "purple",
    "trailer": "white",
    "construction_vehicle": "blue",
    "pedestrian": "green",
    "barrier": "pink",
    "traffic_cone": "red",
    "bicycle": "orange",
    "motorcycle": "cyan",
    "default": "magenta",
}