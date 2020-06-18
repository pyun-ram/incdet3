from det3.utils.utils import is_param, proc_param
from det3.methods.second.core.voxelizer import VoxelizerV1

def build(voxelizer_cfg):
    '''
    @voxelizer_cfg: dict
        e.g.voxelizer_cfg = {
        "type": "VoxelizerV1",
        "@voxel_size": [0.05, 0.05, 0.1],
        "@point_cloud_range": [0, -40, -3, 70.4, 40, 1],
        "@max_num_points": 5,
        "@max_voxels": 20000
        }
    '''
    class_name = voxelizer_cfg["type"]
    if class_name == "VoxelizerV1":
        builder = VoxelizerV1
    else:
        raise NotImplementedError
    params = {proc_param(k):v
        for k, v in voxelizer_cfg.items() if is_param(k)}
    voxelizer = builder(**params)
    return voxelizer

if __name__ == "__main__":
    voxelizer_cfg = {
        "type": "VoxelizerV1",
        "@voxel_size": [0.05, 0.05, 0.1],
        "@point_cloud_range": [0, -40, -3, 70.4, 40, 1],
        "@max_num_points": 5,
        "@max_voxels": 20000
        }
    voxelizer = build(voxelizer_cfg)
