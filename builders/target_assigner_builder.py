from det3.utils.utils import is_param, proc_param
from det3.methods.second.core.box_coder import BoxCoderV1
from det3.methods.second.core.anchor_generator import AnchorGeneratorBEV
from det3.methods.second.core.target_assigner import TaskAssignerV1
from det3.methods.second.core.similarity_calculator import NearestIoUSimilarity

def build_box_coder(box_coder_cfg):
    class_name = box_coder_cfg["type"]
    if class_name == "BoxCoderV1":
        builder = BoxCoderV1
    else:
        raise NotImplementedError
    box_coder_params = {proc_param(k): v
        for k, v in box_coder_cfg.items() if is_param(k)}
    return builder(**box_coder_params)

def build_anchor_generator(anchor_generator_cfg):
    class_name = anchor_generator_cfg["type"]
    if class_name == "AnchorGeneratorBEV":
        builder = AnchorGeneratorBEV
    else:
        raise NotImplementedError
    params = {proc_param(k):v
        for k, v in anchor_generator_cfg.items() if is_param(k)}
    return builder(**params)

def build_similarity_calculator(similarity_calculator_cfg):
    class_name = similarity_calculator_cfg["type"]
    if class_name == "NearestIoUSimilarity":
        builder = NearestIoUSimilarity
    else:
        raise NotImplementedError
    params = {proc_param(k):v
        for k, v in similarity_calculator_cfg.items() if is_param(k)}
    return builder(**params)

def build(target_assigner_cfg):
    class_name = target_assigner_cfg["type"]
    if class_name == "TaskAssignerV1":
        builder = TaskAssignerV1
        params = {proc_param(k):v
            for k, v in target_assigner_cfg.items() if is_param(k)}
        # build box_coder
        box_coder = build_box_coder(target_assigner_cfg["box_coder"])
        params["box_coder"] = box_coder
        # build anchors & region similarity calculators
        classsettings_cfgs = [v
            for k, v in target_assigner_cfg.items() if "class_settings" in k]
        anchor_generators = []
        similarity_calculators = []
        classes = params["classes"]
        for cls in classes:
            classsetting_cfg = [itm for itm in classsettings_cfgs
                if itm["AnchorGenerator"]["@class_name"] == cls][0]
            anchor_generator_cfg = classsetting_cfg["AnchorGenerator"]
            anchor_generator = build_anchor_generator(anchor_generator_cfg)
            anchor_generators.append(anchor_generator)
            similarity_calculator_cfg = classsetting_cfg["SimilarityCalculator"]
            similarity_calculator = build_similarity_calculator(similarity_calculator_cfg)
            similarity_calculators.append(similarity_calculator)
        params["anchor_generators"] = anchor_generators
        params["region_similarity_calculators"] = similarity_calculators
    else:
        raise NotImplementedError
    return builder(**params)

if __name__ == "__main__":
    from incdet3.configs.dev_cfg import cfg, modify_cfg
    modify_cfg(cfg)
    target_assigner = build(cfg.TARGETASSIGNER)
