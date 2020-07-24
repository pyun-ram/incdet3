from det3.ops import read_txt, write_txt
from copy import deepcopy
from pathlib import Path
from incdet3.utils import bcolors
from time import sleep
def change_txt(template_txt_list, replace_dict):
    txt_list = deepcopy(template_txt_list)
    num_changes = 0
    for i, line in enumerate(txt_list):
        if "${" in line:
            start = line.find("${")
            end = line.find("}")
            change_key = line[start+2:end]
            change_value = replace_dict[change_key]
            txt_list[i] = line[:start] + change_value + line[end+1:]
            print(bcolors.BOLD + change_key + bcolors.ENDC)
            print(bcolors.BOLD + line + bcolors.ENDC)
            print(bcolors.BOLD + txt_list[i] + bcolors.ENDC)
            print("===========")
            num_changes += 1
    print(bcolors.WARNING + f"total changes: {num_changes}" + bcolors.ENDC)
    sleep(1)
    return txt_list

def create_replace_dict(mode, idx):
    classes10 = ["car", "pedestrian", "barrier", "truck", "traffic_cone",
        "trailer", "construction_vehicle", "motorcycle", "bicycle", "bus"]
    if mode in ["finetuning", "jointtraining", "lwf"]:
        all_cls = "[" + ", ".join([f"\"{classes10[i]}\""for i in range(idx)]) + "]"
        old_cls = "[" + ", ".join([f"\"{classes10[i]}\""for i in range(idx-1)]) + "]"
        last_tag = (f"July22-expnusc-5+seq-{mode}-{idx-1}"
            if idx != 6 else f"July22-expnusc-5+5-train_from_scratch")
        resume_ckpt_path = f"\"saved_weights/{last_tag}/IncDetMain-{50000+15000*(idx-6)}.tckpt\""
        res = {
            "total_training_steps": f"{50+(idx-6)*15}e3",
            "all_classes": all_cls,
            "old_classes": old_cls,
            "resume_ckpt_path": resume_ckpt_path,
            "num_old_classes": f"{idx-1}",
            "num_old_anchor_per_loc": f"{2*(idx-1)}"
        }
    else:
        raise NotImplementedError
    return res

if __name__ == "__main__":
    # template_path
    g_template_path = "configs/exp-nusc-5+seq/lwf.py"
    save_dir = Path(g_template_path).parent
    template_name = Path(g_template_path).name.split(".")[0]
    for i in range(6, 11):
        save_path = str(Path(save_dir)/f"{template_name}-{i}.py")
        replace_dict = create_replace_dict(mode=template_name, idx=i)
        template_txt_list = read_txt(g_template_path)
        txt_list = change_txt(template_txt_list, replace_dict)
        write_txt(txt_list, save_path)