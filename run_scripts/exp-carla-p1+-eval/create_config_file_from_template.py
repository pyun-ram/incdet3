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

def create_replace_dict(mode, task, domain, cv, reuse_tag):
    if mode in ["finetuning", "jointtraining", "lwf"]:
        if domain == "domain1":
            dataset_name = "CARLA-P1-TASK1"
        elif domain == "domain2":
            dataset_name = "CARLA-P1-TASK2"
        elif domain == "domain3":
            dataset_name = "CARLA-P1-TASK3"
        else:
            raise NotImplementedError
        mode2resumemode = {
            "finetuning": "finetune",
            "jointtraining": "jointtrain",
            "lwf": "lwf"
        }
        resume_ckpt_path = "\""
        resume_ckpt_path += "saved_weights/July24-expcarlap1+/July24-expcarlap1+-"
        resume_ckpt_path += f"{mode2resumemode[mode]}-"
        resume_ckpt_path += f"cv{cv}-"
        if task == "task3":
            resume_ckpt_path += f"cyc-"
        elif task == "task2":
            pass
        else:
            raise NotImplementedError
        if mode == "lwf":
            resume_ckpt_path += f"{reuse_tag}_bias_32/"
        else:
            resume_ckpt_path += f"{reuse_tag}/"
        if task == "task3":
            resume_ckpt_path += f"IncDetMain-25000.tckpt"
        elif task == "task2":
            resume_ckpt_path += f"IncDetMain-20000.tckpt"
        else:
            raise NotImplementedError
        resume_ckpt_path += "\""
        bool_reuse_anchor_for_cls = "True" if reuse_tag == "reuse" else "False"
        res = {
            "dataset_name": dataset_name,
            "resume_ckpt_path": resume_ckpt_path,
            "bool_reuse_anchor_for_cls": bool_reuse_anchor_for_cls,
        }
    else:
        raise NotImplementedError
    return res

def main(mode, task, domain, cv, reuse_tag):
    template_path = f"configs/exp-carla-p1+-eval/{mode}_{task}.py"
    save_path = f"configs/exp-carla-p1+-eval/{mode}_{task}_{domain}_{reuse_tag}_{cv}.py"
    template_txt = read_txt(template_path)
    replace_dict = create_replace_dict(mode, task, domain, cv, reuse_tag)
    res_txt = change_txt(template_txt, replace_dict)
    write_txt(res_txt, save_path)

if __name__ == "__main__":
    # template_path
    mode = "lwf" # finetuning, jointtraining, lwf
    for task in ["task2", "task3"]:
        if task == "task2":
            domain_list = ["domain1", "domain2"]
        elif task == "task3":
            domain_list = ["domain1", "domain2", "domain3"]
        else:
            raise NotImplementedError
        for domain in domain_list:
            for cv in range(5):
                for reuse_tag in ["noreuse", "reuse"]:
                    main(mode, task, domain, cv, reuse_tag)