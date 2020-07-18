import tabulate
import numpy as np
from det3.ops import read_pkl

tag_dict = {
"July14-expcarlarmore-finetuning_cv0_task2_domain3": "logs/July14-expcarlarmore-finetuning_cv0_task2_domain3/test_results.pkl",
"July14-expcarlarmore-finetuning_cv1_task2_domain3": "logs/July14-expcarlarmore-finetuning_cv1_task2_domain3/test_results.pkl",
"July14-expcarlarmore-finetuning_cv2_task2_domain3": "logs/July14-expcarlarmore-finetuning_cv2_task2_domain3/test_results.pkl",
"July14-expcarlarmore-finetuning_cv3_task2_domain3": "logs/July14-expcarlarmore-finetuning_cv3_task2_domain3/test_results.pkl",
"July14-expcarlarmore-finetuning_cv4_task2_domain3": "logs/July14-expcarlarmore-finetuning_cv4_task2_domain3/test_results.pkl",
"July14-expcarlarmore-lwf_cv0_task2_domain3": "logs/July14-expcarlarmore-lwf_cv0_task2_domain3/test_results.pkl",
"July14-expcarlarmore-lwf_cv1_task2_domain3": "logs/July14-expcarlarmore-lwf_cv1_task2_domain3/test_results.pkl",
"July14-expcarlarmore-lwf_cv2_task2_domain3": "logs/July14-expcarlarmore-lwf_cv2_task2_domain3/test_results.pkl",
"July14-expcarlarmore-lwf_cv3_task2_domain3": "logs/July14-expcarlarmore-lwf_cv3_task2_domain3/test_results.pkl",
"July14-expcarlarmore-lwf_cv4_task2_domain3": "logs/July14-expcarlarmore-lwf_cv4_task2_domain3/test_results.pkl",
"July14-expcarlarmore-jointtraining_cv0_task2_domain3": "logs/July14-expcarlarmore-jointtraining_cv0_task2_domain3/test_results.pkl",
"July14-expcarlarmore-jointtraining_cv1_task2_domain3": "logs/July14-expcarlarmore-jointtraining_cv1_task2_domain3/test_results.pkl",
"July14-expcarlarmore-jointtraining_cv2_task2_domain3": "logs/July14-expcarlarmore-jointtraining_cv2_task2_domain3/test_results.pkl",
"July14-expcarlarmore-jointtraining_cv3_task2_domain3": "logs/July14-expcarlarmore-jointtraining_cv3_task2_domain3/test_results.pkl",
"July14-expcarlarmore-jointtraining_cv4_task2_domain3": "logs/July14-expcarlarmore-jointtraining_cv4_task2_domain3/test_results.pkl",
}

statistic_dict = {
    "joint_training": [
        "July14-expcarlarmore-jointtraining_cv0_task2_domain3",
        "July14-expcarlarmore-jointtraining_cv1_task2_domain3",
        "July14-expcarlarmore-jointtraining_cv2_task2_domain3",
        "July14-expcarlarmore-jointtraining_cv3_task2_domain3",
        "July14-expcarlarmore-jointtraining_cv4_task2_domain3",
    ],
    "lwf": [
        "July14-expcarlarmore-lwf_cv0_task2_domain3",
        "July14-expcarlarmore-lwf_cv1_task2_domain3",
        "July14-expcarlarmore-lwf_cv2_task2_domain3",
        "July14-expcarlarmore-lwf_cv3_task2_domain3",
        "July14-expcarlarmore-lwf_cv4_task2_domain3",
    ],
    "fine_tuning": [
        "July14-expcarlarmore-finetuning_cv0_task2_domain3",
        "July14-expcarlarmore-finetuning_cv1_task2_domain3",
        "July14-expcarlarmore-finetuning_cv2_task2_domain3",
        "July14-expcarlarmore-finetuning_cv3_task2_domain3",
        "July14-expcarlarmore-finetuning_cv4_task2_domain3",
    ],
}

valid_classes = ["Car", "Pedestrian"]
header = [
    "Case",
    "Car.Easy", "Car.Mod", "Car.Hard",
    "Ped.Easy", "Ped.Mod", "Ped.Hard",
    "Cyc.Easy", "Cyc.Mod", "Cyc.Hard"
    ]
content = []
data = {}
for tag, pkl_path in tag_dict.items():
    ctt = [tag]
    res = read_pkl(pkl_path)
    res = res['detail']['eval.carla']['carla']
    data_ = {
        "Car": res["Car"]["3d@0.50"],
        "Pedestrian": res["Pedestrian"]["3d@0.25"] if "Pedestrian" in valid_classes else [0, 0 ,0],
        "Cyclist": res["Cyclist"]["3d@0.25"] if "Cyclist" in valid_classes else [0, 0 ,0],
    }
    ctt += data_["Car"]
    ctt += data_["Pedestrian"]
    ctt += data_["Cyclist"]
    data[tag] = data_
    content.append(ctt)
print(tabulate.tabulate(content, headers=header, tablefmt="pipe"))

content = []
for statistic_tag in ["joint_training", "lwf", "fine_tuning"]:
    ctt = [statistic_tag]
    tags = statistic_dict[statistic_tag]
    for cls in ["Car", "Pedestrian", "Cyclist"]:
        for i in range(3): #0: easy, 1: Mode, 2: Hard
            statistic_data = [data[tag][cls][i] for tag in tags]
            statistic_mean = np.mean(statistic_data)
            statistic_std = np.std(statistic_data)
            ctt += [f"{statistic_mean:.1f}+-{statistic_std:.1f}"]
    content.append(ctt)
print(tabulate.tabulate(content, headers=header, tablefmt="pipe"))