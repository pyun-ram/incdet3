import tabulate
from det3.ops import read_pkl
tag_dict = {
    "July04-expcarlasmall-lwf-cv1": "logs/July04-expcarlasmall-lwf-cv1/test_results.pkl",
    "July04-expcarlasmall-lwf-cv2": "logs/July04-expcarlasmall-lwf-cv2/test_results.pkl",
    "July04-expcarlasmall-lwf-cv3": "logs/July04-expcarlasmall-lwf-cv3/test_results.pkl",
    "July04-expcarlasmall-jointtraining-cv1": "logs/July04-expcarlasmall-jointtraining-cv1/test_results.pkl",
    "July04-expcarlasmall-jointtraining-cv2": "logs/July04-expcarlasmall-jointtraining-cv2/test_results.pkl",
    "July04-expcarlasmall-jointtraining-cv3": "logs/July04-expcarlasmall-jointtraining-cv3/test_results.pkl",
    "July04-expcarlasmall-finetune-cv1": "logs/July04-expcarlasmall-finetune-cv1/test_results.pkl",
    "July04-expcarlasmall-finetune-cv2": "logs/July04-expcarlasmall-finetune-cv2/test_results.pkl",
    "July04-expcarlasmall-finetune-cv3": "logs/July04-expcarlasmall-finetune-cv3/test_results.pkl",
}
header = ["Case", "Car.Easy", "Car.Mod", "Car.Hard", "Ped.Easy", "Ped.Mod", "Ped.Hard"]
content = []
for tag, pkl_path in tag_dict.items():
    ctt = [tag]
    res = read_pkl(pkl_path)
    res = res['detail']['eval.carla']['carla']
    ctt += res["Car"]["3d@0.50"]
    ctt += res["Pedestrian"]["3d@0.25"]
    content.append(ctt)
print(tabulate.tabulate(content, headers=header, tablefmt="plain"))