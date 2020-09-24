import torch
from incdet3.models.ewc_func import ewc_measure_distance

if __name__ == "__main__":
    diff = torch.arange(6).reshape(2,3).cuda()
    loss_type = "l2"
    beta = 1
    weights = torch.arange(6).reshape(2,3).cuda()
    dist = ewc_measure_distance(diff, loss_type, beta, weights)
    print(f"diff : {diff}")
    print(f"weights : {weights}")
    print(f"result : {dist}")