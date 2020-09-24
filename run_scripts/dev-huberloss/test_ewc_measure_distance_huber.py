import torch
from incdet3.models.ewc_func import ewc_measure_distance
from matplotlib import pyplot as plt

if __name__ == "__main__":
    diff = torch.linspace(-1, 1, 100)
    loss_type = "huber"
    beta = 0.5
    weights = torch.ones_like(diff)
    dist = ewc_measure_distance(diff, loss_type, beta, weights)
    diff_np = diff.numpy()
    dist_np = dist.numpy()
    plt.figure()
    plt.plot(diff_np, dist_np, label="huber")
    dist = ewc_measure_distance(diff, "l2", beta, weights)
    diff_np = diff.numpy()
    dist_np = dist.numpy()
    plt.plot(diff_np, dist_np, label="l2")
    plt.legend()
    plt.savefig("./tmp.png")

    diff = torch.arange(9).reshape(3,3).cuda()
    loss_type = "huber"
    beta = 5
    weights = torch.arange(9).reshape(3,3).cuda()
    dist = ewc_measure_distance(diff, "huber", beta, weights)
    print(f"diff : {diff}")
    print(f"weights : {weights}")
    print(f"result : {dist}")

