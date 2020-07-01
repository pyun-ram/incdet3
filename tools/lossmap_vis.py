'''
 File Created: Wed Jul 01 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''
import argparse
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
g_data = None

def read_pkl(path:str):
    with open(path, 'rb') as f:
        pkl = pickle.load(f)
    return pkl

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 

class LossMapFormatter(object):
    def __init__(self, im):
        self.im = im
        global g_data
        # num_anchor_per_loc, H, W, num_cls
        self.cls_preds = sigmoid(g_data['cls_preds'].detach().cpu().numpy())
        self.loss_cls = g_data['loss_cls'].reshape(*self.cls_preds.shape).detach().cpu().numpy()
        self.cls_targets = g_data['cls_targets'].reshape(*self.cls_preds.shape[:-1]).detach().cpu().numpy()
    def __call__(self, x, y):
        loss = self.loss_cls[:, int(y), int(x), :].sum()
        fg_score = self.cls_preds[:, int(y), int(x), :].max(axis=-1)
        target = self.cls_targets[:, int(y), int(x)]
        return f'x={x:.1f}, y={y:.1f},'+\
               f'loss: {loss:.2f}, ' +\
               f'fg_score: {[f"{itm:.2f}"for itm in fg_score.flatten()]}, ' +\
               f'target: {target.flatten()}'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Loss Map')
    parser.add_argument('--pkl-path', type=str, help='pkl path')
    # load pickle
    args = parser.parse_args()
    g_data = read_pkl(args.pkl_path)
    # visualize loss map with output cls_pred and target
    data = g_data
    cls_preds = data['cls_preds']
    loss_cls = data['loss_cls'].reshape(*cls_preds.shape)
    cls_targets = data['cls_targets'].reshape(*cls_preds.shape[:-1])

    # output loss_cls
    print("loss_cls is ", loss_cls.sum())
    print("loss_cls of old_anchor old classes: ", loss_cls[:2, :, :, :1].sum())
    print("loss_cls of new_anchor old classes: ", loss_cls[2:, :, :, :1].sum())
    print("loss_cls of old_anchor new classes: ", loss_cls[:2, :, :, 1:].sum())
    print("loss_cls of new_anchor new classes: ", loss_cls[2:, :, :, 1:].sum())
    # visualize loss_map(all)
    loss_map = loss_cls.clone()
    loss_map = loss_map.sum(dim=(0, -1)).squeeze()
    loss_map_vis = (loss_map - loss_map.min()) / loss_map.max() * 255.0
    loss_map_vis = loss_map_vis.detach().cpu().numpy().astype(np.uint8)

    # visualize cls_preds(all)
    preds_map = cls_preds.clone()
    preds_map = torch.sigmoid(preds_map)
    preds_map = preds_map.max(dim=0)[0]
    preds_map = preds_map.max(dim=-1)[0]
    preds_map_vis = (preds_map - 0.0) / 1.0 * 255.0
    preds_map_vis = preds_map_vis.detach().cpu().numpy().astype(np.uint8)

    # visualize cls_targets(all)
    targets_map = cls_targets.clone()
    targets_map[targets_map==-1] = 1
    targets_map = targets_map.sum(dim=(0)).squeeze()
    targets_map_vis = (targets_map - (0.0)) / 2.0 * 255.0
    targets_map_vis = targets_map_vis.detach().cpu().numpy().astype(np.uint8)

    fig, ax = plt.subplots()
    im = ax.imshow(loss_map_vis, interpolation='none', vmin=0, vmax=255, cmap='jet')
    ax.format_coord = LossMapFormatter(im)
    plt.title("loss")

    fig, ax = plt.subplots()
    im = plt.imshow(preds_map_vis, interpolation='none', vmin=0, vmax=255, cmap='jet')
    ax.format_coord = LossMapFormatter(im)
    plt.title("preds")

    fig, ax = plt.subplots()
    im = plt.imshow(targets_map_vis, interpolation='none', vmin=0, vmax=255, cmap='jet')
    ax.format_coord = LossMapFormatter(im)
    plt.title("targets")
    plt.show()
    plt.close()