import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

import argparse
import os
import shutil
import sys
import random
import time
import numpy as np
from datetime import datetime
from tqdm import tqdm

from dataset import GFIELoader
from model_zoo import MultiNet

# from tensorboardX import SummaryWriter

# device = torch.device(f'cuda:{cuda_val}' if torch.cuda.is_available() else 'cpu')

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def argmax_pts(heatmap):

    idx=np.unravel_index(heatmap.argmax(),heatmap.shape)
    pred_y,pred_x=map(float,idx)

    return pred_x,pred_y

def euclid_dist(pred,target):
    pred = pred.to("cpu")
    target = target.to("cpu")

    batch_dist=0.
    batch_size=pred.shape[0]
    pred_H,pred_W=pred.shape[1:]

    for b_idx in range(batch_size):

        pred_x,pred_y=argmax_pts(pred[b_idx])
        norm_p=np.array([pred_x,pred_y])/np.array([pred_W,pred_H])

        sample_target=target[b_idx]
        sample_target=sample_target.view(-1,2).numpy()

        sample_dist=np.linalg.norm(sample_target-norm_p)

        batch_dist+=sample_dist

    euclid_dist=batch_dist/float(batch_size)

    return euclid_dist

def run():

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    epoch = 10

    dataloader = GFIELoader()
    train_loader = dataloader.train_loader
    val_loader = dataloader.val_loader
    test_loader = dataloader.test_loader

    net = MultiNet()
    net.to(device)
    print(net)

    criterion = [nn.MSELoss(reduction="none"), nn.CosineSimilarity()]
    lr = 0.001
    weight_decay = 0.005
    optimizer = optim.Adam(
        net.parameters(), lr=lr, weight_decay=weight_decay
    )

    net.train()
    all_vec_losses = list()
    all_l2_losses = list()
    all_total_losses = list()
    all_dist_losses = list()
    for cur in range(epoch):
        epoch_losses = []
        epoch_l2_losses = []
        epoch_total_losses = []
        epoch_dist_losses = []

        # net.train()
        loader_capacity=len(train_loader)
        pbar=tqdm(total=loader_capacity)
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            x_simg, x_himg, x_headloc, x_depthimg = data["sceneimg"], data["headimg"], data["headloc"], data["depthimg"]
            y_gaze_vector, y_gaze_target2d, y_gaze_heatmap = data["gaze_vector"], data["gaze_target2d"], data["gaze_heatmap"]

            bs = x_simg.size(0)

            x_simg = x_simg.to(device)
            x_himg = x_himg.to(device)
            x_headloc = x_headloc.to(device)
            x_depthimg = x_depthimg.to(device)

            y_gaze_heatmap = y_gaze_heatmap.to(device)
            y_gaze_vector = y_gaze_vector.to(device)
            y_gaze_target2d = y_gaze_target2d.to(device)

            preds = net(x_simg, x_himg, x_headloc, x_depthimg)

            # print(preds["pred_gazedirection"])
            pred_gazedirection = preds["pred_gazedirection"].squeeze()
            pred_heatmap = preds["pred_heatmap"]
            pred_heatmap = pred_heatmap.squeeze()
            # print(pred_gazedirection)

            l2_loss=criterion[0](pred_heatmap,y_gaze_heatmap)
            l2_loss=torch.mean(l2_loss,dim=1)
            l2_loss = torch.mean(l2_loss, dim=1)
            l2_loss=torch.sum(l2_loss)/bs
            epoch_l2_losses.append(l2_loss.item())

            loss = 1- criterion[1](pred_gazedirection, y_gaze_vector)
            # vec_loss=10 * (torch.sum(loss)/bs)
            vec_loss=torch.sum(loss)/bs
            epoch_losses.append(vec_loss.item())
            # print("Epoch {cur}, Loss: ", vec_loss.item())

            total_loss = l2_loss*10000 + vec_loss*10
            epoch_total_losses.append(total_loss.item())

            try:
                distrain_avg = euclid_dist(pred_heatmap, y_gaze_target2d)
                # print(distrain_avg)
                epoch_dist_losses.append(distrain_avg)
            except Exception:
                epoch_dist_losses.append(0)

            total_loss.backward()
            optimizer.step()

            pbar.set_description("Epoch: [{0}]".format(cur))
            pbar.set_postfix(loss=vec_loss.item(), l2_loss=l2_loss.item(), total_loss=total_loss.item(), distrain_avg=distrain_avg)
            pbar.update(1)

        pbar.close()
        all_vec_losses.append((sum(epoch_losses)/len(epoch_losses)))
        all_l2_losses.append((sum(epoch_l2_losses)/len(epoch_l2_losses)))
        all_total_losses.append((sum(epoch_total_losses)/len(epoch_total_losses)))
        all_dist_losses.append((sum(epoch_dist_losses)/len(epoch_dist_losses)))

    with open('vec_losses.txt', 'w') as file:
        for item in all_vec_losses:
            file.write(str(item) + '\n')

    with open('l2_losses.txt', 'w') as file:
        for item in all_l2_losses:
            file.write(str(item) + '\n')

    with open('total_losses.txt', 'w') as file:
        for item in all_total_losses:
            file.write(str(item) + '\n')

    with open('dist.txt', 'w') as file:
        for item in all_dist_losses:
            file.write(str(item) + '\n')



if __name__ == "__main__":
    run()
