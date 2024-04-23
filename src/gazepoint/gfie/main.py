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



def run():

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    epoch = 25

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

    all_losses = list()
    for cur in range(epoch):
        epoch_losses = []

        net.train()
        loader_capacity=len(train_loader)
        pbar=tqdm(total=loader_capacity)
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            x_simg, x_himg = data["sceneimg"], data["headimg"]
            y_gaze_vector, y_gaze_target2d = data["gaze_vector"], data["gaze_target2d"]

            bs = x_simg.size(0)

            x_simg = x_simg.to(device)
            x_himg = x_himg.to(device)
            y_gaze_vector = y_gaze_vector.to(device)
            y_gaze_target2d = y_gaze_target2d.to(device)

            preds = net(x_simg, x_himg)
            # print(preds["pred_gazedirection"])
            pred_gazedirection = preds["pred_gazedirection"].squeeze()
            # print(pred_gazedirection)

            loss = 1- criterion[1](pred_gazedirection, y_gaze_vector)
            # vec_loss=10 * (torch.sum(loss)/bs)
            vec_loss=torch.sum(loss)/bs
            epoch_losses.append(vec_loss.item())
            # print("Epoch {cur}, Loss: ", vec_loss.item())

            vec_loss.backward()
            optimizer.step()

            pbar.set_description("Epoch: [{0}]".format(epoch))
            pbar.set_postfix(loss=vec_loss.item())
            pbar.update(1)

        pbar.close()
        all_losses.append(sum(epoch_losses)/len(epoch_losses))

    with open('my_list.txt', 'w') as file:
        for item in all_losses:
            file.write(item + '\n')




if __name__ == "__main__":
    run()
