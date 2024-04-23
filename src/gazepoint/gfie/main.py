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

from dataset import GFIELoader
from model_zoo import MultiNet

from tensorboardX import SummaryWriter

# device = torch.device(f'cuda:{cuda_val}' if torch.cuda.is_available() else 'cpu')

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def run():

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    epoch = 10

    dataloader = GFIELoader()
    train_loader = dataloader.train_loader
    val_loader = dataloader.val_loader
    test_loader = dataloader.test_loader

    for batch in train_loader:
        print(batch[0])
        break

    net = MultiNet()
    net.to(device)
    criterion = [nn.MSELoss(reduction="none"), nn.CosineSimilarity()]
    lr = 0.001
    weight_decay = 0.005
    optimizer = optim.Adam(
        net.parameters(), lr=lr, weight_decay=weight_decay
    )


if __name__ == "__main__":
    run()
