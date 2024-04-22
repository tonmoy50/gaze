import numpy as np
import pandas as pd
import os

from PIL import Image

import torch

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional  as TF

import matplotlib.pyplot as plt


FILE_ROOT = os.path.dirname(os.path.abspath(__file__))
print(FILE_ROOT)

class GFIEDataset(Dataset):
    def __init__(self) -> None:
        rgb_image = os.path.join(FILE_ROOT, "rgb")
        