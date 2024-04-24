import numpy as np
import pandas as pd
import os

from PIL import Image

import torch

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt


# FILE_ROOT = os.path.dirname(os.path.abspath(__file__))
FILE_ROOT = "/nfs/mareep/data/ssd1/nhaldert/datasets/GFIE_dataset"


class GFIELoader(object):

    def __init__(self):

        self.train_gaze = GFIEDataset("train")
        self.val_gaze = GFIEDataset("valid")
        self.test_gaze = GFIEDataset("test")

        self.train_loader = DataLoader(
            self.train_gaze,
            batch_size=64,
            num_workers=4,
            shuffle=True,
            collate_fn=collate_func,
        )

        self.val_loader = DataLoader(
            self.val_gaze,
            batch_size=8,
            num_workers=4,
            shuffle=False,
            collate_fn=collate_func,
        )

        self.test_loader = DataLoader(
            self.test_gaze,
            batch_size=8,
            num_workers=4,
            shuffle=False,
            collate_fn=collate_func,
        )


class GFIEDataset(Dataset):
    def __init__(self, dstype) -> None:
        self.input_size = 224
        self.heatmap_output_size = 64

        self.rgb_image_path = os.path.join(FILE_ROOT, "rgb")
        self.depth_path = os.path.join(FILE_ROOT, "depth")

        if dstype == "train":
            df = pd.read_csv(os.path.join(FILE_ROOT, "train_annotation.txt"))
        elif dstype == "valid":
            df = pd.read_csv(os.path.join(FILE_ROOT, "valid_annotation.txt"))
        elif dstype == "test":
            df = pd.read_csv(os.path.join(FILE_ROOT, "test_annotation.txt"))
        else:
            raise NotImplementedError

        self.length=len(df)
        self.X_train = df[
            [
                "scene_id",
                "frame_id",
                "h_x_min",
                "h_y_min",
                "h_x_max",
                "h_y_max",
                "eye_u",
                "eye_v",
                "eye_X",
                "eye_Y",
                "eye_Z",
            ]
        ]
        self.Y_train = df[["gaze_u", "gaze_v", "gaze_X", "gaze_Y", "gaze_Z"]]
        self.length = len(df)
        self.dstype = dstype

        transform_list = []
        transform_list.append(transforms.Resize((self.input_size, self.input_size)))
        transform_list.append(transforms.ToTensor())
        # transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):

        (
            scene_id,
            frame_index,
            h_x_min,
            h_y_min,
            h_x_max,
            h_y_max,
            eye_u,
            eye_v,
            eye_X,
            eye_Y,
            eye_Z,
        ) = self.X_train.iloc[index]
        gaze_u, gaze_v, gaze_X, gaze_Y, gaze_Z = self.Y_train.iloc[index]

        rgb_path = os.path.join(
            self.rgb_image_path,
            self.dstype,
            "scene{}".format(int(scene_id)),
            "{:04}.jpg".format(int(frame_index)),
        )
        depth_path=os.path.join(self.depth_path,self.dstype,"scene{}".format(int(scene_id)),"{:04}.npy".format(int(frame_index)))

        depthimg=np.load(depth_path)
        depthimg[np.isnan(depthimg)]=0
        depthimg=depthimg.astype(np.float32)
        depthimg=Image.fromarray(depthimg)

        img = Image.open(rgb_path)
        img = img.convert("RGB")
        width, height = img.size



        headimg = img.crop((int(h_x_min), int(h_y_min), int(h_x_max), int(h_y_max)))

        head_channel = get_head_box_channel(h_x_min, h_y_min, h_x_max, h_y_max, width, height, resolution=self.input_size, coordconv=False).unsqueeze(0)

        gaze_vector = np.array([gaze_X - eye_X, gaze_Y - eye_Y, gaze_Z - eye_Z])
        norm_gaze_vector = (
            1.0 if np.linalg.norm(gaze_vector) <= 0.0 else np.linalg.norm(gaze_vector)
        )
        gaze_vector = gaze_vector / norm_gaze_vector
        gaze_vector = torch.from_numpy(gaze_vector)
        gaze_target2d = torch.from_numpy(np.array([gaze_u, gaze_v]))

        gaze_heatmap = torch.zeros(self.heatmap_output_size, self.heatmap_output_size)
        gaze_heatmap = draw_labelmap(gaze_heatmap, [gaze_u * self.heatmap_output_size, gaze_v * self.heatmap_output_size], 3, type='Gaussian')

        data = dict()
        # Train
        data["sceneimg"] = self.transform(img)
        data["headimg"] = self.transform(headimg)
        data["headloc"] = head_channel
        data["depthimg"] = self.transform(depthimg)

        # Label
        data["gaze_heatmap"] = gaze_heatmap
        data["gaze_vector"] = gaze_vector
        data["gaze_target2d"] = gaze_target2d
        return data

    def __len__(self):
        return self.length


def collate_func(batch):
    batch_data = dict()

    batch_data["sceneimg"] = []
    batch_data["headimg"] = []
    batch_data["headloc"] = []
    batch_data["depthimg"] = []

    batch_data["gaze_heatmap"] = []
    batch_data["gaze_vector"] = []
    batch_data["gaze_target2d"] = []

    for data in batch:
        batch_data["sceneimg"].append(data["sceneimg"])
        batch_data["headimg"].append(data["headimg"])
        batch_data["headloc"].append(data["headloc"])
        batch_data["depthimg"].append(data["depthimg"])

        batch_data["gaze_heatmap"].append(data["gaze_heatmap"])
        batch_data["gaze_vector"].append(data["gaze_vector"])
        batch_data["gaze_target2d"].append(data["gaze_target2d"])

    batch_data["sceneimg"] = torch.stack(batch_data["sceneimg"], 0)
    batch_data["headimg"] = torch.stack(batch_data["headimg"], 0)
    batch_data["headloc"] = torch.stack(batch_data["headloc"], 0)
    batch_data["depthimg"] = torch.stack(batch_data["depthimg"], 0)


    batch_data["gaze_heatmap"] = torch.stack(batch_data["gaze_heatmap"], 0)
    batch_data["gaze_vector"] = torch.stack(batch_data["gaze_vector"], 0)
    batch_data["gaze_target2d"] = torch.stack(batch_data["gaze_target2d"], 0)

    return batch_data


def get_head_box_channel(x_min, y_min, x_max, y_max, width, height, resolution, coordconv=False):
    head_box = np.array([x_min/width, y_min/height, x_max/width, y_max/height])*resolution
    head_box = head_box.astype(int)
    head_box = np.clip(head_box, 0, resolution-1)
    if coordconv:
        unit = np.array(range(0,resolution), dtype=np.float32)
        head_channel = []
        for i in unit:
            head_channel.append([unit+i])
        head_channel = np.squeeze(np.array(head_channel)) / float(np.max(head_channel))
        head_channel[head_box[1]:head_box[3],head_box[0]:head_box[2]] = 0
    else:
        head_channel = np.zeros((resolution,resolution), dtype=np.float32)
        head_channel[head_box[1]:head_box[3],head_box[0]:head_box[2]] = 1
    head_channel = torch.from_numpy(head_channel)
    return head_channel


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def to_numpy(tensor):
    ''' tensor to numpy '''
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def draw_labelmap(img, pt, sigma, type='Gaussian'):
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] += g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    if np.max(img)!=0:
        img = img/np.max(img) # normalize heatmap so it has max value of 1
    return to_torch(img)