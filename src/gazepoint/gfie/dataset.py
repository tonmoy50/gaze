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


FILE_ROOT = os.path.dirname(os.path.abspath(__file__))


class GFIELoader(object):

    def __init__(self):

        self.train_gaze = GFIEDataset("train")
        self.val_gaze = GFIEDataset("valid")
        self.test_gaze = GFIEDataset("test")

        self.train_loader = DataLoader(
            self.train_gaze,
            batch_size=8,
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
        self.rgb_image_path = os.path.join(FILE_ROOT, "rgb")
        if dstype == "train":
            df = pd.read_csv(os.path.join(FILE_ROOT, "train_annotation.txt"))
        elif dstype == "val":
            df = pd.read_csv(os.path.join(FILE_ROOT, "valid_annotation.txt"))
        elif dstype == "test":
            df = pd.read_csv(os.path.join(FILE_ROOT, "test_annotation.txt"))
        else:
            raise NotImplementedError

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
        img = Image.open(rgb_path)
        img = img.convert("RGB")

        headimg = img.crop((int(h_x_min), int(h_y_min), int(h_x_max), int(h_y_max)))

        gaze_vector = np.array([gaze_X - eye_X, gaze_Y - eye_Y, gaze_Z - eye_Z])
        norm_gaze_vector = (
            1.0 if np.linalg.norm(gaze_vector) <= 0.0 else np.linalg.norm(gaze_vector)
        )
        gaze_vector = gaze_vector / norm_gaze_vector
        gaze_vector = torch.from_numpy(gaze_vector)

        gaze_target2d = torch.from_numpy(np.array([gaze_u, gaze_v]))

        data = dict()
        # Train
        data["sceneimg"] = img
        data["headimg"] = headimg

        # Label
        data["gaze_vector"] = [gaze_vector]
        data["gaze_target2d"] = gaze_target2d
        return data


def collate_func():
    batch = dict()

    batch["sceneimg"] = []
    batch["headimg"] = []

    batch["gaze_vector"] = []
    batch["gaze_target2d"] = []

    for data in batch:
        batch["sceneimg"].append(data["sceneimg"])
        batch["headimg"].append(data["headimg"])

        batch["gaze_vector"].append(data["gaze_vector"])
        batch["gaze_target2d"].append(data["gaze_target2d"])

    batch["sceneimg"] = torch.stack(batch["sceneimg"], 0)
    batch["headimg"] = torch.stack(batch["headimg"], 0)

    batch["gaze_vector"] = torch.stack(batch["gaze_vector"], 0)
    batch["gaze_target2d"] = torch.stack(batch["gaze_target2d"], 0)

    return batch
