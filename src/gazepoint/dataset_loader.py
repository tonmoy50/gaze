from scipy.io import loadmat
import numpy as np

import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset

import os
import random

from utils import data_transforms, get_paste_kernel, kernel_map


class GazeDataset(Dataset):
    def __init__(self, logging, root_dir, mat_file, training="train"):
        assert training in set(["train", "test"])
        self.root_dir = root_dir
        self.mat_file = mat_file
        self.training = training

        anns = loadmat(self.mat_file)
        self.bboxes = anns[self.training + "_bbox"]
        self.gazes = anns[self.training + "_gaze"]
        self.paths = anns[self.training + "_path"]
        self.eyes = anns[self.training + "_eyes"]
        self.meta = anns[self.training + "_meta"]
        self.image_num = self.paths.shape[0]

        self.logging = logging
        self.logging.info("%s contains %d images" % (self.mat_file, self.image_num))

    def generate_data_field(self, eye_point):
        """eye_point is (x, y) and between 0 and 1"""
        height, width = 224, 224
        x_grid = np.array(range(width)).reshape([1, width]).repeat(height, axis=0)
        y_grid = np.array(range(height)).reshape([height, 1]).repeat(width, axis=1)
        grid = np.stack((x_grid, y_grid)).astype(np.float32)

        x, y = eye_point
        x, y = x * width, y * height

        grid -= np.array([x, y]).reshape([2, 1, 1]).astype(np.float32)
        norm = np.sqrt(np.sum(grid**2, axis=0)).reshape([1, height, width])
        # avoid zero norm
        norm = np.maximum(norm, 0.1)
        grid /= norm
        return grid

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        image_path = self.paths[idx][0][0]
        image_path = os.path.join(self.root_dir, image_path)

        box = self.bboxes[0, idx][0]
        eye = self.eyes[0, idx][0]
        # todo: process gaze differently for training or testing
        gaze = self.gazes[0, idx].mean(axis=0)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if random.random() > 0.5 and self.training == "train":
            eye = [1.0 - eye[0], eye[1]]
            gaze = [1.0 - gaze[0], gaze[1]]
            image = cv2.flip(image, 1)

        # crop face
        x_c, y_c = eye
        x_0 = x_c - 0.15
        y_0 = y_c - 0.15
        x_1 = x_c + 0.15
        y_1 = y_c + 0.15
        if x_0 < 0:
            x_0 = 0
        if y_0 < 0:
            y_0 = 0
        if x_1 > 1:
            x_1 = 1
        if y_1 > 1:
            y_1 = 1
        h, w = image.shape[:2]
        face_image = image[int(y_0 * h) : int(y_1 * h), int(x_0 * w) : int(x_1 * w), :]
        # process face_image for face net
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = Image.fromarray(face_image)
        face_image = data_transforms[self.training](face_image)
        # process image for saliency net
        # image = image_preprocess(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = data_transforms[self.training](image)

        # generate gaze field
        gaze_field = self.generate_data_field(eye_point=eye)
        # generate heatmap
        heatmap = get_paste_kernel(
            (224 // 4, 224 // 4), gaze, kernel_map, (224 // 4, 224 // 4)
        )
        """
        direction = gaze - eye
        norm = (direction[0] ** 2.0 + direction[1] ** 2.0) ** 0.5
        if norm <= 0.0:
            norm = 1.0

        direction = direction / norm
        """
        sample = {
            "image": image,
            "face_image": face_image,
            "eye_position": torch.FloatTensor(eye),
            "gaze_field": torch.from_numpy(gaze_field),
            "gt_position": torch.FloatTensor(gaze),
            "gt_heatmap": torch.FloatTensor(heatmap).unsqueeze(0),
        }

        return sample
