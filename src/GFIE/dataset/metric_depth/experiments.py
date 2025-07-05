import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}

encoder = "vitl"  # or 'vits', 'vitb'
dataset = "hypersim"  # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 10  # 20 for indoor model, 80 for outdoor model

model = DepthAnythingV2(**{**model_configs[encoder], "max_depth": max_depth})
model.load_state_dict(
    torch.load(
        f"/nobackup/nhaldert/depth_anything_v2_metric_{dataset}_{encoder}.pth",
        map_location="cuda",
    )
)

model.eval()
model.to("cuda")

raw_img = cv2.imread(
    "/nfs/mareep/data/ssd1/nhaldert/datasets/GFIE_dataset/rgb/train/scene1/0000.jpg"
)  # HxWx3 RGB image in BGR format
depth = model.infer_image(raw_img)  # HxW depth map in meters in numpy
print(depth)

import numpy as np

# Load the .npy file
data = np.load(
    "/nfs/mareep/data/ssd1/nhaldert/datasets/GFIE_dataset/depth/train/scene1/0000.npy"
)

# Print the values
data[np.isnan(data)] = 0  # Replace NaN values with 0

print(np.mean(data), np.mean(depth))
