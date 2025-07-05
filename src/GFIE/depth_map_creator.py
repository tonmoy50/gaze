import cv2
import torch
import numpy as np
import pandas as pd
import os


# import warnings

# warnings.filterwarnings("ignore", message="xFormers not available")

from dataset.metric_depth.depth_anything_v2.dpt import DepthAnythingV2


def get_depth_model(rgb_path):

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
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
    return model
    # raw_img = cv2.imread(
    #     "/nfs/mareep/data/ssd1/nhaldert/datasets/GFIE_dataset/rgb/train/scene1/0000.jpg"
    # )  # HxWx3 RGB image in BGR format
    # depth = model.infer_image(raw_img)  # HxW depth map in meters in numpy
    # return depth


if __name__ == "__main__":
    root = "/nfs/mareep/data/ssd1/nhaldert/datasets/GFIE_dataset"
    temp_root = "/nobackup/nhaldert/datasets/GFIE_dataset"
    train_file = os.path.join(root, "train_annotation.txt")

    df_train = pd.read_csv(train_file)
    df_train = df_train[0 : int(df_train.shape[0] * 0.10)]
    print(df_train.shape)

    rgb_path = os.path.join(root, "rgb", "train")

    model = get_depth_model(rgb_path)

    for index, row in df_train.iterrows():
        print("Processing index:", index)
        scene_id = row["scene_id"]
        frame_index = row["frame_id"]
        img_path = os.path.join(
            rgb_path,
            "scene{}".format(int(scene_id)),
            "{:04}.jpg".format(int(frame_index)),
        )

        # print("RGB path:", rgb_path)
        dephimg = cv2.imread(img_path)  # HxWx3 RGB image in BGR format
        depthimg = model.infer_image(dephimg)

        if not os.path.exists(
            os.path.join(
                temp_root,
                "depth",
                "train",
                "scene{}".format(int(scene_id)),
            )
        ):
            os.makedirs(
                os.path.join(
                    temp_root, "depth", "train", "scene{}".format(int(scene_id))
                )
            )

        depth_path = os.path.join(
            temp_root,
            "depth",
            "train",
            "scene{}".format(int(scene_id)),
            "{:04}.npy".format(int(frame_index)),
        )
        np.save(depth_path, depthimg)
