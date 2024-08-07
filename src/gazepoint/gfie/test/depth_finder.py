import os
import numpy as np

from tqdm import tqdm

from PIL import Image
from transformers import pipeline


# DATASET_DIR = "/nfs/magnezone/data/ssd/nhaldert/datasets/Gaze360"
DATASET_DIR = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.join(os.path.dirname(os.path.abspath(__file__)))
                    )
                )
            )
        )
    ),
    "Data",
    "gaze360",
)


def depth_image_creator(lines, depth_estimator):

    for i in tqdm(range(len(lines))):
        img = Image.open(os.path.join(DATASET_DIR, "imgs", lines[i].split(" ")[0]))
        depth = depth_estimator(img)["depth"]
        new_path = lines[i].split(" ")[0].split("/")
        new_path[1] = "body"
        new_path = "/".join(new_path)
        if not os.path.exists(
            os.path.join(DATASET_DIR, "depths", "/".join(new_path.split("/")[:-1]))
        ):
            os.makedirs(
                os.path.join(
                    os.path.join(
                        DATASET_DIR,
                        "depths",
                        "/".join(new_path.split("/")[:-1]),
                    )
                )
            )
        # print("Saving to ", os.path.join(DATASET_DIR, "depths", new_path.replace("jpg", "npy")))
        np.save(
            os.path.join(
                DATASET_DIR,
                "depths",
                new_path.replace("jpg", "npy"),
            ),
            depth,
        )


def main():
    checkpoint = "vinvino02/glpn-nyu"
    checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
    depth_estimator = pipeline(
        "depth-estimation",
        model=checkpoint,
        device=0,
        features=256,
        out_channels=[256, 512, 1024, 1024],
    )

    # with open(os.path.join(DATASET_DIR, "train.txt"), "r") as f:
    #     lines = f.readlines()
    # depth_image_creator(lines, depth_estimator)

    with open(os.path.join(DATASET_DIR, "validation.txt"), "r") as f:
        lines = f.readlines()
    depth_image_creator(lines, depth_estimator)

    with open(os.path.join(DATASET_DIR, "test.txt"), "r") as f:
        lines = f.readlines()
    depth_image_creator(lines, depth_estimator)


if __name__ == "__main__":
    main()
