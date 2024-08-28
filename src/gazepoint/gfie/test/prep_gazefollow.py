import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from PIL import Image
from transformers import pipeline


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASET_DIR = os.path.join(
#     os.path.dirname(
#         os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CUR_DIR))))
#     ),
#     "Data",
#     "gazefollow_extended",
# )

DATASET_DIR = "/nfs/magnezone/data/ssd/nhaldert/datasets/Gazefollow/gazefollow_extended"


def prep_annotations(filename, dtype):
    with open(
        os.path.join(DATASET_DIR, filename),
        "r",
    ) as f:
        lines = f.readlines()

    image_path = list()
    id = list()
    body_bbox_x = list()
    body_bbox_y = list()
    body_bbox_width = list()
    body_bbox_height = list()
    eye_x = list()
    eye_y = list()
    gaze_x = list()
    gaze_y = list()
    head_bbox_x_min = list()
    head_bbox_y_min = list()
    head_bbox_x_max = list()
    head_bbox_y_max = list()
    in_or_out = list()

    for line in lines:
        (
            image_path_,
            id_,
            body_bbox_x_,
            body_bbox_y_,
            body_bbox_width_,
            body_bbox_height_,
            eye_x_,
            eye_y_,
            gaze_x_,
            gaze_y_,
            head_bbox_x_min_,
            head_bbox_y_min_,
            head_bbox_x_max_,
            head_bbox_y_max_,
            in_or_out_,
            # meta1_,
            # meta2_,
        ) = line.split(",")[:15]

        if in_or_out_ != "-1":

            image_path.append(image_path_)
            id.append(id_)
            body_bbox_x.append(body_bbox_x_)
            body_bbox_y.append(body_bbox_y_)
            body_bbox_width.append(body_bbox_width_)
            body_bbox_height.append(body_bbox_height_)
            eye_x.append(eye_x_)
            eye_y.append(eye_y_)
            gaze_x.append(gaze_x_)
            gaze_y.append(gaze_y_)
            head_bbox_x_min.append(head_bbox_x_min_)
            head_bbox_y_min.append(head_bbox_y_min_)
            head_bbox_x_max.append(head_bbox_x_max_)
            head_bbox_y_max.append(head_bbox_y_max_)
            in_or_out.append(in_or_out_)

    df = pd.DataFrame(
        {
            "image_path": image_path,
            "id": id,
            "body_bbox_x": body_bbox_x,
            "body_bbox_y": body_bbox_y,
            "body_bbox_width": body_bbox_width,
            "body_bbox_height": body_bbox_height,
            "eye_x": eye_x,
            "eye_y": eye_y,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "head_bbox_x_min": head_bbox_x_min,
            "head_bbox_y_min": head_bbox_y_min,
            "head_bbox_x_max": head_bbox_x_max,
            "head_bbox_y_max": head_bbox_y_max,
            "in_or_out": in_or_out,
        }
    )

    if dtype != "train":
        save_filename = filename.replace(".txt", ".csv")
        df.to_csv(os.path.join(DATASET_DIR, save_filename), index=False)
    else:
        df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)
        df_train.to_csv(
            os.path.join(DATASET_DIR, "train_annotations_release.csv"), index=False
        )
        df_val.to_csv(
            os.path.join(DATASET_DIR, "valid_annotations_release.csv"), index=False
        )
    print("success")


def make_depth_image(path, depth_estimator):

    with open(path, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        img = Image.open(os.path.join(DATASET_DIR, line.split(",")[0]))
        depth = depth_estimator(img)["depth"]

        new_path = os.path.join(
            DATASET_DIR, "depth", "/".join(line.split(",")[0].split("/")[:-1])
        )
        # print(new_path)

        if not os.path.exists(
            os.path.join(
                DATASET_DIR, "depth", "/".join(line.split(",")[0].split("/")[:-1])
            )
        ):
            os.makedirs(
                os.path.join(
                    DATASET_DIR, "depth", "/".join(line.split(",")[0].split("/")[:-1])
                )
            )

        np.save(
            os.path.join(
                DATASET_DIR, "depth", line.split(",")[0].replace("jpg", "npy")
            ),
            depth,
        )

    # img = Image.open(os.path.join(DATASET_DIR, "imgs", lines[i].split(" ")[0]))
    # depth = depth_estimator(img)["depth"]


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

    prep_annotations(filename="train_annotations_release.txt", dtype="train")
    prep_annotations(filename="test_annotations_release.txt", dtype="test")
    # make_depth_image(
    #     os.path.join(DATASET_DIR, "train_annotations_release.txt"), depth_estimator
    # )
    # make_depth_image(
    #     os.path.join(DATASET_DIR, "test_annotations_release.txt"), depth_estimator
    # )


if __name__ == "__main__":
    main()
