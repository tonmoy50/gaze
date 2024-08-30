import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from PIL import Image
from transformers import pipeline


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CUR_DIR))))
    ),
    "Data",
    "gazefollow_extended",
)

# DATASET_DIR = "/nfs/magnezone/data/ssd/nhaldert/datasets/Gazefollow/gazefollow_extended"


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

        if (
            in_or_out_ != "-1"
            and head_bbox_x_max_ > head_bbox_x_min_
            and head_bbox_y_max_ > head_bbox_y_min_
        ):

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


def test_headcrop():

    test_list = [
        # [
        #     "train/00000053/00053478.jpg",
        #     334.6043149260193,
        #     -3.2492913909010497,
        #     406.41575785824045,
        #     105.30288978571232,
        # ],
        # [
        #     "train/00000024/00024518.jpg",
        #     206.63296000000003,
        #     -3.552713678800501e-15,
        #     344.1664,
        #     218.284,
        # ],
        # [
        #     "train/00000111/00111384.jpg",
        #     172.78807283893175,
        #     160.73920078834607,
        #     276.6928146192028,
        #     292.68173003313467,
        # ],
        # [
        #     "train/00000038/00038112.jpg",
        #     81.46813991357931,
        #     101.52798406271853,
        #     123.81813991357933,
        #     152.34798406271852,
        # ],
        # [
        #     "train/00000083/00083456.jpg",
        #     429.442,
        #     151.38,
        #     475.17999999999995,
        #     224.22199999999998,
        # ],
        # [
        #     "train/00000024/00024259.jpg",
        #     622.1757838326795,
        #     138.1329372443677,
        #     663.2025082954099,
        #     183.55538218524765,
        # ],
        # ["train/00000049/00049697.jpg", 594.442, 137.44, 640.1800000000001, 195.036],
        # [
        #     "train/00000067/00067778.jpg",
        #     226.57225359179643,
        #     174.35437102585865,
        #     305.3418637953064,
        #     274.1292106169712,
        # ],
        # [
        #     "train/00000048/00048806.jpg",
        #     64.56101027580678,
        #     237.7800747468525,
        #     230.50115141198677,
        #     435.3881817487158,
        # ],
        # [
        #     "train/00000018/00018326.jpg",
        #     367.3475085725949,
        #     0.0,
        #     636.4665655628237,
        #     321.1684570235038,
        # ],
        # [
        #     "train/00000062/00062054.jpg",
        #     99.97965811082281,
        #     153.5686002849138,
        #     174.51565811082287,
        #     255.2086002849138,
        # ],
        # [
        #     "train/00000036/00036308.jpg",
        #     178.33949406261866,
        #     56.67741134082102,
        #     409.08438511459406,
        #     384.23834059642246,
        # ],
        # ["train/00000027/00027158.jpg", 205.9264, 117.348, 316.20096, 281.3272],
        # [
        #     "train/00000024/00024523.jpg",
        #     137.4626358866377,
        #     130.91317863027038,
        #     220.47831588663774,
        #     234.99253863027036,
        # ],
        # [
        #     "train/00000101/00101818.jpg",
        #     328.23275974121384,
        #     90.55761020436522,
        #     397.83227273712055,
        #     168.06615876798858,
        # ],
        # [
        #     "train/00000043/00043884.jpg",
        #     0.0,
        #     0.0,
        #     266.67159813606656,
        #     310.2782745608322,
        # ],
        # [
        #     "train/00000000/00000936.jpg",
        #     30.479999999999997,
        #     216.04,
        #     77.91199999999999,
        #     290.576,
        # ],
        # [
        #     "train/00000085/00085008.jpg",
        #     72.43416985065103,
        #     0.0,
        #     419.2177801452449,
        #     505.5335293224901,
        # ],
        # ["train/00000098/00098697.jpg", 74, 178, 97, 210],
        ["train/00000098/00098697.jpg", 74, 97, 178, 210],
    ]

    # sel = 0
    for sel in range(len(test_list)):
        # img = Image.open(os.path.join(DATASET_DIR, "train/00000053/00053478.jpg"))
        img = Image.open(os.path.join(DATASET_DIR, test_list[sel][0]))
        # Image._show(img)

        # cropped_head = img.crop(
        #     (334.6043149260193, -3.2492913909010497, 406.41575785824045, 105.30288978571232)
        # )
        cropped_head = img.crop(
            (
                int(test_list[sel][1]),
                int(test_list[sel][2]),
                int(test_list[sel][3]),
                int(test_list[sel][4]),
            )
        )
        Image._show(cropped_head)


def main():
    # checkpoint = "vinvino02/glpn-nyu"
    # checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
    # depth_estimator = pipeline(
    #     "depth-estimation",
    #     model=checkpoint,
    #     device=0,
    #     features=256,
    #     out_channels=[256, 512, 1024, 1024],
    # )

    # prep_annotations(filename="train_annotations_release.txt", dtype="train")
    # prep_annotations(filename="test_annotations_release.txt", dtype="test")
    # make_depth_image(
    #     os.path.join(DATASET_DIR, "train_annotations_release.txt"), depth_estimator
    # )
    # make_depth_image(
    #     os.path.join(DATASET_DIR, "test_annotations_release.txt"), depth_estimator
    # )

    test_headcrop()


if __name__ == "__main__":
    main()
