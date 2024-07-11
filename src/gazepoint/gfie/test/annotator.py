import cv2
from PIL import Image, ImageDraw
import numpy as np

import os


CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def draw_hbbox(img, hbbox):

    ImageDraw.Draw(img).rectangle(hbbox, outline="red", width=3)
    return img


def draw_point(img, start, end):
    ImageDraw.Draw(img).point(
        (start, end),
        fill="yellow",
    )
    return img


def draw_circle(img, center, radius):
    ImageDraw.Draw(img).point(center, fill="blue")
    ImageDraw.Draw(img).ellipse(
        (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ),
        outline="blue",
        width=3,
    )
    return img


def draw_3d_point(img, point):
    point[0] = point[0]
    point[1] = point[1]
    point[2] = point[2]

    camera_param = {
        "fx": 910.78759766,
        "fy": 910.21258545,
        "cx": 961.65966797,
        "cy": 554.11016846,
    }
    camera_mat = np.array(
        [
            [camera_param["fx"], 0, camera_param["cx"]],
            [0, camera_param["fy"], camera_param["cy"]],
            [0, 0, 1],
        ],
        np.float32,
    )
    # Distortion coefficients are five numbers that model the amount of radial and
    #  tangential distortion in an image. They are used to correct distortion in 2D patterns,
    #   set a detection window around the center of distortion, and generate straight lines.
    #    The coefficients can be obtained from camera calibration tools.
    dist_coeffs = np.zeros((5, 1), np.float32)

    # A vector quantity whose magnitude is proportional to the amount or speed of a rotation,
    # and whose direction is perpendicular to the plane of that rotation
    rotation_vec = np.array([0.0, 0.0, 0.0], np.float32)
    rotV, _ = cv2.Rodrigues(rotation_vec)
    # print(rotV)

    # A translation vector, therefore, is a vector that describes the movement of a point, body,
    # or system from one position to another, without rotation or deformation
    translation_vec = np.array([1.0, 0.0, 0.0], np.float32)

    point_3d = np.float32(
        [[point[0], 0, 0], [0, point[1], 0], [0, 0, point[2]], [0, 0, 0]]
    ).reshape(-1, 3)
    # print(point_3d)

    projected_points, _ = cv2.projectPoints(
        np.array([point_3d], np.float32),
        rotation_vec,
        translation_vec,
        camera_mat,
        dist_coeffs,
    )
    # print(projected_points.astype(int))

    img = cv2.line(
        np.array(img),
        projected_points[0][0].astype(int),
        projected_points[3][0].astype(int),
        (255, 0, 0),
        1,
    )
    img = cv2.line(
        np.array(img),
        projected_points[1][0].astype(int),
        projected_points[3][0].astype(int),
        (0, 255, 0),
        1,
    )
    img = cv2.line(
        np.array(img),
        projected_points[2][0].astype(int),
        projected_points[3][0].astype(int),
        (0, 255, 0),
        1,
    )

    return Image.fromarray(img)


def work_on_rgb_image():
    img_path = os.path.join(CUR_DIR, "0000.jpg")
    annotations = {
        "index": 1,
        "scene_id": 1,
        "frame_id": 1,
        # Head bounding box
        "h_x_min": 1309,
        "h_y_min": 331,
        "h_x_max": 1447,
        "h_y_max": 491,
        # 2D gaze point
        "gaze_u": 1313.0,
        "gaze_v": 834.0,
        # 3D gaze point
        "gaze_X": 0.624,
        "gaze_Y": 0.497,
        "gaze_Z": 1.617,
        # 2D eye point
        "eye_u": 1348,
        "eye_v": 450,
        # 3D eye point
        "eye_X": 0.71,
        "eye_Y": -0.192,
        "eye_Z": 1.675,
    }

    gaze_vec = (
        annotations["gaze_X"] - annotations["eye_X"],
        annotations["gaze_Y"] - annotations["eye_Y"],
        annotations["gaze_Z"] - annotations["eye_Z"],
    )
    # print(gaze_vec)

    img = Image.open(img_path)
    # Image._show(img)

    new_img = draw_hbbox(
        img,
        [
            annotations["h_x_min"],
            annotations["h_y_min"],
            annotations["h_x_max"],
            annotations["h_y_max"],
        ],
    )
    new_img = draw_circle(
        new_img,
        (int(annotations["gaze_u"]), int(annotations["gaze_v"])),
        10,
    )
    new_img = draw_circle(
        new_img,
        (int(annotations["eye_u"]), int(annotations["eye_v"])),
        10,
    )
    # new_img = draw_3d_point(
    #     new_img,
    #     [
    #         annotations["gaze_X"],
    #         annotations["gaze_Y"],
    #         annotations["gaze_Z"],
    #     ],
    # )
    new_img = draw_3d_point(
        new_img, [annotations["eye_X"], annotations["eye_Y"], annotations["eye_Z"]]
    )

    Image._show(new_img)


def work_on_depth_image():
    depth_img_path = os.path.join(CUR_DIR, "0193.npy")
    depth_img = np.load(depth_img_path)
    depth_img[np.isnan(depth_img)] = 0
    print(depth_img.shape)
    depth_img = depth_img.astype(np.float32)
    depth_img = Image.fromarray(depth_img)
    Image._show(depth_img)


def main():

    work_on_rgb_image()
    # work_on_depth_image()


if __name__ == "__main__":
    main()
