from fov_module import view_depthimg

from PIL import Image
import numpy as np
import os

import torch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from torch import nn


CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def argmax_pts(heatmap):

    idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    pred_y, pred_x = map(float, idx)

    return pred_x, pred_y


def strategy3dGazeFollowing(depthmap, pred_gh, pred_gv, eye_3d, campara, ratio=0.1):

    img_H, img_W = depthmap.shape

    # get the center of 2d proposal area
    output_h, output_w = pred_gh.shape

    pred_center = list(argmax_pts(pred_gh))  # From img_utils.py of GFIE
    pred_gazetarget_2d = np.array(
        [pred_center[0] / output_w, pred_center[1] / output_h]
    )

    pred_center[0] = pred_center[0] * img_W / output_w
    pred_center[1] = pred_center[1] * img_H / output_h

    # get the proposal rectangle area
    pu_min = pred_center[0] - img_W * ratio / 2
    pu_max = pred_center[0] + img_W * ratio / 2

    pv_min = pred_center[1] - img_H * ratio / 2
    pv_max = pred_center[1] + img_H * ratio / 2

    if pu_min < 0:
        pu_min, pu_max = 0, img_W * ratio
    elif pu_max > img_W:
        pu_max, pu_min = img_W, img_W - img_W * ratio

    if pv_min < 0:
        pv_min, pv_max = 0, img_H * ratio
    elif pv_max > img_H:
        pv_max, pv_min = img_H, img_H - img_H * ratio

    pu_min, pu_max, pv_min, pv_max = map(int, [pu_min, pu_max, pv_min, pv_max])

    # unproject to 3d proposal area
    range_depthmap = depthmap[pv_min:pv_max, pu_min:pu_max]
    fx, fy, cx, cy = campara

    range_space_DW = np.linspace(pu_min, pu_max - 1, pu_max - pu_min)
    range_space_DH = np.linspace(pv_min, pv_max - 1, pv_max - pv_min)
    # print(range_space_DW, range_space_DH)
    [range_space_xx, range_space_yy] = np.meshgrid(range_space_DW, range_space_DH)

    range_space_X = (range_space_xx - cx) * range_depthmap / fx
    range_space_Y = (range_space_yy - cy) * range_depthmap / fy
    range_space_Z = range_depthmap
    # print(range_space_X, range_space_Y, range_space_Z)

    proposal_3d = np.dstack([range_space_X, range_space_Y, range_space_Z])
    print(proposal_3d)

    matrix_T = proposal_3d - eye_3d

    norm_value = np.linalg.norm(matrix_T, axis=2, keepdims=True)
    norm_value[norm_value <= 0] = 1
    matrix_T = matrix_T / norm_value

    # filter out the invalid depth
    matrix_T[range_depthmap == 0] = 0

    # find the
    gaze_vector_similar_set = np.dot(matrix_T, pred_gv)

    max_index_u, max_index_v = argmax_pts(gaze_vector_similar_set)

    pred_gazetarget_3d = proposal_3d[int(max_index_v), int(max_index_u)]

    pred_gazevector = matrix_T[int(max_index_v), int(max_index_u)]

    pred_gazetarget_3d = np.array(pred_gazetarget_3d).reshape(-1, 3)
    pred_gazetarget_2d = np.array(pred_gazetarget_2d).reshape(-1, 2)
    pred_gazevector = pred_gazevector.reshape(-1, 3)

    return {
        "pred_gazetarget_3d": pred_gazetarget_3d,
        "pred_gazetarget_2d": pred_gazetarget_2d,
        "pred_gazevector": pred_gazevector,
    }


def to_numpy(tensor):
    """tensor to numpy"""
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def draw_labelmap(img, pt, sigma, type="Gaussian"):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == "Gaussian":
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    elif type == "Cauchy":
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma**2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0] : img_y[1], img_x[0] : img_x[1]] += g[g_y[0] : g_y[1], g_x[0] : g_x[1]]
    if np.max(img) != 0:
        img = img / np.max(img)  # normalize heatmap so it has max value of 1
    return to_torch(img)


def main():
    depthimg = np.load(os.path.join(CUR_DIR, "0000.npy"))
    depthimg[np.isnan(depthimg)] = 0
    # depthimg = depthimg.astype(np.float32)
    # depthimg = np.array(depthimg)
    # print(depthimg)
    grayscale_img, heatmap_img = view_depthimg(depthimg)
    # Image._show(heatmap_img)
    # print(heatmap_img.size, type(heatmap_img))
    fx, fy, cx, cy = 910.78759766, 910.21258545, 961.65966797, 554.11016846
    eye_u, eye_v = 1348.0, 455.0
    eye_X, eye_Y, eye_Z = 0.71, -0.192, 1.675
    gaze_u, gaze_v = 1313.0, 834.0
    gaze_X, gaze_Y, gaze_Z = 0.624, 0.497, 1.617
    gaze_vector = np.array([gaze_X - eye_X, gaze_Y - eye_Y, gaze_Z - eye_Z])

    print(depthimg[455][1348])

    X, Y, Z = 0.71, -0.192, 1.673

    # Create a new figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the 3D point
    ax.scatter(X, Y, Z, c="r", marker="o")

    # Set labels
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    # Set title
    ax.set_title("3D Point Visualization")

    # Set limits for better visualization
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 20)

    # Show plot
    plt.show()

    # result = strategy3dGazeFollowing(
    #     np.array(depthimg),
    #     np.array(heatmap_img).reshape(
    #         np.array(heatmap_img).shape[0],
    #         np.array(heatmap_img).shape[1] * np.array(heatmap_img).shape[2],
    #     ),
    #     gaze_vector,
    #     np.array([0.71, -0.192, 1.675]),
    #     [fx, fy, cx, cy],
    # )

    # gaze_vector = np.array([gaze_X - eye_X, gaze_Y - eye_Y, gaze_Z - eye_Z])
    # print(gaze_vector)
    # norm_gaze_vector = (
    #     1.0 if np.linalg.norm(gaze_vector) <= 0.0 else np.linalg.norm(gaze_vector)
    # )
    # print(norm_gaze_vector)
    # gaze_vector = gaze_vector / norm_gaze_vector
    # print(gaze_vector)
    # gaze_vector = torch.from_numpy(gaze_vector)

    # # # generate the heat map label
    # gaze_heatmap = torch.zeros(224, 224)  # set the size of the output

    # rgb_img = Image.open(os.path.join(CUR_DIR, "0000.jpg"))

    # gaze_heatmap = draw_labelmap(
    #     # np.array(rgb_img),
    #     gaze_heatmap,
    #     [gaze_u * 224, gaze_v * 224],
    #     3,
    #     type="Gaussian",
    # )

    # Image._show(Image.fromarray(to_numpy(gaze_heatmap)))


if __name__ == "__main__":
    main()
