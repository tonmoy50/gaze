import torch

import numpy as np
import os
from PIL import Image

import skimage
import cv2


CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def gen_matrix_T(rgbimg, depthimg, eye_X, eye_Y, eye_Z):
    final_width, final_height = rgbimg.shape[0], rgbimg.shape[1]
    input_size = 224
    offset_x, offset_y = 0, 0

    # Generate the matrix_T
    depthmap = np.resize(depthimg, (input_size, input_size))
    # depthmap = depthimg.resize((input_size, input_size), Image.BICUBIC)
    depthmap = np.array(depthmap)

    # scale proportionally
    scale_width, scale_height = final_width / input_size, final_height / input_size

    # construct empty matrix
    matrix_T_DW = np.linspace(0, input_size - 1, input_size)
    matrix_T_DH = np.linspace(0, input_size - 1, input_size)
    [matrix_T_xx, matrix_T_yy] = np.meshgrid(matrix_T_DW, matrix_T_DH)

    # construct matrix_T according to Eq 3. in paper
    fx, fy, cx, cy = 910.78759766, 910.21258545, 961.65966797, 554.11016846

    matrix_T_X = (matrix_T_xx * scale_width + offset_x - cx) * depthmap / fx
    matrix_T_Y = (matrix_T_yy * scale_height + offset_y - cy) * depthmap / fy
    matrix_T_Z = depthmap

    matrix_T = np.dstack((matrix_T_X, matrix_T_Y, matrix_T_Z))
    matrix_T = matrix_T.reshape([-1, 3])
    matrix_T = matrix_T.reshape([input_size, input_size, 3])

    matrix_T = matrix_T - np.array([eye_X, eye_Y, eye_Z])

    norm_value = np.linalg.norm(matrix_T, axis=2, keepdims=True)
    norm_value[norm_value <= 0] = 1

    matrix_T = matrix_T / norm_value

    # convert it to tensor
    matrix_T = torch.from_numpy(matrix_T).float()

    return matrix_T


def get_fov(matrix_T, gazevector, simg_shape, alpha=3):
    relu = torch.nn.ReLU()
    bs = matrix_T.shape[0]
    h, w = simg_shape[2:]

    # gazevector=gazevector.unsqueeze(2)

    matrix_T = matrix_T.reshape([bs, -1, 3])
    # print(matrix_T)

    F = torch.matmul(matrix_T, gazevector)
    # F = F.reshape([bs, 1, h, w])

    F_alpha = relu(F)
    F_alpha = torch.pow(F_alpha, alpha)

    SFoVheatmap = [F, F_alpha]
    return SFoVheatmap


def view_depthimg(img):
    # stretch to full dynamic range
    stretch = skimage.exposure.rescale_intensity(
        img, in_range="image", out_range=(0, 255)
    ).astype(np.uint8)

    # convert to 3 channels
    stretch = cv2.merge([stretch, stretch, stretch])

    # define colors
    color1 = (0, 0, 255)  # red
    color2 = (0, 165, 255)  # orange
    color3 = (0, 255, 255)  # yellow
    color4 = (255, 255, 0)  # cyan
    color5 = (255, 0, 0)  # blue
    color6 = (128, 64, 64)  # violet
    colorArr = np.array(
        [[color1, color2, color3, color4, color5, color6]], dtype=np.uint8
    )

    # resize lut to 256 (or more) values
    lut = cv2.resize(colorArr, (256, 1), interpolation=cv2.INTER_LINEAR)

    # apply lut
    result = cv2.LUT(stretch, lut)

    # create gradient image
    grad = np.linspace(0, 255, 512, dtype=np.uint8)
    grad = np.tile(grad, (20, 1))
    grad = cv2.merge([grad, grad, grad])

    grad_img = Image.fromarray(stretch)
    Image._show(grad_img)
    final_img = Image.fromarray(result)
    Image._show(final_img)


def main():
    rgbimg = np.array(Image.open(os.path.join(CUR_DIR, "0000.jpg")))

    depthimg = np.load(os.path.join(CUR_DIR, "0000.npy"))
    depthimg[np.isnan(depthimg)] = 0
    depthimg = depthimg.astype(np.float32)
    depthimg = np.array(depthimg)
    # depthimg = Image.fromarray(depthimg)
    # Image._show(depthimg)

    eye_X, eye_Y, eye_Z = 0.71, -0.192, 1.675
    matrix_T = gen_matrix_T(rgbimg, depthimg, eye_X, eye_Y, eye_Z)
    f, f_alpha = get_fov(matrix_T, torch.tensor([0.1, 0.2, 0.3]), (1, 1, 224, 224))

    # view_depthimg(f_alpha.numpy())
    view_depthimg(depthimg)


if __name__ == "__main__":
    main()
