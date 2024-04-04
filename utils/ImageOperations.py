import numpy as np
import torch
import torchvision.transforms.functional as functional


def read_rgb_bands(sat_image):
    blue_band = sat_image[:, :, 1]
    green_band = sat_image[:, :, 2]
    red_band = sat_image[:, :, 3]

    rgb_image = np.zeros((sat_image.shape[0], sat_image.shape[1], 3))
    rgb_image[:, :, 0] = red_band
    rgb_image[:, :, 1] = green_band
    rgb_image[:, :, 2] = blue_band
    return rgb_image


def apply_padding(image_tensor, image_shape):
    n, m = image_tensor.shape[:2]
    on, om = image_shape[:2]
    pad_n = on - n
    pad_m = om - m

    pad_top = pad_n // 2
    pad_bottom = pad_n - pad_top
    pad_left = pad_m // 2
    pad_right = pad_m - pad_left

    padded_image_tensor = torch.nn.functional.pad(image_tensor, (0, 0, pad_left, pad_right, pad_top, pad_bottom))

    return padded_image_tensor


def trim(image_tensor, objective_shape):
    objective_width, objective_height = objective_shape
    width, height = image_tensor.shape[:2]

    x_cut = int((width - objective_width) / 2)
    y_cut = int((height - objective_height) / 2)

    return functional.crop(image_tensor, x_cut, y_cut, y_cut, x_cut)


def downscale(image_tensor, objective_shape):
    downscaled_image = functional.resize(image_tensor, objective_shape)
    return downscaled_image

