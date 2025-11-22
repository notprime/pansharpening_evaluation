import numpy as np
import scipy.ndimage as ft
from math import ceil, floor

from .utils import (pad_for_sliding_window,
                    pad_to_power_of_2_channels,
                    normalize_block,
                    cayley_dickson_property_1d,
                    cayley_dickson_property_2d)


def Q(outputs, labels, block_size = 32):
    """
    Universal Image Quality Index (UIQI).
    Measures the similarity between two images using correlation, luminance,
    and contrast comparison within sliding windows. ---> TO CHECK

    Parameters:
        outputs (np.ndarray): First image. Shape: (H, W, C)
        labels (np.ndarray): Second image. Shape: (H, W, C)
        block_size (int): Window size for local statistics. Default: 32

    Returns:
        float: Q index (higher is better, max=1)
    
    References:
        [Wang02] Z. Wang and A. C. Bovik, "A universal image quality index,"
                 IEEE Signal Processing Letters, vol. 9, no. 3, pp. 81-84, March 2002.
    """

    N = block_size ** 2
    nbands = labels.shape[-1]
    kernel = np.ones((block_size, block_size))
    pad_size = floor((kernel.shape[0] - 1) / 2)
    outputs_sq = outputs ** 2
    labels_sq = labels ** 2
    outputs_labels = outputs * labels

    quality = np.zeros(nbands)

    for i in range(nbands):
        outputs_sum = ft.convolve(outputs[:, :, i], kernel)
        labels_sum = ft.convolve(labels[:, :, i], kernel)

        outputs_sq_sum = ft.convolve(outputs_sq[:, :, i], kernel)
        labels_sq_sum = ft.convolve(labels_sq[:, :, i], kernel)
        outputs_labels_sum = ft.convolve(outputs_labels[:, :, i], kernel)
        outputs_sum = outputs_sum[pad_size:-pad_size, pad_size:-pad_size]
        labels_sum = labels_sum[pad_size:-pad_size, pad_size:-pad_size]

        outputs_sq_sum = outputs_sq_sum[pad_size:-pad_size, pad_size:-pad_size]
        labels_sq_sum = labels_sq_sum[pad_size:-pad_size, pad_size:-pad_size]
        outputs_labels_sum = outputs_labels_sum[pad_size:-pad_size, pad_size:-pad_size]

        outputs_labels_sum_mul = outputs_sum * labels_sum
        outputs_labels_sum_mul_sq = outputs_sum ** 2 + labels_sum ** 2

        numerator = 4 * (N * outputs_labels_sum - outputs_labels_sum_mul) * outputs_labels_sum_mul
        denominator_temp = N * (outputs_sq_sum + labels_sq_sum) - outputs_labels_sum_mul_sq
        denominator = denominator_temp * outputs_labels_sum_mul_sq

        index = (denominator_temp == 0) & (outputs_labels_sum_mul_sq != 0)

        quality_map = np.ones(denominator.shape)
        quality_map[index] = 2 * outputs_labels_sum_mul[index] / outputs_labels_sum_mul_sq[index]
        index = denominator != 0
        quality_map[index] = numerator[index] / denominator[index]
        quality[i] = np.mean(quality_map)

    return np.mean(quality).item()


def q_index_metric(im1, im2, size):
    """
        Q2n calculation on a window of dimension (size, size).
        Auxiliary function for Q2n calculation.

        Parameters:
            im1 (np.ndarray): First image patch. Shape: (size, size, C)
            im2 (np.ndarray): Second image patch. Shape: (size, size, C)
            size (int): Window size

        Returns:
            q (np.ndarray): Q2n value for this window
    """

    im1 = im1.astype(np.double)
    im2 = im2.astype(np.double)
    im2 = np.concatenate([np.expand_dims(im2[:, :, 0], -1), -im2[:, :, 1:]], axis = -1)

    depth = im1.shape[-1]

    for i in range(depth):
        im1[:, :, i], m, s = normalize_block(im1[:, :, i])
        if m == 0:
            if i == 0:
                im2[:, :, i] = im2[:, :, i] - m + 1
            else:
                im2[:, :, i] = -(-im2[:, :, i] - m + 1)
        else:
            if i == 0:
                im2[:, :, i] = ((im2[:, :, i] - m) / s) + 1
            else:
                im2[:, :, i] = -(((-im2[:, :, i] - m) / s) + 1)

    m1 = np.mean(im1, axis = (0, 1))
    m2 = np.mean(im2, axis = (0, 1))

    mod_q1m = np.sqrt(np.sum(m1 ** 2))
    mod_q2m = np.sqrt(np.sum(m2 ** 2))

    mod_q1 = np.sqrt(np.sum(im1 ** 2, axis = -1))
    mod_q2 = np.sqrt(np.sum(im2 ** 2, axis = -1))

    term2 = mod_q1m * mod_q2m
    term4 = mod_q1m ** 2 + mod_q2m ** 2
    temp = (size ** 2) / (size ** 2 - 1)
    int1 = temp * np.mean(mod_q1 ** 2)
    int2 = temp * np.mean(mod_q2 ** 2)
    int3 = temp * (mod_q1m ** 2 + mod_q2m ** 2)
    term3 = int1 + int2 - int3

    mean_bias = 2 * term2 / term4

    if term3 == 0:
        q = np.zeros((1, 1, depth), dtype = 'float64')
        q[:, :, -1] = mean_bias
    else:
        cbm = 2 / term3
        qu = cayley_dickson_property_2d(im1, im2)
        qm = cayley_dickson_property_1d(m1, m2)

        qv = temp * np.mean(qu, axis = (0, 1))
        q = qv - temp * qm
        q = q * mean_bias * cbm

    return q


def Q2n_map(outputs, labels, q_block_size = 32, q_shift = 32):
    """
        Compute Q2n map using sliding windows.

        Parameters:
            outputs (np.ndarray): First image. Shape: (H, W, C)
            labels (np.ndarray): Second image. Shape: (H, W, C)
            q_block_size (int): Window size. Default: 32
            q_shift (int): Stride. Default: 32

        Returns:
            q2n_index_map (np.ndarray): Q2n map. Shape: (n_windows_h, n_windows_w)
    """

    height, width, depth = labels.shape
    stepx = ceil(height / q_shift)
    stepy = ceil(width / q_shift)

    values = np.zeros((stepx, stepy, depth))

    for j in range(stepx):
        for i in range(stepy):
            values[j, i, :]  = q_index_metric(
                labels[j * q_shift:j * q_shift + q_block_size, i * q_shift: i * q_shift + q_block_size, :],
                outputs[j * q_shift:j * q_shift + q_block_size, i * q_shift: i * q_shift + q_block_size, :],
                q_block_size)

    q2n_index_map = np.sqrt(np.sum(values ** 2, axis = -1))

    return q2n_index_map