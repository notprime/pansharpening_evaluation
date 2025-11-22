"""
Utility functions
"""

import numpy as np
from math import ceil, log2
from skimage.transform.integral import integral_image as integral


def local_cross_correlation(img_1, img_2, half_width):
    """
        Cross-Correlation Field computation.
        Parameters
        ----------
        img_1 : Numpy Array
            First image on which calculate the cross-correlation. Dimensions: H, W
        img_2 : Numpy Array
            Second image on which calculate the cross-correlation. Dimensions: H, W
        half_width : int
            The semi-size of the window on which calculate the cross-correlation
        Return
        ------
        L : Numpy array
            The cross-correlation map between img_1 and img_2
    """

    w = int(half_width)
    ep = 1e-20

    if (len(img_1.shape)) != 3:
        img_1 = np.expand_dims(img_1, axis = -1)
    if (len(img_2.shape)) != 3:
        img_2 = np.expand_dims(img_2, axis = -1)

    img_1 = np.pad(img_1.astype(np.float64), ((w, w), (w, w), (0, 0)))
    img_2 = np.pad(img_2.astype(np.float64), ((w, w), (w, w), (0, 0)))

    img_1_cum = np.zeros(img_1.shape)
    img_2_cum = np.zeros(img_2.shape)
    for i in range(img_1.shape[-1]):
        img_1_cum[:, :, i] = integral(img_1[:, :, i]).astype(np.float64)
    for i in range(img_2.shape[-1]):
        img_2_cum[:, :, i] = integral(img_2[:, :, i]).astype(np.float64)

    img_1_mu = (img_1_cum[2 * w:, 2 * w:, :] - img_1_cum[:-2 * w, 2 * w:, :] - img_1_cum[2 * w:, :-2 * w, :]
                + img_1_cum[:-2 * w, :-2 * w, :]) / (4 * w ** 2)
    img_2_mu = (img_2_cum[2 * w:, 2 * w:, :] - img_2_cum[:-2 * w, 2 * w:, :] - img_2_cum[2 * w:, :-2 * w, :]
                + img_2_cum[:-2 * w, :-2 * w, :]) / (4 * w ** 2)

    img_1 = img_1[w:-w, w:-w, :] - img_1_mu
    img_2 = img_2[w:-w, w:-w, :] - img_2_mu

    img_1 = np.pad(img_1.astype(np.float64), ((w, w), (w, w), (0, 0)))
    img_2 = np.pad(img_2.astype(np.float64), ((w, w), (w, w), (0, 0)))

    i2 = img_1 ** 2
    j2 = img_2 ** 2
    ij = img_1 * img_2

    i2_cum = np.zeros(i2.shape)
    j2_cum = np.zeros(j2.shape)
    ij_cum = np.zeros(ij.shape)

    for i in range(i2_cum.shape[-1]):
        i2_cum[:, :, i] = integral(i2[:, :, i]).astype(np.float64)
    for i in range(j2_cum.shape[-1]):
        j2_cum[:, :, i] = integral(j2[:, :, i]).astype(np.float64)
    for i in range(ij_cum.shape[-1]):
        ij_cum[:, :, i] = integral(ij[:, :, i]).astype(np.float64)

    sig2_ij_tot = (ij_cum[2 * w:, 2 * w:, :] - ij_cum[:-2 * w, 2 * w:, :] - ij_cum[2 * w:, :-2 * w, :]
                   + ij_cum[:-2 * w, :-2 * w, :])
    sig2_ii_tot = (i2_cum[2 * w:, 2 * w:, :] - i2_cum[:-2 * w, 2 * w:, :] - i2_cum[2 * w:, :-2 * w, :]
                   + i2_cum[:-2 * w, :-2 * w, :])
    sig2_jj_tot = (j2_cum[2 * w:, 2 * w:, :] - j2_cum[:-2 * w, 2 * w:, :] - j2_cum[2 * w:, :-2 * w, :]
                   + j2_cum[:-2 * w, :-2 * w, :])

    sig2_ij_tot = np.clip(sig2_ij_tot, ep, sig2_ij_tot.max())
    sig2_ii_tot = np.clip(sig2_ii_tot, ep, sig2_ii_tot.max())
    sig2_jj_tot = np.clip(sig2_jj_tot, ep, sig2_jj_tot.max())

    xcorr = sig2_ij_tot / ((sig2_ii_tot * sig2_jj_tot) ** 0.5 + ep)

    return xcorr


def normalize_block(im):
    """
    Auxiliary function, normalize image block for Q2n computation.

    Parameters:
        im (np.ndarray): Image block. Shape: (H, W)

    Returns:
        tuple: (y, m, s), where:
                y (np.ndarray): normalized im
                m (float): mean of im
                s (float): standard deviation of im
        
    """

    m = np.mean(im)
    s = np.std(im, ddof = 1)

    if s == 0:
        s = 1e-10

    y = ((im - m) / s) + 1

    return y, m, s


def cayley_dickson_property_1d(onion1, onion2):
    """
    Cayley-Dickson construction for 1-D arrays.
    Auxiliary function, used in hypercomplex Q2n calculation.

    Parameters:
        onion1 (np.ndarray): First 1-D array
        onion2 (np.ndarray): Second 1-D array
    
    Returns:
        ris (np.ndarray): Result of Cayley-Dickson construction

    """

    n = len(onion1)

    if n > 1:
        half_pos = n // 2
        a = onion1[:half_pos]
        b = onion1[half_pos:]

        neg = np.ones(b.shape)
        neg[1:] = -1

        b = b * neg
        c = onion2[:half_pos]
        d = onion2[half_pos:]
        d = d * neg

        if n == 2:
            ris = np.concatenate([(a * c) - (d * b), (a * d) + (c * b)])
        else:
            ris1 = cayley_dickson_property_1d(a, c)

            ris2 = cayley_dickson_property_1d(d, b * neg)
            ris3 = cayley_dickson_property_1d(a * neg, d)
            ris4 = cayley_dickson_property_1d(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = np.concatenate([aux1, aux2])
    else:
        ris = onion1 * onion2

    return ris


def cayley_dickson_property_2d(onion1, onion2):
    """
    Cayley-Dickson construction for 2-D arrays.
    Auxiliary function, used in hypercomplex Q2n calculation.

    Parameters:
        onion1 (np.ndarray): First multispectral image. Shape: (H, W, C), with C = Bands
        onion2 (np.ndarray): Second multispectral image. Shape: (H, W, C), with C = Bands
    
    Returns:
        ris (np.ndarray): Result of Cayley-Dickson construction

    """

    dim3 = onion1.shape[-1]
    if dim3 > 1:
        half_pos = dim3 // 2

        a = onion1[:, :, :half_pos]
        b = onion1[:, :, half_pos:]
        b = np.concatenate([np.expand_dims(b[:, :, 0], -1), -b[:, :, 1:]], axis = -1)

        c = onion2[:, :, :half_pos]
        d = onion2[:, :, half_pos:]
        d = np.concatenate([np.expand_dims(d[:, :, 0], -1), -d[:, :, 1:]], axis = -1)

        if dim3 == 2:
            ris = np.concatenate([(a * c) - (d * b), (a * d) + (c * b)], axis = -1)
        else:
            ris1 = cayley_dickson_property_2d(a, c)
            ris2 = cayley_dickson_property_2d(d,
                                              np.concatenate([np.expand_dims(b[:, :, 0], -1), -b[:, :, 1:]], axis = -1))
            ris3 = cayley_dickson_property_2d(np.concatenate([np.expand_dims(a[:, :, 0], -1), -a[:, :, 1:]], axis = -1),
                                              d)
            ris4 = cayley_dickson_property_2d(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4

            ris = np.concatenate([aux1, aux2], axis = -1)
    else:
        ris = onion1 * onion2

    return ris


def pad_to_power_of_2_channels(img):
    """
    Pad the channel dimension to the nearest power of 2.
    Required for Cayley-Dickson construction in Q2n,
    as Q2n hypercomplex math requires power-of-2 channels
    
    Parameters:
        img (np.ndarray): Image. Shape: (H, W, C)
        
    Returns:
        img (np.ndarray): Padded image. Shape: (H, W, C') where C' is power of 2
    """
    height, width, depth = img.shape
    
    if ceil(log2(depth)) - log2(depth) != 0:
        exp_difference = 2 ** ceil(log2(depth)) - depth
        diff_zeros = np.zeros((height, width, exp_difference), dtype=img.dtype)
        img = np.concatenate([img, diff_zeros], axis=-1)
    
    return img


def pad_for_sliding_window(img, block_size, shift):
    """
    Pad image to ensure complete coverage by sliding windows.
    
    Parameters:
        img (np.ndarray): Image. Shape: (H, W, C)
        block_size (int): Window size
        shift (int): Stride
        
    Returns:
        img (np.ndarray): Padded image
    """
    height, width, depth = img.shape
    
    stepx = ceil(height / shift)
    stepy = ceil(width / shift)
    
    est1 = (stepx - 1) * shift + block_size - height
    est2 = (stepy - 1) * shift + block_size - width
    
    if est1 > 0 or est2 > 0:
        pad_h = max(0, est1)
        pad_w = max(0, est2)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    
    return img


def preprocess_for_metrics(sharp, pan, hs, ratio=6):
    """
    Preprocess images to ensure dimension compatibility.
    This function expects the channel dimension to be the last one.
    
    This ensures that:
    1. HR dimensions are divisible by ratio
    2. HR dimensions / ratio == LR dimensions
    
    Args:
        sharp: Sharpened image (H, W, C)
        pan: Panchromatic (H, W, 1)
        hs: Hyperspectral (H/ratio, W/ratio, C)
        ratio: Resolution ratio
        
    Returns:
        tuple: (sharp_padded, pan_padded, hs_padded)
    """
    
    hr_h, hr_w = sharp.shape[:2]
    lr_h, lr_w = hs.shape[:2]
    
    # Step 1: Ensure HR dimensions are divisible by ratio
    # This is CRITICAL for D_s metric
    pad_h = (ratio - (hr_h % ratio)) % ratio
    pad_w = (ratio - (hr_w % ratio)) % ratio
    
    if pad_h > 0 or pad_w > 0:
        sharp = np.pad(sharp, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        pan = np.pad(pan, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    
    # Step 2: Ensure LR dimensions match HR/ratio
    expected_lr_h = sharp.shape[0] // ratio
    expected_lr_w = sharp.shape[1] // ratio
    
    lr_pad_h = expected_lr_h - lr_h
    lr_pad_w = expected_lr_w - lr_w
    
    if lr_pad_h > 0 or lr_pad_w > 0:
        hs = np.pad(hs, ((0, lr_pad_h), (0, lr_pad_w), (0, 0)), mode='edge')
    
    # Verify alignment
    assert sharp.shape[0] == pan.shape[0]
    assert sharp.shape[1] == pan.shape[1]
    assert sharp.shape[0] // ratio == hs.shape[0]
    assert sharp.shape[1] // ratio == hs.shape[1]
    
    return sharp, pan, hs