"""
Implementation of the HQNR framework
to compute quality metrics for pansharpening evaluation,
leveraging Dask for optimized parallel computation.

Main metrics:
- D_lambda_khan: spectral distortion index
- D_s: spatial distortion index
- HQNR: Hybrid Quality with No Reference, where HQNR = (1 - D_lambda_khan) * (1 - D_s)

Usage examples:
    python main.py sharp.tif pan.tif hs.tif
    python main.py sharp.tif pan.tif hs.tif --ratio 6 --output results.json
    python main.py sharp.tif pan.tif hs.tif --config config.yaml

References:
    [Musto24] Musto, Tricomi, Bruno, Pasquali, "Advancing Prisma Pansharpening: 
              A Deep Learning Approach with Synthetic Data Pretraining 
              and Transfer Learning", WHISPERS, 2023
    [Scarpa21] Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality 
               assessment for pansharpening.", arXiv preprint arXiv:2108.06144
    [Garzelli09] A. Garzelli and F. Nencini, "Hypercomplex quality assessment 
                 of multi/hyper-spectral images," IEEE GRSL, 2009.
    [Alparone08] L. Alparone et al., "Multispectral and Panchromatic Data 
                 Fusion Assessment Without Reference"
    [Vivone20] G. Vivone et al., "A New Benchmark Based on Recent Advances in 
               Multispectral Pansharpening", IEEE GRSM, 2020.

Author: Riccardo Musto
"""


#import paths
import numpy as np
import scipy.ndimage as ft
import rasterio
import dataset.syn_dataset_gen as data_gen
from pansharpening_metrics import Q2n
import multiprocessing as mp


from math import ceil, floor, log2, sqrt
from skimage.transform.integral import integral_image as integral

import torch
from torch import nn

import dask
import dask.array as da
from dask.distributed import LocalCluster, Client


class MetricsConfig:
    """
    Configuration parameters for pansharpening metrics computation.
    
    Attributes:
        q_block_size (int): Window size for Q2n/Q index calculation.
            Default: 32. Typical values: 16, 32, 64.
            Larger values = smoother results but less spatial detail.
            
        q_shift (int): Stride for sliding window in Q2n calculation.
            Default: 32 (non-overlapping windows).
            Use q_shift < q_block_size for overlapping windows.
            
        dask_chunk_size (tuple): Spatial chunk size for Dask parallelization.
            Default: (64, 64). Should be multiple of q_block_size.
            Larger chunks = less overhead but more memory per worker.
            For 32x32 blocks: (64, 64) processes 4 blocks at once.
            For 16x16 blocks: (64, 64) processes 16 blocks at once.
            
        n_workers (float): Fraction of CPU cores to use for Dask.
            Default: 0.9 (90% of available cores).
            Range: 0.1 to 1.0.
            
        exponent (int): Exponent for D_lambda and D_s calculation.
            Default: 1 (standard L1 norm).
            Higher values penalize larger differences more.
    """
    
    def __init__(self, 
                 q_block_size=32,
                 q_shift=32,
                 dask_chunk_size=(64, 64),
                 n_workers=0.9,
                 exponent=1):
        self.q_block_size = q_block_size
        self.q_shift = q_shift
        self.dask_chunk_size = dask_chunk_size
        self.n_workers = n_workers
        self.exponent = exponent
        
    def validate(self):
        """Validate configuration parameters."""
        assert self.q_block_size > 0, "q_block_size must be positive"
        assert self.q_shift > 0, "q_shift must be positive"
        assert all(c > 0 for c in self.dask_chunk_size), "chunk sizes must be positive"
        assert 0 < self.n_workers <= 1, "n_workers must be between 0 and 1"
        
        # Warning if chunk size is not a multiple of block size
        if any(c % self.q_block_size != 0 for c in self.dask_chunk_size):
            print(f"Warning: dask_chunk_size {self.dask_chunk_size} is not a multiple "
                  f"of q_block_size {self.q_block_size}. This may be inefficient.")


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


def pad_to_compute(outputs, labels, q_block_size = 32, q_shift = 32):
    height, width, depth = labels.shape
    stepx = ceil(height / q_shift)
    stepy = ceil(width / q_shift)

    if stepy <= 0:
        stepx = 1
        stepy = 1

    est1 = (stepx - 1) * q_shift + q_block_size - height
    est2 = (stepy - 1) * q_shift + q_block_size - width

    if (est1 != 0) and (est2 != 0):
        labels = np.pad(labels, ((0, est1), (0, est2), (0, 0)), mode='reflect')
        outputs = np.pad(outputs, ((0, est1), (0, est2), (0, 0)), mode='reflect')

        outputs = outputs.astype(outputs.dtype)
        labels = labels.astype(labels.dtype)

    height, width, depth = labels.shape

    if ceil(log2(depth)) - log2(depth) != 0:
        exp_difference = 2 ** (ceil(log2(depth))) - depth
        diff_zeros = np.zeros((height, width, exp_difference), dtype="float64")
        labels = np.concatenate([labels, diff_zeros], axis=-1)
        outputs = np.concatenate([outputs, diff_zeros], axis=-1)

    return outputs, labels

def D_lambda_khan(outputs, inputs, ratio = 6, sensor = 'PRISMA', config=None):
    """
    Compute Khan's Spectral Distortion Index using Q2n on image pairs.
    This metric measures spectral consistency by comparing the Q index 
    between all pairs of bands.

    Parameters:
        outputs (np.ndarray): Pansharpened image. Shape: (H, W, C)
        inputs (np.ndarray): Low-resolution MS image. Shape: (H, W, C)
        config (MetricsConfig): Configuration parameters. If None, uses defaults.

    Returns:
        d_lambda (float): D_lambda_khan index (lower is better, range [0, 1])
    """

    """if config is None:
        config = MetricsConfig() """


    # MTF-based low-pass filtering and decimation of the pansharpened image
    # aggiungere su repo github: bisogna modificare resize_images se si usano altri sensori!!! --> todo
    lpfd_outputs = data_gen.resize_images(img_ms=outputs,  # ---> TODO <---
                                          ratio=ratio, # <--- da gestire
                                          sensor=sensor) # <--- da gestire

    # Prepare data
    # Pad images for windowed computation
    lpfd_outputs = pad_for_sliding_window(lpfd_outputs,
                                          config.q_block_size,
                                          config.q_shift)
    inputs = pad_for_sliding_window(inputs, 
                                    config.q_block_size,
                                    config.q_shift)
    
    # Pad channels to power of 2
    lpfd_outputs = pad_to_power_of_2_channels(lpfd_outputs)
    inputs = pad_to_power_of_2_channels(inputs)

    """lpfd_outputs, inputs = pad_to_compute(outputs=lpfd_outputs, 
                                          labels=inputs,
                                          q_block_size=block_size,
                                          q_shift=block_size)"""

    height, width, depth = inputs.shape

    # Create Dask arrays for parallel processing
    arr_out = da.from_array(lpfd_outputs, 
                            chunks = (config.dask_chunk_size[0],
                                      config.dask_chunk_size[1],
                                      depth))
    arr_inp = da.from_array(inputs, 
                            chunks = (config.dask_chunk_size[0],
                                      config.dask_chunk_size[1],
                                      depth))
    
    # Compute Q2n map in parallel
    results = da.map_blocks(Q2n_map, 
                            arr_out, 
                            arr_inp, 
                            q_block_size = config.q_block_size,
                            q_shift = config.q_shift,
                            dtype = np.float32,
                            chunks = (config.dask_chunk_size[0] // config.q_block_size,
                                      config.dask_chunk_size[1] // config.q_block_size),
                            drop_axis = -1).compute()

    q2n_index = np.mean(np.asarray(results))

    d_lambda = 1 - q2n_index

    return d_lambda


def compute_D_s(sharp, pan, hs, red_pan, block_size = 32, exponent = 1):
    """
    Compute spatial distortion for a single band.
    
    Parameters:
        sharp (np.ndarray): Sharpened band. Shape: (H, W, 1)
        pan (np.ndarray): Panchromatic image. Shape: (H, W, 1)
        red_pan (np.ndarray): Reduced PAN. Shape: (H/ratio, W/ratio, 1)
        hs (np.ndarray): Original HS band. Shape: (H/ratio, W/ratio, 1)
        block_size (int): Window size for Q index
        exponent (int): Exponent for distortion calculation
        
    Returns:
        np.ndarray: D_s value for this band. Shape: (1, 1, 1)
    """

    Q_high = Q(sharp, pan, block_size = block_size)
    Q_low = Q(hs, red_pan, block_size = block_size)
    metric = np.abs(Q_high - Q_low) ** exponent

    return np.array([[[metric]]])

def D_s(sharp, pan, hs, ratio = 6, sensor = 'Prisma', config=None):
    """
    Compute spatial distortion index.
    
    This metric measures spatial consistency by comparing the Q index
    at high and low resolutions for each band.
    
    Parameters:
        sharp (np.ndarray): Pansharpened image. Shape: (H, W, C)
        pan (np.ndarray): Panchromatic image. Shape: (H, W, 1) or (H, W)
        hs (np.ndarray): Low-resolution HS image. Shape: (H/ratio, W/ratio, C)
        ratio (int): Resolution ratio. Default: 6
        sensor (str): Sensor name for MTF filtering. Default: 'PRISMA'
        config (MetricsConfig): Configuration parameters. If None, uses defaults.
        
    Returns:
        float: D_s index (lower is better, range [0,1])
        
    Note:
        Requires a function to downsample PAN (e.g., using sensor MTF).
        If not available, use simple downsampling as approximation.
    """

    """if config is None:
        config = MetricsConfig()"""
    
    nbands = sharp.shape[-1]
    lr_shape = hs.shape
    hr_shape = sharp.shape

    # Ensure PAN has channel dimension
    if len(pan.shape) == 2:
        pan = np.expand_dims(pan, axis = -1)

    # Downsample PAN to match HS resolution
    # ---> VEDERE BENE CHE SUCCEDE IN RESIZE_PAN, COSì CAPISCO COME GESTIRE EXPAND_DIMS!!! <--- TODO
    red_pan = data_gen.resize_pan(pan.squeeze(), ratio = ratio, sensor = sensor)
    red_pan = np.expand_dims(red_pan, axis = -1) # <--- è dopo resize_pan, viene tolta la terza dim nella f?

    arr_red_pan = da.from_array(red_pan, chunks = (lr_shape[0], lr_shape[1], 1)).astype(np.float32)
    arr_hs = da.from_array(hs, chunks = (lr_shape[0], lr_shape[1], 1)).astype(np.float32)
    arr_pan = da.from_array(pan, chunks = (hr_shape[0], hr_shape[1], 1)).astype(np.float32)
    arr_sharp = da.from_array(sharp, chunks = (hr_shape[0], hr_shape[1], 1)).astype(np.float32)

    results = da.map_blocks(compute_D_s, 
                            arr_sharp, 
                            arr_pan, 
                            arr_hs,
                            arr_red_pan, 
                            block_size = config.q_block_size,
                            exponent = config.exponent,
                            chunks = (1, 1, 1)).compute()

    D_s_index = (results.sum() / nbands) ** (1 / config.exponent)

    return D_s_index


def preprocess_for_metrics(sharp, pan, hs, ratio=6):
    """
    Preprocess images to ensure dimension compatibility.
    
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


def compute_metrics(sharp, pan, hs, ratio = 6, sensor = 'PRISMA', config = None,
                    use_dask_cluster = True):
    """
    Compute all pansharpening quality metrics (D_lambda, D_s, HQNR).
    
    Parameters:
        sharp (np.ndarray): Pansharpened image. Shape: (H, W, C)
        pan (np.ndarray): Panchromatic image. Shape: (H, W, 1) or (H, W)
        hs (np.ndarray): Low-resolution HS image. Shape: (H/ratio, W/ratio, C)
        ratio (int): Resolution ratio. Default: 6
        sensor (str): Sensor name. Default: 'PRISMA'
        config (MetricsConfig): Configuration. If None, uses defaults.
        use_dask_cluster (bool): Whether to start a Dask cluster. Default: True
        
    Returns:
        dict: Dictionary with keys 'D_lambda', 'D_s', 'HQNR'
        
    Example:
        >>> config = MetricsConfig(q_block_size=32, dask_chunk_size=(64, 64))
        >>> metrics = compute_metrics(sharp, pan, hs, config=config)
        >>> print(f"HQNR: {metrics['HQNR']:.4f}")
    """

    if config is None:
        config = MetricsConfig()

    config.validate()

    # CRITICAL: align dimensions for ratio compatibility:
    # high-resolution dimensions must be divisible by ratio
    sharp, pan, hs = preprocess_for_metrics(sharp, pan, hs, ratio)

    results = {}

    if use_dask_cluster:
        n_workers = int(config.n_workers * mp.cpu_count())
        with LocalCluster(n_workers = n_workers,
                          processes = True,
                          memory_limit = "auto",
                          threads_per_worker = 1) as cluster, Client(cluster) as client:
            
            print(f"Computing D_s with {n_workers} workers...")
            results['D_s'] = D_s(sharp, pan, hs, ratio = ratio, sensor = sensor, config = config)

            print(f"Computing D_lambda with {n_workers} workers...")
            results['D_lambda'] = D_lambda_khan(sharp, hs, config = config)
    
    else:
        # Compute without cluster, for debugging: runs sequentially 
        results['D_s'] = D_s(sharp, pan, hs, ratio = ratio, sensor = sensor, config = config)
        results['D_lambda'] = D_lambda_khan(sharp, hs, config = config)
    
    results['HQNR'] =  (1 - results['D_lambda']) * (1 - results['D_s'])

    return results


if __name__ == "__main__":

    """
    Example usage of the HQNR protocol.
    
    This example shows how to:
    1. Load images (replace with your actual loading code)
    2. Configure metrics parameters
    3. Compute quality metrics
    """

    hs_path = "/your/hs_path"
    pan_path = "/your/pan_path"
    sharp_path = "/your/sharp_path"

    # For debugging
    np.random.seed(42)
    sharp = np.random.rand(3072, 3072, 180).astype(np.float32)
    pan = np.random.rand(3072, 3072, 1).astype(np.float32)
    hs = np.random.rand(512, 512, 180).astype(np.float32)

    config = MetricsConfig(
        q_block_size = 32,              # Window size for quality index
        q_shift = 32,                   # Shift for the window
        dask_chunk_size = (64, 64),     # Number of chunks to be processed in parallel, based on q_block_size
        n_workers = 0.9,                # Percentage of CPU cores used
        exponent = 1)                   # Standard L1 norm
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(
        sharp = sharp,
        pan = pan,
        hs = hs,
        ratio = 6,              # PRISMA ratio: 30m/5m = 6
        sensor = 'PRISMA',
        config = config,
        use_dask_cluster = True
    )
    
    # Display results
    print("\n" + "="*50)
    print("PANSHARPENING QUALITY METRICS")
    print("="*50)
    print(f"D_lambda (spectral distortion): {metrics['D_lambda']:.6f}")
    print(f"D_s (spatial distortion):       {metrics['D_s']:.6f}")
    print(f"HQNR (overall quality):         {metrics['HQNR']:.6f}")
    print("="*50)
    print("\nInterpretation:")
    print("  - D_lambda closer to 0 = better spectral preservation")
    print("  - D_s closer to 0 = better spatial preservation")
    print("  - HQNR closer to 1 = better overall quality")


    # =========================================================================
    # REAL DATA EXAMPLE (uncomment and adapt to your needs)
    # =========================================================================
    """
    import rasterio
    
    print("\nLoading real data...")
    
    # Load images
    hs_path = "/path/to/hyperspectral.tif"
    pan_path = "/path/to/panchromatic.tif"
    sharp_path = "/path/to/sharpened.tif"
    
    # Read with rasterio (format: C x H x W)
    hs = rasterio.open(hs_path).read()
    pan = rasterio.open(pan_path).read()
    sharp = rasterio.open(sharp_path).read()
    
    # Convert to H x W x C format
    hs = hs.transpose((1, 2, 0)).astype(np.float32)
    pan = pan.transpose((1, 2, 0)).astype(np.float32)
    sharp = sharp.transpose((1, 2, 0)).astype(np.float32)
    
    # Optional: crop to region of interest
    # hs = hs[y1:y2, x1:x2, :]
    # pan = pan[y1*6:y2*6, x1*6:x2*6, :]  # Scale by ratio
    # sharp = sharp[y1*6:y2*6, x1*6:x2*6, :]
    
    print(f"Image shapes:")
    print(f"  HS: {hs.shape}")
    print(f"  PAN: {pan.shape}")
    print(f"  Sharp: {sharp.shape}")
    
    # Configure for large images
    config = MetricsConfig(
        q_block_size=32,
        q_shift=32,
        dask_chunk_size=(128, 128),  # Larger chunks for big images
        n_workers=0.9,
        exponent=1
    )
    
    # Compute metrics
    metrics = compute_metrics(
        sharp=sharp,
        pan=pan,
        hs=hs,
        ratio=6,
        sensor='PRISMA',
        config=config,
        use_dask_cluster=True
    )
    
    print(f"\nResults:")
    print(f"  D_lambda: {metrics['D_lambda']:.6f}")
    print(f"  D_s: {metrics['D_s']:.6f}")
    print(f"  HQNR: {metrics['HQNR']:.6f}")
    """
    
    






    data_paths = paths.prs20241021_424
    hs_path = save_path + data_paths['nwb']
    pan_path = data_paths['pan']
    #nwb_path = save_path + data_paths['nwb']
    sharp_path = save_path + data_paths['sharp'] + model_name + '.tif'
    lr_pix1, lr_pix2, hr_pix1, hr_pix2 = data_paths['px_range']

    # remember: C x W x H, PRISMA x6 (30m -> 5m)
    print(f"Working with {sharp_path.split('/')[-1]}")
    hs = rasterio.open(hs_path).read() # C x W x H
    pan = rasterio.open(pan_path).read() # 1 x 6W x 6H
    sharp = rasterio.open(sharp_path).read() # C x 6W x 6H

    hr_tile = 192  # for 5 meter spatial resolution
    lr_tile = int(hr_tile / 6)  # for 30 meter spatial resolution
    hr_row_tiles = (pan.shape[1] / hr_tile)
    hr_col_tiles = (pan.shape[2] / hr_tile)
    lr_row_tiles = (hs.shape[1] / lr_tile)
    lr_col_tiles = (hs.shape[2] / lr_tile)
    hr_rows = int((hr_tile * np.ceil(hr_row_tiles)) - pan.shape[1])
    hr_cols = int((hr_tile * np.ceil(hr_col_tiles)) - pan.shape[2])
    lr_rows = int((lr_tile * np.ceil(lr_row_tiles)) - hs.shape[1])
    lr_cols = int((lr_tile * np.ceil(lr_col_tiles)) - hs.shape[2])

    # Zero pad images
    pan = np.pad(np.squeeze(pan, axis=0), ((0, hr_rows), (0, hr_cols)))
    pan = pan[None, ...]
    hs = np.pad(hs, ((0, 0), (0, lr_rows), (0, lr_cols)))
    sharp = np.pad(sharp, ((0, 0), (0, hr_rows), (0, hr_cols)))

    hs = hs[:, lr_pix1:lr_pix2, lr_pix1:lr_pix2]
    pan = pan[:, hr_pix1:hr_pix2, hr_pix1:hr_pix2]
    sharp = sharp[:, hr_pix1:hr_pix2, hr_pix1:hr_pix2]


    # quindi Canali su ultima dim
    hs = hs.transpose((1, 2, 0)).astype(np.float32) # W x H x C
    pan = pan.transpose((1, 2, 0)).astype(np.float32) # 6W x 6H x 1
    sharp = sharp.transpose((1, 2, 0)).astype(np.float32) # 6W x 6H x C
    print(sharp.shape, pan.shape, hs.shape)

    with LocalCluster(n_workers = int(0.9 * mp.cpu_count()),
                      processes = True, memory_limit = "auto",
                      threads_per_worker = 1,
                      ) as cluster, Client(cluster) as client:
 
        print(hs_path)
        start = time.time()
        print("Computing D_s... ")
        d_s = D_s(sharp, pan, hs)
        print(f"D_s computed in {time.time() - start} seconds")
        print("Computing D_lambda... ")
        start = time.time()
        d_l = D_lambda_khan(sharp, hs)
        print(f"D_lambda computed in {time.time() - start}")
        print("D_s = ", d_s)
        print("D_lambda = ", d_l)
        hqnr = (1 - d_l) * (1 - d_s)
        print("HQNR = ", hqnr)

