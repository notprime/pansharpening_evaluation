"""
MTF-based downsampling for hyperspectral and panchromatic images.

This module implements sensor-specific Modulation Transfer Function (MTF) filtering.

References:
    [Wald97] L. Wald et al., "Fusion of satellites of different spatial resolutions",
             PE&RS, 1997.
    [Aiazzi06] B. Aiazzi et al., "MTF-tailored Multiscale Fusion", IEEE TGRS, 2006.

"""


import math
import numpy as np
import torch
from torch import nn
from skimage import transform

from .sensors import SENSOR_MTF



def fspecial_gauss(size, sigma):
    """
    Create a 2D Gaussian filter kernel (mimics MATLAB's fspecial('gaussian')).
    
    Args:
        size (tuple): Kernel dimensions (H, W)
        sigma (float): Standard deviation of Gaussian
    
    Returns:
        h (np.ndarray): Normalized Gaussian filter kernel
    """

    m, n = [(ss - 1.) / 2. for ss in size]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    # Gaussian formula
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))

    # Zero out very small values
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    # Normalize
    sumh = h.sum()
    if sumh != 0:
        h /= sumh

    return h


def fir_filter_wind(hd, w):
    """
    Compute FIR filter using window method.
    
    Args:
        hd (np.ndarray): Desired frequency response (2D)
        w (np.ndarray): Window function (2D)
    
    Returns:
        h (np.ndarray): FIR filter kernel
    """

    # Apply FFT for filter design
    hd = np.rot90(np.fft.fftshift(np.rot90(hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)

    # Apply window and normalize
    h = h * w
    h = np.clip(h, a_min = 0, a_max = np.max(h))
    h = h / np.sum(h)

    return h


def nyquist_filter_generator(nyquist_freq, ratio, kernel_size):
    """
    Generate MTF-based filter kernels for each band.
    
    Creates sensor-specific low-pass filters based on Modulation Transfer Function
    characteristics. Only squared kernels have been implemented.
    
    Args:
        nyquist_freq (np.ndarray or list): MTF frequencies for each band
        ratio (int): Downsampling ratio (e.g., 6 for PRISMA)
        kernel_size (int): Filter kernel size (typically 41)
    
    Returns:
        kernel (np.ndarray): Filter kernels. Shape: (kernel_size, kernel_size, nbands)
    
    """

    assert isinstance(nyquist_freq, (np.ndarray, list)), 'Error: GNyq must be a list or a ndarray'

    if isinstance(nyquist_freq, list):
        nyquist_freq = np.asarray(nyquist_freq)

    nbands = nyquist_freq.shape[0]
    kernel = np.zeros((kernel_size, kernel_size, nbands))  # generic kernel (for normalization purpose)

    fcut = 1.0 / float(ratio)  # Cutoff frequency

    for j in range(nbands):
        # Compute Gaussian parameter based on MTF
        alpha = np.sqrt(((kernel_size - 1) * (fcut / 2)) ** 2 / (-2 * np.log(nyquist_freq[j])))
        
        # Generate Gaussian filter
        H = fspecial_gauss((kernel_size, kernel_size), alpha)
        Hd = H / np.max(H)

        # Apply Kaiser window
        h = np.kaiser(kernel_size, 0.5)

        # Generate final filter
        kernel[:, :, j] = np.real(fir_filter_wind(Hd, h))

    return kernel


def mtf_kernel_to_torch(h):
    """
    Convert MTF kernel to PyTorch Conv2d format.
    
    Args:
        h (np.ndarray): MTF kernel. Shape: (kernel_size, kernel_size, nbands)
    
    Returns:
        h (torch.Tensor): Kernel in Conv2d format. Shape: (nbands, 1, kernel_size, kernel_size)
    """

    h = np.moveaxis(h, -1, 0)
    h = np.expand_dims(h, axis = 1)
    h = h.astype(np.float32)
    h = torch.from_numpy(h).type(torch.float32)

    return h


def get_sensor_mtf(sensor, nbands = None):
    """
    Get MTF parameters for a specific sensor.
    
    Args:
        sensor (str): Sensor name (e.g., 'PRISMA', 'WV3', 'QB')
        nbands (int): Number of bands (required for PRISMA)
    
    Returns:
        dict: Dictionary with 'GNyq' and 'GNyqPan' keys
    
    Raises:
        ValueError: If sensor is unknown or nbands not provided for PRISMA
    """
    sensor = sensor.upper() if sensor else None
    
    if sensor not in SENSOR_MTF:
        raise ValueError(f"Unknown sensor: {sensor}. Available: {list(SENSOR_MTF.keys())}")
    
    mtf = SENSOR_MTF[sensor].copy()
    
    # Special handling for PRISMA (uniform MTF across all bands)
    if sensor == 'PRISMA':
        if nbands is None:
            raise ValueError("nbands must be provided for PRISMA sensor")
        mtf['GNyq'] = mtf['GNyq'] * np.ones(nbands)
    
    return mtf


def resize_pan(img_pan, ratio, sensor = 'PRISMA', mtf = None, apply_mtf = True):
    """
    Downsample panchromatic image with MTF filtering.
    
    Args:
        img_pan (np.ndarray): Panchromatic image. Shape: (H, W)
        ratio (int): Downsampling ratio (e.g., 6 for PRISMA: 5m -> 30m)
        sensor (str): Sensor name (default: 'PRISMA')
        mtf (dict): Custom MTF parameters. If None, uses sensor defaults.
        apply_mtf (bool): Whether to apply MTF filtering (default: True)
    
    Returns:
        img_pan_lr (np.ndarray): Downsampled PAN image. Shape: (H/ratio, W/ratio)
    """

    # Compute output size
    out_h = math.floor(img_pan.shape[0] / ratio)
    out_w = math.floor(img_pan.shape[1] / ratio)
    pan_scale = (out_h, out_w)

    # Simple downsampling (no MTF)
    if not apply_mtf or (sensor is None and mtf is None):
        return transform.resize(img_pan, pan_scale, order = 3)
    
    # Get MTF parameters
    if mtf is not None:
        GNyqPan = np.array([mtf['GNyqPan']])
    else:
        sensor_mtf = get_sensor_mtf(sensor)
        GNyqPan = np.array([sensor_mtf['GNyqPan']])

    # Generate MTF filter
    kernel_size = 41
    h = nyquist_filter_generator(GNyqPan, ratio, kernel_size)
    h = mtf_kernel_to_torch(h)

    # Apply MTF filtering using Conv2d
    img_pan_4d = np.expand_dims(img_pan, [0, 1]) # Add batch and channel dims
    conv = nn.Conv2d(in_channels = 1, 
                     out_channels = 1, 
                     padding = math.ceil(kernel_size / 2),
                     kernel_size = h.shape, 
                     groups = 1, 
                     bias = False, 
                     padding_mode = 'replicate')
    
    conv.weight.data = h
    conv.weight.requires_grad = False

    img_pan_filtered = conv(torch.from_numpy(img_pan_4d)).numpy()
    img_pan_filtered = np.squeeze(img_pan_filtered)
    img_pan_lr = transform.resize(img_pan_filtered, pan_scale, order = 0)

    return img_pan_lr


def resize_hs(img_hs, ratio, sensor = 'PRISMA', mtf = None):
    """
    Downsample hyperspectral image with MTF filtering.
    
    Applies band-specific MTF filtering before downsampling to preserve
    spectral information and avoid aliasing.
    
    Args:
        img_hs (np.ndarray): Hyperspectral image. Shape: (H, W, C)
        ratio (int): Downsampling ratio (e.g., 6 for PRISMA)
        sensor (str): Sensor name (default: 'PRISMA')
        mtf (dict): Custom MTF parameters. Keys: 'GNyq' (array of MTF per band)
    
    Returns:
        img_hs_lr (np.ndarray): Downsampled HS image. Shape: (H/ratio, W/ratio, C)
    """

    nbands = img_hs.shape[-1]

    # Compute output size
    out_h = math.floor(img_hs.shape[0] / ratio)
    out_w = math.floor(img_hs.shape[1] / ratio)
    hs_scale = (out_h, out_w, nbands)

    # Simple downsampling (no MTF)
    if sensor is None and mtf is None:
        return transform.resize(img_hs, hs_scale, order = 3)
    
    # Get MTF parameters
    if mtf is not None:
        GNyq = mtf['GNyq']
    else:
        sensor_mtf = get_sensor_mtf(sensor, nbands = nbands)
        GNyq = sensor_mtf['GNyq']

    # Generate MTF filter
    kernel_size = 41
    h = nyquist_filter_generator(GNyq, ratio, kernel_size)
    h = mtf_kernel_to_torch(h)

    # Prepare for Conv2d: (H, W, C) -> (1, C, H, W)
    img_hs_4d = np.moveaxis(img_hs, -1, 0)  # (C, H, W)
    img_hs_4d = np.expand_dims(img_hs_4d, axis=0)  # (1, C, H, W)

    # Apply MTF filtering (grouped convolution - one filter per band)
    conv = nn.Conv2d(in_channels = nbands, 
                     out_channels = nbands, 
                     padding = math.ceil(kernel_size / 2),
                     kernel_size = h.shape, 
                     groups = nbands, # each band filtered independently
                     bias = False, 
                     padding_mode = 'replicate',
                     dtype = h.dtype)
    
    conv.weight.data = h
    conv.weight.requires_grad = False

    # Filter
    img_hs_filtered = conv(torch.from_numpy(img_hs_4d).float()).numpy()
    img_hs_filtered = np.squeeze(img_hs_filtered)  # Remove batch dim
    img_hs_filtered = np.moveaxis(img_hs_filtered, 0, -1)  # (H, W, C)

    # Downsample
    img_hs_lr = transform.resize(img_hs_filtered, hs_scale, order=0)
    img_hs_lr = np.clip(img_hs_lr, 0., 1.)
    
    return img_hs_lr
