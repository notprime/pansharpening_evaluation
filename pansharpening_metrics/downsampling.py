import math
import numpy as np
from dataset import spectral_tools as st
import torch
from torch import nn
from skimage import transform


def fspecial_gauss(size, sigma):
    """
        Function to mimic the 'fspecial' gaussian MATLAB function
        Parameters
        ----------
        size : Tuple
            The dimensions of the kernel. Dimension: H, W
        sigma : float
            The frequency of the gaussian filter
        Return
        ------
        h : Numpy array
            The Gaussian Filter of sigma frequency and size dimension
        """

    m, n = [(ss - 1.) / 2. for ss in size]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh

    return h


def fir_filter_wind(hd, w):
    """
        Compute fir filter with window method
        Parameters
        ----------
        hd : float
            Desired frequency response (2D)
        w : Numpy Array
            The filter kernel (2D)
        Return
        ------
        h : Numpy array
            The fir Filter
    """

    hd = np.rot90(np.fft.fftshift(np.rot90(hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    h = np.clip(h, a_min = 0, a_max = np.max(h))
    h = h / np.sum(h)

    return h


def nyquist_filter_generator(nyquist_freq, ratio, kernel_size):
    """
        Compute the estimeted MTF filter kernels.
        Parameters
        ----------
        nyquist_freq : Numpy Array or List
            The MTF frequencies
        ratio : int
            The resolution scale which elapses between MS and PAN.
        kernel_size : int
            The size of the kernel (Only squared kernels have been implemented).
        Return
        ------
        kernel : Numpy array
            The filter based on Modulation Transfer Function.
    """

    assert isinstance(nyquist_freq, (np.ndarray, list)), 'Error: GNyq must be a list or a ndarray'

    if isinstance(nyquist_freq, list):
        nyquist_freq = np.asarray(nyquist_freq)

    nbands = nyquist_freq.shape[0]

    kernel = np.zeros((kernel_size, kernel_size, nbands))  # generic kerenel (for normalization purpose)

    fcut = 1 / np.double(ratio)

    for j in range(nbands):
        alpha = np.sqrt(((kernel_size - 1) * (fcut / 2)) ** 2 / (-2 * np.log(nyquist_freq[j])))
        H = fspecial_gauss((kernel_size, kernel_size), alpha)
        Hd = H / np.max(H)
        h = np.kaiser(kernel_size, 0.5)

        kernel[:, :, j] = np.real(fir_filter_wind(Hd, h))

    return kernel


def mtf_kernel_to_torch(h):
    """
        Compute the estimated MTF filter kernels for the supported satellites and calculate the spatial bias between
        each Multi-Spectral band and the Panchromatic (to implement the coregistration feature).
        Parameters
        ----------
        h : Numpy Array
            The filter based on Modulation Transfer Function.
        Return
        ------
        h : Tensor array
            The filter based on Modulation Transfer Function reshaped to Conv2d kernel format.
        """

    h = np.moveaxis(h, -1, 0)
    h = np.expand_dims(h, axis = 1)
    h = h.astype(np.float32)
    h = torch.from_numpy(h).type(torch.float32)

    return h


def resize_pan(img_pan, ratio, sensor = None, mtf = None, apply_mtf_to_pan = True):

    GNyqPan = []
    if (sensor is None) & (mtf is None):
        PAN_scale = (math.floor(img_pan.shape[0] / ratio), math.floor(img_pan.shape[1] / ratio))
        I_PAN_LR = transform.resize(img_pan, PAN_scale, order=3)
        return I_PAN_LR
    elif (sensor == 'QB') & (mtf is None):
        GNyqPan = np.asarray([0.15])
    elif ((sensor == 'Ikonos') or (sensor == 'IKONOS')) & (mtf is None):
        GNyqPan = np.asarray([0.17])
    elif (sensor == 'GeoEye1' or sensor == 'GE1') & (mtf is None):
        GNyqPan = np.asarray([0.16])
    elif (sensor == 'WV2') & (mtf is None):
        GNyqPan = np.asarray([0.11])
    elif (sensor == 'WV3') & (mtf is None):
        GNyqPan = np.asarray([0.5])
    elif (sensor == 'PRISMA') & (mtf is None):
        GNyqPan = np.asarray([0.22])
    elif mtf is not None:
        GNyqPan = np.asarray([mtf['GNyqPan']])

    N = 41
    PAN_scale = (math.floor(img_pan.shape[0] / ratio), math.floor(img_pan.shape[1] / ratio))

    if apply_mtf_to_pan:
        img_pan = np.expand_dims(img_pan, [0, 1])

        h = nyquist_filter_generator(GNyqPan, ratio, N)
        h = mtf_kernel_to_torch(h)

        conv = nn.Conv2d(in_channels=1, out_channels=1, padding=math.ceil(N / 2),
                         kernel_size=h.shape, groups=1, bias=False, padding_mode='replicate')

        conv.weight.data = h
        conv.weight.requires_grad = False

        I_PAN_LP = conv(torch.from_numpy(img_pan)).numpy()
        I_PAN_LP = np.squeeze(I_PAN_LP)
        I_PAN_LR = transform.resize(I_PAN_LP, PAN_scale, order=0)

    else:
        I_PAN_LR = transform.resize(img_pan, PAN_scale, order=3)

    return I_PAN_LR


def resize_hs(img_ms, ratio, sensor = None, mtf = None):
    """
    CHANNEL ON LAST DIM
        Function to perform a downscale of all the data provided by the satellite.
        It downsamples the data of the scale value.
        To more detail please refers to
        [1] G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. Orn Ulfarsson, L. Alparone, J. Chanussot -
            A new benchmark based on recent advances in multispectral pansharpening: Revisiting pansharpening with
            classical and emerging pansharpening methods
        [2] L. Wald, (1) T. Ranchin, (2) M. Mangolini - Fusion of satellites of different spatial resolutions:
            assessing the quality of resulting images
        [3] B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, M. Selva - MTF-tailored Multiscale Fusion of
            High-resolution MS and Pan Imagery
        [4] M. Ciotola, S. Vitale, A. Mazza, G. Poggi, G. Scarpa - Pansharpening by convolutional neural networks in
            the full resolution framework
        Parameters
        ----------
        img_ms : Numpy Array
            stack of Multi-Spectral bands. Dimension: H, W, B
        ratio : int
            the resolution scale which elapses between MS and PAN.
        sensor : str
            The name of the satellites which has provided the images.
        mtf : Dictionary
            The desired Modulation Transfer Frequencies with which perform the low pass filtering process.
            Example of usage:
                MTF = {'GNyq' : np.asarray([0.21, 0.2, 0.3, 0.4]), 'GNyqPan': 0.5}
        Return
        ------
        I_MS_LR : Numpy array
            the stack of Multi-Spectral bands downgraded by the ratio factor
        """
    GNyq = []
    if (sensor is None) & (mtf is None):
        MS_scale = (math.floor(img_ms.shape[0] / ratio), math.floor(img_ms.shape[1] / ratio), img_ms.shape[2])
        I_MS_LR = transform.resize(img_ms, MS_scale, order = 3)
        return I_MS_LR
    elif (sensor == 'QB') & (mtf is None):
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22])  # Bands Order: B,G,R,NIR
    elif ((sensor == 'Ikonos') or (sensor == 'IKONOS')) & (mtf is None):
        GNyq = np.asarray([0.26, 0.28, 0.29, 0.28])  # Bands Order: B,G,R,NIR
    elif (sensor == 'GeoEye1' or sensor == 'GE1') & (mtf is None):
        GNyq = np.asarray([0.23, 0.23, 0.23, 0.23])  # Bands Order: B, G, R, NIR
    elif (sensor == 'WV2') & (mtf is None):
        GNyq = 0.35 * np.ones(7)
        GNyq = np.append(GNyq, 0.27)
    elif (sensor == 'WV3') & (mtf is None):
        GNyq = np.asarray([0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315])
    elif (sensor == 'PRISMA') & (mtf is None):
        GNyq = 0.28 * np.ones(img_ms.shape[-1])
    elif mtf is not None:
        GNyq = mtf['GNyq']

    N = 41

    b = img_ms.shape[-1]

    img_ms = np.moveaxis(img_ms, -1, 0)
    img_ms = np.expand_dims(img_ms, axis = 0)
    h = nyquist_filter_generator(GNyq, ratio, N)
    h = mtf_kernel_to_torch(h)

    conv = nn.Conv2d(in_channels = b, out_channels = b, padding = math.ceil(N / 2),
                     kernel_size = h.shape, groups = b, bias = False, padding_mode = 'replicate',
                     dtype = h.dtype)

    conv.weight.data = h

    conv.weight.requires_grad = False

    I_MS_LP = conv(torch.from_numpy(img_ms).float()).numpy()
    I_MS_LP = np.squeeze(I_MS_LP)
    I_MS_LP = np.moveaxis(I_MS_LP, 0, -1)
    MS_scale = (math.floor(I_MS_LP.shape[0] / ratio), math.floor(I_MS_LP.shape[1] / ratio), I_MS_LP.shape[2])

    I_MS_LR = transform.resize(I_MS_LP, MS_scale, order = 0)
    I_MS_LR = np.clip(I_MS_LR, 0., 1.)

    return I_MS_LR