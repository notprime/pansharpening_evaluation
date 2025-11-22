"""
Main quality metrics: D_lambda_khan, D_s, and HQNR
"""


import numpy as np
import dask.array as da
from dask.distributed import LocalCluster, Client
import multiprocessing as mp

from .config import MetricsConfig
from .quality_indices import Q, Q2n_map
from .downsampling import resize_hs, resize_pan
from .utils import pad_for_sliding_window, pad_to_power_of_2_channels, preprocess_for_metrics


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
    lpfd_outputs = resize_hs(img_ms = outputs,
                                 ratio = ratio,
                                 sensor = sensor)

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

def D_s(sharp, pan, hs, ratio = 6, sensor = 'PRISMA', config = None):
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
        print(f"Debug --- ENTRATI QUI, AGGIUNTA LAST DIM COME CHANNEL A PAN")
        pan = np.expand_dims(pan, axis = -1)

    # Downsample PAN to match HS resolution
    # ---> VEDERE BENE CHE SUCCEDE IN RESIZE_PAN, COSì CAPISCO COME GESTIRE EXPAND_DIMS!!! <--- TODO
    red_pan = resize_pan(pan.squeeze(), ratio = ratio, sensor = sensor)
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

    results = {}

    if use_dask_cluster:
        n_workers = int(config.n_workers * mp.cpu_count())
        with LocalCluster(n_workers = n_workers,
                          processes = True,
                          memory_limit = "auto",
                          threads_per_worker = 1) as cluster, Client(cluster) as client:
            
            print(f"Debug --- sharp: {sharp.shape} - pan: {pan.shape} - hs: {hs.shape}")
            
            print(f"Computing D_s with {n_workers} workers...")
            results['D_s'] = D_s(sharp, pan, hs, ratio = ratio, sensor = sensor, config = config)

            print(f"Computing D_lambda with {n_workers} workers...")
            results['D_lambda'] = D_lambda_khan(sharp, hs, sensor = sensor, config = config)
    
    else:
        # Compute without cluster, for debugging: runs sequentially 
        results['D_s'] = D_s(sharp, pan, hs, ratio = ratio, sensor = sensor, config = config)
        results['D_lambda'] = D_lambda_khan(sharp, hs, config = config)
    
    results['HQNR'] =  (1 - results['D_lambda']) * (1 - results['D_s'])

    return results