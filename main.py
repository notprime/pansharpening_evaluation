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

### NEW IMPORTS
import argparse
import json
import yaml
from pathlib import Path
import time

import numpy as np
import rasterio

from pansharpening_metrics import (
    compute_metrics,
    MetricsConfig,
    preprocess_for_metrics
)



### OLD IMPORTS
#import paths
import numpy as np
import rasterio
import multiprocessing as mp


from math import ceil, floor, log2, sqrt
from skimage.transform.integral import integral_image as integral

import torch
from torch import nn

import dask
import dask.array as da
from dask.distributed import LocalCluster, Client


def load_data(path):
    """
    Load image from file.
    
    Supports TIFF (via rasterio) and numpy arrays.
    Channels are expected to be the first dimension.
    Returns image in HWC format.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    print(f"Loading {path.name}...")

    # Try rasterio first
    try:
        with rasterio.open(path) as src:
            data = src.read() # Should return C x H x W
            return data.transpose(1, 2, 0).astype(np.float32) # Convert to H x W x C
    except:
        # Fallback to numpy
        data = np.load(path)
        return data.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description = 'Compute pansharpening quality metrics (D_lambda_khan, D_s, HQNR)',
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
Examples:
  python main.py sharp.tif pan.tif hs.tif
  python main.py sharp.tif pan.tif hs.tif --ratio 6 --output results.json
  python main.py sharp.tif pan.tif hs.tif --config config.yaml
  python main.py sharp.tif pan.tif hs.tif --config config.yaml --n_workers 0.7
"""
    )

    # Required arguments (positional)
    parser.add_argument('sharp', help = 'Path to sharpened hyperspectral image')
    parser.add_argument('pan', help = 'Path to panchromatic image')
    parser.add_argument('hs', help = 'Path to low-resolution hyperspectral image')
    
    # Optional arguments
    parser.add_argument('--ratio', type = int, default = 6,
                       help='Resolution ratio (default: 6)')
    parser.add_argument('--sensor', default = 'PRISMA',
                       help='Sensor name (default: PRISMA)')
    parser.add_argument('--config', type = str,
                       help='Path to YAML configuration file')
    parser.add_argument('--output', '-o', type = str,
                       help='Save results to JSON file')
    
    # Configuration overrides
    parser.add_argument('--q_block_size', type = int,
                       help = 'Override Q block size')
    parser.add_argument('--n_workers', type = float,
                       help = 'Override worker fraction (0-1)')
    parser.add_argument('--dask_chunk_size', type = int, nargs = 2, metavar = ('H', 'W'),
                       help='Override Dask chunk size (e.g., --dask_chunk_size 128 128)')
    
    args = parser.parse_args()

    print("="*70)
    print("PANSHARPENING QUALITY METRICS")
    print("="*70)

    # Load data
    print("\nLoading data...")
    try:
        sharp = load_data(args.sharp) 
        pan = load_data(args.pan) 
        hs = load_data(args.hs) 
    except Exception as e:
        print(f"\nError loading images: {e}")
        return 1
    
    print(f"\nImage shapes:")
    print(f"  Sharp: {sharp.shape}")
    print(f"  PAN:   {pan.shape}")
    print(f"  HS:    {hs.shape}")

    # Preprocess data
    print("\nPreprocessing...")
    try:
        """
        CRITICAL: align dimensions for ratio compatibility:
        high-resolution dimensions must be divisible by ratio
        """
        sharp, pan, hs = preprocess_for_metrics(sharp, pan, hs, args.ratio)
    except Exception as e:
        print(f"\nError during preprocessing: {e}")
        return 1
    
    print(f"After preprocessing:")
    print(f"  Sharp: {sharp.shape}")
    print(f"  PAN:   {pan.shape}")
    print(f"  HS:    {hs.shape}")

    # Create configuration
    if args.config:
        print(f"\nLoading config from: {args.config}")
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = MetricsConfig(**config_dict)
    else:
        config = MetricsConfig()
    
    # Override with CLI arguments
    if args.q_block_size:
        config.q_block_size = args.q_block_size
    if args.n_workers:
        config.n_workers = args.n_workers
    if args.dask_chunk_size:
        config.dask_chunk_size = tuple(args.dask_chunk_size)
    
    config.validate()

    # Compute metrics
    print("\nComputing metrics...")
    print(f"Configuration: q_block_size={config.q_block_size}, "
          f"chunk_size={config.dask_chunk_size}, n_workers={config.n_workers}")
    
    try:
        metrics = compute_metrics(
            sharp = sharp,
            pan = pan,
            hs = hs,
            ratio = args.ratio,
            sensor = args.sensor,
            config = config
        )
    except Exception as e:
        print(f"\nError during computation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"D_lambda (spectral): {metrics['D_lambda']:.6f}")
    print(f"D_s (spatial):       {metrics['D_s']:.6f}")
    print(f"HQNR (overall):      {metrics['HQNR']:.6f}")
    print("="*70)

    # Save results if requested
    if args.output:
        output_data = {
            'metrics': {
                'D_lambda': float(metrics['D_lambda']),
                'D_s': float(metrics['D_s']),
                'HQNR': float(metrics['HQNR'])
            },
            'configuration': {
                'q_block_size': config.q_block_size,
                'q_shift': config.q_shift,
                'dask_chunk_size': list(config.dask_chunk_size),
                'n_workers': config.n_workers,
                'exponent': config.exponent
            },
            'input_files': {
                'sharp': str(args.sharp),
                'pan': str(args.pan),
                'hs': str(args.hs)
            },
            'parameters': {
                'ratio': args.ratio,
                'sensor': args.sensor
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())