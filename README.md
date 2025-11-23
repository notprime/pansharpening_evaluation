#Pansharpening Quality Metrics

Efficient quality assessment metrics for hyperspectral pansharpening evaluation, optimized for large images using Dask.
Overview

This package implements reference-free quality metrics for evaluating pansharpening results, namely the HQNR protocol:

    D_λ (D_lambda): Spectral distortion index
    D_s: Spatial distortion index
    HQNR: Hybrid Quality with No Reference = (1 - D_λ) × (1 - D_s)

Installation
bash

# Clone repository
git clone https://github.com/yourusername/pansharpening-metrics.git
cd pansharpening-metrics

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .

Quick Start
Command Line (Simplest)
bash

# Basic usage
python main.py sharp.tif pan.tif hs.tif

# With configuration
python main.py sharp.tif pan.tif hs.tif --config config.yaml

# Save results
python main.py sharp.tif pan.tif hs.tif --output results.json

Python API (Recommended for Scripts)
python

from pansharpening_metrics import compute_metrics, MetricsConfig
import rasterio

# Load images (H x W x C format)
sharp = rasterio.open('sharp.tif').read().transpose(1, 2, 0)
pan = rasterio.open('pan.tif').read().transpose(1, 2, 0)
hs = rasterio.open('hs.tif').read().transpose(1, 2, 0)

# Configure (optional)
config = MetricsConfig(q_block_size=32, n_workers=0.9)

# Compute metrics
metrics = compute_metrics(sharp, pan, hs, ratio=6, config=config)

# Results
print(f"D_lambda: {metrics['D_lambda']:.4f}")
print(f"D_s:      {metrics['D_s']:.4f}")
print(f"HQNR:     {metrics['HQNR']:.4f}")

Usage
Command Line Interface

Basic usage:
bash

python main.py <sharp> <pan> <hs>

All options:
bash

python main.py sharp.tif pan.tif hs.tif \
    --ratio 6 \
    --sensor PRISMA \
    --config config.yaml \
    --output results.json \
    --q_block_size 32 \
    --n_workers 0.9 \
    --dask_chunk_size 128 128

Arguments:

    sharp: Sharpened hyperspectral image (required)
    pan: Panchromatic image (required)
    hs: Low-resolution hyperspectral image (required)
    --ratio: Resolution ratio (default: 6)
    --sensor: Sensor name (default: PRISMA)
    --config: YAML configuration file
    --output, -o: Save results to JSON
    --q_block_size: Override Q window size
    --n_workers: Override worker fraction (0-1)
    --dask_chunk_size: Override chunk size (H W)

Python API

Import:
python

from pansharpening_metrics import (
    compute_metrics,      # Main function
    MetricsConfig,        # Configuration
    D_lambda_khan,        # Individual metrics
    D_s,
    normalize_inputs,     # Utilities
    preprocess_for_metrics
)

Basic usage:
python

# With default configuration
metrics = compute_metrics(sharp, pan, hs, ratio=6)

With custom configuration:
python

config = MetricsConfig(
    q_block_size=32,
    q_shift=32,
    dask_chunk_size=(128, 128),
    n_workers=0.9,
    exponent=1
)
metrics = compute_metrics(sharp, pan, hs, ratio=6, config=config)

Using presets:
python

config = MetricsConfig.balanced()      # Recommended
# config = MetricsConfig.conservative() # Low memory
# config = MetricsConfig.aggressive()   # High performance

metrics = compute_metrics(sharp, pan, hs, config=config)

Individual metrics:
python

from pansharpening_metrics import D_lambda_khan, D_s
from pansharpening_metrics.downsampling import simple_downsample_multiband

# Compute separately
sharp_lr = simple_downsample_multiband(sharp, ratio=6)
d_lambda = D_lambda_khan(sharp_lr, hs, config=config)
d_s = D_s(sharp, pan, hs, ratio=6, config=config)
hqnr = (1 - d_lambda) * (1 - d_s)

Configuration
Configuration File (YAML)

Create a config.yaml:
yaml

q_block_size: 32          # Window size for Q/Q2n
q_shift: 32               # Stride for sliding windows
dask_chunk_size: [128, 128]  # Chunk size for parallelization
n_workers: 0.9            # Fraction of CPU cores to use
exponent: 1               # Exponent for distortion metrics

Use it:
bash

python main.py sharp.tif pan.tif hs.tif --config config.yaml

Configuration Parameters

Parameter	Default	Description
q_block_size	32	Window size for quality indices
q_shift	32	Stride for sliding windows
dask_chunk_size	(64, 64)	Chunk size for Dask parallelization
n_workers	0.9	Fraction of CPU cores to use (0-1)
exponent	1	Exponent for distortion metrics (Lp norm)

Presets
python

# Conservative: low memory usage
config = MetricsConfig.conservative()
# q_block_size=32, chunk_size=(64,64), n_workers=0.5

# Balanced: recommended for most cases
config = MetricsConfig.balanced()
# q_block_size=32, chunk_size=(128,128), n_workers=0.9

# Aggressive: maximum performance
config = MetricsConfig.aggressive()
# q_block_size=32, chunk_size=(256,256), n_workers=0.95, overlapping windows

Image Format Requirements

All images should be in H × W × C format (Height × Width × Channels):

    Hyperspectral (HS): (H, W, C) at low resolution (e.g., 30m)
    Panchromatic (PAN): (H×ratio, W×ratio, 1) at high resolution (e.g., 5m)
    Sharpened: (H×ratio, W×ratio, C) at high resolution

Example for PRISMA (30m → 5m, ratio=6)
python

hs.shape    # (512, 512, 180)    - 30m resolution
pan.shape   # (3072, 3072, 1)    - 5m resolution (512×6=3072)
sharp.shape # (3072, 3072, 180)  - 5m resolution

Format Conversion

If your data is in channels-first format (C × H × W):
python

from pansharpening_metrics import normalize_inputs

# Automatically converts CHW to HWC and validates
sharp, pan, hs = normalize_inputs(sharp, pan, hs, ratio=6)

Examples
Example 1: Basic Usage
bash

python main.py data/sharp.tif data/pan.tif data/hs.tif

Output:

======================================================================
PANSHARPENING QUALITY METRICS
======================================================================

Loading images...
  Loading sharp.tif...
  Loading pan.tif...
  Loading hs.tif...

Image shapes:
  Sharp: (3072, 3072, 180)
  PAN:   (3072, 3072, 1)
  HS:    (512, 512, 180)

Preprocessing...
After preprocessing:
  Sharp: (3072, 3072, 180)
  PAN:   (3072, 3072, 1)
  HS:    (512, 512, 180)

Computing metrics...
Configuration: q_block_size=32, chunk_size=(64, 64), n_workers=0.9

======================================================================
RESULTS
======================================================================
D_lambda (spectral): 0.123456
D_s (spatial):       0.234567
HQNR (overall):      0.678901
======================================================================

Done!

Example 2: Batch Processing
bash

#!/bin/bash
# Process multiple models

for model in model_v1 model_v2 model_v3; do
    echo "Processing $model..."
    python main.py \
        results/${model}_sharp.tif \
        data/pan.tif \
        data/hs.tif \
        --output results/${model}_metrics.json
done

Example 3: Python Script
python

from pansharpening_metrics import compute_metrics, MetricsConfig
import rasterio
import json

# Load images
sharp = rasterio.open('sharp.tif').read().transpose(1, 2, 0)
pan = rasterio.open('pan.tif').read().transpose(1, 2, 0)
hs = rasterio.open('hs.tif').read().transpose(1, 2, 0)

# Configure
config = MetricsConfig.balanced()

# Compute
metrics = compute_metrics(sharp, pan, hs, ratio=6, config=config)

# Save
with open('results.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"HQNR: {metrics['HQNR']:.4f}")

Example 4: With Preprocessing
python

from pansharpening_metrics import (
    compute_metrics, 
    MetricsConfig,
    normalize_inputs,
    preprocess_for_metrics
)

# Load your images (any format)
sharp, pan, hs = load_your_images()

# Normalize format (CHW -> HWC, validation)
sharp, pan, hs = normalize_inputs(sharp, pan, hs, ratio=6)

# Align dimensions (ensures compatibility)
sharp, pan, hs = preprocess_for_metrics(sharp, pan, hs, ratio=6)

# Compute metrics
config = MetricsConfig.balanced()
metrics = compute_metrics(sharp, pan, hs, ratio=6, config=config)

Understanding the Metrics
D_lambda (Spectral Distortion)

Measures how well spectral relationships between bands are preserved.

    Range: [0, 1]
    Interpretation: Lower is better (0 = perfect spectral preservation)
    Method: Compares Q2n index between band pairs at low resolution

D_s (Spatial Distortion)

Measures how well spatial details are preserved.

    Range: [0, 1]
    Interpretation: Lower is better (0 = perfect spatial preservation)
    Method: Compares Q index at high and low resolutions for each band

HQNR (Hybrid Quality with No Reference)

Overall quality combining spectral and spatial preservation.

    Formula: HQNR = (1 - D_λ) × (1 - D_s)
    Range: [0, 1]
    Interpretation: Higher is better (1 = perfect quality)

Performance Tips
Memory Management

For large images (>5000×5000):
python

config = MetricsConfig(
    dask_chunk_size=(256, 256),  # Larger chunks
    n_workers=0.9
)

For limited memory:
python

config = MetricsConfig.conservative()  # Uses smaller chunks, fewer workers

Computation Time

Approximate times on 16-core CPU:

Image Size	Bands	Config	Time
1000×1000	180	Balanced	30-60s
3000×3000	180	Balanced	2-4 min
6000×6000	180	Balanced	10-20 min
6000×6000	180	Aggressive	5-10 min

Project Structure

pansharpening-metrics/
├── pansharpening_metrics/    # Main package
│   ├── __init__.py           # API exports
│   ├── config.py             # MetricsConfig class
│   ├── utils.py              # Utility functions
│   ├── quality_indices.py    # Q and Q2n implementations
│   ├── metrics.py            # D_lambda, D_s, compute_metrics
│   └── downsampling.py       # Downsampling functions
├── main.py                   # CLI entry point
├── config.yaml               # Example configuration
├── requirements.txt          # Dependencies
├── setup.py                  # Package setup
└── README.md                 # This file

Troubleshooting
Import Error
bash

# Install package in development mode
pip install -e .

Memory Error
python

# Use conservative configuration
config = MetricsConfig.conservative()

Or from CLI:
bash

python main.py sharp.tif pan.tif hs.tif --n_workers 0.5 --dask_chunk_size 64 64

Shape Mismatch Error
python

# Use preprocessing functions
from pansharpening_metrics import normalize_inputs, preprocess_for_metrics

sharp, pan, hs = normalize_inputs(sharp, pan, hs, ratio=6)
sharp, pan, hs = preprocess_for_metrics(sharp, pan, hs, ratio=6)

References

    Scarpa et al. (2021): "Full-resolution quality assessment for pansharpening", arXiv:2108.06144
    Garzelli & Nencini (2009): "Hypercomplex quality assessment of multi/hyper-spectral images", IEEE GRSL
    Alparone et al. (2008): "Multispectral and panchromatic data fusion assessment without reference"
    Vivone et al. (2020): "A new benchmark based on recent advances in multispectral pansharpening", IEEE GRSM
    Wang & Bovik (2002): "A universal image quality index", IEEE SPL

License

MIT License - see LICENSE file for details.
Citation

If you use this library in your research, please cite:
bibtex

@software{pansharpening_metrics,
  author = {Riccardo Musto},
  title = {Pansharpening Quality Metrics for Hyperspectral Images},
  year = {2024},
  url = {https://github.com/yourusername/pansharpening-metrics},
  version = {1.0.0}
}

Author

Riccardo Musto
Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
