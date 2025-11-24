# Pansharpening Quality Metrics

Efficient quality assessment metrics for hyperspectral pansharpening evaluation, optimized for large images using Dask.

## Overview

This package implements reference-free quality metrics for evaluating pansharpening results:

- **D_λ (D_lambda)**: Spectral distortion index
- **D_s**: Spatial distortion index  
- **HQNR**: Hybrid Quality with No Reference = (1 - D_λ) × (1 - D_s)

## Installation

```bash
# Clone repository
git clone https://github.com/notprime/pansharpening_evaluation.git
cd pansharpening-metrics

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Start

### Command Line (Simplest)

```bash
# Basic usage
python main.py sharp.tif pan.tif hs.tif

# With configuration
python main.py sharp.tif pan.tif hs.tif --config config.yaml

# Save results
python main.py sharp.tif pan.tif hs.tif --output results.json
```

### Python API (Recommended for Scripts)
First load the data, then preprocess the cubes, and finally compute the metrics. Change the `ratio` value to your needs.

```python
from pansharpening_metrics import compute_metrics, MetricsConfig
import rasterio

# Load images (H x W x C format)
sharp = rasterio.open('sharp.tif').read().transpose(1, 2, 0)
pan = rasterio.open('pan.tif').read().transpose(1, 2, 0)
hs = rasterio.open('hs.tif').read().transpose(1, 2, 0)
sharp, pan, hs = preprocess_for_metrics(sharp, pan, hs, ratio=6)

# Configure (optional)
config = MetricsConfig(q_block_size=32, n_workers=0.9)

# Compute metrics
metrics = compute_metrics(sharp, pan, hs, ratio=6, config=config)

# Results
print(f"D_lambda: {metrics['D_lambda']:.4f}")
print(f"D_s:      {metrics['D_s']:.4f}")
print(f"HQNR:     {metrics['HQNR']:.4f}")
```

## Usage

### Command Line Interface

**Basic usage:**
```bash
python main.py <sharp.tif> <pan.tif> <hs.tif>
```

**All options:**
```bash
python main.py sharp.tif pan.tif hs.tif \
    --ratio 6 \
    --sensor PRISMA \
    --config config.yaml \
    --output results.json \
    --q_block_size 32 \
    --n_workers 0.9 \
    --dask_chunk_size 128 128
```

**Arguments:**
- `sharp.tif`: Sharpened hyperspectral image path (required)
- `pan.tif`: Panchromatic image path (required)
- `hs.tif`: Low-resolution hyperspectral image path (required)
- `--ratio`: Resolution ratio (default: 6)
- `--sensor`: Sensor name (default: PRISMA)
- `--config`: YAML configuration file
- `--output, -o`: Save results to JSON
- `--q_block_size`: Override Q window size
- `--n_workers`: Override worker fraction (0-1)
- `--dask_chunk_size`: Override chunk size (H W)

At the moment, this are the current supported sensors:
- QuickBird;
- Ikonos;
- GeoEye1;
- WorldView-2;
- WorldView-3;
- PRISMA.

If you want to add your own sensor, you have to add it to the `resize_hs` and `resize_pan` functions in `downsampling.py`, by providing the Modulation Transfer Frequencies for the low-pass filtering. 

### Python API

**Import:**
```python
from pansharpening_metrics import (
    compute_metrics,      # Main function
    MetricsConfig,        # Configuration
    D_lambda_khan,        # Individual metrics
    D_s,
    normalize_inputs,     # Utilities
    preprocess_for_metrics
)
```

**Basic usage:**
```python
# With default configuration
metrics = compute_metrics(sharp, pan, hs, ratio=6)
```

**With custom configuration:**
```python
config = MetricsConfig(
    q_block_size=32,
    q_shift=32,
    dask_chunk_size=(128, 128),
    n_workers=0.9,
    exponent=1
)
metrics = compute_metrics(sharp, pan, hs, ratio=6, config=config)
```

**Using presets:**
```python
config = MetricsConfig.balanced()      # Recommended
# config = MetricsConfig.conservative() # Low memory
# config = MetricsConfig.aggressive()   # High performance

metrics = compute_metrics(sharp, pan, hs, config=config)
```


## Configuration

### Configuration File (YAML)

Create a `config.yaml`:

```yaml
q_block_size: 32          # Window size for Q/Q2n
q_shift: 32               # Stride for sliding windows
dask_chunk_size: [128, 128]  # Chunk size for parallelization
n_workers: 0.9            # Fraction of CPU cores to use
exponent: 1               # Exponent for distortion metrics
```

Use it:
```bash
python main.py sharp.tif pan.tif hs.tif --config config.yaml
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `q_block_size` | 32 | Window size for quality indices |
| `q_shift` | 32 | Stride for sliding windows |
| `dask_chunk_size` | (64, 64) | Chunk size for Dask parallelization |
| `n_workers` | 0.9 | Fraction of CPU cores to use (0-1) |
| `exponent` | 1 | Exponent for distortion metrics (Lp norm) |

### Presets

```python
# Conservative: low memory usage
config = MetricsConfig.conservative()
# q_block_size=32, chunk_size=(64,64), n_workers=0.5

# Balanced: recommended for most cases
config = MetricsConfig.balanced()
# q_block_size=32, chunk_size=(128,128), n_workers=0.9

# Aggressive: maximum performance
config = MetricsConfig.aggressive()
# q_block_size=32, chunk_size=(256,256), n_workers=0.95, overlapping windows
```

## Image Format Requirements

All images should be in **H × W × C** format (Height × Width × Channels). This is handled by the `load_data` function in the `main.py` file, so that:

- **Hyperspectral (HS)**: (H, W, C) at low resolution (e.g., 30m)
- **Panchromatic (PAN)**: (H×ratio, W×ratio, 1) at high resolution (e.g., 5m)
- **Sharpened**: (H×ratio, W×ratio, C) at high resolution

### Example for PRISMA (30m → 5m, ratio=6)

```python
hs.shape    # (512, 512, 180)    - 30m resolution
pan.shape   # (3072, 3072, 1)    - 5m resolution (512×6=3072)
sharp.shape # (3072, 3072, 180)  - 5m resolution
```

## Examples

### Example 1: Basic Usage

```bash
python main.py data/sharp.tif data/pan.tif data/hs.tif
```

Output:
```
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
```

### Example 2: Python Script

```python
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
```

## Understanding the Metrics

### D_lambda (Spectral Distortion)

Measures how well spectral relationships between bands are preserved.

- **Range**: [0, 1]
- **Interpretation**: Lower is better (0 = perfect spectral preservation)
- **Method**: Compares Q2n index between band pairs at low resolution

### D_s (Spatial Distortion)

Measures how well spatial details are preserved.

- **Range**: [0, 1]  
- **Interpretation**: Lower is better (0 = perfect spatial preservation)
- **Method**: Compares Q index at high and low resolutions for each band

### HQNR (Hybrid Quality with No Reference)

Overall quality combining spectral and spatial preservation.

- **Formula**: HQNR = (1 - D_λ) × (1 - D_s)
- **Range**: [0, 1]
- **Interpretation**: Higher is better (1 = perfect quality)

## Performance Tips

### Memory Management

**For large images (>5000×5000):**
```python
config = MetricsConfig(
    dask_chunk_size=(256, 256),  # Larger chunks
    n_workers=0.9
)
```

## Project Structure

```
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
```

## Troubleshooting

### Import Error

```bash
# Install package in development mode
pip install -e .
```

## References


References:

1. **Musto et al. (2024)**: "Advancing Prisma Pansharpening: A Deep Learning Approach with Synthetic Data Pretraining and Transfer Learning", WHISPERS
2. **Scarpa et al. (2021)**: "Full-resolution quality assessment for pansharpening", arXiv:2108.06144
3. **Garzelli & Nencini (2009)**: "Hypercomplex quality assessment of multi/hyper-spectral images", IEEE GRSL
4. **Alparone et al. (2008)**: "Multispectral and panchromatic data fusion assessment without reference"
5. **Vivone et al. (2020)**: "A new benchmark based on recent advances in multispectral pansharpening", IEEE GRSM


## License

MIT License - see LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{pansharpening_metrics,
  author = {Riccardo Musto},
  title = {Pansharpening Quality Metrics for Hyperspectral Images},
  year = {2025},
  url = {https://github.com/notprime/pansharpening_evaluation},
  version = {1.0.0}
}
```

## Author

Riccardo Musto

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
