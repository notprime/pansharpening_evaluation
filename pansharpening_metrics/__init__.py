"""
Pansharpening Quality Metrics

Quality assessment metrics for hyperspectral pansharpening evaluation.

Main Features:
    - D_lambda: Spectral distortion index
    - D_s: Spatial distortion index
    - HQNR: Hybrid quality with no reference
"""

__version__ = "1.0.0"

from .config import MetricsConfig
from .metrics import (
    compute_metrics, 
    D_lambda_khan, D_s, 
    preprocess_for_metrics)
from .quality_indices import Q, Q2n_map

### DA RIVEDERE, capire cosa importare da utils

__all__ = [
    'MetricsConfig',
    'compute_metrics',
    'D_lambda_khan',
    'D_s',
    'preprocess_for_metrics',
    'Q',
    'Q2n_map',
    '__version__',
]