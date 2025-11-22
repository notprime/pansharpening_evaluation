"""
Configuration class for metrics computation
"""

class MetricsConfig:
    """
    Configuration parameters for pansharpening metrics computation.
    
    Attributes:
        q_block_size (int): Window size for Q2n/Q index calculation.
            Default: 32. Typical values: 16, 32, 64.
            Larger values = smoother results but less spatial detail.
            
        q_shift (int): Stride for sliding window in Q2n calculation.
            Default: 32 (non-overlapping windows if also q_block_size = 32).
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
            
    @classmethod
    def balanced(cls):
        """Balanced configuration (recommended)."""
        return cls(q_block_size=32, q_shift=32, dask_chunk_size=(128, 128), 
                   n_workers=0.9, exponent=1)
    
    @classmethod
    def conservative(cls):
        """Conservative configuration for low-memory systems."""
        return cls(q_block_size=32, q_shift=32, dask_chunk_size=(64, 64), 
                   n_workers=0.5, exponent=1)
    
    @classmethod
    def aggressive(cls):
        """Aggressive configuration for high-end systems."""
        return cls(q_block_size=32, q_shift=16, dask_chunk_size=(256, 256), 
                   n_workers=0.95, exponent=1)
    
    # You can also implement your own configuration!
    @classmethod
    def custom(cls):
        """Aggressive configuration for high-end systems."""
        return cls(q_block_size=32, q_shift=32, dask_chunk_size=(64, 64), 
                   n_workers=0.9, exponent=1)