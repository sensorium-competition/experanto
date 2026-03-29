"""
Experanto: Interpolating recordings and stimuli in neuroscience experiments.

Main exports:
    - Experiment: Load and query multi-modal experiment data
    - ChunkDataset: PyTorch Dataset for training
    - Interpolator: Base class for modality interpolators
"""

from .experiment import Experiment
from .datasets import ChunkDataset
from .interpolators import Interpolator

__all__ = [
    "Experiment",
    "ChunkDataset", 
    "Interpolator",
]

__version__ = "0.1.0"