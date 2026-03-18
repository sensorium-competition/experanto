import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .datasets import ChunkDataset
from .utils import (
    FastSessionDataLoader,
    LongCycler,
    MultiEpochsDataLoader,
    SessionConcatDataset,
)

logger = logging.getLogger(__name__)


def get_multisession_dataloader(
    paths: List[str],
    configs: Optional[Union[DictConfig, Dict, List[Union[DictConfig, Dict]]]] = None,
    shuffle_keys: bool = False,
    **kwargs,
) -> LongCycler:
def get_multisession_concat_dataloader(...):
    """Create a concatenated multi-session dataloader.

    Unlike :func:`get_multisession_dataloader`, ...
    """

    # Input validation
    if not paths or not isinstance(paths, list):
        raise ValueError("paths must be a non-empty list of strings")

    if any(not isinstance(p, str) for p in paths):
        raise TypeError("All elements in paths must be strings")

    if configs is None and "config" in kwargs:
        configs = kwargs.pop("config")

    if isinstance(configs, list) and len(configs) != len(paths):
     raise ValueError("Length of configs must match length of paths")

    # Create datasets
    datasets = []
    session_names = []

    start_time = time.time()
    for i, (path, cfg) in enumerate(zip(paths, configs)):
        # Create dataset with deterministic seed
        path_hash = hash(path) % 10000
        dataset_seed = seed + path_hash if seed is not None else None

        # Set specific seed for this dataset if needed
        if hasattr(cfg.get("dataset", {}), "seed") and dataset_seed is not None:
            cfg["dataset"]["seed"] = dataset_seed
        if "dataset" in cfg:
            cfg = cfg["dataset"]
        try:
            # Assuming ChunkDataset is defined elsewhere
            dataset = ChunkDataset(path, **cfg)
            session_name = dataset.data_key

            # Only add datasets with non-zero length
            if len(dataset) > 0:
                datasets.append(dataset)
                session_names.append(session_name)
        except Exception as e:
             logger.warning(f"Error creating dataset for {path}: {str(e)}")

    if not datasets:
        return None

    # Create the concatenated dataset
    concat_dataset = SessionConcatDataset(datasets, session_names)

    # Get dataloader config from the first config
    if dataloader_config is None:
        dataloader_config = dict(configs[0].get("dataloader", {}))

    # Create the dataloader with our simplified implementation
    return FastSessionDataLoader(dataset=concat_dataset, seed=seed, **dataloader_config)
