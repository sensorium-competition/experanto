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
    configs: Union[DictConfig, Dict, List[Union[DictConfig, Dict]]] = None,
    shuffle_keys: bool = False,
    **kwargs,
) -> DataLoader:
    """Create a multi-session dataloader from multiple experiment paths.

    Creates a :class:`ChunkDataset` for each path and wraps them in a
    :class:`LongCycler` that yields ``(session_key, batch)`` pairs.
    The cycler continues until the longest session is exhausted.

    Parameters
    ----------
    paths : list of str
        Paths to experiment directories.
    configs : dict, DictConfig, or list, optional
        Configuration for each dataset. If a single config is provided,
        it will be applied to all datasets. If a list is provided, it
        should match the length of ``paths``. Each config should have
        ``dataset`` and ``dataloader`` keys.
    shuffle_keys : bool, default=False
        Whether to shuffle the order of session keys.
    **kwargs
        Additional keyword arguments. Supports ``config`` as an alias
        for ``configs``.

    Returns
    -------
    LongCycler
        A dataloader-like object that yields ``(session_key, batch)`` tuples.
        Iterates until the longest session is exhausted.

    See Also
    --------
    get_multisession_concat_dataloader : Alternative that concatenates sessions.
    LongCycler : The underlying multi-session iterator.

    Examples
    --------
    >>> from experanto.dataloaders import get_multisession_dataloader
    >>> from experanto.configs import DEFAULT_CONFIG
    >>> paths = ['/path/to/exp1', '/path/to/exp2']
    >>> loader = get_multisession_dataloader(paths, configs=DEFAULT_CONFIG)
    >>> for session_key, batch in loader:
    ...     print(f"Session: {session_key}, batch shape: {batch['responses'].shape}")
    """

    if configs is None and "config" in kwargs:
        configs = kwargs.pop("config")

    # Convert single config to list for uniform handling
    if isinstance(configs, (DictConfig, dict)):
        configs = [configs] * len(paths)

    dataloaders = {}
    for i, (path, cfg) in enumerate(zip(paths, configs)):
        # TODO use saved meta dict to find data key
        if "dynamic" in path:
            dataset_name = path.split("dynamic")[1].split("-Video")[0]
        elif "_gaze" in path:
            dataset_name = path.split("_gaze")[0].split("datasets/")[1]
        else:
            dataset_name = f"session_{i}"
        dataset = ChunkDataset(path, **cfg.dataset)
        dataloaders[dataset_name] = MultiEpochsDataLoader(
            dataset,
            **cfg.dataloader,
        )

    return LongCycler(dataloaders)


def get_multisession_concat_dataloader(
    paths: List[str],
    configs: Union[Dict, List[Dict]] = None,
    seed: Optional[int] = 0,
    dataloader_config: Optional[Dict] = None,
    **kwargs,
) -> "FastSessionDataLoader":
    """Create a concatenated multi-session dataloader.

    Unlike :func:`get_multisession_dataloader`, this function concatenates
    all sessions into a single dataset and uses batch sampling to ensure
    each batch contains samples from only one session. This is more
    memory-efficient and provides better shuffling across sessions.

    Parameters
    ----------
    paths : list of str
        Paths to experiment directories.
    configs : dict or list of dict, optional
        Configuration for each dataset. If a single config is provided,
        it will be applied to all datasets. Each config should have
        ``dataset`` and ``dataloader`` keys.
    seed : int, default=0
        Random seed for reproducibility. Each dataset gets a deterministic
        seed derived from this value and its path hash.
    dataloader_config : dict, optional
        Configuration for the dataloader (batch_size, num_workers, etc.).
        If None, uses the dataloader config from the first config.
    **kwargs
        Additional keyword arguments. Supports ``config`` as an alias
        for ``configs``.

    Returns
    -------
    FastSessionDataLoader or None
        A dataloader that yields ``(session_key, batch)`` tuples.
        Returns None if no valid datasets could be created.

    See Also
    --------
    get_multisession_dataloader : Alternative using separate dataloaders.
    FastSessionDataLoader : The underlying dataloader implementation.
    SessionConcatDataset : Dataset that concatenates multiple sessions.

    Examples
    --------
    >>> from experanto.dataloaders import get_multisession_concat_dataloader
    >>> from experanto.configs import DEFAULT_CONFIG
    >>> paths = ['/path/to/exp1', '/path/to/exp2', '/path/to/exp3']
    >>> loader = get_multisession_concat_dataloader(paths, configs=DEFAULT_CONFIG)
    >>> for session_key, batch in loader:
    ...     print(f"Session: {session_key}")
    """
    if configs is None and "config" in kwargs:
        configs = kwargs.pop("config")

    # Convert single config to list for uniform handling
    if not isinstance(configs, list):
        configs = [configs] * len(paths)

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
            warnings.warn(f"Error creating dataset for {path}: {str(e)}")

    if not datasets:
        return None

    # Create the concatenated dataset
    concat_dataset = SessionConcatDataset(datasets, session_names)

    # Get dataloader config from the first config
    if dataloader_config is None:
        dataloader_config = dict(configs[0].get("dataloader", {}))

    # Create the dataloader with our simplified implementation
    return FastSessionDataLoader(dataset=concat_dataset, seed=seed, **dataloader_config)
