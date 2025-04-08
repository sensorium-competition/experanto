from typing import Any, List, Optional, Tuple, Union, Dict
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import numpy as np
import time
import warnings

from .datasets import ChunkDataset

from .utils import MultiEpochsDataLoader, LongCycler, SessionConcatDataset, FastSessionDataLoader




def get_multisession_dataloader(paths: List[str],
                                configs: Union[DictConfig, Dict, List[Union[DictConfig, Dict]]] = None,
                                shuffle_keys: bool = False,
                                **kwargs) -> DataLoader:
    """
    Create a multisession dataloader from a list of paths and corresponding configs.
    Args:
        paths (List[str]): List of paths to the datasets.
        configs (Union[DictConfig, Dict, List[Union[DictConfig, Dict]]]): Configuration for each dataset.
            If a single config is provided, it will be applied to all datasets.
            If a list is provided, it should match the length of paths.
        shuffle_keys (bool): Whether to shuffle the keys of the dataloaders.
        **kwargs: Additional keyword arguments for dataset and dataloader configuration.
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
        dataloaders[dataset_name] = MultiEpochsDataLoader(dataset,
                                                          **cfg.dataloader,)



    return LongCycler(dataloaders)


def get_multisession_concat_dataloader(paths: List[str],
                                       configs: Union[Dict, List[Dict]] = None,
                                       seed: Optional[int] = 0,
                                       **kwargs) -> 'FastSessionDataLoader':
    """
    Creates a multi-session dataloader using SessionConcatDataset and SessionDataLoader.
    Returns (session_key, batch) pairs during iteration.

    Args:
        paths: List of paths to dataset files
        configs: Configuration for datasets (single config or list of configs)
        seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
        prefetch_factor: Prefetch factor for data loading
        **kwargs: Additional arguments

    Returns:
        SessionDataLoader instance or None if no valid datasets found
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
        # Extract session name
        if "dynamic" in path:
            session_name = path.split("dynamic")[1].split("-Video")[0]
        elif "_gaze" in path:
            session_name = path.split("_gaze")[0].split("datasets/")[1]
        else:
            session_name = f"session_{i}"

        # Create dataset with deterministic seed
        path_hash = hash(path) % 10000
        dataset_seed = seed + path_hash if seed is not None else None

        # Set specific seed for this dataset if needed
        if hasattr(cfg.get('dataset', {}), 'seed') and dataset_seed is not None:
            cfg['dataset']['seed'] = dataset_seed

        # Create and append the dataset
        try:
            # Assuming ChunkDataset is defined elsewhere
            dataset = globals()["ChunkDataset"](path, **cfg['dataset'])

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
    dl_config = dict(configs[0].get('dataloader', {}))

    # Create the dataloader with our simplified implementation
    return FastSessionDataLoader(
        dataset=concat_dataset,
        seed=seed,
        **dl_config
    )
