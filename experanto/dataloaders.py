from typing import Any, List, Optional, Tuple, Union, Dict
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import numpy as np
import time

from .datasets import ChunkDataset

from .utils import MultiEpochsDataLoader, LongCycler, OptimizedSessionConcatDataset, SimpleStatefulDataLoader



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
                                       configs: Union[DictConfig, Dict, List[Union[DictConfig, Dict]]] = None,
                                       seed: Optional[int] = 0,
                                       num_workers = None,
                                       prefetch_factor = None,
                                       **kwargs) -> SimpleStatefulDataLoader:
    """
    Creates a multi-session dataloader using SessionConcatDataset and SimpleStatefulDataLoader.
    Returns (session_key, batch) pairs during iteration, just like the LongCycler.
    """
    if configs is None and "config" in kwargs:
        configs = kwargs.pop("config")

    # Convert single config to list for uniform handling
    if isinstance(configs, (DictConfig, dict)):
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
        if hasattr(cfg.dataset, 'seed') and dataset_seed is not None:
            cfg.dataset.seed = dataset_seed

        # Create and append the dataset
        try:
            dataset = globals()["ChunkDataset"](path, **cfg.dataset)
            # Only add datasets with non-zero length
            if len(dataset) > 0:
                datasets.append(dataset)
                session_names.append(session_name)
                print(f"Created dataset {i}: {session_name} from {path}, length = {len(dataset)}")
            else:
                print(f"Skipping empty dataset {i}: {session_name} from {path}")
        except Exception as e:
            warnings.warn(f"Error creating dataset for {path}: {str(e)}")

    if not datasets:
        print("No valid datasets found with non-zero length")
        return None

    print(f"Dataset creation took {time.time() - start_time:.2f} seconds")

    # Create the concatenated dataset
    concat_dataset = OptimizedSessionConcatDataset(datasets, session_names)

    # Get dataloader config from the first config
    dl_config = dict(configs[0].dataloader)
    if num_workers is not None:
        dl_config["num_workers"] = num_workers
    if prefetch_factor == 0:
        dl_config["prefetch_factor"] = None

    # Create the stateful dataloader
    return SimpleStatefulDataLoader(
        dataset=concat_dataset,
        seed=seed,
        **dl_config
    )


def maybe_get_validation_concat_loader(cfg, paths, max_sessions=None):
    """
    Creates validation dataloader using SessionConcatDataset approach.
    Returns (session_key, batch) pairs during iteration, just like the previous implementation.

    Args:
        cfg: Configuration object
        paths: List of paths to dataset files
        max_sessions: Maximum number of sessions to load

    Returns:
        SimpleStatefulDataLoader instance or None if no validation loaders could be created
    """
    config = deepcopy(cfg)
    config.dataset.modality_config.screen.sample_stride = config.dataset.modality_config.screen.chunk_size
    config.dataset.modality_config.screen.include_blanks = False
    config.dataset.modality_config.screen.valid_condition = {"tier": "validation"}

    # Set validation seed
    val_seed = cfg.get("seed", 42) + 10000 if cfg.get("seed") is not None else None

    # Limit the number of paths
    if max_sessions is not None and len(paths) > max_sessions:
        paths = paths[:max_sessions]

    valid_datasets = []
    valid_session_names = []

    for i, path in enumerate(paths):
        try:
            # Extract session name
            if "dynamic" in path:
                session_name = path.split("dynamic")[1].split("-Video")[0]
            elif "_gaze" in path:
                session_name = path.split("_gaze")[0].split("datasets/")[1]
            else:
                session_name = f"val_session_{i}"

            # Create dataset
            dataset = globals()["ChunkDataset"](path, **config.dataset)

            # Only add non-empty datasets
            if len(dataset) > 0:
                valid_datasets.append(dataset)
                valid_session_names.append(session_name)
                print(f"Added validation dataset: {session_name} (length = {len(dataset)})")
            else:
                print(f"Skipping empty validation dataset: {session_name}")

        except Exception as e:
            print(f"Error creating validation dataset for {path}: {str(e)}")

    if not valid_datasets:
        print("No valid validation datasets found with non-zero length")
        return None

    # Create the concatenated dataset
    concat_dataset = OptimizedSessionConcatDataset(valid_datasets, valid_session_names)

    # Get dataloader config
    dl_config = dict(config.dataloader)
    if num_workers is not None:
        dl_config["num_workers"] = num_workers
    if prefetch_factor == 0:
        dl_config["prefetch_factor"] = None

    # Create the stateful dataloader
    return SimpleStatefulDataLoader(
        dataset=concat_dataset,
        seed=val_seed,
        **dl_config
    )

