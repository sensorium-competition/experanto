from typing import Any, List, Optional, Tuple, Union, Dict
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .datasets import ChunkDataset

from .utils import MultiEpochsDataLoader, LongCycler, StatefulLongCycler, PooledStatefulDataLoader



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


def get_pooled_multisession_dataloader(paths: List[str],
                                       configs: Union[DictConfig, Dict, List[Union[DictConfig, Dict]]] = None,
                                       seed: Optional[int] = 0,
                                       max_workers: int = 20,
                                       **kwargs) -> StatefulLongCycler:
    """
    Creates a multi-session dataloader with deterministic seeds for reproducibility.
    """
    if configs is None and "config" in kwargs:
        configs = kwargs.pop("config")

    # Convert single config to list for uniform handling
    if isinstance(configs, (DictConfig, dict)):
        configs = [configs] * len(paths)

    # Initialize the worker pool
    PooledStatefulDataLoader.configure_pool(max_workers=max_workers)

    # Create a reproducible hash-based seed for each dataset based on its path
    # This ensures consistent seeds even if dataset order changes
    base_rng = np.random.RandomState(seed)

    dataloaders = {}
    for i, (path, cfg) in enumerate(zip(paths, configs)):
        # Extract dataset name in a consistent way
        if "dynamic" in path:
            dataset_name = path.split("dynamic")[1].split("-Video")[0]
        elif "_gaze" in path:
            dataset_name = path.split("_gaze")[0].split("datasets/")[1]
        else:
            dataset_name = f"session_{i}"

        # Generate a deterministic seed based on the dataset path
        # This ensures the same dataset always gets the same seed regardless of order
        path_hash = hash(path) % 10000  # Convert path to a consistent number
        dataset_seed = seed + path_hash  # Combine with base seed

        # Create dataset
        dataset = globals()["ChunkDataset"](path, **cfg.dataset)

        # Get dataloader config
        dl_config = dict(cfg.dataloader)

        # Calculate workers - make this deterministic too
        if 'num_workers' in dl_config:
            requested_workers = dl_config['num_workers']
            # Base allocation on dataset name for consistency
            workers_for_dataset = max(1, max_workers // len(paths))
            dl_config['num_workers'] = min(requested_workers, workers_for_dataset)

        # Use PooledStatefulDataLoader with deterministic seed
        dataloaders[dataset_name] = PooledStatefulDataLoader(
            dataset,
            seed=dataset_seed,  # Use deterministic seed based on path
            **dl_config
        )

    # Use StatefulLongCycler with the pooled dataloaders
    return StatefulLongCycler(dataloaders, seed=seed)