from typing import Any, List, Optional, Tuple, Union, Dict
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .datasets import ChunkDataset
from .utils import MultiEpochsDataLoader, LongCycler, StatefulDataLoader, StatefulLongCycler


def get_multisession_dataloader(paths: List[str],
                                configs: Union[DictConfig, Dict, List[Union[DictConfig, Dict]]] = None, 
                                **kwargs) -> DataLoader:

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


def get_stateful_multisession_dataloader(paths: List[str],
                                         configs: Union[DictConfig, Dict, List[Union[DictConfig, Dict]]] = None,
                                         seed: Optional[int] = 0,
                                         **kwargs) -> StatefulLongCycler:
    """
    Creates a multi-session dataloader using StatefulDataLoader and StatefulLongCycler
    for fault-tolerant training.

    Args:
        paths: List of paths to dataset files for each session.
        configs: A single config or a list of configs for each session.
                 Each config should contain 'dataset' and 'dataloader' keys.
        seed: Random seed for initializing dataloaders and cycler.
        **kwargs: Additional keyword arguments.

    Returns:
        A StatefulLongCycler instance wrapping the session dataloaders.
    """
    if configs is None and "config" in kwargs:
        configs = kwargs.pop("config")

    # Convert single config to list for uniform handling
    if isinstance(configs, (DictConfig, dict)):
        configs = [configs] * len(paths)

    dataloaders = {}
    for i, (path, cfg) in enumerate(zip(paths, configs)):
        # TODO: Refine dataset name extraction logic if needed
        if "dynamic" in path:
            dataset_name = path.split("dynamic")[1].split("-Video")[0]
        elif "_gaze" in path:
            dataset_name = path.split("_gaze")[0].split("datasets/")[1]
        else:
            dataset_name = f"session_{i}"

        dataset = ChunkDataset(path, **cfg.dataset)
        # Use StatefulDataLoader
        dataloaders[dataset_name] = StatefulDataLoader(dataset,
                                                       seed=seed + i, # Ensure different seeds per loader if desired
                                                       **cfg.dataloader)

    # Use StatefulLongCycler
    return StatefulLongCycler(dataloaders, seed=seed)

