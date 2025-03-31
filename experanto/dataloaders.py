from typing import Any, List, Optional, Tuple, Union, Dict
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .datasets import ChunkDataset
from .utils import MultiEpochsDataLoader, LongCycler, ShuffledLongCycler


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
    if shuffle_keys:
        return ShuffledLongCycler(dataloaders, shuffle_keys=True)
    else:
        return LongCycler(dataloaders)
