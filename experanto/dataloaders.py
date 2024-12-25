from typing import Any, List, Optional, Tuple
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .datasets import ChunkDataset
from .utils import MultiEpochsDataLoader, LongCycler


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
        dataset_name = f"session_{i}"
        dataset = ChunkDataset(path, **cfg.dataset)
        dataloaders[dataset_name] = MultiEpochsDataLoader(dataset,
                                                          **cfg.dataloader,)
    return LongCycler(dataloaders)