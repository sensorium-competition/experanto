from typing import Any, List, Optional, Tuple
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .datasets import ChunkDataset
from .utils import MultiEpochsDataLoader, LongCycler


def get_multisession_dataloader(paths, config: DictConfig,) -> DataLoader:
    dataloaders = {}
    for i, path in enumerate(paths):
        dataset_name = i
        dataset = ChunkDataset(path, **config.dataset,)
        dataloaders[dataset_name] = MultiEpochsDataLoader(dataset,
                                               **config.dataloader,
                                               )
    return LongCycler(dataloaders)