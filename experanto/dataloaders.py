from typing import Any, List, Optional, Tuple
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from .datasets import ChunkDataset,MonkeyFixation, MonkeyFixationGazeCrop
from .utils import MultiEpochsDataLoader, LongCycler


def get_multisession_dataloader(paths, config: DictConfig) -> DataLoader:
    dataloaders = {}
    for i, path in enumerate(paths):
        dataset_name = path.split("dynamic")[1].split("-Video")[0]
        dataset = ChunkDataset(path, **config.dataset,)
        dataloaders[dataset_name] = MultiEpochsDataLoader(dataset,
                                               **config.dataloader,
                                               )
    return LongCycler(dataloaders)

def get_multisession_monkeyloader(
    paths,
    dataset_cls=MonkeyFixation,
    config: DictConfig = None
) -> DataLoader:
    dataloaders = {}
    for i, path in enumerate(paths):
        dataset_name = path.split("dynamic")[1].split("-Video")[0]
        dataset = dataset_cls(path, **config.dataset)
        dataloaders[dataset_name] = MultiEpochsDataLoader(dataset, **config.dataloader)

    return LongCycler(dataloaders)


def get_multisession_monkeyloaders(paths, dataset_cls=MonkeyFixation, config: DictConfig = None, split=True, return_val=True):
    """
    Returns a LongCycler of MultiEpochsDataLoaders across multiple sessions.
    
    Parameters:
        paths (list): List of paths to session root folders.
        dataset_cls (type): Dataset class to instantiate (e.g., MonkeyFixation).
        config (DictConfig): Hydra-style config with 'dataset' and 'dataloader' keys.
        split (bool): Whether to split into train/val using .split() method of dataset.
        return_val (bool): Whether to return validation loaders as well.
    
    Returns:
        train_cycler (LongCycler): Combined loader for training.
        val_cycler (LongCycler, optional): Combined loader for validation (if return_val=True).
    """
    train_loaders = {}
    val_loaders = {}

    for path in paths:
        dataset_name = path.split("dynamic")[1].split("-Video")[0]
        dataset = dataset_cls(path, **config.dataset)

        if split:
            train_ds, val_ds = dataset.split(train_frac=0.8, seed=42)
        else:
            train_ds, val_ds = dataset, None

        train_loader = MultiEpochsDataLoader(train_ds, **config.dataloader)
        train_loaders[dataset_name] = train_loader

        if return_val and val_ds is not None:
            val_loader = MultiEpochsDataLoader(val_ds, **config.dataloader)
            val_loaders[dataset_name] = val_loader

    train_cycler = LongCycler(train_loaders)
    
    if return_val:
        val_cycler = LongCycler(val_loaders)
        return train_cycler, val_cycler
    
    return train_cycler

def build_nnvision_loader_dict(train_cycler, val_cycler=None, test_cycler=None):
    loader_dict = {
        "train": {'all_sessions':train_cycler},
    }
    
    if val_cycler is not None:
        loader_dict["val"] = {'all_sessions':val_cycler}

    if test_cycler is not None:
        loader_dict["test"] = {'all_sessions':test_cycler}

    # Get n_neurons from training loader
    n_neurons = {
        name: next(iter(loader)).get("responses", torch.empty(0)).shape[1]
        for name, loader in train_cycler.loaders.items()
    }
    loader_dict["n_neurons"] = {'all_sessions':n_neurons}

    return loader_dict