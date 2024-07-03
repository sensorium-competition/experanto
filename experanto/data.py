import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from .experiment import Experiment


class Mouse2pDataset(Dataset):
    def __init__(self, root_folder: str, sampling_rate: float, chunk_size: int) -> None:
        self.root_folder = Path(root_folder)
        self.sampling_rate = sampling_rate
        self.chunk_size = chunk_size
        self._experiment = Experiment(root_folder, sampling_rate)
        self._first_stim_sample, self._last_stim_sample = self._experiment.get_valid_range("screen")

    def __len__(self):
        return int((self._last_stim_sample - self._first_stim_sample) / self.chunk_size)

    def __getitem__(self, idx):
        s = idx * self.chunk_size
        chunk_idx = slice(s, s + self.chunk_size)
        return list(self._experiment[chunk_idx].values())

    def get_device_names(self):
        return self._experiment.device_names
    