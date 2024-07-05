from __future__ import annotations

from collections import namedtuple, Iterable
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from .experiment import Experiment


class Mouse2pChunkedDataset(Dataset):
    def __init__(self, root_folder: str, sampling_rate: float, chunk_size: int) -> None:
        self.root_folder = Path(root_folder)
        self.sampling_rate = sampling_rate
        self.chunk_size = chunk_size
        self._experiment = Experiment(root_folder)
        self.device_names = self._experiment.device_names
        self.start_time, self.end_time = self._experiment.get_valid_range("screen")
        self._sample_times = np.arange(
            self.start_time, self.end_time, 1.0 / self.sampling_rate
        )
        self.DataPoint = namedtuple("DataPoint", self.device_names)

    def __len__(self):
        return int(len(self._sample_times) / self.chunk_size)

    def __getitem__(self, idx):
        s = idx * self.chunk_size
        times = self._sample_times[s : s + self.chunk_size]
        data, _ = self._experiment.interpolate(times)
        return self.DataPoint(*list(data.values()))
