import numpy as np
from pathlib import Path
from collections import namedtuple
from collections.abc import Sequence, abstractmethod

Interval = namedtuple("Interval", ["start", "end"])


class Block(Sequence):

    def __init__(self, root_folder: str, samplingrate: float) -> None:
        self.root_folder = Path(root_folder)
        self.samplingrate = samplingrate

    @property
    @abstractmethod
    def time_idx(self) -> np.ndarray:
        ...
        
    @abstractmethod
    def interpolate(self, times: np.ndarray) -> np.ndarray:
        ...

    def __getitem__(self, idx):
        return self.interpolate(self.time_idx[idx])

    def __len__(self) -> int:
        return self.time_idx.size


class DataBlock(Block):

    def __init__(self, root_folder: str, samplingrate: float, start_time=None) -> None: 
        super().__init__(root_folder, samplingrate)
        # load start and end times of stimuli
        self.start_times = np.load(self.root_folder / "start_times.npy")
        self.end_times = np.load(self.root_folder / "end_times.npy")

        # extract time interval of the block
        self.interval = Interval(self.start_times.min(), self.end_times.max())
        self.reset_time_index(start_time or self.interval.start)

    def reset_time_index(self, start_time) -> None:
        # create time index
        dt = 1.0 / self.samplingrate
        self._time_idx = np.arange(start_time, self.interval.end + dt, dt)



