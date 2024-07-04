import numpy as np
from collections import namedtuple
from collections.abc import Sequence
from pathlib import Path
from .interpolators import Interpolator
import re

Interval = namedtuple("Interval", ["start", "end"])


class Experiment(Sequence):

    def __init__(self, root_folder: str, sampling_rate: float) -> None:
        self.root_folder = Path(root_folder)
        self.sampling_rate = sampling_rate
        self._devices = dict()
        self.device_names = list()
        self.start_time = np.inf
        self.end_time = -np.inf
        self._load_devices()


    def _load_devices(self) -> None:
        # Populate devices by going through subfolders
        # Assumption: blocks are sorted by start time
        device_folders = [d for d in self.root_folder.iterdir() if d.is_dir()]
        self.device_names = [f.name for f in device_folders]
        for d in device_folders:
            print("Parsing {} data... ".format(d.name), end="")
            self._devices[d.name] = Interpolator.create(d)
            ts = self._devices[d.name].timestamps.flatten()
            self.start_time = min(self.start_time, ts[0])
            self.end_time = max(self.end_time, ts[-1])
            print("done")

        self._sample_times = np.arange(self.start_time, self.end_time, 1.0 / self.sampling_rate) 


    def __getitem__(self, idx) -> dict:
        # allow indexing with e[idx,dev] as a shortcut for e[idx][dev]
        if isinstance(idx, tuple) and len(idx) == 2:
            idx, dev = idx
        else:
            dev = None
        
        if isinstance(idx, int):
            idx = slice(idx, idx+1)

        assert isinstance(idx, slice) and (idx.step is None or idx.step == 1), \
            "Only integer indices or slices with step 1 are supported"
        assert isinstance(dev, str) or dev is None, "Second index must be a string"

        t = self._sample_times[idx]

        if dev is None:
            return_value = {}
            for dev, interpolator in self._devices.items():
                values, valid = interpolator.interpolate(self._sample_times[idx])
                return_value[dev] = np.full((len(t), ) + values.shape[1:], np.nan, dtype=values.dtype)
                return_value[dev][valid] = values
        elif isinstance(dev, str):
            assert dev in self._devices, "Unknown device '{}'".format(dev)
            values, valid = self._devices[dev].interpolate(self._sample_times[idx])
            return_value = np.full((len(t), ) + values.shape[1:], np.nan, dtype=values.dtype)
            return_value[valid] = values

        return return_value

    def __len__(self) -> int:
        return len(self._sample_times)

    def get_sample_index(self, t):
        return np.searchsorted(self._sample_times, t)

    def get_valid_range(self, device_name) -> tuple:
        s = np.searchsorted(self._sample_times, self._devices[device_name][0].timestamps[0])
        e = np.searchsorted(self._sample_times, self._devices[device_name][-1].timestamps[-1])
        return s, e
