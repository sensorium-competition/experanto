import numpy as np
from collections import namedtuple
from collections.abc import Sequence
from pathlib import Path
from .interpolators import Interpolator
import re


class Experiment:

    def __init__(self, root_folder: str) -> None:
        self.root_folder = Path(root_folder)
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
            dev = Interpolator.create(d)
            self._devices[d.name] = dev
            self.start_time = dev.start_time
            self.end_time = dev.end_time
            print("done")

    def interpolate(self, times: slice, device=None) -> tuple[np.ndarray, np.ndarray]:
        if device is None:
            values = {}
            valid = {}
            for d, interp in self._devices.items():
                values[d], valid[d] = interp.interpolate(times)
        elif isinstance(device, str):
            assert device in self._devices, "Unknown device '{}'".format(device)
            values, valid = self._devices[device].interpolate(times)
        return values, valid

    # def __getitem__(self, idx) -> dict:
    #     # allow indexing with e[idx,dev] as a shortcut for e[idx][dev]
    #     if isinstance(idx, tuple) and len(idx) == 2:
    #         idx, dev = idx
    #     else:
    #         dev = None
        
    #     if isinstance(idx, int):
    #         idx = slice(idx, idx+1)

    #     assert isinstance(idx, slice) and (idx.step is None or idx.step == 1), \
    #         "Only integer indices or slices with step 1 are supported"
    #     assert isinstance(dev, str) or dev is None, "Second index must be a string"

    #     t = self._sample_times[idx]

    #     if dev is None:
    #         return_value = {}
    #         for dev, interpolator in self._devices.items():
    #             values, valid = interpolator.interpolate(self._sample_times[idx])
    #             return_value[dev] = np.full((len(t), ) + values.shape[1:], np.nan, dtype=values.dtype)
    #             return_value[dev][valid] = values
    #     elif isinstance(dev, str):
    #         assert dev in self._devices, "Unknown device '{}'".format(dev)
    #         values, valid = self._devices[dev].interpolate(self._sample_times[idx])
    #         return_value = np.full((len(t), ) + values.shape[1:], np.nan, dtype=values.dtype)
    #         return_value[valid] = values

    #     return return_value

    # def __len__(self) -> int:
    #     return len(self._sample_times)

    def get_valid_range(self, device_name) -> tuple:
        return self._devices[device_name].start_time, self._devices[device_name].end_time
