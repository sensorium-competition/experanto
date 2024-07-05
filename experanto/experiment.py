import numpy as np
from collections import namedtuple
from collections.abc import Sequence
from pathlib import Path
from .interpolators import Interpolator
import re
import logging

log = logging.getLogger(__name__)


class Experiment:

    def __init__(self, root_folder: str) -> None:
        self.root_folder = Path(root_folder)
        self._devices = dict()
        self.start_time = np.inf
        self.end_time = -np.inf
        self._load_devices()

    def _load_devices(self) -> None:
        # Populate devices by going through subfolders
        # Assumption: blocks are sorted by start time
        device_folders = [d for d in self.root_folder.iterdir() if d.is_dir()]

        for d in device_folders:
            log.info(f"Parsing {d.name} data... ")
            dev = Interpolator.create(d)
            self._devices[d.name] = dev
            self.start_time = dev.start_time
            self.end_time = dev.end_time
            log.info("Parsing finished")
            
    @property
    def devive_names(self):
        return tuple(self._devices.keys())        

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

 

    def get_valid_range(self, device_name) -> tuple:
        return tuple(self._devices[device_name].valid_interval)
