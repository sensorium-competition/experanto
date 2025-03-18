from __future__ import annotations

import logging
import re
from collections import namedtuple
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .interpolators import Interpolator

log = logging.getLogger(__name__)

# see .configs.py for the definition of DEFAULT_MODALITY_CONFIG
DEFAULT_MODALITY_CONFIG = dict()

class Experiment:
    def __init__(
        self,
        root_folder: str,
        modality_config: dict = DEFAULT_MODALITY_CONFIG,
        cache_data: bool = False,
    ) -> None:
        """
        root_folder: path to the data folder
        interp_config: dict for configuring interpolators, like
            interp_config = {"screen": {...}, {"eye_tracker": {...}, }
        cache_data: if True, loads and keeps all trial data in memory
        """
        self.root_folder = Path(root_folder)
        self.devices = dict()
        self.start_time = np.inf
        self.end_time = -np.inf
        self.modality_config = modality_config
        self.cache_data = cache_data
        self._load_devices()

    def _load_devices(self) -> None:
        # Populate devices by going through subfolders
        # Assumption: blocks are sorted by start time
        device_folders = [d for d in self.root_folder.iterdir() if (d.is_dir())]

        for d in device_folders:
            if d.name not in self.modality_config:
                log.info(f"Skipping {d.name} data... ")
                continue
            log.info(f"Parsing {d.name} data... ")
            dev = Interpolator.create(d, **self.modality_config[d.name]["interpolation"], cache_data=self.cache_data)
            self.devices[d.name] = dev
            self.start_time = dev.start_time
            self.end_time = dev.end_time
            log.info("Parsing finished")

    @property
    def device_names(self):
        return tuple(self.devices.keys())

    def interpolate(self, times: slice, device=None) -> tuple[np.ndarray, np.ndarray]:
        if device is None:
            values = {}
            valid = {}
            for d, interp in self.devices.items():
                values[d], valid[d] = interp.interpolate(times)
        elif isinstance(device, str):
            assert device in self.devices, "Unknown device '{}'".format(device)
            values, valid = self.devices[device].interpolate(times)
        return values, valid

    def get_valid_range(self, device_name) -> tuple:
        return tuple(self.devices[device_name].valid_interval)
