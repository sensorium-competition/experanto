from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from .configs import DEFAULT_MODALITY_CONFIG
from .interpolators import Interpolator

log = logging.getLogger(__name__)


class Experiment:
    def __init__(
        self,
        root_folder: str,
        modality_config: dict = DEFAULT_MODALITY_CONFIG,
        cache_data: bool = False,
    ) -> None:
        """
        root_folder: path to the data folder
        modality_config: dict for configuring interpolators
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
        # Assumption: blocks are sorted by start time
        device_folders = [d for d in self.root_folder.iterdir() if d.is_dir()]

        for d in device_folders:
            if d.name not in self.modality_config:
                log.info(f"Skipping {d.name} data... ")
                continue

            log.info(f"Parsing {d.name} data... ")
            interp_conf = self.modality_config[d.name]["interpolation"]

            if (
                isinstance(interp_conf, (dict, DictConfig))
                and "_target_" in interp_conf
            ):
                dev = instantiate(
                    interp_conf, root_folder=d, cache_data=self.cache_data
                )
                if not isinstance(dev, Interpolator):
                    raise ValueError(
                        "Instantiated object must inherit from Interpolator class."
                    )

            elif isinstance(interp_conf, Interpolator):
                dev = interp_conf

            else:
                warnings.warn(
                    "Falling back to original Interpolator creation logic.", UserWarning
                )
                dev = Interpolator.create(d, cache_data=self.cache_data, **interp_conf)

            self.devices[d.name] = dev
            self.start_time = min(self.start_time, dev.start_time)
            self.end_time = max(self.end_time, dev.end_time)
            log.info("Parsing finished")

    @property
    def device_names(self):
        return tuple(self.devices.keys())

    def interpolate(
        self,
        times: np.ndarray,
        device: Union[str, Interpolator, None] = None,
        return_valid: bool = False,
    ) -> Union[tuple[dict, dict], dict, tuple[np.ndarray, np.ndarray], np.ndarray]:
        if device is None:
            values, valid = {}, {}
            for d, interp in self.devices.items():
                res = interp.interpolate(times, return_valid=return_valid)
                if return_valid:
                    vals, vlds = res
                    values[d], valid[d] = vals, vlds
                else:
                    values[d] = res
            return (values, valid) if return_valid else values

        elif isinstance(device, str):
            if device not in self.devices:
                raise KeyError(f"Unknown device '{device}'")
            return self.devices[device].interpolate(times, return_valid=return_valid)

        raise ValueError(f"Unsupported device type: {type(device)}")

    def get_valid_range(self, device_name: str) -> tuple[float, float]:
        return tuple(self.devices[device_name].valid_interval)
