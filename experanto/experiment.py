from __future__ import annotations

import logging
import re
from collections import namedtuple
from collections.abc import Sequence
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .configs import DEFAULT_MODALITY_CONFIG
from .interpolators import Interpolator
from .intervals import TimeInterval

log = logging.getLogger(__name__)


class Experiment:
    def __init__(
        self,
        root_folder: str,
        modality_config: dict = DEFAULT_MODALITY_CONFIG,
        cache_data: bool = False,
        interpolate_precision: int = 5,
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
        self.scale_precision = 10**interpolate_precision
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
            dev = Interpolator.create(
                d,
                cache_data=self.cache_data,
                **self.modality_config[d.name]["interpolation"],
            )
            self.devices[d.name] = dev
            self.start_time = dev.start_time
            self.end_time = dev.end_time
            log.info("Parsing finished")

    @property
    def device_names(self):
        return tuple(self.devices.keys())

    def interpolate(
        self, times: slice, device: str = None
    ) -> tuple[np.ndarray, np.ndarray]:
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

    def get_data_for_interval(
        self,
        interval: TimeInterval,
        target_sampling_rates: Optional[Union[float, dict[str, float]]] = None,
        devices: Optional[Union[str, List[str]]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieve interpolated data for a given time interval across specified devices.

        Parameters
        ----------
        interval : TimeInterval
            The time interval for which data should be retrieved.
        target_sampling_rates : float or dict[str, float], optional
            The target sampling rate(s) in Hz. If a single float is provided, it is applied to all devices.
            If a dictionary is provided, it should map device names to their respective sampling rates.
            If None or a device is not specified in the dictionary, the default sampling rate from the modality config is used.
        devices : str or list of str, optional
            The device(s) to retrieve data for. If None, all available devices (`self.devices`) are used.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary mapping each device name to its corresponding interpolated data as a NumPy array.

        Raises
        ------
        AssertionError
            If a specified device is not found in `self.devices`.
        ValueError
            If no sampling rate is specified or available for a device.
        """
        devices = self._resolve_devices(devices)
        target_sampling_rates = self._resolve_sampling_rates(devices, target_sampling_rates)

        out = {}
        for device in devices:
            target_sampling_rate = target_sampling_rates[device]
            if target_sampling_rate is None:
                raise ValueError(
                    f"Target sampling rate for device '{device}' is not specified."
                )

            start_time = int(round(interval.start * self.scale_precision))
            end_time = int(round(interval.end * self.scale_precision))
            offset = int(
                round(
                    self.modality_config[device].get("offset", 0) * self.scale_precision
                )
            )
            time_delta = int(round((1.0 / target_sampling_rate) * self.scale_precision))
            # Generate times as ints - important as for np.floats the summation is not associative
            times = np.arange(start_time + offset, end_time + offset, time_delta)
            # Scale everything back to truncated values
            times = times.astype(np.float64) / self.scale_precision

            data, _ = self.interpolate(times, device=device)
            out[device] = data
        return out

    def get_data_for_chunks(
        self,
        start_time: float,
        chunk_sizes: Optional[Union[int, dict[str, int]]] = None,
        target_sampling_rates: Optional[Union[float, dict[str, float]]] = None,
        devices: Optional[Union[str, List[str]]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieve interpolated data for a fixed number of timesteps (chunk size) per device.

        Parameters
        ----------
        chunk_sizes : int or dict[str, int], optional
            Number of time steps to retrieve per device. If a single int is provided, it is used for all devices.
            If a dictionary is provided, it should map device names to their respective number of timesteps.
            If None, default chunk sizes defined in the configuration are used.
        target_sampling_rates : float or dict[str, float], optional
            Target sampling rate(s) in Hz. If a single float is provided, it is applied to all devices.
            If a dictionary is provided, it should map device names to their respective sampling rates.
            If None or a device is not specified in the dictionary, the default sampling rate from the modality config is used.
        devices : str or list of str, optional
            Devices to retrieve data for. If None, all available devices (`self.devices`) are used.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary mapping each device name to its corresponding interpolated data array 
            of shape `(chunk_size, ...)`.

        Raises
        ------
        AssertionError
            If a specified device is not found in `self.devices`.
        ValueError
            If no sampling rate is specified or available for a device.
        """
        devices = self._resolve_devices(devices)
        chunk_sizes = self._resolve_chunk_sizes(devices, chunk_sizes)
        target_sampling_rates = self._resolve_sampling_rates(devices, target_sampling_rates)

        start_time = int(round(start_time * self.scale_precision))
        out = {}
        timestamps = {}
        for device in devices:
            chunk_size = chunk_sizes[device]
            target_sampling_rate = target_sampling_rates[device]
            if target_sampling_rate is None:
                raise ValueError(
                    f"Target sampling rate for device '{device}' is not specified."
                )

            offset = int(
                round(
                    self.modality_config[device].get("offset", 0) * self.scale_precision
                )
            )
            time_delta = int(round((1.0 / target_sampling_rate) * self.scale_precision))
            # Generate times as ints - important as for np.floats the summation is not associative
            times = start_time + offset + np.arange(chunk_size) * time_delta
            # Scale everything back to truncated values
            times = times.astype(np.float64) / self.scale_precision

            data, _ = self.interpolate(times, device=device)
            out[device] = data
            timestamps[device] = times
        return out, timestamps

    def _resolve_devices(self, devices):
        if devices is None:
            return list(self.devices.keys())
        if isinstance(devices, str):
            devices = [devices]
        for device in devices:
            assert device in self.devices, f"Unknown device '{device}'"
        return devices

    def _resolve_sampling_rates(self, devices, rates):
        if rates is None:
            return {
                d: self.modality_config[d].get("sampling_rate") for d in devices
            }
        elif isinstance(rates, (int, float)):
            return {d: rates for d in devices}
        else:
            return rates

    def _resolve_chunk_sizes(self, devices, chunk_sizes):
        if chunk_sizes is None:
            return {
                d: self.modality_config[d].get("chunk_size") for d in devices
            }
        elif isinstance(chunk_sizes, (int, float)):
            return {d: int(chunk_sizes) for d in devices}
        else:
            return chunk_sizes

    def _get_device_offset(self, device: str) -> int:
        return int(round(self.modality_config[device].get("offset", 0) * self.scale_precision))
