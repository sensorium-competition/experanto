from __future__ import annotations

import functools
import importlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torchvision
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import Compose, Lambda, ToTensor

from .configs import DEFAULT_MODALITY_CONFIG
from .experiment import Experiment
from .interpolators import ImageTrial, VideoTrial
from .intervals import (
    TimeInterval,
    find_intersection_between_two_interval_arrays,
    get_stats_for_valid_interval,
)
from .utils import add_behavior_as_channels, replace_nan_with_batch_mean

# see .configs.py for the definition of DEFAULT_MODALITY_CONFIG
DEFAULT_MODALITY_CONFIG = dict()


class SimpleChunkedDataset(Dataset):
    def __init__(
        self,
        root_folder: str,
        sampling_rate: float,
        chunk_size: int,
        interp_config: dict = DEFAULT_MODALITY_CONFIG,
    ) -> None:
        self.root_folder = Path(root_folder)
        self.sampling_rate = sampling_rate
        self.chunk_size = chunk_size
        self._experiment = Experiment(
            root_folder,
            interp_config,
        )
        self.device_names = self._experiment.device_names
        self.start_time, self.end_time = self._experiment.get_valid_range("screen")
        self._sample_times = np.arange(
            self.start_time, self.end_time, 1.0 / self.sampling_rate
        )

    def __len__(self):
        return int(len(self._sample_times) / self.chunk_size)

    def __getitem__(self, idx):
        s = idx * self.chunk_size
        times = self._sample_times[s : s + self.chunk_size]
        data = self._experiment.interpolate(times, return_valid=False)
        assert isinstance(data, dict)
        phase_shifts = self._experiment.devices["responses"]._phase_shifts
        timestamps_neurons = (times - times.min())[:, None] + phase_shifts[None, :]
        data["timestamps"] = timestamps_neurons

        # Hack-2: add batch dimension for screen
        if len(data["screen"].shape) != 4:
            data["screen"] = data["screen"][:, None, ...]
        return data


class ChunkDataset(Dataset):
    """PyTorch Dataset for chunked experiment data.

    This dataset loads an experiment and provides temporally-chunked samples
    suitable for training neural networks. Each sample contains synchronized
    data from all modalities (e.g., screen, responses, eye_tracker, treadmill)
    at a given time window.

    Parameters
    ----------
    root_folder : str
        Path to the experiment directory containing modality subfolders.
    global_sampling_rate : float, optional
        Sampling rate (Hz) applied to all modalities. If None, uses
        per-modality rates from ``modality_config``.
    global_chunk_size : int, optional
        Number of samples per chunk for all modalities. If None, uses
        per-modality sizes from ``modality_config``.
    add_behavior_as_channels : bool, default=False
        If True, concatenates behavioral data as additional image channels.
        Deprecated: use separate modality outputs instead.
    replace_nans_with_means : bool, default=False
        If True, replaces NaN values with column means.
    cache_data : bool, default=False
        If True, keeps loaded data in memory for faster access.
    out_keys : iterable, optional
        Which modalities to include in output. Defaults to all modalities
        plus 'timestamps'.
    normalize_timestamps : bool, default=True
        If True, normalizes timestamps relative to recording start.
    modality_config : dict
        Configuration for each modality including sampling rates, transforms,
        filters, and interpolation settings. See Notes for structure.
    seed : int, optional
        Random seed for reproducible shuffling of valid time points.
    safe_interval_threshold : float, default=0.5
        Safety margin (in seconds) to exclude from edges of valid intervals.
    interpolate_precision : int, default=5
        Number of decimal places for time precision. Prevents floating-point
        accumulation errors during interpolation.


    Attributes
    ----------
    data_key : str
        Unique identifier for this dataset, extracted from metadata.
    device_names : tuple
        Names of loaded modalities.
    start_time : float
        Start of valid time range (after applying safety threshold).
    end_time : float
        End of valid time range (after applying safety threshold).

    See Also
    --------
    Experiment : Lower-level interface for data access.
    get_multisession_dataloader : Load multiple datasets.
    experanto.configs : Default configuration values.

    Notes
    -----
    The dataset handles:

    - Per-modality sampling rates and chunk sizes
    - Time offset alignment between modalities
    - Data normalization and transforms
    - Filtering based on trial conditions and data quality
    - Reproducible random sampling with seeds

    The ``modality_config`` is a nested dictionary with per-modality settings. The following is an example of a modality config for a screen, responses, eye_tracker, and treadmill:

    .. code-block:: yaml

        screen:
          sampling_rate: null
          chunk_size: null
          valid_condition:
            tier: test
            stim_type: stimulus.Frame
          offset: 0
          sample_stride: 4
          include_blanks: false
          transforms:
            ToTensor:
              _target_: torchvision.transforms.ToTensor
            Normalize:
              _target_: torchvision.transforms.Normalize
              mean: 80.0
              std: 60.0
            Resize:
              _target_: torchvision.transforms.Resize
              size:
              - 144
              - 256
            CenterCrop:
              _target_: torchvision.transforms.CenterCrop
              size: 144
          interpolation: {}
        responses:
          sampling_rate: null
          chunk_size: null
          offset: 0.1
          transforms:
            normalize_variance_only: true # old standardize: true
          interpolation:
            interpolation_mode: nearest_neighbor
        eye_tracker:
          sampling_rate: null
          chunk_size: null
          offset: 0
          transforms:
            normalize: true
          interpolation:
            interpolation_mode: nearest_neighbor
        treadmill:
          sampling_rate: null
          chunk_size: null
          offset: 0
          transforms:
            normalize: true
          interpolation:
            interpolation_mode: nearest_neighbor

    Examples
    --------
    >>> from experanto.datasets import ChunkDataset
    >>> from experanto.configs import DEFAULT_MODALITY_CONFIG
    >>> dataset = ChunkDataset(
    ...     '/path/to/experiment',
    ...     global_sampling_rate=30,
    ...     global_chunk_size=60,
    ...     modality_config=DEFAULT_MODALITY_CONFIG,
    ... )
    >>> len(dataset)
    1000
    >>> sample = dataset[0]
    >>> sample['screen'].shape
    torch.Size([1, 60, 144, 256])
    >>> sample['responses'].shape
    torch.Size([16, 500])
    """

    def __init__(
        self,
        root_folder: str,
        global_sampling_rate: Optional[float] = None,
        global_chunk_size: Optional[int] = None,
        add_behavior_as_channels: bool = False,
        replace_nans_with_means: bool = False,
        cache_data: bool = False,
        out_keys: Optional[Iterable] = None,
        normalize_timestamps: bool = True,
        modality_config: dict = DEFAULT_MODALITY_CONFIG,
        seed: Optional[int] = None,
        safe_interval_threshold: float = 0.5,
        interpolate_precision: int = 5,
    ) -> None:
        self.root_folder = Path(root_folder)
        self.data_key = self.get_data_key_from_root_folder(root_folder)
        self.interpolate_precision = interpolate_precision
        self.scale_precision = 10**self.interpolate_precision

        self.modality_config = modality_config
        self.chunk_sizes, self.sampling_rates, self.chunk_s = {}, {}, {}
        for device_name in self.modality_config.keys():
            cfg = self.modality_config[device_name]
            self.chunk_sizes[device_name] = global_chunk_size or cfg.chunk_size
            self.sampling_rates[device_name] = global_sampling_rate or cfg.sampling_rate

        self.add_behavior_as_channels = add_behavior_as_channels
        self.replace_nans_with_means = replace_nans_with_means
        self.sample_stride = self.modality_config.screen.sample_stride  # type: ignore[union-attr]
        self._experiment = Experiment(
            root_folder,
            modality_config,
            cache_data=cache_data,
        )
        self.device_names = self._experiment.device_names
        self.out_keys = out_keys or (list(self.device_names) + ["timestamps"])
        self.normalize_timestamps = normalize_timestamps

        # Determine the intersection of valid time ranges across all devices
        max_start_time = -np.inf
        min_end_time = np.inf
        if not self.device_names:
            raise ValueError(
                "No devices found in the experiment to determine valid time range."
            )

        for device_name in self.device_names:
            start, end = self._experiment.get_valid_range(device_name)
            max_start_time = max(max_start_time, start)
            min_end_time = min(min_end_time, end)

        # Check if we found any valid finite range after iteration
        if max_start_time == -np.inf or min_end_time == np.inf:
            raise ValueError(
                f"Could not determine a finite valid time range from any device. Calculated range: ({max_start_time}, {min_end_time})"
            )

        # Apply the safety margin
        self.start_time = max_start_time + safe_interval_threshold
        self.end_time = min_end_time - safe_interval_threshold

        if self.start_time >= self.end_time:
            raise ValueError(
                f"No valid overlapping time interval found across all devices after applying safety threshold. "
                f"Original range: ({max_start_time:.4f}, {min_end_time:.4f}), "
                f"Threshold: {safe_interval_threshold:.4f}, "
                f"Adjusted range: ({self.start_time:.4f}, {self.end_time:.4f})"
            )

        self._read_trials()
        self.initialize_statistics()
        self._screen_sample_times = np.arange(
            self.start_time, self.end_time, 1.0 / self.sampling_rates["screen"]
        )
        # iterate over the valid condition in modality_config["screen"]["valid_condition"] to get the indices of self._screen_sample_times that meet all criteria
        self._full_valid_sample_times_filtered = self.get_full_valid_sample_times(
            filter_for_valid_intervals=True
        )
        # self._full_valid_sample_times_unfiltered = self.get_full_valid_sample_times(filter_for_valid_intervals=False)

        # the _valid_screen_times are the indices from which the starting points for the chunks will be taken
        # sampling stride is used to reduce the number of starting points by the stride
        # default of stride is 1, so all starting points are used
        self._valid_screen_times = self._full_valid_sample_times_filtered[
            :: self.sample_stride
        ]

        self.transforms = self.initialize_transforms()

        self.seed = seed
        self._rng = np.random.RandomState(seed) if seed is not None else np.random

    def _read_trials(self) -> None:
        screen = self._experiment.devices["screen"]
        self._trials = [t for t in screen.trials]
        start_idx = np.array([t.first_frame_idx for t in self._trials])
        self._start_times = screen.timestamps[start_idx]
        self._end_times = np.append(screen.timestamps[start_idx[1:]], np.inf)
        self.meta_conditions = {}
        for k in ["modality", "valid_trial", "tier"]:
            self.meta_conditions[k] = [
                t.get_meta(k) if t.get_meta(k) is not None else "blank"
                for t in self._trials
            ]

    def initialize_statistics(self) -> None:
        """Initialize normalization statistics for each device.

        Loads mean and standard deviation values from each device's meta folder
        and stores them in ``self._statistics`` for use during data transforms.
        """
        self._statistics = {}
        for device_name in self.device_names:
            self._statistics[device_name] = {}
            # If modality should be normalized, load respective statistics from file.
            if self.modality_config[device_name].transforms.get("normalization", False):
                mode = self.modality_config[device_name].transforms.normalization
                means = np.load(
                    self._experiment.devices[device_name].root_folder / "meta/means.npy"
                )
                stds = np.load(
                    self._experiment.devices[device_name].root_folder / "meta/stds.npy"
                )
                if device_name == "responses":
                    # same as in neuralpredictors before
                    # https://github.com/sinzlab/neuralpredictors/blob/2b420058b2c0c029842ba739829114ddfa0f8b50/neuralpredictors/data/transforms.py#L375-L378
                    threshold = 0.01 * np.nanmean(stds)
                    idx = stds[0, :] < threshold  # response std shape: (1, n_neurons)
                    stds[0, idx] = (
                        threshold  # setting stds which are smaller than threshold to threshold
                    )

                # if mode is a dict, it will override the means and stds
                if not isinstance(mode, str):
                    means = np.array(mode.get("means", means))
                    stds = np.array(mode.get("stds", stds))
                if mode == "normalize_variance_only":
                    # If modality should only be adjusted by variance (old "standardize"), set means to 0.
                    means = np.zeros_like(means)
                elif mode == "recompute_responses":
                    means = np.zeros_like(means)
                    stds = np.nanstd(self._experiment.devices["responses"]._data, 0)[
                        None, ...
                    ]
                elif mode == "recompute_behavior":
                    means = np.nanmean(self._experiment.devices[device_name]._data, 0)[
                        None, ...
                    ]
                    stds = np.nanstd(self._experiment.devices[device_name]._data, 0)[
                        None, ...
                    ]
                elif mode == "screen_default":
                    means = np.array((80))
                    stds = np.array((60))

                self._statistics[device_name]["mean"] = means.reshape(
                    1, -1
                )  # (n, 1) -> (1, n) for broadcasting in __get_item__
                self._statistics[device_name]["std"] = stds.reshape(
                    1, -1
                )  # same as above

    @staticmethod
    def add_channel_function(x):
        if len(x.shape) == 3:
            return torch.from_numpy(x[:, None, ...])
        else:
            return torch.from_numpy(x)

    def initialize_transforms(self):
        """Initialize data transforms for each device based on modality config."""
        transforms = {}
        for device_name in self.device_names:
            if device_name == "screen":
                add_channel = Lambda(self.add_channel_function)
                transform_list: List[Any] = []

                for v in self.modality_config.screen.transforms.values():  # type: ignore[union-attr]
                    if isinstance(v, dict):  # config dict
                        module = instantiate(v)
                        if isinstance(module, torch.nn.Module):
                            transform_list.append(module)

                transform_list.insert(0, add_channel)
            else:
                transform_list: List[Any] = [ToTensor()]

            # Normalization.
            if self.modality_config[device_name].transforms.get("normalization", False):
                transform_list.append(
                    torchvision.transforms.Normalize(
                        self._statistics[device_name]["mean"],
                        self._statistics[device_name]["std"],
                    )
                )

            transforms[device_name] = Compose(transform_list)
        return transforms

    def _get_callable_filter(self, filter_config):
        """Return a callable filter function from config or an existing callable.

        Notes
        -----
        Handles partial instantiation using hydra.utils.instantiate.

        Parameters
        ----------
        filter_config : dict, DictConfig, or callable
            Either a config dictionary/DictConfig specifying a filter (with
            '__target__'), or an already-instantiated callable filter function.

        Returns
        -------
        callable
            The final filter function ready to be called with `device_`.
        """
        # Check if it's already a callable (function)
        if callable(filter_config):
            # print(f"DEBUG: callable(filter_config) returned True for type {type(filter_config)}. Returning config directly.")
            return filter_config

        # Check if it's a config that needs instantiation
        if (
            isinstance(filter_config, (dict, DictConfig))
            and "__target__" in filter_config
        ):
            try:
                # Manually handle instantiation for factory pattern with __partial__=True
                target_str = filter_config["__target__"]
                module_path, func_name = target_str.rsplit(".", 1)

                # Import the module and get the factory function
                module = importlib.import_module(module_path)
                factory_func = getattr(module, func_name)

                # Prepare arguments for the factory function (excluding special keys)
                args = {
                    k: v
                    for k, v in filter_config.items()
                    if k not in ("__target__", "__partial__")
                }

                # Call the factory function with its arguments to get the actual implementation function
                implementation_func = factory_func(**args)  # type: ignore[reportCallIssue]
                return implementation_func

            except (ImportError, AttributeError, KeyError, TypeError) as e:
                raise TypeError(
                    f"Failed to manually instantiate filter from config {filter_config}: {e}"
                )

        raise TypeError(
            f"Filter config must be either callable or a valid config dict with __target__, got {type(filter_config)}"
        )

    def get_valid_intervals_from_filters(
        self, visualize: bool = False
    ) -> List[TimeInterval]:
        valid_intervals: Optional[List[TimeInterval]] = None
        for modality in self.modality_config:
            if "filters" in self.modality_config[modality]:
                device = self._experiment.devices[modality]
                for filter_name, filter_config in self.modality_config[modality][
                    "filters"
                ].items():
                    # Get the final callable filter function
                    filter_function = self._get_callable_filter(filter_config)
                    valid_intervals_: List[TimeInterval] = filter_function(device_=device)  # type: ignore[assignment]
                    if visualize:
                        print(f"modality: {modality}, filter: {filter_name}")
                        visualization_string = get_stats_for_valid_interval(
                            valid_intervals_, self.start_time, self.end_time
                        )
                        print(visualization_string)
                    if valid_intervals is None:
                        valid_intervals = valid_intervals_
                    else:
                        valid_intervals = find_intersection_between_two_interval_arrays(
                            valid_intervals, valid_intervals_
                        )

        return valid_intervals if valid_intervals is not None else []

    def get_condition_mask_from_meta_conditions(
        self, valid_conditions_sum_of_product: List[dict]
    ) -> np.ndarray:
        """Create a boolean mask for trials satisfying given conditions.

        Parameters
        ----------
        valid_conditions_sum_of_product : list of dict
            Condition dictionaries combined with OR logic, where conditions
            within each dictionary use AND logic.

        Returns
        -------
        np.ndarray
            Boolean mask indicating which trials satisfy at least one set of
            conditions.

        Notes
        -----
        For example,
        ``[{'tier': 'train', 'stim_type': 'natural'}, {'tier': 'blank'}]``
        matches trials that are either (train AND natural) OR blank.
        """
        all_conditions: Optional[np.ndarray] = None
        for valid_conditions_product in valid_conditions_sum_of_product:
            conditions_of_product = None
            for k, valid_condition in valid_conditions_product.items():
                trial_conditions = self.meta_conditions[k]
                condition_mask = np.array(
                    [condition == valid_condition for condition in trial_conditions]
                )
                if conditions_of_product is None:
                    conditions_of_product = condition_mask
                else:
                    conditions_of_product &= condition_mask
            if all_conditions is None:
                all_conditions = conditions_of_product
            else:
                all_conditions |= conditions_of_product
        if all_conditions is None:
            return np.array([], dtype=bool)
        return all_conditions

    def get_screen_sample_mask_from_meta_conditions(
        self,
        satisfy_for_next: int,
        valid_conditions_sum_of_product: List[dict],
        filter_for_valid_intervals: bool = True,
    ) -> np.ndarray:
        """Create a boolean mask for screen samples satisfying given conditions.

        Parameters
        ----------
        satisfy_for_next : int
            Number of consecutive samples that must satisfy conditions.
        valid_conditions_sum_of_product : list of dict
            Condition dictionaries combined with OR logic, where conditions
            within each dictionary use AND logic.
        filter_for_valid_intervals : bool, default=True
            Whether to apply interval-based filtering.

        Returns
        -------
        numpy.ndarray
            Boolean array matching screen sample times, True where conditions are met.
        """
        all_conditions = self.get_condition_mask_from_meta_conditions(
            valid_conditions_sum_of_product
        )
        sample_mask = np.zeros_like(self._screen_sample_times, dtype=bool)
        valid_indices = np.where(all_conditions)[0]

        filter_valid_intervals = (
            self.get_valid_intervals_from_filters(visualize=False)
            if filter_for_valid_intervals
            else None
        )
        # filter_valid_intervals = None

        if len(valid_indices) > 0:
            starts = self._start_times[valid_indices]
            ends = self._end_times[valid_indices]

            # Create TimeIntervals from starts and ends
            trial_intervals = [
                TimeInterval(start, end) for start, end in zip(starts, ends)
            ]

            # If we have filter_valid_intervals, find intersection with trial intervals
            if filter_valid_intervals:
                # Find intersection between trial intervals and filter valid intervals
                valid_intervals = find_intersection_between_two_interval_arrays(
                    trial_intervals, filter_valid_intervals
                )
            else:
                valid_intervals = trial_intervals

            # Apply mask only for the intersected intervals
            for interval in valid_intervals:
                mask = (self._screen_sample_times >= interval.start) & (
                    self._screen_sample_times < interval.end
                )
                sample_mask |= mask

        if satisfy_for_next > 1:
            windows = np.lib.stride_tricks.sliding_window_view(
                sample_mask, satisfy_for_next
            )
            sample_mask = np.all(windows, axis=1)

        return sample_mask

    def get_full_valid_sample_times(
        self, filter_for_valid_intervals: bool = True
    ) -> np.ndarray:
        """Get all valid chunk starting times based on meta conditions.

        Iterates through sample times and checks if they can be used as chunk
        start times (i.e., the next ``chunk_size`` points are all valid based
        on the previous meta condition filtering).

        Parameters
        ----------
        filter_for_valid_intervals : bool, default=True
            Whether to apply interval-based filtering.

        Returns
        -------
        numpy.ndarray
            Array of valid starting time points.
        """

        # Calculate all possible end indices
        chunk_size = self.chunk_sizes["screen"]
        n_samples = len(self._screen_sample_times) - chunk_size + 1
        possible_indices = np.arange(n_samples)

        # Check duration condition vectorized
        duration_mask = (
            self._screen_sample_times[possible_indices + chunk_size - 1] < self.end_time
        )

        # this assumes that the valid_condition is a single condition
        valid_conditions = self.modality_config["screen"]["valid_condition"]
        if not isinstance(valid_conditions, (list, tuple, ListConfig)):
            valid_conditions = [valid_conditions]

        valid_conditions = list(valid_conditions)

        if self.modality_config["screen"]["include_blanks"]:
            additional_valid_conditions = {"tier": "blank"}
            valid_conditions.append(additional_valid_conditions)

        sample_mask_from_meta_conditions = self.get_screen_sample_mask_from_meta_conditions(
            chunk_size, valid_conditions, filter_for_valid_intervals  # type: ignore[arg-type]
        )

        final_mask = duration_mask & sample_mask_from_meta_conditions

        return self._screen_sample_times[possible_indices[final_mask]]

    def shuffle_valid_screen_times(self) -> None:
        """
        Shuffle valid screen times using the dataset's random number generator
        for reproducibility.
        """
        times = self._full_valid_sample_times_filtered
        if self.seed is not None:
            self._valid_screen_times = np.sort(
                self._rng.choice(
                    times, size=len(times) // self.sample_stride, replace=False
                )
            )
        else:
            self._valid_screen_times = np.sort(
                np.random.choice(
                    times, size=len(times) // self.sample_stride, replace=False
                )
            )

    def get_data_key_from_root_folder(cls, root_folder):
        """Extract a data key from the root folder path.

        Checks for a meta.json file and extracts the data_key or scan_key.

        Parameters
        ----------
        root_folder : str or Path
            Path to the root folder containing the dataset.

        Returns
        -------
        str
            The extracted data key, or folder name if meta.json doesn't
            exist or lacks data_key.
        """
        # Convert Path object to string if necessary
        root_folder = str(root_folder)

        # Construct the path to meta.json
        meta_file_path = os.path.join(root_folder, "meta.json")

        # Initialize meta as an empty dict
        meta = {}

        # Check if the file exists before trying to open it
        if os.path.isfile(meta_file_path):
            try:
                with open(meta_file_path, "r") as file:
                    meta = json.load(file)

                # Get data_key from meta if it exists
                if "data_key" in meta:
                    return meta["data_key"]
                elif "scan_key" in meta:
                    key = meta["scan_key"]
                    data_key = f"{key['animal_id']}-{key['session']}-{key['scan_idx']}"
                    return data_key
                if "dynamic" in root_folder:
                    dataset_name = root_folder.split("dynamic")[1].split("-Video")[0]
                    return dataset_name
                elif "_gaze" in root_folder:
                    dataset_name = root_folder.split("_gaze")[0].split("datasets/")[1]
                    return dataset_name
                else:
                    print(
                        f"No 'data_key' found in {meta_file_path}, using folder name instead"
                    )
            except json.JSONDecodeError:
                print(f"Error: {meta_file_path} is not a valid JSON file")
            except Exception as e:
                print(f"Error loading {meta_file_path}: {str(e)}")
        else:
            print(f"No metadata file found at {meta_file_path}")
        return os.path.basename(root_folder)

    def __len__(self):
        return len(self._valid_screen_times)

    def __getitem__(self, idx: int) -> dict:
        """Return a single data sample at the given index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        dict
            Dictionary containing data for each modality in ``out_keys``, e.g.:

            - ``'screen'``: torch.Tensor of shape ``(C, T, H, W)``
            - ``'responses'``: torch.Tensor of shape ``(T, N_neurons)``
            - ``'eye_tracker'``: torch.Tensor of shape ``(T, N_features)``
            - ``'treadmill'``: torch.Tensor of shape ``(T, N_features)``
            - ``'timestamps'``: dict mapping modality names to time arrays

            Where ``T`` is the chunk size (may differ per modality),
            ``C`` is channels, ``H`` is height, ``W`` is width.
        """
        out = {}
        timestamps = {}
        s = self._valid_screen_times[idx]
        for device_name in self.device_names:
            sampling_rate = self.sampling_rates[device_name]
            chunk_size = self.chunk_sizes[device_name]
            chunk_s = chunk_size / sampling_rate

            # convert everything to int to avoid numerical issues
            start_time = int(round(s * self.scale_precision))
            offset = int(
                round(self.modality_config[device_name].offset * self.scale_precision)
            )
            time_delta = int(round((1.0 / sampling_rate) * self.scale_precision))
            # Generate times as ints - important as for np.floats the summation is not associative
            times = start_time + offset + np.arange(chunk_size) * time_delta
            # scale everything back to truncated values
            times = times.astype(np.float64) / self.scale_precision

            data = self._experiment.interpolate(
                times, device=device_name, return_valid=False
            )
            out[device_name] = self.transforms[device_name](data).squeeze(
                0
            )  # remove dim0 for response/eye_tracker/treadmill
            # TODO: find better convention for image, video, color, gray channels. This makes the monkey data same as mouse.
            if device_name == "screen":
                if out[device_name].shape[-1] == 3:
                    out[device_name] = out[device_name].permute(0, 3, 1, 2)
                if out[device_name].shape[0] == chunk_size:
                    out[device_name] = out[device_name].transpose(0, 1)
            # all signals are interpolated for the same times, so no phase shifts adjustment is needed
            times = torch.from_numpy(times)
            if self.normalize_timestamps:
                times = times - self._experiment.devices["responses"].start_time
                times = times.to(torch.float32).contiguous()
            timestamps[device_name] = times

        out["timestamps"] = timestamps

        # deprecated
        if self.add_behavior_as_channels:
            out = add_behavior_as_channels(out)

        final_out = {}
        for key in out:
            if key in self.out_keys:
                if key == "timestamps":
                    final_out[key] = out[key]
                elif not out[key].is_contiguous():
                    final_out[key] = out[key].contiguous()
                else:
                    final_out[key] = out[key]

        return final_out

    def get_state(self) -> Dict[str, Any]:
        """Return the current state of the dataset's RNG."""
        return {
            "rng_state": self._rng.get_state() if self.seed is not None else None,
            "valid_screen_times": self._valid_screen_times.copy(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the dataset's RNG state."""
        if state["rng_state"] is not None and self.seed is not None:
            self._rng.set_state(state["rng_state"])
        self._valid_screen_times = state["valid_screen_times"]
