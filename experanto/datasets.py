from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms.v2 import ToTensor, Compose, Lambda
from omegaconf import OmegaConf
from hydra.utils import instantiate

from .experiment import Experiment
from .interpolators import ImageTrial, VideoTrial
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
        self.DataPoint = namedtuple("DataPoint", self.device_names)

    def __len__(self):
        return int(len(self._sample_times) / self.chunk_size)

    def __getitem__(self, idx):
        s = idx * self.chunk_size
        times = self._sample_times[s : s + self.chunk_size]
        data, _ = self._experiment.interpolate(times)
        phase_shifts = self._experiment.devices["responses"]._phase_shifts
        timestamps_neurons = (times - times.min())[:, None] + phase_shifts[None, :]
        data["timestamps"] = timestamps_neurons

        # Hack-2: add batch dimension for screen
        if len(data["screen"].shape) != 4:
            data["screen"] = data["screen"][:, None, ...]
        return data


class Mouse2pStaticImageDataset(Dataset):
    def __init__(
        self,
        root_folder: str,
        tier: str,
        offset: float,
        stim_duration: float,
        interp_config: dict = DEFAULT_MODALITY_CONFIG,
    ) -> None:
        self.root_folder = Path(root_folder)
        self.tier = tier
        self.offset = offset
        self.stim_duration = stim_duration
        self._experiment = Experiment(
            root_folder,
            interp_config,
        )
        self.device_names = self._experiment.device_names
        self.DataPoint = namedtuple("DataPoint", self.device_names)
        self._read_trials()

    def _read_trials(self):
        screen = self._experiment.devices["screen"]
        self._trials = [
            t
            for t in screen.trials
            if isinstance(t, ImageTrial) and t.get_meta("tier") == self.tier
        ]
        s_idx = np.array([t.first_frame_idx for t in self._trials])
        if len(s_idx):
            self._start_times = screen.timestamps[s_idx]
        else:
            self._start_times = np.array([])

    def __len__(self):
        return len(self._trials)

    def __getitem__(self, idx):
        assert isinstance(idx, int), "Index must be an integer"
        data = dict()
        for device_name, device in self._experiment.devices.items():
            if device_name == "screen":
                times = self._start_times[idx]
            else:
                Fs = device.sampling_rate
                times = (
                    self._start_times[idx]
                    + self.offset
                    + np.arange(0, self.stim_duration, 1.0 / Fs)
                )
            d, _ = device.interpolate(times)
            data[device_name] = d.mean(axis=0)
        return self.DataPoint(**data)


class Mouse2pVideoDataset(Dataset):
    def __init__(
        self,
        root_folder: str,
        tier: str,
        stim_duration: float,
        sampling_rate: float,
        subsample: bool,
        cut: bool,
        add_channel: bool,
        channel_pos: int,
        interp_config: dict = DEFAULT_MODALITY_CONFIG,
    ) -> None:
        """
        this dataloader returns the full the video resampled to the new freq rate
        subsampling frames ideally should be outside dataset but in the dataloader augmentations

        :param root_folder: path to the data folder
        :param tier: train/test/validation
        :param stim_duration: how many frames to take from the video
        :param sampling_rate: sampling rate to interpolate
        :param subsample: if we sample longer video from the non-start position
        :param cut: if we cut the video up to the stim_duration length or not
        :param add_channel: if video does not have channels, this flag shows if to add it
        :param channel_pos: if add_channel True and no channels are in the video, this would be the position to add it
        """
        self.root_folder = Path(root_folder)
        self.tier = tier
        self.sampling_rate = sampling_rate
        self.stim_duration = stim_duration
        self._experiment = Experiment(
            root_folder,
            interp_config,
        )
        self.device_names = self._experiment.device_names
        # this is needed to match sensorium order only
        if "screen" in self.device_names and "responses" in self.device_names:
            start = ["screen", "responses"]
            self.device_names = tuple(start) + tuple(
                set(self.device_names).difference(set(start))
            )
        self.subsample = subsample
        self.cut = cut
        self.add_channel = add_channel
        self.channel_pos = channel_pos
        assert (
            0 <= channel_pos < 4
        ), "channels could be extended only for positions [0,3]"

        self.start_time, self.end_time = self._experiment.get_valid_range("screen")
        self.DataPoint = namedtuple("DataPoint", self.device_names)
        self.MetaNeuro = namedtuple(
            "MetaNeuro", ["cell_motor_coordinates", "unit_ids", "fields"]
        )
        self._read_trials()

    def _read_trials(self):
        # reads all videos from valid tiers and saves times for them
        # also have saves the start and end time if test videos are in between
        # todo
        screen = self._experiment.devices["screen"]
        self._trials = [
            t
            for t in screen.trials
            if isinstance(t, VideoTrial) and t.get_meta("tier") == self.tier
        ]
        s_idx = np.array([t.first_frame_idx for t in self._trials])
        # todo - not sure if it should be t.first_frame_idx + t.num_frames
        e_idx = np.array([t.first_frame_idx + t.num_frames - 1 for t in self._trials])
        # todo - this uses the assumption that sampling_rate is less or equal the sampling rate of the screen stimuli
        if self.cut:
            assert all(
                [t.num_frames >= self.stim_duration for t in self._trials]
            ), "stim_duration should be smaller"
        if len(s_idx):
            self._start_times = screen.timestamps[s_idx]
            self._end_times = screen.timestamps[e_idx]
        else:
            self._start_times = np.array([])
            self._end_times = np.array([])

    def __len__(self):
        return len(self._trials)

    @property
    def neurons(self):
        loc_meta = {
            "cell_motor_coordinates": [],
            "unit_ids": [],
            "fields": [],
        }
        if "responses" in self._experiment.devices.keys():
            # todo - make it lazy loading? and read-only properties?
            root_folder = self._experiment.devices["responses"].root_folder
            meta = self._experiment.devices["responses"].load_meta()
            if "neuron_properties" in meta:
                cell_motor_coordinates = np.load(
                    root_folder / meta["neuron_properties"]["cell_motor_coordinates"]
                )
                unit_ids = np.load(root_folder / meta["neuron_properties"]["unit_ids"])
                fields = np.load(root_folder / meta["neuron_properties"]["fields"])

                loc_meta = {
                    "cell_motor_coordinates": cell_motor_coordinates,
                    "unit_ids": unit_ids,
                    "fields": fields,
                }

        return self.MetaNeuro(**loc_meta)

    def __getitem__(self, idx):
        """

        :param idx: idx of the video
        :return: this would return video in data['screen'] with shape of [t, h, w]
        """
        fs = self.sampling_rate
        times = np.arange(self._start_times[idx], self._end_times[idx], 1 / fs)
        # get all times possible
        # cut is needed
        if self.cut:
            if self.subsample:
                start = np.random.randint(0, len(times) - self.stim_duration)
                times = times[start : start + self.stim_duration]
            else:
                times = times[: self.stim_duration]

        data, _ = self._experiment.interpolate(times)

        if self.add_channel and len(data["screen"].shape) != 4:
            data["screen"] = np.expand_dims(data["screen"], axis=self.channel_pos)
        # this hack matches the shape for sensorium models
        if "responses" in data:
            data["responses"] = data["responses"].T
        return self.DataPoint(**data)


class ChunkDataset(Dataset):
    def __init__(
        self,
        root_folder: str,
        global_sampling_rate: None,
        global_chunk_size: None,
        add_behavior_as_channels: bool = False,
        replace_nans_with_means: bool = False,
        modality_config: dict = DEFAULT_MODALITY_CONFIG,
        add_behavior_as_channels: bool = False,
        replace_nans_with_means: bool = False,
    ) -> None:
        """
        The full modality config is a nested dictionary.
        The following is an example of a modality config for a screen, responses, eye_tracker, and treadmill:

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
            standardize: true
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
        """
        self.root_folder = Path(root_folder)
        self.modality_config = instantiate(modality_config)
        self.chunk_sizes, self.sampling_rates, self.chunk_s = {}, {}, {}
        for device_name in self.modality_config.keys():
            cfg = self.modality_config[device_name]
            self.chunk_sizes[device_name] = global_chunk_size or cfg.chunk_size
            self.sampling_rates[device_name] = global_sampling_rate or cfg.sampling_rate
        self.add_behavior_as_channels = add_behavior_as_channels
        self.replace_nans_with_means = replace_nans_with_means
        self.sample_stride = self.modality_config.screen.sample_stride
        self._experiment = Experiment(
            root_folder,
            modality_config,
        )
        self.device_names = self._experiment.device_names
        self.start_time, self.end_time = self._experiment.get_valid_range("screen")
        self._read_trials()
        self.initialize_statistics()
        self._screen_sample_times = np.arange(
            self.start_time, self.end_time, 1.0 / self.sampling_rates["screen"]
        )
        # iterate over the valid condition in modality_config["screen"]["valid_condition"] to get the indices of self._screen_sample_times that meet all criteria
        self._sample_in_meta_condition = self.get_sample_in_meta_condition()
        self._full_valid_sample_times = self.get_full_valid_sample_times()

        # the _valid_screen_times are the indices from which the starting points for the chunks will be taken
        # sampling stride is used to reduce the number of starting points by the stride
        # default of stride is 1, so all starting points are used
        self._valid_screen_times = self._full_valid_sample_times[::self.sample_stride]
        self.transforms = self.initialize_transforms()

    def _read_trials(self) -> None:
        screen = self._experiment.devices["screen"]
        self._trials = [t for t in screen.trials]
        start_idx = np.array([t.first_frame_idx for t in self._trials])
        self._start_times = screen.timestamps[start_idx]
        self._end_times = np.append(screen.timestamps[start_idx[1:]], np.inf)
        self.meta_conditions = {}
        for k in self.modality_config.screen.valid_condition.keys():
            self.meta_conditions[k] = [t.get_meta(k) if t.get_meta(k) is not None else "blank" for t in self._trials]

    def initialize_statistics(self) -> None:
        """
        Initializes the statistics for each device based on the modality config.
        :return:
            instantiates self._statistics with the mean and std for each device
        """
        self._statistics = {}
        for device_name in self.device_names:
            self._statistics[device_name] = {}
            # If modality should be normalized, load respective statistics from file.
            if self.modality_config[device_name].transforms.get("normalization", False):
                mode = self.modality_config[device_name].transforms.normalization
                assert mode in ['standardize', 'normalize', "recompute_responses", "screen_default", "recompute_behavior"], f"Unknown mode {mode}"
                assert mode in ['standardize', 'normalize', "response_hack", "screen_hack", "behavior_hack"]
                means = np.load(self._experiment.devices[device_name].root_folder / "meta/means.npy")
                stds = np.load(self._experiment.devices[device_name].root_folder / "meta/stds.npy")
                if mode == 'standardize':
                    # If modality should only be standarized, set means to 0.
                    means = np.zeros_like(means)
                elif mode == 'recompute_responses':
                    means = np.zeros_like(means)
                    stds = np.nanstd(self._experiment.devices["responses"]._data, 0)[None, ...]
                elif mode == 'recompute_behavior':
                    means = np.nanmean(self._experiment.devices[device_name]._data, 0)[None, ...]
                    stds = np.nanstd(self._experiment.devices[device_name]._data, 0)[None, ...]
                elif mode == 'screen_default':
                    means = np.array((80))
                    stds = np.array((60))

                self._statistics[device_name]["mean"] = means.reshape(1, -1)  # (n, 1) -> (1, n) for broadcasting in __get_item__
                self._statistics[device_name]["std"] = stds.reshape(1, -1)  # same as above

    def initialize_transforms(self):
        """
        Initializes the transforms for each device based on the modality config.
        :return:
        """
        transforms = {}
        for device_name in self.device_names:
            if device_name == "screen":
                add_channel = Lambda(lambda x: torch.from_numpy(x[:, None, ...]) if len(x.shape) == 3 else torch.from_numpy(x))
                transform_list = [v for v in self.modality_config.screen.transforms.values() if isinstance(v, torch.nn.Module)]
                transform_list.insert(0, add_channel)
            else:
                transform_list = [ToTensor()]
            
            # Normalization.
            if self.modality_config[device_name].transforms.get("normalization", False):
                transform_list.append(
                    torchvision.transforms.Normalize(self._statistics[device_name]["mean"], self._statistics[device_name]["std"])
                )

            transforms[device_name] = Compose(transform_list)
        return transforms

    def get_sample_in_meta_condition(self) -> dict:
        """
        iterates through all stimuli, selects the ones which match the meta conditions (tiers or stimuli types) and creates a mask to select the correct times using `self._screen_sample_times` as the clock/reference times
           for example:
              if meta_conditions = {"tier": [train,train, ...], "stim_type": [type1, type2, ...]}
              and valid_condition = {"tier": train, "stim_type": type2}
           then the output would be {"tier": [True, True, ...], "stim_type": [False, True, ...]}
        """

        sample_in_meta_condition = {}
        for k, v in self.modality_config["screen"]["valid_condition"].items():
            # Pre-allocate a boolean array
            result = np.zeros_like(self._screen_sample_times, dtype=bool)
            
            # Create masks for all trials at once
            valid_conditions = np.array([
                condition == v or (condition == "blank" and self.modality_config["screen"]["include_blanks"])
                for condition in self.meta_conditions[k]
            ])
            
            # Only process valid trials
            valid_indices = np.where(valid_conditions)[0]
            if len(valid_indices) > 0:
                starts = self._start_times[valid_indices]
                ends = self._end_times[valid_indices]
                
                # Vectorized comparison for all valid trials
                for start, end in zip(starts, ends):
                    mask = (self._screen_sample_times >= start) & (self._screen_sample_times < end)
                    result |= mask
                    
            sample_in_meta_condition[k] = result

        return sample_in_meta_condition



    def get_full_valid_sample_times(self) -> Iterable:
        """
        Returns all valid sample times that can be used as starting points for chunks.
        A sample time is valid if:
        1. The chunk starting at this time does not extend beyond the end time
        2. All samples in the chunk satisfy the meta conditions specified in the config
           (e.g., all frames belong to the correct tier and stimulus type)

        Returns:
            Iterable: Array of valid sample times that can be used as chunk starting points
        """

        # Calculate all possible end indices
        chunk_size = self.chunk_sizes["screen"]
        n_samples = len(self._screen_sample_times) - chunk_size
        possible_indices = np.arange(n_samples)
        
        # Check duration condition vectorized
        duration_mask = self._screen_sample_times[possible_indices + chunk_size] < self.end_time
        
        # Initialize all_conditions array with duration mask
        all_conditions = duration_mask.copy()  # Make a copy to ensure correct shape

        # Check meta conditions vectorized
        for k, v in self._sample_in_meta_condition.items():
            # Create a sliding window view of the meta condition array
            windows = np.lib.stride_tricks.sliding_window_view(v[:n_samples + chunk_size], chunk_size)[:n_samples]
            # Check if all values in each window are True
            condition_mask = np.all(windows, axis=1)
            # Ensure shapes match
            assert condition_mask.shape == all_conditions.shape, f"Shape mismatch: {condition_mask.shape} vs {all_conditions.shape}"
            # Combine with previous conditions
            all_conditions &= condition_mask
        
        # Get the valid times
        valid_times = self._screen_sample_times[possible_indices[all_conditions]]
        return valid_times


    def shuffle_valid_screen_times(self) -> None:
        """
        convenience function to randomly select new starting points for each chunk. Use this in training after each epoch.
        If the sample stride is 1, all starting points will be used and this convenience function is not needed.
        If the sample stride is larger than 1, this function will shuffle the starting points and select a subset of them.
        """
        times = self._full_valid_sample_times
        self._valid_screen_times = np.sort(np.random.choice(times, size=len(times) // self.sample_stride, replace=False))

    def __len__(self):
        return len(self._valid_screen_times)

    def __getitem__(self, idx) -> dict:
        out = {}
        s = self._valid_screen_times[idx]
        for device_name in self.device_names:
            sampling_rate = self.sampling_rates[device_name]
            chunk_size = self.chunk_sizes[device_name]
            chunk_s = chunk_size / sampling_rate

            times = np.linspace(s, s + chunk_s, chunk_size, endpoint=False)
            times = times + self.modality_config[device_name].offset
            data, _ = self._experiment.interpolate(times, device=device_name)

            if self.replace_nans_with_means:
                if np.any(np.isnan(data)):
                    data = replace_nan_with_batch_mean(data)

            out[device_name] = self.transforms[device_name](data).squeeze(0) # remove dim0 for response/eye_tracker/treadmill
            # TODO: find better convention for image, video, color, gray channels. This makes the monkey data same as mouse.
            if device_name == "screen":
                if out[device_name].shape[-1] == 3:
                    out[device_name] = out[device_name].permute(0, 3, 1, 2)

        if self.add_behavior_as_channels:
            out = add_behavior_as_channels(out)
        if self._experiment.devices["responses"].use_phase_shifts:
            phase_shifts = self._experiment.devices["responses"]._phase_shifts
            times = (times - times.min())[:, None] + phase_shifts[None, :]
        else:
            times = times - times.min()
        out["timestamps"] = torch.from_numpy(times)

        return out

