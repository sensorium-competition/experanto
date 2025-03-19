from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import v2
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
        out_keys: Optional[Iterable] = None,
        modality_config: dict = DEFAULT_MODALITY_CONFIG,
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
        self.out_keys = out_keys or self.device_names
        self.start_time, self.end_time = self._experiment.get_valid_range("screen")
        self._read_trials()
        self.initialize_statistics()
        self._screen_sample_times = np.arange(
            self.start_time, self.end_time, 1.0 / self.sampling_rates["screen"]
        )
        # iterate over the valid condition in modality_config["screen"]["valid_condition"] to get the indices of self._screen_sample_times that meet all criteria
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
        for k in ["modality", "valid_trial"] + list(self.modality_config.screen.valid_condition.keys()):
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

                means = np.load(self._experiment.devices[device_name].root_folder / "meta/means.npy")
                stds = np.load(self._experiment.devices[device_name].root_folder / "meta/stds.npy")

                # if mode is a dict, it will override the means and stds
                if not isinstance(mode, str):
                    means = np.array(mode.get("means", means))
                    stds = np.array(mode.get("stds", stds))

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

    @staticmethod
    def add_channel_function(x):
        if len(x.shape) == 3:
            return torch.from_numpy(x[:, None, ...])
        else:
            return torch.from_numpy(x)

    def initialize_transforms(self):
        """
        Initializes the transforms for each device based on the modality config.
        :return:
        """
        transforms = {}
        for device_name in self.device_names:
            if device_name == "screen":
                add_channel = Lambda(self.add_channel_function)
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
    
        
    def get_condition_mask_from_meta_conditions(self, valid_conditions_sum_of_product: List[dict]) -> np.ndarray:
        """Creates a boolean mask for trials that satisfy any of the given condition combinations.
        
        Args:
            valid_conditions_sum_of_product: List of dictionaries, where each dictionary represents a set of
                conditions that should be satisfied together (AND). Multiple dictionaries are combined with OR.
                Example: [{'tier': 'train', 'stim_type': 'natural'}, {'tier': 'blank'}] matches trials that
                are either (train AND natural) OR blank.

        Returns:
            np.ndarray: Boolean mask indicating which trials satisfy at least one set of conditions.
        """
        all_conditions = None

        for valid_conditions_product in valid_conditions_sum_of_product:

            conditions_of_product = None

            for k, valid_condition in valid_conditions_product.items():

                trial_conditions = self.meta_conditions[k]

                condition_mask = np.array([condition == valid_condition for condition in trial_conditions])

                if conditions_of_product is None:
                    conditions_of_product = condition_mask
                else:
                    conditions_of_product &= condition_mask

            if all_conditions is None:
                all_conditions = conditions_of_product
            else:
                all_conditions |= conditions_of_product

        return all_conditions
    
    def get_screen_sample_mask_from_meta_conditions(self, satisfy_for_next: int, valid_conditions_sum_of_product: List[dict]) -> np.ndarray:
        """Creates a boolean mask indicating which screen samples satisfy the given conditions.

        Args:
            satisfy_for_next: Number of consecutive samples that must satisfy conditions
            valid_conditions_sum_of_product: List of condition dictionaries combined with OR logic,
                where conditions within each dictionary use AND logic

        Returns:
            Boolean array matching screen sample times, True where conditions are met
        """

        all_conditions = self.get_condition_mask_from_meta_conditions(valid_conditions_sum_of_product)

        sample_mask = np.zeros_like(self._screen_sample_times, dtype=bool)

        valid_indices = np.where(all_conditions)[0]

        if len(valid_indices) > 0:
            starts = self._start_times[valid_indices]
            ends = self._end_times[valid_indices]

            for start, end in zip(starts, ends):
                mask = (self._screen_sample_times >= start) & (self._screen_sample_times < end)
                sample_mask |= mask

        if satisfy_for_next > 1:
            windows = np.lib.stride_tricks.sliding_window_view(sample_mask, satisfy_for_next)
            sample_mask = np.all(windows, axis=1)

        return sample_mask

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
        n_samples = len(self._screen_sample_times) - chunk_size + 1
        possible_indices = np.arange(n_samples)
        
        # Check duration condition vectorized
        duration_mask = self._screen_sample_times[possible_indices + chunk_size - 1] < self.end_time

        # this assumes that the valid_condition is a single condition
        valid_conditions = [self.modality_config["screen"]["valid_condition"]]

        if self.modality_config["screen"]["include_blanks"]:
            additional_valid_conditions = {"tier": "blank"}  
            valid_conditions.append(additional_valid_conditions)

        sample_mask_from_meta_conditions = self.get_screen_sample_mask_from_meta_conditions(chunk_size, valid_conditions)

        final_mask = duration_mask & sample_mask_from_meta_conditions

        return self._screen_sample_times[possible_indices[final_mask]]
     

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
        timestamps = {}
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

            if device_name == 'responses':
                if self._experiment.devices["responses"].use_phase_shifts:
                    phase_shifts = self._experiment.devices["responses"]._phase_shifts
                    times = (times - times.min())[:, None] + phase_shifts[None, :]

            timestamps[device_name] = torch.from_numpy(times)
        out["timestamps"] = timestamps
        if self.add_behavior_as_channels:
            out = add_behavior_as_channels(out)

        # remove any keys that are not in the out_keys
        out = {k: out[k] for k in self.out_keys if k in out}

        return out


from experanto.utils import add_behavior_as_channels, replace_nan_with_batch_mean
from typing import Optional, Union, List


class TargetedChunkDataset(ChunkDataset):
    def __init__(
            self,
            root_folder: str,
            global_sampling_rate: None,
            global_chunk_size: None,
            modality_config: dict = DEFAULT_MODALITY_CONFIG,
            add_behavior_as_channels: bool = False,
            replace_nans_with_means: bool = False,
            sample_times: Optional[Iterable] = None,
            history: float = 0.,
            valid_time_difference: list = [0.04, 0.160],

    ) -> None:
        """
        :param sample_times: a list of timestamps. Each timestamp will be turned into a chunk
        :param history: history in seconds of where the chunk should start
        :param valid_time_difference: the interval in seconds, relative to the timestamp of interest
        """
        super().__init__(root_folder=root_folder,
                         global_sampling_rate=global_sampling_rate,
                         global_chunk_size=global_chunk_size,
                         modality_config=modality_config,
                         add_behavior_as_channels=add_behavior_as_channels,
                         replace_nans_with_means=replace_nans_with_means,
                         )

        self._sample_times = sample_times
        self.history = history
        self.valid_time_difference = valid_time_difference
        self.target_marks = {}
        self.extended = False        

        # check if all offsets in the modality config are zero:
        for device_name in self.device_names:
            if self.modality_config[device_name].offset != 0:
                raise ValueError(f"Offset in modality config for {device_name} has to be Zero for this dataloader. ")
        
        # set the sample times for the targeted chunks
        tier = self.modality_config["screen"]["valid_condition"]["tier"]
        self._set_sample_times_for_targeted_chunks(tier, True, True)

    def __getitem__(self, idx) -> dict:
        out = {}
        if not self.extended:
            s = self._valid_screen_times[idx] # s is the start time of the sample of interest
        else:
            s = self._extended_valid_screen_times[idx]
        for device_name in self.device_names:
            sampling_rate = self.sampling_rates[device_name]
            chunk_size = self.chunk_sizes[device_name]
            chunk_s = chunk_size / sampling_rate

            times = np.linspace(s, s + chunk_s, chunk_size, endpoint=False)
            times = times - self.history
            data, _ = self._experiment.interpolate(times, device=device_name)
            if device_name == "responses":
                relative_times = times - s
                delta_min = self.valid_time_difference[0]
                delta_max = self.valid_time_difference[1]
                valid_mask = ((relative_times) >= delta_min) & ((relative_times) <= delta_max)
                valid_indices = np.where(valid_mask)[0] # these are the indices of the samples that are of interest

            if self.replace_nans_with_means:
                if np.any(np.isnan(data)):
                    data = replace_nan_with_batch_mean(data)

            out[device_name] = self.transforms[device_name](data).squeeze(
                0)  # remove dim0 for response/eye_tracker/treadmill
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
        out["valid_indices"] = torch.from_numpy(valid_indices)
        return out
    
    def _set_sample_times_for_targeted_chunks(
            self,
            tier: Optional[Union[str, List[str]]],
            valid_trials_only: bool = False,
            use_first_frame: bool = True
    ) -> None:
        """Sets valid sample times by filtering trials based on tier and validity.

        Args:
            tier: Trial tier(s) to include (e.g., 'train', 'test').
            valid_trials_only: If True, only includes valid trials.
            use_first_frame: If True, uses only first frame timestamp per trial, else uses all frames.
        """

        # Get all trials
        trials = np.array(self._trials)
        
        valid_conditions = [{"tier" : tier}]
        if valid_trials_only:
            valid_conditions[0]["valid_trial"] = True

        condition_mask = self.get_condition_mask_from_meta_conditions(valid_conditions)
        
        trials = trials[condition_mask]

        # Get timestamps
        timestamps = self._experiment.devices['screen'].timestamps
        
        # Extract times based on first_frame or all frames
        sample_times = []
        if use_first_frame:
            for trial in trials:
                first_frame_idx = trial.get_meta('first_frame_idx')
                if first_frame_idx is not None:
                    sample_times.append(timestamps[first_frame_idx])
        else:
            for trial in trials:
                frame_indices = trial.get_meta('frame_idx')
                if frame_indices is not None:
                    sample_times.extend(timestamps[frame_indices])
                    
        self._valid_screen_times = np.array(sample_times)

    def prepare_for_multisampling(self) -> None:

        self.extend_valid_screen_times()
        self.create_target_marks()
        self.generate_index_maps_to_target_marks()        

    def extend_valid_screen_times(self) -> None:
        """
        Extend valid screen times by generating and filtering potential times.

        This method calculates potential valid screen times by applying shifts 
        to existing valid times and filters them to match closely with the full 
        valid sample times. The result is stored in `_extended_valid_screen_times`, 
        and the `extended` flag is set to True.
        """
        T_response = 1 / self.sampling_rates['responses']
        response_chunk_duration = self.chunk_sizes['responses'] * T_response
        epsilon = T_response * 0.5
        max_shift_count = int((response_chunk_duration - self.valid_time_difference[0] - epsilon) / T_response)
        
        # Create all possible shifts for each valid screen time using broadcasting
        shifts = np.arange(max_shift_count) * T_response
        potential_times = self._valid_screen_times[:, None] - shifts[None, :]
        potential_times = potential_times.flatten()
        # Keep only times that are close to any time in full_valid_sample_times
        valid_mask = np.zeros_like(potential_times, dtype=bool)
        for full_time in self._full_valid_sample_times:
            valid_mask |= np.isclose(potential_times, full_time, atol=5e-3, rtol=0)
        
        self._extended_valid_screen_times = np.sort(potential_times[valid_mask])
        self.extended = True

    def create_target_marks(self) -> None:
        """
        Initializes and populates the target marks for each trial.

        This method filters trials based on specified tiers and validity, 
        calculates marking times within an image, and organizes target marks 
        by image ID and sub-index. Each mark contains groundtruth and prediction 
        lists for further evaluation.
        """
        target_marks = {}
        
        trials = self._trials

        # Filter by tier if specified
        if self.tier is not None:
            if isinstance(self.tier, str):
                self.tier = [self.tier]
            trials = [t for t in trials if t.get_meta('tier') in self.tier]
            
        # Filter by valid_trial if specified
        if self.valid_trials_only:
            trials = [t for t in trials if t.get_meta('valid_trial')]        

        response_sampling_duration = 1 / self.sampling_rates['responses']
        response_chunk_duration = self.chunk_sizes['responses'] * response_sampling_duration

        marking_times_in_an_image = [t for t in np.arange(0, response_chunk_duration, response_sampling_duration) if t >= self.valid_time_difference[0] and t <= self.valid_time_difference[1]]

        target_marks["marking_times_in_an_image"] = marking_times_in_an_image

        target_marks["marks"] = {}

        target_marks["trial_index_to_image_name"] = {}

        for i, trial in enumerate(trials):

            image_id = str(trial.get_meta('image_id'))


            # Find all keys with this image_id prefix and get max sub-index
            matching_keys = [k for k in target_marks["marks"].keys() if k.startswith(image_id + "_")]
            next_sub_idx = max([int(k.split("_")[-1]) for k in matching_keys]) + 1 if len(matching_keys) > 0 else 0
            target_marks["marks"][image_id + f"_{next_sub_idx}"] = [[] for _ in marking_times_in_an_image]

            target_marks["trial_index_to_image_name"][i] = image_id + f"_{next_sub_idx}"

            target_marks["marks"][image_id + f"_{next_sub_idx}"] = [{
                "groundtruth" : [],
                "predictions" : []
            } for _ in marking_times_in_an_image]

        self.target_marks = target_marks


    def generate_index_maps_to_target_marks(self) -> None:

        """
        Map each extended valid screen timestamp to the response frames that align
        with its original screen anchor points. For each extended timestamp, this
        method identifies which anchor points fall within its response window,
        then stores the indices of matching frames and their timestamps. This
        ensures consistent indexing between screen and response data when their
        sampling rates differ, facilitating the correct insertion of predictions
        and ground truth.
        """        

        response_sampling_duration = 1 / self.sampling_rates['responses']
        response_chunk_duration = self.chunk_sizes['responses'] * response_sampling_duration

        anchor_points = self._valid_screen_times
        anchor_points_right = self._valid_screen_times + self.valid_time_difference[1]
        anchor_points_left = self._valid_screen_times + self.valid_time_difference[0]

        self.target_marks["index_maps_to_target_marks"] = {}

        for index in range(len(self._extended_valid_screen_times)):

            self.target_marks["index_maps_to_target_marks"][index] = {}

            insert_screen_time = self._extended_valid_screen_times[index]
            insert_screen_time_response_end = insert_screen_time + response_chunk_duration

            response_times = np.linspace(insert_screen_time, insert_screen_time_response_end, self.chunk_sizes['responses'], endpoint=False)

            target_images_mask = anchor_points_right - insert_screen_time >= 0
            target_images_mask &= insert_screen_time_response_end - anchor_points_left >= 0

            # this is to ensure that those we are going to insert are for the original trial not for the others that also inside the response window
            # we could also get those in the response window but then due to screen sampling rate not being equal to response sampling rate, the indices actually do not match
            # Check if time differences are (approximately) integer multiples of response_sampling_duration
            time_diffs = insert_screen_time - anchor_points
            multiples = time_diffs / response_sampling_duration
            rounded_multiples = np.round(multiples)
            is_integer_multiple = np.abs(multiples - rounded_multiples) < 1e-10  # Small threshold for floating point comparison
            target_images_mask &= is_integer_multiple

            target_images_indices = np.where(target_images_mask)[0]

            target_anchor_points_left = anchor_points_left[target_images_indices]
            target_anchor_points_right = anchor_points_right[target_images_indices]

            response_times = response_times[None, :]
            target_anchor_points_left = target_anchor_points_left[:, None]
            target_anchor_points_right = target_anchor_points_right[:, None]

            temp1 = response_times - target_anchor_points_left
            temp2 = target_anchor_points_right - response_times

            target_mark_indices = temp1 >= 0
            target_mark_indices &= temp2 >= 0

            for j, target_image_index in enumerate(target_images_indices):
                corresponding_response_times = response_times[0][target_mark_indices[j]]
                self.target_marks["index_maps_to_target_marks"][index][target_image_index] = (target_mark_indices[j], corresponding_response_times)


    def insert_result_into_target_marks(self, index, logits, groundtruth):
        """
        Inserts prediction results into target marks for a given index,
        automatically determining how many slots (`num_points`) are possible
        based on the sampling rate and valid time difference. Indices for
        the `True` values in the mask are assigned accordingly.
        """
        # Compute the total number of points that fit into the valid time window
        response_duration = 1 / self.sampling_rates['responses']
        valid_time_difference = self.valid_time_difference[1] - self.valid_time_difference[0]
        num_points = int(valid_time_difference / response_duration) + 1

        for image_idx, (mask, _) in self.target_marks["index_maps_to_target_marks"][index].items():
            true_idxs = np.flatnonzero(mask)
            if not true_idxs.size:
                continue  # No True values => nothing to insert

            # Decide whether to start from 0 or from the end (num_points - number_of_trues)
            start_idx = 0 if true_idxs[0] == 0 else (num_points - len(true_idxs))
            assigned_idxs = range(start_idx, start_idx + len(true_idxs))

            image_name = self.target_marks["trial_index_to_image_name"][image_idx]
            for mask_i, assign_i in zip(true_idxs, assigned_idxs):
                self.target_marks["marks"][image_name][assign_i]["groundtruth"].append(groundtruth[mask_i])
                self.target_marks["marks"][image_name][assign_i]["predictions"].append(logits[mask_i])

    def evaluate_target_marks(self, averaged_across_images: bool = True) -> None:
        """
        Evaluates the target marks using Pearson correlation coefficient.

        Args:
            averaged_across_images (bool): If True, averages results across images 
                                           with the same ID. Defaults to True.

        This method computes the Pearson correlation for predictions against 
        groundtruth data, optionally averaging across images with the same ID. 
        It handles cases with insufficient data by skipping them.
        """
        from torchmetrics import PearsonCorrCoef

        assert len(self.target_marks["marks"]) > 0, "No target marks found"
        first_image = list(self.target_marks["marks"].keys())[0]
        assert len(self.target_marks["marks"][first_image]) > 0, "No marks found for first image"
        assert "groundtruth" in self.target_marks["marks"][first_image][0], "No groundtruth found in first mark"
        assert len(self.target_marks["marks"][first_image][0]["groundtruth"]) > 0, "Empty groundtruth found"
        num_neurons = self.target_marks["marks"][first_image][0]["groundtruth"].shape[0]

        pearson = PearsonCorrCoef(num_outputs = num_neurons)

        image_ids = []

        if averaged_across_images:
            for image_name, image_data in self.target_marks["marks"].items():
                image_ids.append(image_name.split("_")[0])
            image_ids = list(set(image_ids))
        else:
            image_ids = [image_name for image_name in self.target_marks["marks"].keys()]

        num_weird_images = 0

        num_of_marks_for_each_image = len(self.target_marks["marking_times_in_an_image"])
        
        for image_id in image_ids:

            target_image_names = [image_name for image_name in self.target_marks["marks"].keys() if image_name.startswith(image_id)]

            groundtruth_datas = [[] for _ in range(num_of_marks_for_each_image)]
            predictions_datas = [[] for _ in range(num_of_marks_for_each_image)]
            
            for target_image_name in target_image_names:


                for i in range(num_of_marks_for_each_image):

                    groundtruth_datas[i].extend(self.target_marks["marks"][target_image_name][i]["groundtruth"])
                    predictions_datas[i].extend(self.target_marks["marks"][target_image_name][i]["predictions"])


            for i in range(num_of_marks_for_each_image):
                # taking the mean across all the samples for each neuron
                groundtruth_datas[i] = np.array(groundtruth_datas[i]).mean(axis=0)
                predictions_datas[i] = np.array(predictions_datas[i]).mean(axis=0)

            groundtruth_datas = np.array(groundtruth_datas)
            predictions_datas = np.array(predictions_datas)

            if len(groundtruth_datas.shape) == 1:
                num_weird_images += 1
                continue

            groundtruth_datas = torch.tensor(groundtruth_datas)
            predictions_datas = torch.tensor(predictions_datas)

            pearson.update(predictions_datas, groundtruth_datas)

        result = pearson.compute()

        return result
