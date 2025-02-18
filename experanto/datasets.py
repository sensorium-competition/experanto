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
import yaml
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt

from .configs import DEFAULT_MODALITY_CONFIG
from .experiment import Experiment
from .interpolators import ImageTrial, VideoTrial
from .utils import GazeBasedCrop, replace_nans_with_neighbors, get_validation_split


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
        self._statistics = {}
        for device_name in self.device_names:
            self._statistics[device_name] = {}
            # If modality should be normalized, load respective statistics from file.
            if self.modality_config[device_name].transforms.get("normalization", False):
                mode = self.modality_config[device_name].transforms.normalization
                assert mode in ['standardize', 'normalize']
                means = np.load(self._experiment.devices[device_name].root_folder / "meta/means.npy")
                stds = np.load(self._experiment.devices[device_name].root_folder / "meta/stds.npy")
                if mode == 'standardize':
                    # If modality should only be standarized, set means to 0.
                    means = np.zeros_like(means)

                self._statistics[device_name]["mean"] = means.reshape(1, -1)  # (n, 1) -> (1, n) for broadcasting in __get_item__
                self._statistics[device_name]["std"] = stds.reshape(1, -1)  # same as above

    def initialize_transforms(self):
        transforms = {}
        for device_name in self.device_names:
            if device_name == "screen":
                #add_channel = Lambda(lambda x: torch.from_numpy(x[:, None, ...]) if len(x.shape) == 3 else torch.from_numpy(x))
                # Handle adding a channel dimension for both tensor and array inputs
                add_channel = Lambda(lambda x: x[:, None, ...] if isinstance(x, torch.Tensor) and len(x.shape) == 3
                                 else torch.from_numpy(x[:, None, ...]) if isinstance(x, np.ndarray) and len(x.shape) == 3
                                 else x)
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
        iterates through all
           for example:
              if meta_conditions = {"tier": [train,train, ...], "stim_type": [type1, type2, ...]}
              and valid_condition = {"tier": train, "stim_type": type2}
           then the output would be {"tier": [True, True, ...], "stim_type": [False, True, ...]}
        """
        sample_in_meta_condition = {}
        for k, v in self.modality_config["screen"]["valid_condition"].items():
            sample_in_meta = []
            for i, (condition, start, end) in enumerate(zip(self.meta_conditions[k], self._start_times, self._end_times)):
                if condition == v or (condition == "blank" and self.modality_config["screen"]["include_blanks"]):
                    sample_in_meta.append(
                        (self._screen_sample_times >= start) & (self._screen_sample_times < end)
                    )
            sample_in_meta_condition[k] = np.stack(sample_in_meta).sum(0).astype(bool)
        return sample_in_meta_condition

    def get_full_valid_sample_times(self, ) -> Iterable:
        """
        iterates through all sample times and checks if they are in the meta condition of interest
        :return:
        """
        valid_times = []
        for i, _ in enumerate(self._screen_sample_times[:-self.chunk_sizes["screen"]]):
            maybe_all_true = []
            correct_duration = self._screen_sample_times[i + self.chunk_sizes["screen"]] < self.end_time
            maybe_all_true.append(correct_duration)
            for k, v in self._sample_in_meta_condition.items():
                maybe_all_true.append(np.all(v[i: i + self.chunk_sizes["screen"]]))
            if np.all(maybe_all_true):
                valid_times.append(self._screen_sample_times[i])
        return np.stack(valid_times)

    def shuffle_valid_screen_times(self) -> None:
        """
        convenience function to randomly select new starting points for each chunk. Use this in training after each epoch.
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
            print(device_name)
            data, _ = self._experiment.interpolate(times, device=device_name)
            out[device_name] = self.transforms[device_name](data).squeeze(0) # remove dim0 for response/eye_tracker/treadmill

        phase_shifts = self._experiment.devices["responses"]._phase_shifts
        times_with_phase_shifts = (times - times.min())[:, None] + phase_shifts[None, :]
        out["timestamps"] = torch.from_numpy(times_with_phase_shifts)
        return out
    

class MonkeyFixation(Dataset):
    def __init__(
        self,
        root_folder: str,
        global_sampling_rate: None,
        global_chunk_size: None,
        modality_config: dict = DEFAULT_MODALITY_CONFIG,
    ) -> None:
        self.root_folder = Path(root_folder)
        self.modality_config = instantiate(modality_config)
        self.chunk_sizes, self.sampling_rates, self.chunk_s = {}, {}, {}
        for device_name in self.modality_config.keys():
            cfg = self.modality_config[device_name]
            self.chunk_sizes[device_name] = global_chunk_size or cfg.chunk_size
            self.sampling_rates[device_name] = global_sampling_rate or cfg.sampling_rate

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

        self.transforms = self.initialize_transforms()

        # Dynamically filter swap times based on meta_conditions
        self._swap_times_valid_trial = self.get_swap_time_valid_conditions()
        self.DataPoint = namedtuple("DataPoint", self.device_names)

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
        self._statistics = {}
        for device_name in self.device_names:
            self._statistics[device_name] = {}
            if self.modality_config[device_name].transforms.get("normalization", False):
                mode = self.modality_config[device_name].transforms.normalization
                assert mode in ['standardize', 'normalize']
                means = np.load(self._experiment.devices[device_name].root_folder / "meta/means.npy")
                stds = np.load(self._experiment.devices[device_name].root_folder / "meta/stds.npy")
                if mode == 'standardize':
                    means = np.zeros_like(means)

                self._statistics[device_name]["mean"] = means.reshape(1, -1)
                self._statistics[device_name]["std"] = stds.reshape(1, -1)

    def initialize_transforms(self):
        transforms = {}
        for device_name in self.device_names:
            if device_name == "screen":
                add_channel = Lambda(lambda x: x[:, None, ...] if isinstance(x, torch.Tensor) and len(x.shape) == 3
                                 else torch.from_numpy(x[:, None, ...]) if isinstance(x, np.ndarray) and len(x.shape) == 3
                                 else x)
                transform_list = [v for v in self.modality_config.screen.transforms.values() if isinstance(v, torch.nn.Module)]
                transform_list.insert(0, add_channel)
                transforms[device_name] = Compose(transform_list)
            else:
                transform_list = [ToTensor()]
            if self.modality_config[device_name].transforms.get("normalization", False):
                transform_list.append(
                    torchvision.transforms.Normalize(self._statistics[device_name]["mean"], self._statistics[device_name]["std"])
                )
            transforms[device_name] = Compose(transform_list)
        return transforms

    def get_swap_time_valid_conditions(self) -> np.ndarray:
        """
        Determines valid swap times based on the meta conditions in the modality configuration.
        """
        valid_conditions = self.modality_config["screen"]["valid_condition"]
        include_blanks = self.modality_config["screen"]["include_blanks"]

        valid_indices = np.ones(len(self.meta_conditions["valid_trial"]), dtype=bool)

        for condition_key, condition_value in valid_conditions.items():
            condition_array = np.array(self.meta_conditions[condition_key])
            valid_indices &= (condition_array == condition_value) | (
                (condition_array == "blank") & include_blanks
            )
        self._valid_indices = valid_indices
        # Return the swap times that satisfy all conditions
        return self._experiment.devices['screen'].timestamps[valid_indices]


    def shuffle_valid_screen_times(self) -> None:
        times = self._full_valid_sample_times
        self._valid_screen_times = np.sort(np.random.choice(times, size=len(times) // self.sample_stride, replace=False))
    

    @staticmethod
    def get_batch_aligned_split(n_images, train_frac, batch_size, seed):
        if seed is not None:
            np.random.seed(seed)

        indices = np.arange(n_images)
        np.random.shuffle(indices)

        # Compute training size and align up to the next batch
        train_size = int(n_images * train_frac)
        train_size = ((train_size + batch_size - 1) // batch_size) * batch_size  # Align up
        train_size = min(train_size, n_images)  # Ensure it doesn't exceed total images

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        return train_indices, val_indices

    def split(self, train_frac=0.8, seed=None):
        """
        Split the dataset into train and validation subsets using the same logic as the static loader.
        """
        # Get train and validation indices using the static loader's logic
        train_indices, val_indices = get_validation_split(len(self), train_frac, seed)

        # Create train and validation subsets
        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)
        return train_dataset, val_dataset

    def __len__(self):
        return len(self._swap_times_valid_trial)

    def __getitem__(self, idx) -> dict:
        out = {}
        s = self._swap_times_valid_trial[idx]  # Select the swap time for this index
        for device_name in self.device_names:
            sampling_rate = self.sampling_rates[device_name]
            chunk_size = self.chunk_sizes[device_name]
            chunk_s = chunk_size / sampling_rate
            times = np.linspace(s, s + chunk_s, chunk_size, endpoint=False)
            times = times + self.modality_config[device_name].offset
            if device_name == "screen":
                (data,meta), _ = self._experiment.interpolate(times, device=device_name)
                out['image_id'] = meta
            else:
                data, _ = self._experiment.interpolate(times, device=device_name)
            out[device_name] = self.transforms[device_name](data).squeeze(0)
            
        out["timestamps"] = torch.from_numpy(times)
        datapoint = {'screen':out['screen'], 'responses':out['responses']}
        return out
        #return self.DataPoint(**datapoint)


class MonkeyFixationGazeCrop(MonkeyFixation):
    def __init__(
        self,
        root_folder: str,
        global_sampling_rate: None,
        global_chunk_size: None,
        modality_config: dict = DEFAULT_MODALITY_CONFIG,
    ) -> None:
        
        super().__init__(root_folder, global_sampling_rate, global_chunk_size, modality_config)
        gaze_params = ['destRect','bgColor','fixSpotColor','fixSpotLocation','monitorCenter','stimulusLocation']
        self.gaze_params = {each: self._experiment.devices['screen'].load_meta()[each] for each in gaze_params}
        self.gaze_cropper = GazeBasedCrop(
            crop_size=(100,100),
            pixel_per_degree=62.86,
            monitor_center=[1920/2,1080/2],
            dest_rect=[0, 0, 1920, 1080],  # Full-screen stimulus
        )

    def __getitem__(self, idx) -> dict:
        out = {}
        s = self._swap_times_valid_trial[idx]  # Select the swap time for this index
        for device_name in self.device_names:
            sampling_rate = self.sampling_rates[device_name]
            chunk_size = self.chunk_sizes[device_name]
            chunk_s = chunk_size / sampling_rate
            times = np.linspace(s, s + chunk_s, chunk_size, endpoint=False)
            times = times + self.modality_config[device_name].offset
            if device_name == "screen":
                (data,meta), _ = self._experiment.interpolate(times, device=device_name)
                out['image_id'] = meta
            else:
                data, _ = self._experiment.interpolate(times, device=device_name)
            out[device_name] = self.transforms[device_name](data).squeeze(0)
        
        # Apply Gaze-Based Cropping for each image in the sequence
        screen_images = out["screen"]  # Shape (T, C, H, W) if multiple images
        if screen_images.dim() == 4:  # Multiple images (T, C, H, W)
            cropped_screens = torch.stack([
                self.gaze_cropper(img, out['eye_tracker'], self.gaze_params['fixSpotLocation'], self.gaze_params['StimulusLocation'], dynamic=True) for img in screen_images
            ])
        else:  # Single image case
            cropped_screens = self.gaze_cropper(screen_images, out['eye_tracker'], self.gaze_params['fixSpotLocation'], self.gaze_params['stimulusLocation'], dynamic=False)

        out["screen"] = cropped_screens
        plt.imshow(out['screen'][0], cmap='gray')
        return out