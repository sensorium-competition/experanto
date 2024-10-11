from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .experiment import Experiment
from .interpolators import ImageTrial, VideoTrial, DEFAULT_INTERP_CONFIG

class SimpleChunkedDataset(Dataset):
    def __init__(
        self,
        root_folder: str,
        sampling_rate: float,
        chunk_size: int,
        interp_config: dict = DEFAULT_INTERP_CONFIG,
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
        return self.DataPoint(*list(data.values()))


class Mouse2pStaticImageDataset(Dataset):
    def __init__(
        self,
        root_folder: str,
        tier: str,
        offset: float,
        stim_duration: float,
        interp_config: dict = DEFAULT_INTERP_CONFIG,
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
        interp_config: dict = DEFAULT_INTERP_CONFIG,
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
