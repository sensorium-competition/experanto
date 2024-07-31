from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterable
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from .experiment import Experiment
from .interpolators import ImageTrial


class Mouse2pChunkedDataset(Dataset):
    def __init__(self, root_folder: str, sampling_rate: float, chunk_size: int) -> None:
        self.root_folder = Path(root_folder)
        self.sampling_rate = sampling_rate
        self.chunk_size = chunk_size
        self._experiment = Experiment(root_folder)
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

        # Hack-1 add timestamps for each neuron
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
            stim_duration: float
            ) -> None:
        self.root_folder = Path(root_folder)
        self.tier = tier
        self.offset = offset
        self.stim_duration = stim_duration
        self._experiment = Experiment(root_folder)
        self.device_names = self._experiment.device_names
        self.DataPoint = namedtuple("DataPoint", self.device_names)
        self._read_trials()
    
    def _read_trials(self):
        screen = self._experiment.devices["screen"]
        self._trials = [
            t for t in screen.trials
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
                times = self._start_times[idx] + self.offset + np.arange(
                    0, self.stim_duration, 1.0 / Fs)
            d, _ = device.interpolate(times)
            if device_name == "screen":
                data[device_name] = (d.mean(axis=0)[None, ...] - 124) / 64
            else:
                data[device_name] = d.mean(axis=0)
        return self.DataPoint(**data)

    

class StaticNoDatapoint(Dataset):
    def __init__(
            self, 
            root_folder: str, 
            tier: str, 
            offset: float, 
            stim_duration: float
            ) -> None:
        self.root_folder = Path(root_folder)
        self.tier = tier
        self.offset = offset
        self.stim_duration = stim_duration
        self._experiment = Experiment(root_folder)
        self.device_names = self._experiment.device_names
        self._read_trials()
    
    def _read_trials(self):
        screen = self._experiment.devices["screen"]
        self._trials = [
            t for t in screen.trials
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
        data = []
        for device_name, device in self._experiment.devices.items():
            if device_name == "screen":
                times = self._start_times[idx]
            else:
                Fs = device.sampling_rate
                times = self._start_times[idx] + self.offset + np.arange(
                    0, self.stim_duration, 1.0 / Fs)
            d, _ = device.interpolate(times)
            if device_name == "screen":
                data.append((d.mean(axis=0)[None, ...] - 124) / 64)
            else:
                data.append(d.mean(axis=0))
        return tuple(data)

    
class StaticDict(Dataset):
    def __init__(
            self, 
            root_folder: str, 
            tier: str, 
            offset: float, 
            stim_duration: float
            ) -> None:
        self.root_folder = Path(root_folder)
        self.tier = tier
        self.offset = offset
        self.stim_duration = stim_duration
        self._experiment = Experiment(root_folder)
        self.device_names = self._experiment.device_names
        self._read_trials()
    
    def _read_trials(self):
        screen = self._experiment.devices["screen"]
        self._trials = [
            t for t in screen.trials
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
                times = self._start_times[idx] + self.offset + np.arange(
                    0, self.stim_duration, 1.0 / Fs)
            d, _ = device.interpolate(times)
            if device_name == "screen":
                data[device_name] = (d.mean(axis=0)[None, ...] - 124) / 64
            else:
                data[device_name] = d.mean(axis=0)
        return data