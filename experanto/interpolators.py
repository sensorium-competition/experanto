from __future__ import annotations

import os
import re
import typing
import warnings
from abc import abstractmethod
from pathlib import Path

import cv2
import numpy as np
import numpy.lib.format as fmt
import yaml

from .utils import linear_interpolate_1d_sequence, linear_interpolate_sequences


class TimeInterval(typing.NamedTuple):
    start: float
    end: float

    def __contains__(self, time):
        return self.start <= time < self.end

    def intersect(self, times):
        return (times >= self.start) & (times < self.end)

    def __repr__(self) -> str:
        return f"TimeInterval [{self.start}, {self.end})"

    def __iter__(self):
        return iter((self.start, self.end))


class Interpolator:
    def __init__(self, root_folder: str) -> None:
        self.root_folder = Path(root_folder)
        meta = self.load_meta()
        self.start_time = None
        self.end_time = None
        # Valid interval can be different to start time and end time.
        self.valid_interval = None

    def load_meta(self):
        with open(self.root_folder / "meta.yml") as f:
            meta = yaml.load(f, Loader=yaml.SafeLoader)
        return meta

    @abstractmethod
    def interpolate(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...
        # returns interpolated signal and boolean mask of valid samples

    def __contains__(self, times: np.ndarray):
        return np.any(self.valid_times(times))

    @staticmethod
    def create(root_folder: str, **kwargs) -> "Interpolator":
        with open(Path(root_folder) / "meta.yml", "r") as file:
            meta_data = yaml.load(file, Loader=yaml.SafeLoader)
        modality = meta_data.get("modality")
        class_name = modality.capitalize() + "Interpolator"
        assert class_name in globals(), f"Unknown modality: {modality}"
        return globals()[class_name](root_folder, **kwargs)

    def valid_times(self, times: np.ndarray) -> np.ndarray:
        return self.valid_interval.intersect(times)


class SequenceInterpolator(Interpolator):
    def __init__(
        self,
        root_folder: str,
        keep_nans: bool = False,
        interpolation_mode: str = "nearest_neighbor",
        interp_window: int = 5,
        normalize: bool = False,
        normalize_subtract_mean: bool = False,
        normalize_std_threshold: typing.Optional[float] = None,  # or 0.01
        **kwargs,
    ) -> None:
        """
        interpolation_mode - nearest neighbor or linear
        keep_nans - if we keep nans in linear interpolation
        interp_window - how many points before or after the target period
            are considered for interpolation
        """
        super().__init__(root_folder)
        meta = self.load_meta()
        self.keep_nans = keep_nans
        self.interp_window = interp_window
        self.interpolation_mode = interpolation_mode
        self.normalize = normalize
        self.normalize_subtract_mean = normalize_subtract_mean
        self.normalize_std_threshold = normalize_std_threshold
        self.sampling_rate = meta["sampling_rate"]
        self.time_delta = 1.0 / self.sampling_rate
        self.start_time = meta["start_time"]
        self.end_time = meta["end_time"]
        # Valid interval can be different to start time and end time.
        self.valid_interval = TimeInterval(self.start_time, self.end_time)

        self.use_phase_shifts = meta["phase_shift_per_signal"]
        if meta["phase_shift_per_signal"]:
            self._phase_shifts = np.load(self.root_folder / "meta/phase_shifts.npy")
            self.valid_interval = TimeInterval(
                self.start_time + np.max(self._phase_shifts),
                self.end_time + np.min(self._phase_shifts),
            )
        self.n_signals = meta["n_signals"]
        # read .npy or .mem (memmap) file
        if (self.root_folder / "data.npy").exists():
            self._data = np.load(self.root_folder / "data.npy")
        else:
            self._data = np.memmap(
                self.root_folder / "data.mem",
                dtype=meta["dtype"],
                mode="r",
                shape=(meta["n_timestamps"], meta["n_signals"]),
            )
        if self.normalize:
            self.normalize_init()

    def normalize_init(self):
        self.mean = np.load(self.root_folder / "meta/means.npy")
        self.std = np.load(self.root_folder / "meta/stds.npy")
        assert (
            self.mean.shape[0] == self.n_signals
        ), f"mean shape does not match: {self.mean.shape} vs {self._data.shape}"
        assert (
            self.std.shape[0] == self.n_signals
        ), f"std shape does not match: {self.std.shape} vs {self._data.shape}"
        self.mean = self.mean.T
        self.std = self.std.T
        if self.normalize_std_threshold:
            threshold = self.normalize_std_threshold * np.nanmean(self.std)
            idx = self.std > threshold
            self._precision = np.ones_like(self.std) / threshold
            self._precision[idx] = 1 / self.std[idx]
        else:
            self._precision = 1 / self.std

    #         if len(self._precision.shape) == 1:
    #             self._precision = self._precision.reshape(1, -1)

    def normalize_data(self, data):
        if self.normalize_subtract_mean:
            data = data - self.mean
        data = data * self._precision
        return data

    def interpolate(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # valid is an array of boolean, right
        valid = self.valid_times(times)
        valid_times = times[valid]
        if self.use_phase_shifts:
            idx = np.floor(
                (
                    valid_times[:, np.newaxis]
                    - self._phase_shifts[np.newaxis, :]
                    - self.start_time
                )
                / self.time_delta
            ).astype(int)
            data = np.take_along_axis(self._data, idx, axis=0)
        else:
            idx = np.floor((valid_times - self.start_time) / self.time_delta).astype(
                int
            )
            data = self._data[idx]
        if self.interpolation_mode == "nearest_neighbor":
            return data, valid
        elif self.interpolation_mode == "linear":
            # we are interested to take the data a bit before and after to have better interpolation
            # if the target time sequence starts / ends with nan
            if len(idx.shape) == 1:
                start_idx = int(max(0, idx[0] - self.interp_window))
                end_idx = int(
                    min(
                        idx[-1] + self.interp_window,
                        np.floor(
                            (self.valid_interval.end - self.valid_interval.start)
                            / self.time_delta
                        ),
                    )
                )

                # time is always first dim
                array = self._data[start_idx:end_idx]
                orig_times = (
                    np.arange(start_idx, end_idx) * self.time_delta
                    + self.valid_interval.start
                )
                assert (
                    array.shape[0] == orig_times.shape[0]
                ), "times and data should be same length before interpolation"
                data = linear_interpolate_sequences(
                    array, orig_times, valid_times, self.keep_nans
                )

            else:
                # this probably should be changed to be more efficient
                start_idx = np.where(
                    idx[0, :] - self.interp_window > 0,
                    idx[0, :] - self.interp_window,
                    0,
                ).astype(int)
                max_idx = np.floor(
                    (self.valid_interval.end - self.valid_interval.start)
                    / self.time_delta
                ).astype(int)
                end_idx = np.where(
                    idx[-1, :] + self.interp_window < max_idx,
                    idx[-1, :] + self.interp_window,
                    max_idx,
                ).astype(int)
                data = np.full((len(valid_times), self._data.shape[-1]), np.nan)
                for n_idx, st_idx in enumerate(start_idx):
                    local_data = self._data[st_idx : end_idx[n_idx], n_idx]
                    local_time = (
                        np.arange(st_idx, end_idx[n_idx]) * self.time_delta
                        + self.valid_interval.start
                    )
                    assert (
                        local_data.shape[0] == local_time.shape[0]
                    ), "times and data should be same length before interpolation"

                    data[:, n_idx] = linear_interpolate_1d_sequence(
                        local_data, local_time, valid_times, self.keep_nans
                    )
            if self.normalize:
                data = self.normalize_data(data)
            return data, valid
        else:
            raise NotImplementedError(
                f"interpolation_mode should be linear or nearest_neighbor"
            )


class ScreenInterpolator(Interpolator):
    def __init__(
        self,
        root_folder: str,
        rescale: bool = False,
        rescale_size: typing.Optional[tuple(int, int)] = None,
        normalize: bool = False,
        **kwargs,
    ) -> None:
        """
        rescale would rescale images to the _image_size if true
        """
        super().__init__(root_folder)
        self.timestamps = np.load(self.root_folder / "timestamps.npy")
        self.start_time = self.timestamps[0]
        self.end_time = self.timestamps[-1]
        self.valid_interval = TimeInterval(self.start_time, self.end_time)
        self.rescale = rescale
        self._parse_trials()

        # create mapping from image index to file index
        self._num_frames = [t.num_frames for t in self.trials]
        self._first_frame_idx = [t.first_frame_idx for t in self.trials]
        self._data_file_idx = np.concatenate(
            [np.full(t.num_frames, i) for i, t in enumerate(self.trials)]
        )
        # infer image size
        if not rescale_size:
            for m in self.trials:
                if m.image_size is not None:
                    self._image_size = m.image_size
                    break
        else:
            self._image_size = rescale_size
        self.normalize = normalize
        if self.normalize:
            self.normalize_init()

    def normalize_init(self):
        self.mean = np.load(self.root_folder / "meta/means.npy")
        self.std = np.load(self.root_folder / "meta/stds.npy")
        if self.rescale:
            self.mean = self.rescale_frame(self.mean.T).T
            self.std = self.rescale_frame(self.std.T).T
        assert (
            self.mean.shape == self._image_size
        ), f"mean size is different: {self.mean.shape} vs {self._image_size}"
        assert (
            self.std.shape == self._image_size
        ), f"std size is different: {self.std.shape} vs {self._image_size}"

    def normalize_data(self, data):
        return (data - self.mean) / self.std

    def _combine_metadatas(self) -> None:
        
        # Function to check if a file is a numbered yml file
        def is_numbered_yml(file_name):
            return re.fullmatch(r"\d{5}\.yml", file_name) is not None

        # Initialize an empty dictionary to store all YAML contents
        all_yaml_data = {}

        # Get block subfolders and sort by number
        meta_files = [
            f
            for f in (self.root_folder / "meta").iterdir()
            if f.is_file() and is_numbered_yml(f.name)
        ]
        meta_files.sort(key=lambda f: int(os.path.splitext(f.name)[0]))        

        # Read each YAML file and store under its filename
        for meta_file in meta_files:
            with open(meta_file, 'r') as file:
                file_base_name = meta_file.stem # here it should be the yaml file's name without the .yml extension
                yaml_content = yaml.safe_load(file)
                all_yaml_data[file_base_name] = yaml_content

        output_path = self.root_folder / "combined_meta.yml"
        with open(output_path, 'w') as file:
            yaml.dump(all_yaml_data, file)

    def read_combined_meta(self) -> None:

        if not (self.root_folder / "combined_meta.yml").exists():
            print("Combining metadatas...")
            self._combine_metadatas()

        with open(self.root_folder / "combined_meta.yml", 'r') as file:
            self.combined_meta = yaml.safe_load(file)
        
        metadatas = []
        keys = []
        for key, value in self.combined_meta.items():
            metadatas.append(value)
            keys.append(key)

        return metadatas, keys
    
    def _parse_trials(self) -> None:

        self.trials = []

        metadatas, keys = self.read_combined_meta()

        for key, metadata in zip(keys, metadatas):

            data_file_name = self.root_folder / "data" / f"{key}.npy"
            self.trials.append(ScreenTrial.create(data_file_name, metadata))

    def interpolate(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = self.valid_times(times)
        valid_times = times[valid]
        valid_times += 1e-4  # add small offset to avoid numerical issues

        assert np.all(np.diff(valid_times) > 0), "Times must be sorted"
        idx = (
            np.searchsorted(self.timestamps, valid_times) - 1
        )  # convert times to frame indices
        assert np.all(
            (idx >= 0) & (idx < len(self.timestamps))
        ), "All times must be within the valid range"
        data_file_idx = self._data_file_idx[idx]

        # Go through files, load them and extract all frames
        unique_file_idx = np.unique(data_file_idx)
        out = np.zeros([len(valid_times)] + list(self._image_size))
        for u_idx in unique_file_idx:
            data = self.trials[u_idx].get_data()
            # TODO: establish convention of dimensons for time/channels. Then we can remove this
            if ((len(data.shape) == 2) or (data.shape[-1] == 3)) and (len(data.shape) < 4):
                data = np.expand_dims(data, axis=0)
            idx_for_this_file = np.where(self._data_file_idx[idx] == u_idx)
            if self.rescale:
                orig_size = data[idx[idx_for_this_file] - self._first_frame_idx[u_idx]]
                out[idx_for_this_file] = np.stack(
                    [
                        self.rescale_frame(np.asarray(frame, dtype=np.float32).T).T
                        for frame in orig_size
                    ]
                )
            else:
                out[idx_for_this_file] = data[
                    idx[idx_for_this_file] - self._first_frame_idx[u_idx]
                ]
        if self.normalize:
            out = self.normalize_data(out)
        return out, valid

    def rescale_frame(self, frame: np.array) -> np.array:
        """
        Changes the resolution of the image to this size.
        Returns: Rescaled image
        """
        return cv2.resize(frame, self._image_size, interpolation=cv2.INTER_AREA).astype(
            np.float32
        )


class ScreenTrial:
    def __init__(
        self,
        data_file_name: str,
        meta_data: dict,
        image_size: tuple,
        first_frame_idx: int,
        num_frames: int,
    ) -> None:
        self.data_file_name = data_file_name
        self._meta_data = meta_data
        self.modality = meta_data.get("modality")
        self.image_size = image_size
        self.first_frame_idx = first_frame_idx
        self.num_frames = num_frames

    @staticmethod
    def create(data_file_name: str, meta_data: dict) -> "ScreenTrial":
        modality = meta_data.get("modality")
        class_name = modality.capitalize() + "Trial"
        assert class_name in globals(), f"Unknown modality: {modality}"
        return globals()[class_name](data_file_name, meta_data)

    def get_data(self) -> np.array:
        return np.load(self.data_file_name)

    def get_meta(self, property: str):
        return self._meta_data.get(property)


class ImageTrial(ScreenTrial):
    def __init__(self, data_file_name, meta_data) -> None:
        super().__init__(
            data_file_name,
            meta_data,
            tuple(meta_data.get("image_size")),
            meta_data.get("first_frame_idx"),
            1,
        )


class VideoTrial(ScreenTrial):
    def __init__(self, data_file_name, meta_data) -> None:
        super().__init__(
            data_file_name,
            meta_data,
            tuple(meta_data.get("image_size")),
            meta_data.get("first_frame_idx"),
            meta_data.get("num_frames"),
        )


class BlankTrial(ScreenTrial):
    def __init__(self, data_file_name, meta_data) -> None:
        super().__init__(
            data_file_name,
            meta_data,
            tuple(meta_data.get("image_size")),
            meta_data.get("first_frame_idx"),
            1,
        )
        self.interleave_value = meta_data.get("interleave_value")

    def get_data(self) -> np.array:
        return np.full((1,) + self.image_size, self.interleave_value)
