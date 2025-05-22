from __future__ import annotations

import json
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

from .intervals import TimeInterval
from .utils import linear_interpolate_1d_sequence, linear_interpolate_sequences


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
    def create(root_folder: str, cache_data: bool = False, **kwargs) -> "Interpolator":
        with open(Path(root_folder) / "meta.yml", "r") as file:
            meta_data = yaml.load(file, Loader=yaml.SafeLoader)
        modality = meta_data.get("modality")
        class_name = modality.capitalize() + "Interpolator"
        assert class_name in globals(), f"Unknown modality: {modality}"
        return globals()[class_name](root_folder, cache_data, **kwargs)

    def valid_times(self, times: np.ndarray) -> np.ndarray:
        return self.valid_interval.intersect(times)


class SequenceInterpolator(Interpolator):
    def __init__(
        self,
        root_folder: str,
        cache_data: bool = False,  # already cached, put it here for consistency
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

        is_memmap = isinstance(self._data, np.memmap)
        if cache_data and is_memmap:
            self._data = np.array(self._data).astype(
                np.float32
            )  # Convert memmap to ndarray

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

    def normalize_data(self, data):
        if self.normalize_subtract_mean:
            data = data - self.mean
        data = data * self._precision
        return data

    def interpolate(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        valid = self.valid_times(times)
        valid_times = times[valid]
        if self.interpolation_mode == "nearest_neighbor":
            if self.use_phase_shifts:
                idx_lower = np.floor(
                    (
                        valid_times[:, np.newaxis]
                        - self._phase_shifts[np.newaxis, :]
                        - self.start_time
                    )
                    / self.time_delta
                ).astype(int)
                data = np.take_along_axis(self._data, idx_lower, axis=0).astype(
                    np.float32
                )
            else:
                idx_lower = np.floor(
                    (valid_times - self.start_time) / self.time_delta
                ).astype(int)
                data = self._data[idx_lower].astype(np.float32)
            return data, valid

        elif self.interpolation_mode == "linear":

            if self.use_phase_shifts:
                idx_lower = np.floor(
                    (
                        valid_times[:, np.newaxis]
                        - self._phase_shifts[np.newaxis, :]
                        - self.start_time
                    )
                    / self.time_delta
                ).astype(int)
            else:
                idx_lower = np.floor(
                    (valid_times - self.start_time) / self.time_delta
                ).astype(int)

            idx_upper = idx_lower + 1
            overflow_mask = (idx_upper >= self._data.shape[0]) | (idx_lower < 0)
            compute_mask = ~overflow_mask

            if self.use_phase_shifts:

                valid_times = valid_times[:, None]
                interpolated = np.full(
                    (valid_times.shape[0], idx_lower.shape[1], 1), np.nan
                )

                for dim in range(idx_upper.shape[1]):

                    dim_mask = compute_mask[:, dim]

                    idx_lower_single_dim = idx_lower[dim_mask, dim]
                    idx_upper_single_dim = idx_upper[dim_mask, dim]

                    times_lower = (idx_lower_single_dim * self.time_delta)[:, None]
                    times_upper = (idx_upper_single_dim * self.time_delta)[:, None]
                    denom = times_upper - times_lower

                    time_dim = valid_times[dim_mask] - self._phase_shifts[dim]

                    lower_numerator = times_upper - time_dim
                    upper_numerator = time_dim - times_lower

                    lower_signal_ratio = lower_numerator / denom
                    upper_signal_ratio = upper_numerator / denom

                    data_lower = self._data[idx_lower_single_dim, dim][:, None]
                    data_upper = self._data[idx_upper_single_dim, dim][:, None]

                    interpolated[dim_mask, dim] = (
                        lower_signal_ratio * data_lower
                        + upper_signal_ratio * data_upper
                    )

                # Combine all masks
                combined_mask = overflow_mask.any(axis=0)
                valid = valid[~combined_mask]
                interpolated = np.squeeze(interpolated)

            else:

                idx_upper = idx_upper[compute_mask]
                idx_lower = idx_lower[compute_mask]

                times_lower = idx_lower * self.time_delta
                times_upper = idx_upper * self.time_delta
                denom = times_upper - times_lower

                times_valid = valid_times[compute_mask]

                lower_signal_ratio = ((times_upper - times_valid) / denom)[:, None]
                upper_signal_ratio = ((times_valid - times_lower) / denom)[:, None]

                data_lower = self._data[idx_lower]
                data_upper = self._data[idx_upper]

                interpolated = np.full(
                    (valid_times.shape[0], data_lower.shape[1]), np.nan
                )
                interpolated[compute_mask] = (
                    lower_signal_ratio * data_lower + upper_signal_ratio * data_upper
                )

                valid = valid[~overflow_mask]

            if not self.keep_nans:
                neuron_means = np.nanmean(interpolated, axis=0)
                # Replace NaNs with the column means directly
                np.copyto(interpolated, neuron_means, where=np.isnan(interpolated))

            return interpolated, valid

        else:
            raise NotImplementedError(
                f"interpolation_mode should be linear or nearest_neighbor"
            )


class ScreenInterpolator(Interpolator):
    def __init__(
        self,
        root_folder: str,
        cache_data: bool = False,  # New parameter
        rescale: bool = False,
        rescale_size: typing.Optional[tuple(int, int)] = None,
        normalize: bool = False,
        **kwargs,
    ) -> None:
        """
        rescale would rescale images to the _image_size if true
        cache_data: if True, loads and keeps all trial data in memory
        """
        super().__init__(root_folder)
        self.timestamps = np.load(self.root_folder / "timestamps.npy")
        self.start_time = self.timestamps[0]
        self.end_time = self.timestamps[-1]
        self.valid_interval = TimeInterval(self.start_time, self.end_time)
        self.rescale = rescale
        self.cache_trials = cache_data  # Store the cache preference
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

        # Initialize an empty dictionary to store all contents
        all_data = {}

        # Get meta files and sort by number
        meta_files = [
            f
            for f in (self.root_folder / "meta").iterdir()
            if f.is_file() and is_numbered_yml(f.name)
        ]
        meta_files.sort(key=lambda f: int(os.path.splitext(f.name)[0]))

        # Read each YAML file and store under its filename
        for meta_file in meta_files:
            with open(meta_file, "r") as file:
                file_base_name = meta_file.stem
                yaml_content = yaml.safe_load(file)
                all_data[file_base_name] = yaml_content

        output_path = self.root_folder / "combined_meta.json"
        with open(output_path, "w") as file:
            json.dump(all_data, file)

    def read_combined_meta(self) -> None:
        if not (self.root_folder / "combined_meta.json").exists():
            print("Combining metadatas...")
            self._combine_metadatas()

        with open(self.root_folder / "combined_meta.json", "r") as file:
            self.combined_meta = json.load(file)

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
            # Pass the cache_trials parameter when creating trials
            self.trials.append(
                ScreenTrial.create(
                    data_file_name, metadata, cache_data=self.cache_trials
                )
            )

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
        out = np.zeros([len(valid_times)] + list(self._image_size), dtype=np.float32)
        for u_idx in unique_file_idx:
            data = self.trials[u_idx].get_data()
            # TODO: establish convention of dimensons for time/channels. Then we can remove this
            # TODO: revisit this for on the fly decoding
            if ((len(data.shape) == 2) or (data.shape[-1] == 3)) and (
                len(data.shape) < 4
            ):

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
        cache_data: bool = False,
    ) -> None:
        self.data_file_name = data_file_name
        self._meta_data = meta_data
        self.modality = meta_data.get("modality")
        self.image_size = image_size
        self.first_frame_idx = first_frame_idx
        self.num_frames = num_frames
        self._cached_data = None
        self._cache_data = cache_data
        if self._cache_data:
            self._cached_data = self.get_data_()

    @staticmethod
    def create(
        data_file_name: str, meta_data: dict, cache_data: bool = False
    ) -> "ScreenTrial":
        modality = meta_data.get("modality")
        class_name = modality.lower().capitalize() + "Trial"
        assert class_name in globals(), f"Unknown modality: {modality}"
        return globals()[class_name](data_file_name, meta_data, cache_data=cache_data)

    def get_data_(self) -> np.array:
        """Base implementation for loading/generating data"""
        return np.load(self.data_file_name)

    def get_data(self) -> np.array:
        """Wrapper that handles caching"""
        if self._cached_data is not None:
            return self._cached_data
        return self.get_data_()

    def get_meta(self, property: str):
        return self._meta_data.get(property)


class ImageTrial(ScreenTrial):
    def __init__(self, data_file_name, meta_data, cache_data: bool = False) -> None:
        super().__init__(
            data_file_name,
            meta_data,
            tuple(meta_data.get("image_size")),
            meta_data.get("first_frame_idx"),
            1,
            cache_data=cache_data,
        )


class VideoTrial(ScreenTrial):
    def __init__(self, data_file_name, meta_data, cache_data: bool = False) -> None:
        super().__init__(
            data_file_name,
            meta_data,
            tuple(meta_data.get("image_size")),
            meta_data.get("first_frame_idx"),
            meta_data.get("num_frames"),
            cache_data=cache_data,
        )


class BlankTrial(ScreenTrial):
    def __init__(self, data_file_name, meta_data, cache_data: bool = False) -> None:

        self.interleave_value = meta_data.get("interleave_value")

        super().__init__(
            data_file_name,
            meta_data,
            tuple(meta_data.get("image_size")),
            meta_data.get("first_frame_idx"),
            1,
            cache_data=cache_data,
        )

    def get_data_(self) -> np.array:
        """Override base implementation to generate blank data"""
        return np.full((1,) + self.image_size, self.interleave_value, dtype=np.float32)


class InvalidTrial(ScreenTrial):
    def __init__(self, data_file_name, meta_data, cache_data: bool = False) -> None:

        self.interleave_value = meta_data.get("interleave_value")

        super().__init__(
            data_file_name,
            meta_data,
            tuple(meta_data.get("image_size")),
            meta_data.get("first_frame_idx"),
            1,
            cache_data=cache_data,
        )

    def get_data_(self) -> np.array:
        """Override base implementation to generate blank data"""
        return np.full((1,) + self.image_size, self.interleave_value, dtype=np.float32)
