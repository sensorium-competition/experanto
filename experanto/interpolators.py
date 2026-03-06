from __future__ import annotations

import json
import os
import re
import typing
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Union, cast

import cv2
import numpy as np
import numpy.lib.format as fmt
import yaml
from numba import njit, prange
from scipy.ndimage import gaussian_filter1d

from .intervals import TimeInterval


class Interpolator:
    def __init__(self, root_folder: str) -> None:
        self.root_folder = Path(root_folder)
        self.start_time = None
        self.end_time = None
        # Valid interval can be different to start time and end time.
        self.valid_interval = None

    def load_meta(self):
        with open(self.root_folder / "meta.yml") as f:
            meta = yaml.safe_load(f)
        return meta

    @abstractmethod
    def interpolate(
        self, times: np.ndarray, return_valid: bool = False
    ) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
        ...
        # returns interpolated signal and boolean mask of valid samples

    def __contains__(self, times: np.ndarray):
        return np.any(self.valid_times(times))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def create(root_folder: str, cache_data: bool = False, **kwargs) -> "Interpolator":
        with open(Path(root_folder) / "meta.yml", "r") as file:
            meta_data = yaml.safe_load(file)
        modality = meta_data.get("modality")

        if modality == "sequence":
            if meta_data.get("phase_shift_per_signal", False):
                return PhaseShiftedSequenceInterpolator(
                    root_folder, cache_data, **kwargs
                )
            else:
                return SequenceInterpolator(root_folder, cache_data, **kwargs)
        elif modality == "screen":
            return ScreenInterpolator(root_folder, cache_data, **kwargs)
        elif modality == "time_interval":
            return TimeIntervalInterpolator(root_folder, cache_data, **kwargs)
        elif modality == "spikes":
            return SpikesInterpolator(root_folder, cache_data, **kwargs)
        else:
            raise ValueError(
                f"There is no interpolator for {modality}. Please use 'sequence', 'screen', 'time_interval' as modality or provide a custom interpolator."
            )

    def valid_times(self, times: np.ndarray) -> np.ndarray:
        assert self.valid_interval is not None
        return self.valid_interval.intersect(times)

    def close(self):
        ...
        # generally, nothing to do
        # can be overwritten to close any open files or resources


class SequenceInterpolator(Interpolator):
    def __init__(
        self,
        root_folder: str,
        cache_data: bool = False,  # already cached, put it here for consistency
        keep_nans: bool = False,
        interpolation_mode: str = "nearest_neighbor",
        normalize: bool = False,
        normalize_subtract_mean: bool = False,
        normalize_std_threshold: typing.Optional[float] = None,  # or 0.01
        **kwargs,
    ) -> None:
        """
        interpolation_mode - nearest neighbor or linear
        keep_nans - if we keep nans in linear interpolation
        """
        super().__init__(root_folder)
        meta = self.load_meta()
        self.keep_nans = keep_nans
        self.interpolation_mode = interpolation_mode
        self.normalize = normalize
        self.normalize_subtract_mean = normalize_subtract_mean
        self.normalize_std_threshold = normalize_std_threshold
        self.sampling_rate = meta["sampling_rate"]
        self.time_delta = 1.0 / self.sampling_rate
        self.start_time = meta["start_time"]
        self.end_time = meta["end_time"]
        self.is_mem_mapped = meta["is_mem_mapped"] if "is_mem_mapped" in meta else False
        # Valid interval can be different to start time and end time.
        self.valid_interval = TimeInterval(self.start_time, self.end_time)

        self.n_signals = meta["n_signals"]
        # read .mem (memmap) or .npy file
        if self.is_mem_mapped:
            self._data = np.memmap(
                self.root_folder / "data.mem",
                dtype=meta["dtype"],
                mode="r",
                shape=(meta["n_timestamps"], meta["n_signals"]),
            )

            if cache_data:
                self._data = np.array(self._data).astype(
                    np.float32
                )  # Convert memmap to ndarray
        else:
            self._data = np.load(self.root_folder / "data.npy")

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

    def interpolate(
        self, times: np.ndarray, return_valid: bool = False
    ) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
        valid = self.valid_times(times)
        valid_times = times[valid]

        if len(valid_times) == 0:
            warnings.warn(
                "Sequence interpolation returns empty array, no valid times queried"
            )
            return (
                (np.empty((0, self._data.shape[1])), valid)
                if return_valid
                else np.empty((0, self._data.shape[1]))
            )

        idx_lower = np.floor((valid_times - self.start_time) / self.time_delta).astype(
            int
        )

        if self.interpolation_mode == "nearest_neighbor":
            data = self._data[idx_lower]

            return (data, valid) if return_valid else data

        elif self.interpolation_mode == "linear":
            idx_upper = idx_lower + 1
            overflow_mask = idx_upper >= self._data.shape[0]

            if np.any(idx_lower < 0):  # should not be possible
                warnings.warn(
                    f"Interpolation index {idx_lower} is negative. This should not happen."
                )
                overflow_mask = overflow_mask | idx_lower < 0

            valid = valid[~overflow_mask]

            idx_upper = idx_upper[~overflow_mask]
            idx_lower = idx_lower[~overflow_mask]

            times_lower = idx_lower * self.time_delta
            times_upper = idx_upper * self.time_delta
            denom = times_upper - times_lower

            times_valid = valid_times[~overflow_mask]

            lower_signal_ratio = ((times_upper - times_valid) / denom)[:, None]
            upper_signal_ratio = ((times_valid - times_lower) / denom)[:, None]

            data_lower = self._data[idx_lower]
            data_upper = self._data[idx_upper]

            interpolated = (
                lower_signal_ratio * data_lower + upper_signal_ratio * data_upper
            )

            if not self.keep_nans:
                neuron_means = np.nanmean(interpolated, axis=0)
                # Replace NaNs with the column means directly
                np.copyto(interpolated, neuron_means, where=np.isnan(interpolated))

            return (interpolated, valid) if return_valid else interpolated

        else:
            raise NotImplementedError(
                f"interpolation_mode should be linear or nearest_neighbor"
            )

    def close(self) -> None:
        super().close()
        del self._data


class PhaseShiftedSequenceInterpolator(SequenceInterpolator):
    def __init__(
        self,
        root_folder: str,
        cache_data: bool = False,  # already cached, put it here for consistency
        keep_nans: bool = False,
        interpolation_mode: str = "nearest_neighbor",
        normalize: bool = False,
        normalize_subtract_mean: bool = False,
        normalize_std_threshold: typing.Optional[float] = None,  # or 0.01
        **kwargs,
    ) -> None:
        super().__init__(
            root_folder,
            cache_data=cache_data,
            keep_nans=keep_nans,
            interpolation_mode=interpolation_mode,
            normalize=normalize,
            normalize_subtract_mean=normalize_subtract_mean,
            normalize_std_threshold=normalize_std_threshold,
            **kwargs,
        )

        self._phase_shifts = np.load(self.root_folder / "meta/phase_shifts.npy")
        self.valid_interval = TimeInterval(
            self.start_time
            + (np.max(self._phase_shifts) if len(self._phase_shifts) > 0 else 0),
            self.end_time
            + (np.min(self._phase_shifts) if len(self._phase_shifts) > 0 else 0),
        )

    def interpolate(
        self, times: np.ndarray, return_valid: bool = False
    ) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
        valid = self.valid_times(times)
        valid_times = times[valid]

        if len(valid_times) == 0:
            warnings.warn(
                "Sequence interpolation returns empty array, no valid times queried"
            )
            return (
                (np.empty((0, self._data.shape[1])), valid)
                if return_valid
                else np.empty((0, self._data.shape[1]))
            )

        idx_lower = np.floor(
            (
                valid_times[:, np.newaxis]
                - self._phase_shifts[np.newaxis, :]
                - self.start_time
            )
            / self.time_delta
        ).astype(int)

        if self.interpolation_mode == "nearest_neighbor":
            data = np.take_along_axis(self._data, idx_lower, axis=0)
            return (data, valid) if return_valid else data

        elif self.interpolation_mode == "linear":
            idx_upper = idx_lower + 1
            overflow_mask = idx_upper >= self._data.shape[0]

            if np.any(idx_lower < 0):  # should not be possible
                warnings.warn(
                    f"Interpolation index {idx_lower} is negative. This should not happen."
                )
                overflow_mask = overflow_mask | idx_lower < 0

            valid = valid[~overflow_mask.any(axis=1)]

            times_lower = idx_lower * self.time_delta
            times_upper = idx_upper * self.time_delta
            denom = times_upper - times_lower

            time_dim = valid_times[:, np.newaxis] - self._phase_shifts[np.newaxis, :]

            lower_numerator = times_upper - time_dim
            upper_numerator = time_dim - times_lower

            lower_signal_ratio = lower_numerator / denom
            upper_signal_ratio = upper_numerator / denom

            _, cols = np.indices(idx_lower.shape)
            data_lower = self._data[idx_lower, cols]
            data_upper = self._data[idx_upper, cols]

            interpolated = (
                lower_signal_ratio * data_lower + upper_signal_ratio * data_upper
            )

            if not self.keep_nans:
                neuron_means = np.nanmean(interpolated, axis=0)
                # Replace NaNs with the column means directly
                np.copyto(interpolated, neuron_means, where=np.isnan(interpolated))

            return (interpolated, valid) if return_valid else interpolated

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
        rescale_size: typing.Optional[tuple[int, int]] = None,
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

    def read_combined_meta(self) -> tuple[list, list]:
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

    def interpolate(
        self, times: np.ndarray, return_valid: bool = False
    ) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
        valid = self.valid_times(times)
        valid_times = times[valid]
        valid_times += 1e-4  # add small offset to avoid numerical issues

        assert np.all(np.diff(valid_times) > 0), "Times must be sorted"
        idx = cast(
            np.ndarray, np.searchsorted(self.timestamps, valid_times) - 1
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
        return (out, valid) if return_valid else out

    def rescale_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Changes the resolution of the image to this size.
        Returns: Rescaled image
        """
        return cv2.resize(frame, self._image_size, interpolation=cv2.INTER_AREA).astype(
            np.float32
        )


class TimeIntervalInterpolator(Interpolator):
    def __init__(self, root_folder: str, cache_data: bool = False, **kwargs):
        super().__init__(root_folder)
        self.cache_data = cache_data

        meta = self.load_meta()
        self.meta_labels = meta["labels"]
        self.start_time = meta["start_time"]
        self.end_time = meta["end_time"]
        self.valid_interval = TimeInterval(self.start_time, self.end_time)

        if self.cache_data:
            self.labeled_intervals = {
                label: np.load(self.root_folder / filename)
                for label, filename in self.meta_labels.items()
            }

    def interpolate(
        self, times: np.ndarray, return_valid: bool = False
    ) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Interpolate time intervals for labeled events.

        Given a set of time points and a set of labeled intervals (defined in the
        `meta.yml` file), this method returns a boolean array indicating, for each
        time point, whether it falls within any interval for each label.

        The method uses half-open intervals [start, end), where a timestamp t is
        considered to fall within an interval if start <= t < end. This means the
        start time is inclusive and the end time is exclusive.

        Parameters
        ----------
        times : np.ndarray
            Array of time points to be checked against the labeled intervals.

        Returns
        -------
        out : np.ndarray of bool, shape (len(valid_times), n_labels)
            Boolean array where each row corresponds to a valid time point and each
            column corresponds to a label. `out[i, j]` is True if the i-th valid
            time falls within any interval for the j-th label, and False otherwise.

        Notes
        -----
        - The labels and their corresponding intervals are defined in the `meta.yml`
          file under the `labels` key. Each label points to a `.npy` file containing
          an array of shape (n, 2), where each row is a [start, end) time interval.
        - Typical labels might include 'train', 'validation', 'test', 'saccade',
          'gaze', or 'target'.
        - Only time points within the valid interval (as defined by start_time and
          end_time in meta.yml) are considered; others are filtered out.
        - Intervals where start > end are considered invalid and will trigger a
          warning.
        """
        valid = self.valid_times(times)
        valid_times = times[valid]

        n_labels = len(self.meta_labels)
        n_times = len(valid_times)

        if n_times == 0:
            warnings.warn(
                "TimeIntervalInterpolator returns an empty array, no valid times queried."
            )
            return np.empty((0, n_labels), dtype=bool)

        out = np.zeros((n_times, n_labels), dtype=bool)
        for i, (label, filename) in enumerate(self.meta_labels.items()):
            if self.cache_data:
                intervals = self.labeled_intervals[label]
            else:
                intervals = np.load(self.root_folder / filename, allow_pickle=True)

            if len(intervals) == 0:
                warnings.warn(
                    f"TimeIntervalInterpolator found no intervals for label: {label}"
                )
                continue

            for start, end in intervals:
                if start > end:
                    warnings.warn(
                        f"Invalid interval found for label: {label}, interval: ({start}, {end})"
                    )
                    continue
                # Half-open interval [start, end): inclusive start, exclusive end
                mask = (valid_times >= start) & (valid_times < end)
                out[mask, i] = True

        return (out, valid) if return_valid else out


class ScreenTrial:
    def __init__(
        self,
        data_file_name: Union[str, Path],
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
        data_file_name: Union[str, Path], meta_data: dict, cache_data: bool = False
    ) -> "ScreenTrial":
        modality = meta_data.get("modality")
        assert modality is not None
        class_name = modality.lower().capitalize() + "Trial"
        assert class_name in globals(), f"Unknown modality: {modality}"
        return globals()[class_name](data_file_name, meta_data, cache_data=cache_data)

    def get_data_(self) -> np.ndarray:
        """Base implementation for loading/generating data"""
        return np.load(self.data_file_name)

    def get_data(self) -> np.ndarray:
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

    def get_data_(self) -> np.ndarray:
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

    def get_data_(self) -> np.ndarray:
        """Override base implementation to generate blank data"""
        return np.full((1,) + self.image_size, self.interleave_value, dtype=np.float32)


#  This decorator works on a Python function and does not know how to handle self, so it cannot be a member of a class, here SpikeInterpolator.
# 'parallel=True' allows it to use all CPU cores.
@njit(parallel=True, fastmath=True)
def fast_count_spikes(all_spikes, indices, window_starts, window_ends, out_counts):
    """
    all_spikes: 1D array
    indices: 1D array - start/end of each neuron in all_spikes
    window_starts: 1D array - start times for the query
    window_ends: 1D array
    out_counts: 2D array
    """
    n_batch = len(window_starts)
    n_neurons = len(indices) - 1

    # We parallelize the OUTER loop (the batch).
    # Or we can parallelize the NEURON loop.
    # Since N_Neurons (38k) > Batch (e.g. 128), parallelizing neurons is better.

    for i in prange(n_neurons):
        # 1. Get the slice for this neuron
        # (This is zero-copy in Numba)
        idx_start = indices[i]
        idx_end = indices[i + 1]
        neuron_spikes = all_spikes[idx_start:idx_end]

        # 2. Check all time windows for this neuron
        # Since spikes are sorted, we use binary search
        for b in range(n_batch):
            t0 = window_starts[b]
            t1 = window_ends[b]

            # Binary Search
            # np.searchsorted is supported natively in Numba
            # It finds where t0 and t1 would fit in the sorted array
            c_start = np.searchsorted(neuron_spikes, t0)
            c_end = np.searchsorted(neuron_spikes, t1)

            out_counts[b, i] = c_end - c_start


class SpikesInterpolator(Interpolator):
    """
    Interpolator for spike train data.

    This interpolator reads raw spike times and computes spike counts within
    specified time windows around queried timestamps.

    Data Storage Format:
    --------------------
    The spike data must be stored in a flat 1D binary file named `spikes.npy`
    (dtype: float64) inside the `root_folder`.

    The array contains the actual continuous spike timings (e.g., in seconds).
    The timings must be **blocked by neuron**, and within each neuron's block,
    the spike times must be **sorted in ascending chronological order**.

    A `meta.yml` file in the same folder must provide a `spike_indices` list.
    This list defines the start and end indices for each neuron's block in
    the flat array. For example, if neuron 0 has 50 spikes and neuron 1 has 30
    spikes, `spike_indices` should be `[0, 50, 80]`.

    Parameters:
    -----------
    root_folder : str
        Path to the directory containing `spikes.npy` and `meta.yml`.
    cache_data : bool, optional
        If True, eagerly loads the entire spike array into RAM (`np.load`)
        for faster access. If False, memory-maps the data from disk (`np.memmap`).
        Default is False.
    interpolation_window : float, optional
        The size of the time window used to count spikes, in the same time units
        as the spike data. Default is 0.3.
    interpolation_align : str, optional
        Alignment of the interpolation window relative to the queried time `t`.
        - "center": window is [t - window/2, t + window/2)
        - "left": window is [t, t + window)
        - "right": window is [t - window, t)
        Default is "center".
    smoothing_sigma : float, optional
        Standard deviation for a Gaussian filter applied to the resulting
        spike counts along the time axis. The unit is in number of time steps
        (array indices), not physical time. Set to 0.0 to disable smoothing.
        Default is 0.0.
    """

    def __init__(
        self,
        root_folder: str,
        cache_data: bool = False,
        interpolation_window: float = 0.3,
        interpolation_align: str = "center",
        smoothing_sigma: float = 0.0,
    ):
        super().__init__(root_folder)

        meta = self.load_meta()

        self.start_time = meta["start_time"]
        self.end_time = meta["end_time"]
        self.valid_interval = TimeInterval(self.start_time, self.end_time)

        self.interpolation_window = interpolation_window
        self.interpolation_align = interpolation_align
        self.smoothing_sigma = smoothing_sigma
        self.cache_data = cache_data
        self.is_mem_mapped = meta.get("is_mem_mapped", False)  # read-only memmap

        # Use self.root_folder, defined in the base class
        filename = 'spikes.mem' if self.is_mem_mapped else 'spikes.npy'
        self.dat_path = self.root_folder / filename

        # Ensure indices are typed correctly for Numba
        self.indices = np.array(meta["spike_indices"]).astype(np.int64)
        computed_n_signals = len(self.indices) - 1
        meta_n_signals = meta.get("n_signals")
        if meta_n_signals is not None and meta_n_signals != computed_n_signals:
            raise ValueError(
                f"Mismatch between meta['n_signals'] ({meta_n_signals}) and "
                f"len(spike_indices) - 1 ({computed_n_signals})."
            )
        self.n_signals = (
            meta_n_signals if meta_n_signals is not None else computed_n_signals
        )

        # Check interpolation_align validity
        if self.interpolation_align not in ["center", "left", "right"]:
            raise ValueError(
                f"Unknown alignment mode: {self.interpolation_align}, should be 'center', 'left' or 'right'"
            )

        # The screen times for our experiment are stored in float64. So this should be the same dtype for consistency and to avoid issues with memmap.
        # Use the unified cache_data flag for eager loading
        if self.is_mem_mapped:
            self.spikes = np.memmap(
                self.dat_path,
                dtype=meta.get("dtype", "float64"),
                mode="r",
                shape=(self.indices[-1],),
            )
            if self.cache_data:
                self.spikes = np.array(self.spikes)
        else:
            self.spikes = np.load(self.dat_path)

    def interpolate(
        self, times: np.ndarray, return_valid: bool = False
    ) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
        # 1. Filter for valid times
        valid = self.valid_times(times)
        valid_times = times[valid]

        # Handle edge case where no times are valid
        if len(valid_times) == 0:
            return (
                (np.empty((0, self.n_signals)), valid)
                if return_valid
                else np.empty((0, self.n_signals))
            )

        # 2. Prepare boundaries
        if self.interpolation_align == "center":
            starts = valid_times - self.interpolation_window / 2
            ends = valid_times + self.interpolation_window / 2
        elif self.interpolation_align == "left":
            starts = valid_times
            ends = valid_times + self.interpolation_window
        elif self.interpolation_align == "right":
            starts = valid_times - self.interpolation_window
            ends = valid_times
        else:
            raise ValueError(
                f"Unknown alignment mode: {self.interpolation_align}, should be 'center', 'left' or 'right'"
            )

        # 3. Prepare Output
        # SIZE FIX: Only allocate for the VALID batch size
        # valid_size refers to the number of valid timestamps you are querying at once.
        valid_size = len(valid_times)
        counts = np.zeros((valid_size, self.n_signals), dtype=np.float64)

        # 4. Call Numba Engine
        fast_count_spikes(self.spikes, self.indices, starts, ends, counts)

        # 5. Apply Smoothing (Gaussian Filter)
        if self.smoothing_sigma > 0:
            # We assume 'times' is a sorted, equidistant sequence.
            # If valid_size is 1, smoothing is impossible/no-op.
            if valid_size > 1:
                # Apply Gaussian filter along the time axis (axis 0)
                # Note: sigma is in units of array indices (time steps).
                # If your times are 30Hz (33ms) and you want 100ms smoothing,
                # sigma should be ~3.
                counts = gaussian_filter1d(counts, sigma=self.smoothing_sigma, axis=0)

        # SIGNATURE FIX: Return both data and the mask
        return (counts, valid) if return_valid else counts

    def close(self):
        super().close()
        # Trigger cleanup of memmap
        if hasattr(self, "spikes") and isinstance(self.spikes, np.memmap):
            _mmap_obj = getattr(self.spikes, "_mmap", None)
            if _mmap_obj is not None:
                _mmap_obj.close()
            del self.spikes
