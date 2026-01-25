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


class Interpolator:
    """Abstract base class for time-series interpolation.

    Interpolators load data from a modality folder and provide methods to
    query values at arbitrary time points. Each modality (screen, responses,
    eye_tracker, treadmill) has a specialized interpolator subclass.

    Parameters
    ----------
    root_folder : str
        Path to the modality directory containing data and metadata files.

    Attributes
    ----------
    root_folder : pathlib.Path
        Path to the modality directory.
    start_time : float
        Earliest timestamp in the data.
    end_time : float
        Latest timestamp in the data.
    valid_interval : TimeInterval
        Time range for which interpolation is valid.

    See Also
    --------
    SequenceInterpolator : For 1D time-series data (responses, behaviors).
    ScreenInterpolator : For visual stimuli (images, videos).
    Experiment : High-level interface that manages multiple interpolators.
    """

    def __init__(self, root_folder: str) -> None:
        self.root_folder = Path(root_folder)
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
        """Interpolate data at the specified time points.

        Parameters
        ----------
        times : numpy.ndarray
            1D array of time points (in seconds) at which to interpolate.

        Returns
        -------
        data : numpy.ndarray
            Interpolated values. Shape depends on the modality.
        valid : numpy.ndarray
            Boolean mask indicating which input times were within the
            valid interval.
        """
        ...

    def __contains__(self, times: np.ndarray):
        return np.any(self.valid_times(times))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def create(root_folder: str, cache_data: bool = False, **kwargs) -> "Interpolator":
        """Factory method to create the appropriate interpolator for a modality.

        Reads the ``meta.yml`` file in the folder to determine the modality type
        and instantiates the corresponding interpolator subclass.

        Parameters
        ----------
        root_folder : str
            Path to the modality directory.
        cache_data : bool, default=False
            If True, loads all data into memory for faster access.
        **kwargs
            Additional arguments passed to the interpolator constructor.

        Returns
        -------
        Interpolator
            An instance of the appropriate interpolator subclass.

        Raises
        ------
        ValueError
            If the modality type is not supported.
        """
        with open(Path(root_folder) / "meta.yml", "r") as file:
            meta_data = yaml.load(file, Loader=yaml.SafeLoader)
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
        else:
            raise ValueError(
                f"There is no interpolator for {modality}. Please use 'sequence', 'screen', 'time_interval' as modality or provide a custom interpolator."
            )

    def valid_times(self, times: np.ndarray) -> np.ndarray:
        return self.valid_interval.intersect(times)

    def close(self):
        ...
        # generally, nothing to do
        # can be overwritten to close any open files or resources


class SequenceInterpolator(Interpolator):
    """Interpolator for 1D time-series data (neural responses, behaviors).

    Handles regularly-sampled time-series stored as memory-mapped or NumPy
    arrays. Supports nearest-neighbor and linear interpolation modes.

    Parameters
    ----------
    root_folder : str
        Path to the modality directory containing ``data.mem`` or ``data.npy``.
    cache_data : bool, default=False
        If True, loads memory-mapped data into RAM for faster access.
    keep_nans : bool, default=False
        If False, replaces NaN values with column means during interpolation.
    interpolation_mode : str, default='nearest_neighbor'
        Interpolation method: ``'nearest_neighbor'`` or ``'linear'``.
    normalize : bool, default=False
        If True, normalizes data using stored mean/std statistics.
    normalize_subtract_mean : bool, default=False
        If True, subtracts mean during normalization.
    normalize_std_threshold : float, optional
        Minimum std threshold to prevent division by near-zero values.
    **kwargs
        Additional keyword arguments (ignored).

    Attributes
    ----------
    sampling_rate : float
        Original sampling rate of the data in Hz.
    time_delta : float
        Time between samples (1 / sampling_rate).
    n_signals : int
        Number of signals (e.g., neurons, behavior channels).

    Notes
    -----
    For linear interpolation, values are computed as:

    .. math::

        y(t) = y_0 \\cdot \\frac{t_1 - t}{t_1 - t_0} + y_1 \\cdot \\frac{t - t_0}{t_1 - t_0}

    where :math:`t_0` and :math:`t_1` are the surrounding sample times.
    """

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

    def interpolate(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = self.valid_times(times)
        valid_times = times[valid]

        if len(valid_times) == 0:
            warnings.warn(
                "Sequence interpolation returns empty array, no valid times queried"
            )
            return np.empty((0, self._data.shape[1])), valid

        idx_lower = np.floor((valid_times - self.start_time) / self.time_delta).astype(
            int
        )

        if self.interpolation_mode == "nearest_neighbor":
            data = self._data[idx_lower]

            return data, valid

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

            return interpolated, valid

        else:
            raise NotImplementedError(
                f"interpolation_mode should be linear or nearest_neighbor"
            )

    def close(self) -> None:
        super().close()
        del self._data


class PhaseShiftedSequenceInterpolator(SequenceInterpolator):
    """Sequence interpolator with per-signal phase shifts.

    Extends :class:`SequenceInterpolator` to handle signals recorded with
    different phase offsets (e.g., neurons with different response latencies).
    Each signal is interpolated at its own phase-shifted time.

    Parameters
    ----------
    root_folder : str
        Path to the modality directory. Must contain ``meta/phase_shifts.npy``.
    **kwargs
        All parameters from :class:`SequenceInterpolator`.

    Attributes
    ----------
    _phase_shifts : numpy.ndarray
        Per-signal phase shift values in seconds.
    """

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

    def interpolate(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = self.valid_times(times)
        valid_times = times[valid]

        if len(valid_times) == 0:
            warnings.warn(
                "Sequence interpolation returns empty array, no valid times queried"
            )
            return np.empty((0, self._data.shape[1])), valid

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
            return data, valid

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

            return interpolated, valid

        else:
            raise NotImplementedError(
                f"interpolation_mode should be linear or nearest_neighbor"
            )


class ScreenInterpolator(Interpolator):
    """Interpolator for visual stimuli (images and videos).

    Handles frame-based visual data organized as trials. Each trial can be
    a single image, a video sequence, or a blank screen. Frames are indexed
    by timestamp and retrieved on demand.

    Parameters
    ----------
    root_folder : str
        Path to the screen modality directory containing ``timestamps.npy``,
        ``data/`` folder with trial files, and ``meta/`` folder with metadata.
    cache_data : bool, default=False
        If True, loads all trial data into memory for faster access.
    rescale : bool, default=False
        If True, rescales frames to ``rescale_size``.
    rescale_size : tuple of int, optional
        Target size ``(height, width)`` for rescaling. If None, uses the
        native image size from metadata.
    normalize : bool, default=False
        If True, normalizes frames using stored mean/std statistics.
    **kwargs
        Additional keyword arguments (ignored).

    Attributes
    ----------
    timestamps : numpy.ndarray
        Array of frame timestamps.
    trials : list of ScreenTrial
        List of trial objects containing frame data.

    See Also
    --------
    ImageTrial : Single-frame stimuli.
    VideoTrial : Multi-frame video stimuli.
    BlankTrial : Blank/gray screen stimuli.
    """

    def __init__(
        self,
        root_folder: str,
        cache_data: bool = False,  # New parameter
        rescale: bool = False,
        rescale_size: typing.Optional[tuple[int, int]] = None,
        normalize: bool = False,
        **kwargs,
    ) -> None:
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

    def interpolate(self, times: np.ndarray) -> np.ndarray:
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

        return out


class ScreenTrial:
    """Base class for visual stimulus trials.

    Represents a single trial (stimulus presentation) in a screen recording.
    Subclasses handle different trial types: images, videos, and blanks.

    Parameters
    ----------
    data_file_name : str
        Path to the data file for this trial.
    meta_data : dict
        Metadata dictionary for the trial.
    image_size : tuple
        Frame dimensions ``(height, width)`` or ``(height, width, channels)``.
    first_frame_idx : int
        Index of the first frame in the global timestamp array.
    num_frames : int
        Number of frames in this trial.
    cache_data : bool, default=False
        If True, loads and caches data on initialization.
    """

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
    """Trial containing a single static image."""

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
    """Trial containing a multi-frame video sequence."""

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
    """Trial containing a blank/gray screen (inter-stimulus interval)."""

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
    """Placeholder for invalid or corrupted trials."""

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
