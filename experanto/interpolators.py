from __future__ import annotations

import json
import logging
import os
import re
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import yaml
from numba import njit, prange
from scipy.ndimage import gaussian_filter1d

from .intervals import TimeInterval

logger = logging.getLogger(__name__)


class Interpolator:
    """Abstract base class for time series interpolation.

    Interpolators load data from a modality folder and map time points to
    data values. Each modality (e.g., screen, responses, eye_tracker,
    treadmill) is assigned to a separate interpolator object belonging to
    one of the Interpolator subclasses (e.g., SequenceInterpolator,
    ScreenInterpolator, etc.), but multiple modalities can belong to the same
    class, such as treadmill and eye_tracker both being assigned to the
    SequenceInterpolator subclass.

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
    SequenceInterpolator : For time series data (responses, behaviors).
    ScreenInterpolator : For visual stimuli (images, videos).
    TimeIntervalInterpolator : For labeled time intervals (e.g., train/test splits).
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
            meta = yaml.safe_load(f)
        return meta

    @abstractmethod
    def interpolate(
        self, times: np.ndarray, return_valid: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Map an array of time points to interpolated data values."""
        ...

    def __contains__(self, times: np.ndarray):
        return np.any(self.valid_times(times))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def create(root_folder: str, cache_data: bool = False, **kwargs) -> Interpolator:
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
        with open(Path(root_folder) / "meta.yml") as file:
            meta_data = yaml.safe_load(file)
        modality = meta_data.get("modality")

        if modality == "sequence":
            if meta_data.get("phase_shift_per_signal", False):
                return PhaseShiftedSequenceInterpolator(
                    root_folder, cache_data=cache_data, **kwargs
                )
            else:
                return SequenceInterpolator(
                    root_folder, cache_data=cache_data, **kwargs
                )
        elif modality == "screen":
            use_stimuli_names = kwargs.pop(
                "use_stimuli_names", meta_data.get("use_stimuli_names", False)
            )
            return ScreenInterpolator(
                root_folder,
                cache_data=cache_data,
                use_stimuli_names=use_stimuli_names,
                **kwargs,
            )
        elif modality == "time_interval":
            return TimeIntervalInterpolator(
                root_folder, cache_data=cache_data, **kwargs
            )
        elif modality == "spikes":
            return SpikeInterpolator(root_folder, cache_data=cache_data, **kwargs)
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
    """Interpolator for time series data.

    Handles regularly-sampled time series stored as memory-mapped or NumPy
    arrays. Supports nearest-neighbor and linear interpolation modes.

    Parameters
    ----------
    root_folder : str
        Path to the modality directory containing ``data.mem`` or ``data.npy``.
    cache_data : bool, default=False
        If True, loads memory-mapped data into RAM for faster access.
    keep_nans : bool, default=False
        If False and ``interpolation_mode='linear'``, replaces NaN values with
        column means during interpolation. For ``'nearest_neighbor'``, NaNs are
        left unchanged.
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

        y(t) = y_0 \\cdot \\frac{t_1 - t}{t_1 - t_0} + y_1 \\cdot \\frac{t - t_0}{t_1 - t_0},

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
        normalize_std_threshold: float | None = None,  # or 0.01
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

    def interpolate(
        self, times: np.ndarray, return_valid: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        valid = self.valid_times(times)
        valid_times = times[valid]

        if len(valid_times) == 0:
            warnings.warn(
                "Interpolation returns empty array, no valid times queried.",
                UserWarning,
                stacklevel=2,
            )
            return (
                (np.empty((0, self._data.shape[1]), dtype=self._data.dtype), valid)
                if return_valid
                else np.empty((0, self._data.shape[1]), dtype=self._data.dtype)
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
                    f"Interpolation index {idx_lower} is negative. This should not happen.",
                    UserWarning,
                    stacklevel=2,
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
                "interpolation_mode should be linear or nearest_neighbor"
            )

    def close(self) -> None:
        super().close()

        if hasattr(self, "_data") and isinstance(self._data, np.memmap):
            mmap_obj = getattr(self._data, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()

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
        normalize_std_threshold: float | None = None,  # or 0.01
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
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        valid = self.valid_times(times)
        valid_times = times[valid]

        if len(valid_times) == 0:
            warnings.warn(
                "Interpolation returns empty array, no valid times queried.",
                UserWarning,
                stacklevel=2,
            )
            return (
                (np.empty((0, self._data.shape[1]), dtype=self._data.dtype), valid)
                if return_valid
                else np.empty((0, self._data.shape[1]), dtype=self._data.dtype)
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
                    f"Interpolation index {idx_lower} is negative. This should not happen.",
                    UserWarning,
                    stacklevel=2,
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
                "interpolation_mode should be linear or nearest_neighbor"
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
    use_stimuli_names : bool, default=False
        If True, uses ``stimulus_name`` from metadata to locate data files instead of trial keys.
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
        cache_data: bool = False,
        rescale: bool = False,
        rescale_size: tuple[int, int] | None = None,
        normalize: bool = False,
        use_stimuli_names: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(root_folder)
        self.timestamps = np.load(self.root_folder / "timestamps.npy")
        self.start_time = self.timestamps[0]
        self.end_time = self.timestamps[-1]
        self.valid_interval = TimeInterval(self.start_time, self.end_time)
        self.rescale = rescale
        self.cache_data = cache_data  # Store the cache preference
        self.use_stimuli_names = use_stimuli_names
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
            with open(meta_file) as file:
                file_base_name = meta_file.stem
                yaml_content = yaml.safe_load(file)
                all_data[file_base_name] = yaml_content

        output_path = self.root_folder / "combined_meta.json"
        with open(output_path, "w") as file:
            json.dump(all_data, file)

    def read_combined_meta(self) -> tuple[list, list]:
        if not (self.root_folder / "combined_meta.json").exists():
            logger.info("Combining metadata files...")
            self._combine_metadatas()

        with open(self.root_folder / "combined_meta.json") as file:
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

        for key, metadata in zip(keys, metadatas, strict=True):
            if self.use_stimuli_names:
                stimulus_name = metadata.get("stimulus_name")
                assert (
                    stimulus_name is not None
                ), f"stimulus_name is required in metadata when use_stimuli_names is True, but not found for key: {key}"
                data_file_name = self.root_folder / "data" / f"{stimulus_name}.npy"
            else:
                data_file_name = self.root_folder / "data" / f"{key}.npy"
            self.trials.append(
                ScreenTrial.create(data_file_name, metadata, cache_data=self.cache_data)
            )

    def interpolate(
        self, times: np.ndarray, return_valid: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
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
        """Rescale frame to the configured image size.

        Parameters
        ----------
        frame : np.ndarray
            Input image frame.

        Returns
        -------
        np.ndarray
            Rescaled image as float32.
        """
        return cv2.resize(frame, self._image_size, interpolation=cv2.INTER_AREA).astype(
            np.float32
        )

    def close(self) -> None:
        super().close()

        if hasattr(self, "trials"):
            for trial in self.trials:
                if hasattr(trial, "_cached_data") and isinstance(
                    trial._cached_data, np.memmap
                ):
                    mmap_obj = getattr(trial._cached_data, "_mmap", None)
                    if mmap_obj is not None:
                        mmap_obj.close()
                    del trial._cached_data
            del self.trials


class TimeIntervalInterpolator(Interpolator):
    """Interpolator for labeled time intervals.

    Maps time points to boolean membership in labeled intervals. Given a
    set of time points, returns a boolean array indicating whether each
    point falls within any interval for each label.

    Labels and their intervals are defined in the ``meta.yml`` file under
    the ``labels`` key. Each label points to a ``.npy`` file containing an
    array of shape ``(n, 2)``, where each row is a ``[start, end)``
    half-open time interval. Typical labels include ``'train'``,
    ``'validation'``, ``'test'``, ``'saccade'``, ``'gaze'``, or
    ``'target'``.

    The half-open convention means a timestamp *t* is considered inside an
    interval when ``start <= t < end``. Intervals where ``start > end``
    are treated as invalid and trigger a warning.

    Parameters
    ----------
    root_folder : str
        Path to the modality directory containing ``meta.yml`` and the
        ``.npy`` interval files referenced by its ``labels`` mapping.
    cache_data : bool, default=False
        If True, loads all interval arrays into memory at init time.
    **kwargs
        Additional keyword arguments (ignored).

    Attributes
    ----------
    meta_labels : dict
        Mapping from label names to ``.npy`` filenames.

    Notes
    -----
    - Only time points within the valid interval (as defined by
      ``start_time`` and ``end_time`` in ``meta.yml``) are considered;
      others are filtered out.
    - The ``interpolate`` method returns an array of shape
      ``(n_valid_times, n_labels)`` where ``out[i, j]`` is True if the
      *i*-th valid time falls within any interval for the *j*-th label.
    """

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
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        valid = self.valid_times(times)
        valid_times = times[valid]

        n_labels = len(self.meta_labels)
        n_times = len(valid_times)

        if n_times == 0:
            warnings.warn(
                "Interpolation returns empty array, no valid times queried.",
                UserWarning,
                stacklevel=2,
            )
            return (
                (np.empty((0, n_labels), dtype=bool), valid)
                if return_valid
                else np.empty((0, n_labels), dtype=bool)
            )

        out = np.zeros((n_times, n_labels), dtype=bool)
        for i, (label, filename) in enumerate(self.meta_labels.items()):
            if self.cache_data:
                intervals = self.labeled_intervals[label]
            else:
                intervals = np.load(self.root_folder / filename)

            if len(intervals) == 0:
                warnings.warn(
                    f"TimeIntervalInterpolator found no intervals for label: {label}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            for start, end in intervals:
                if start > end:
                    warnings.warn(
                        f"Invalid interval found for label: {label}, interval: ({start}, {end})",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                # Half-open interval [start, end): inclusive start, exclusive end
                mask = (valid_times >= start) & (valid_times < end)
                out[mask, i] = True

        return (out, valid) if return_valid else out


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
        data_file_name: str | Path,
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
        data_file_name: str | Path,
        meta_data: dict,
        cache_data: bool = False,
    ) -> ScreenTrial:
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

    def get_data_(self) -> np.ndarray:
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

    def get_data_(self) -> np.ndarray:
        """Override base implementation to generate blank data"""
        return np.full((1,) + self.image_size, self.interleave_value, dtype=np.float32)


#  Numba JIT decorator: compiles Python function to fast machine code at runtime (mainly for numerical loops).
#  This decorator does not know how to handle self, so it cannot be a member of a class, here SpikeInterpolator.
# 'parallel=True' allows it to use all CPU cores.
@njit(parallel=True, fastmath=True)
def _fast_count_spikes(all_spikes, indices, window_starts, window_ends, out_counts):
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


class SpikeInterpolator(Interpolator):
    """
    Interpolator for spike train data.

    This interpolator reads raw spike times and computes spike counts within
    specified time windows around queried timestamps.

    Data Storage Format:
    --------------------
    The spike data must be stored in a flat 1D binary file named `spikes.npy` or `spikes.mem`
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
        (array indices), not physical time.
        If your times are 30Hz (33ms) and you want 100ms smoothing,
        sigma should be ~3.
        Set to 0.0 to disable smoothing.
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
        filename = "spikes.mem" if self.is_mem_mapped else "spikes.npy"
        self.dat_path = self.root_folder / filename

        # Ensure indices are typed correctly for Numba
        self.indices = np.array(meta["spike_indices"]).astype(np.int64)
        self.n_signals = len(self.indices) - 1
        meta_n_signals = meta.get("n_signals")
        if meta_n_signals is not None and meta_n_signals != self.n_signals:
            raise ValueError(
                f"Mismatch between meta['n_signals'] ({meta_n_signals}) and "
                f"len(spike_indices) - 1 ({self.n_signals})."
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
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        # 1. Filter for valid times
        valid = self.valid_times(times)
        valid_times = times[valid]

        # Handle edge case where no times are valid
        if len(valid_times) == 0:
            warnings.warn(
                "Interpolation returns empty array, no valid times queried.",
                UserWarning,
                stacklevel=2,
            )
            return (
                (np.empty((0, self.n_signals), dtype=np.float64), valid)
                if return_valid
                else np.empty((0, self.n_signals), dtype=np.float64)
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
        # valid_size refers to the number of valid timestamps you are querying at once.
        valid_size = len(valid_times)
        counts = np.zeros((valid_size, self.n_signals), dtype=np.float64)

        # 4. Call Numba Engine
        _fast_count_spikes(self.spikes, self.indices, starts, ends, counts)

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

        return (counts, valid) if return_valid else counts

    def close(self):
        super().close()
        # Trigger cleanup of memmap
        if hasattr(self, "spikes") and isinstance(self.spikes, np.memmap):
            _mmap_obj = getattr(self.spikes, "_mmap", None)
            if _mmap_obj is not None:
                _mmap_obj.close()
            del self.spikes
