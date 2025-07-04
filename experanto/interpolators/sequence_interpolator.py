from __future__ import annotations

import typing
import warnings

import numpy as np

from ..intervals import TimeInterval
from .base import Interpolator
from .registry import register_interpolator


@register_interpolator(
    lambda meta: meta.get("modality", None) == "sequence", priority=0
)
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
                "interpolation_mode should be linear or nearest_neighbor"
            )

    def close(self) -> None:
        super().close()
        del self._data


@register_interpolator(
    lambda meta: meta.get("modality", None) == "sequence"
    and meta.get("phase_shift_per_signal", False),
    priority=1,
)
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
                "interpolation_mode should be linear or nearest_neighbor"
            )
