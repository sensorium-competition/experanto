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
    
    def __enter__(self):
        return self
    
    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def create(root_folder: str, **kwargs) -> "Interpolator":
        with open(Path(root_folder) / "meta.yml", "r") as file:
            meta_data = yaml.load(file, Loader=yaml.SafeLoader)
        modality = meta_data.get("modality")
        class_name = modality.title().replace("_", "") + "Interpolator"
        assert class_name in globals(), f"Unknown modality: {modality}"
        return globals()[class_name](root_folder, **kwargs)

    def valid_times(self, times: np.ndarray) -> np.ndarray:
        return self.valid_interval.intersect(times)
    
    def close(self):
        ...
        # generally, nothing to do
        # can be overwritten to close any open files or resources


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

    #         if len(self._precision.shape) == 1:
    #             self._precision = self._precision.reshape(1, -1)

    def normalize_data(self, data):
        if self.normalize_subtract_mean:
            data = data - self.mean
        data = data * self._precision
        return data
            

    def interpolate(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        
        valid = self.valid_times(times)
        valid_times = times[valid]        
        idx_lower = np.floor((valid_times - self.start_time) / self.time_delta).astype(
            int
        )

        if self.interpolation_mode == "nearest_neighbor":
            data = self._data[idx_lower]
            return data, valid

        elif self.interpolation_mode == "linear":

            idx_upper = idx_lower + 1
            overflow_mask = (idx_upper >= self._data.shape[0]) | (idx_lower < 0)
            compute_mask = ~overflow_mask
                
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

            interpolated = np.full((valid_times.shape[0], data_lower.shape[1]), np.nan)
            interpolated[compute_mask] = lower_signal_ratio * data_lower + upper_signal_ratio * data_upper

            valid_indices = np.flatnonzero(valid)
            valid[valid_indices[overflow_mask]] = False
                
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
        super().__init__(
            root_folder,
            keep_nans=keep_nans,
            interpolation_mode=interpolation_mode,
            interp_window=interp_window,
            normalize=normalize,
            normalize_subtract_mean=normalize_subtract_mean,
            normalize_std_threshold=normalize_std_threshold,
            **kwargs
        )
        
        self._phase_shifts = np.load(self.root_folder / "meta/phase_shifts.npy")
        self.valid_interval = TimeInterval(
            self.start_time + np.max(self._phase_shifts),
            self.end_time + np.min(self._phase_shifts),
        )

    def interpolate(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = self.valid_times(times)
        valid_times = times[valid]
        
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
            overflow_mask = (idx_upper >= self._data.shape[0]) | (idx_lower < 0)
            compute_mask = ~overflow_mask

            valid_times = valid_times[:, None]
            interpolated = np.full((valid_times.shape[0], idx_lower.shape[1], 1), np.nan)

            for dim in range (idx_upper.shape[1]):

                dim_mask = compute_mask[:,dim]

                idx_lower_single_dim = idx_lower[dim_mask, dim]
                idx_upper_single_dim = idx_upper[dim_mask, dim]

                times_lower = (idx_lower_single_dim * self.time_delta)[:, None]
                times_upper = (idx_upper_single_dim * self.time_delta)[:, None]
                denom = times_upper - times_lower

                time_dim = valid_times[dim_mask] - self._phase_shifts[dim]

                lower_numerator = times_upper - time_dim
                upper_numerator = time_dim - times_lower
                
                lower_signal_ratio = (lower_numerator / denom)
                upper_signal_ratio = (upper_numerator / denom)

                data_lower = self._data[idx_lower_single_dim, dim][:, None]
                data_upper = self._data[idx_upper_single_dim, dim][:, None]


                interpolated[dim_mask, dim] = lower_signal_ratio * data_lower + upper_signal_ratio * data_upper

            valid_indices = np.flatnonzero(valid)
            for mask in overflow_mask.T:
                valid[valid_indices[mask]] = False

            interpolated = np.squeeze(interpolated)
                
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

    def _parse_trials(self) -> None:
        # Function to check if a file is a numbered yml file
        def is_numbered_yml(file_name):
            return re.fullmatch(r"\d{5}\.yml", file_name) is not None

        # Get block subfolders and sort by number
        meta_files = [
            f
            for f in (self.root_folder / "meta").iterdir()
            if f.is_file() and is_numbered_yml(f.name)
        ]
        meta_files.sort(key=lambda f: int(os.path.splitext(f.name)[0]))

        self.trials = []
        for f in meta_files:
            self.trials.append(ScreenTrial.create(f))

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
            if len(data.shape) == 2:
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
        file_name: str,
        meta_data: dict,
        image_size: tuple,
        first_frame_idx: int,
        num_frames: int,
    ) -> None:
        f = Path(file_name)
        self.file_name = f
        self.data_file_name = f.parent.parent / "data" / (f.stem + ".npy")
        self._meta_data = meta_data
        self.modality = meta_data.get("modality")
        self.image_size = image_size
        self.first_frame_idx = first_frame_idx
        self.num_frames = num_frames

    @staticmethod
    def create(file_name: str) -> "ScreenTrial":
        with open(file_name, "r") as file:
            meta_data = yaml.load(file, Loader=yaml.SafeLoader)
        modality = meta_data.get("modality")
        class_name = modality.capitalize() + "Trial"
        assert class_name in globals(), f"Unknown modality: {modality}"
        return globals()[class_name](file_name, meta_data)

    def get_data(self) -> np.array:
        return np.load(self.data_file_name)

    def get_meta(self, property: str):
        return self._meta_data.get(property)


class ImageTrial(ScreenTrial):
    def __init__(self, file_name, data) -> None:
        super().__init__(
            file_name,
            data,
            tuple(data.get("image_size")),
            data.get("first_frame_idx"),
            1,
        )


class VideoTrial(ScreenTrial):
    def __init__(self, file_name, data) -> None:
        super().__init__(
            file_name,
            data,
            tuple(data.get("image_size")),
            data.get("first_frame_idx"),
            data.get("num_frames"),
        )


class BlankTrial(ScreenTrial):
    def __init__(self, file_name, data) -> None:
        super().__init__(
            file_name,
            data,
            tuple(data.get("image_size")),
            data.get("first_frame_idx"),
            1,
        )
        self.interleave_value = data.get("interleave_value")

    def get_data(self) -> np.array:
        return np.full((1,) + self.image_size, self.interleave_value)
