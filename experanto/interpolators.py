from pathlib import Path
import numpy as np
from abc import abstractmethod
import yaml
import numpy.lib.format as fmt
import os
import yaml
import warnings


class TimeInterval:
    def __init__(self, start, end) -> None:
        self.start = start
        self.end = end

    def __contains__(self, time):
        return self.start <= time < self.end

    def intersect(self, times):
        return (times >= self.start) & (times < self.end)
        
        
    def __repr__(self) -> str:
        return f"TimeInterval [{self.start}, {self.end})"


class Interpolator:

    def __init__(self, root_folder: str) -> None:
        self.root_folder = Path(root_folder)
        # self.timestamps = np.load(self.root_folder / "timestamps.npy") # Alex: Move to ImageInterpolator
        meta = self.load_meta()
        self.start_time = meta["start_time"]
        self.end_time = meta["end_time"]
        # Valid interval can be different to start time and end time. 
        self.valid_interval = TimeInterval(self.start_time, self.end_time)

    def load_meta(self):
        with open(self.root_folder / "meta.yml") as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)
        return meta

    @abstractmethod
    def interpolate(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...
        # returns interpolated signal and boolean mask of valid samples

    def __contains__(self, times: np.ndarray):
        return np.any(self.valid_times(times))

    @staticmethod
    def create(root_folder: str) -> "Interpolator":
        with open(Path(root_folder) / 'meta.yml', 'r') as file:
            meta_data = yaml.safe_load(file)
        modality = meta_data.get('modality')
        class_name = modality.capitalize() + "Interpolator"
        assert class_name in globals(), f"Unknown modality: {modality}"
        return globals()[class_name](root_folder)

    def __contains__(self, times: np.ndarray):
        return np.any((times >= self.timestamps[0]) & (times <= self.timestamps[-1]))
    
    def valid_times(self, times: np.ndarray) -> np.ndarray:
        return self.valid_interval.intersect(times)



class SequenceInterpolator(Interpolator):

    def __init__(self, root_folder: str) -> None:
        super().__init__(root_folder)
        meta = self.load_meta()
        self.time_delta = 1./meta["sampling_rate"]

        self.use_phase_shifts = meta["phase_shift_per_signal"]
        if meta["phase_shift_per_signal"]:
            self._phase_shifts = np.load(self.root_folder / "meta/phase_shifts.npy")
            self.valid_interval = TimeInterval(
                self.start_time + np.max(self._phase_shifts),
                self.end_time + np.min(self._phase_shifts),
            )
        self._data = np.load(self.root_folder / "data.npy")

    def interpolate(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = self.valid_times(times)
        valid_times = times[valid]

        if self.use_phase_shifts:
            idx = np.round((valid_times[:, np.newaxis] - self._phase_shifts[np.newaxis, :] - self.start_time) / self.time_delta).astype(int)
            data = np.take_along_axis(self._data, idx, axis=0)
        else:
            idx = np.round((valid_times - self.start_time) / self.time_delta).astype(int)
            data = self._data[idx]
            
        return data, valid


def get_npy_shape(file_path):
    with open(file_path, 'rb') as f:
        version = fmt.read_magic(f)
        fmt._check_version(version)
        shape, fortran_order, dtype = fmt._read_array_header(f, version)
    return shape


class ImageInterpolator(Interpolator):

    def __init__(self, root_folder: str) -> None:
        super().__init__(root_folder)
        self.timestamps = np.load(self.root_folder / "timestamps.npy")

        # create mapping from image index to file index
        self._image_files = list(Path(os.path.join(root_folder, "data")).rglob("*.npy"))
        shape = get_npy_shape(self._image_files[0])
        self._image_size = shape[1:]

    def interpolate(self, times: np.ndarray) -> tuple:
        valid = self.valid_times(times)
        valid_times = times[valid]

        assert np.all(np.diff(valid_times) > 0), "Times must be sorted"
        idx = np.searchsorted(self.timestamps, valid_times) - 1 # convert times to image indices
        
        # Go through files, load them and extract all images
        unique_img_idx = np.unique(idx)
        imgs = np.zeros([len(valid_times)] + list(self._image_size))
        for u_idx in unique_img_idx:
            image = np.load(self._image_files[u_idx])
            idx_for_this_img = np.where(idx == u_idx)
            imgs[idx_for_this_img] = np.repeat(image, len(idx_for_this_img), axis=0)

        # display warning that output is cropped to (36,64)
        warnings.warn("Image output is cropped to (36,64)", UserWarning)
        return imgs[:,:36,:64], valid


class VideoInterpolator(Interpolator):

    def __init__(self, root_folder: str) -> None:
        super().__init__(root_folder)
        self.timestamps = np.load(self.root_folder / "timestamps.npy")

        # create mapping from image index to file index
        self._video_files = list(Path(os.path.join(root_folder, "data")).rglob("*.npy"))
        shape = get_npy_shape(self._video_files[0])
        self._video_size = shape[1:]
        self._num_frames = []
        self._video_file_idx = np.zeros([0], dtype=int)
        for i, f in enumerate(self._video_files):
            shape = get_npy_shape(f)
            assert len(shape) == 3, "Videos must be 3D arrays"
            assert self._video_size == shape[1:], "All videos must have the same size"
            self._num_frames.append(shape[0])
            self._video_file_idx = np.append(self._video_file_idx, np.full(shape[0], i), axis=0)
        self._first_frame_idx = np.cumsum([0] + self._num_frames)


    def interpolate(self, times: np.ndarray) -> tuple:
        valid = self.valid_times(times)
        valid_times = times[valid]

        assert np.all(np.diff(valid_times) > 0), "Times must be sorted"
        idx = np.searchsorted(self.timestamps, valid_times) - 1 # convert times to frame indices
        video_idx = self._video_file_idx[idx]
        
        # Go through files, load them and extract all frames
        unique_vid_idx = np.unique(video_idx)
        vids = np.zeros([len(valid_times)] + list(self._video_size))
        for u_idx in unique_vid_idx:
            video = np.load(self._video_files[u_idx])
            idx_for_this_vid = np.where(self._video_file_idx[idx] == u_idx)
            vids[idx_for_this_vid] = video[idx[idx_for_this_vid] - self._first_frame_idx[u_idx]]

        return vids, valid
    
