
from pathlib import Path
import numpy as np
from abc import abstractmethod
import numpy.lib.format as fmt
import os
import yaml
import warnings
import re


class Interpolator:

    def __init__(self, root_folder: str) -> None:
        self.root_folder = Path(root_folder)
        self.timestamps = np.load(self.root_folder / "timestamps.npy")
    
    @abstractmethod
    def interpolate(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...
        # returns interpolated signal and boolean mask of valid samples

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
        return (times >= self.timestamps[0]) & (times <= self.timestamps[-1])


class SequenceInterpolator(Interpolator):
    
        def __init__(self, root_folder: str) -> None:
            super().__init__(root_folder)
            # quick hack to use only one timestamp for all neurons -- will be fixed in the future
            if self.timestamps.ndim == 2:
                self.timestamps = self.timestamps[:,0]
            print(self.timestamps.shape)
            
            self._data = np.load(self.root_folder / "data.npy")


        def interpolate(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            valid = self.valid_times(times)
            valid_times = times[valid]
            
            # index to the right of the biggest timestamp smaller than valid_times
            right_index = np.searchsorted(self.timestamps, valid_times, side="left") 
            
            # check if the closest timestamp is to the left or right
            closer_to_left = (valid_times - self.timestamps[right_index - 1]) < (self.timestamps[right_index] - valid_times)
            
            # correct timestamp accordingly
            idx = right_index - closer_to_left
            return self._data[idx], valid


class ScreenInterpolator(Interpolator):

    def __init__(self, root_folder: str) -> None:
        super().__init__(root_folder)
        self.timestamps = np.load(self.root_folder / "timestamps.npy")
        self._parse_meta()

        # create mapping from image index to file index
        # print(Path(root_folder) / (m.file_name + ".npy"))
        self._data_files = [Path(root_folder) / "data" / (m.file_name + ".npy") for m in self.meta]
        self._num_frames = [m.num_frames for m in self.meta]
        self._first_frame_idx = [m.first_frame for m in self.meta]
        self._data_file_idx = np.concatenate([np.full(m.num_frames, i) for i, m in enumerate(self.meta)])

        self._image_size = self.meta[0].image_size
        assert np.all([m.image_size == self._image_size for m in self.meta]), 'All files must have the same image size'

    def _parse_meta(self) -> None:
        # Function to check if a file is a numbered yml file
        def is_numbered_yml(file_name):
            return re.fullmatch(r'\d{5}\.yml', file_name) is not None

        # Get block subfolders and sort by number
        meta_files = [f for f in (self.root_folder / "meta").iterdir() if f.is_file() and is_numbered_yml(f.name)]
        # meta_files = list(Path(os.path.join(self._root_folder, "meta")).rglob("*.yml"))
        meta_files.sort(key=lambda f: int(os.path.splitext(f.name)[0]))

        self.meta = []
        for f in meta_files:
            self.meta.append(ScreenMeta.create(f))

    def interpolate(self, times: np.ndarray) -> tuple:
        valid = self.valid_times(times)
        valid_times = times[valid]
        valid_times += 1e-6 # add small offset to avoid numerical issues

        assert np.all(np.diff(valid_times) > 0), "Times must be sorted"
        idx = np.searchsorted(self.timestamps, valid_times) - 1 # convert times to frame indices
        assert np.all((idx >= 0) & (idx < len(self.timestamps))), "All times must be within the valid range"
        data_file_idx = self._data_file_idx[idx]
        
        # Go through files, load them and extract all frames
        unique_file_idx = np.unique(data_file_idx)
        out = np.zeros([len(valid_times)] + list(self._image_size))
        for u_idx in unique_file_idx:
            data = np.load(self._data_files[u_idx])
            idx_for_this_file = np.where(self._data_file_idx[idx] == u_idx)
            out[idx_for_this_file] = data[idx[idx_for_this_file] - self._first_frame_idx[u_idx]]

        return out, valid
    

class ScreenMeta():
    def __init__(self, file_name: str, data: dict, image_size: tuple, first_frame: int, num_frames: int) -> None:
        self.file_name = file_name
        self._data = data
        self.modality = data.get('modality')
        self.image_size = image_size
        self.first_frame = first_frame
        self.num_frames = num_frames

    @staticmethod
    def create(file_name: str) -> "ScreenMeta":
        with open(file_name, 'r') as file:
            meta_data = yaml.safe_load(file)
        modality = meta_data.get('modality')
        class_name = modality.capitalize() + "Meta"
        assert class_name in globals(), f"Unknown modality: {modality}"
        return globals()[class_name](Path(file_name).stem, meta_data)
    

class ImageMeta(ScreenMeta):
    def __init__(self, file_name, data) -> None:
        super().__init__(file_name, data, tuple(data.get("image_size")), data.get("first_frame"), 2)


class VideoMeta(ScreenMeta):
    def __init__(self, file_name, data) -> None:
        super().__init__(file_name, data, tuple(data.get("image_size")), data.get("first_frame"), data.get("num_frames"))

 