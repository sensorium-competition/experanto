
from pathlib import Path
import numpy as np
from abc import abstractmethod
import numpy.lib.format as fmt
import os


class Interpolator:

    def __init__(self, root_folder: str) -> None:
        self.root_folder = Path(root_folder)
        self.timestamps = np.load(self.root_folder / "timesteps.npy")
    
    @abstractmethod
    def interpolate(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...
        # returns interpolated signal and boolean mask of valid samples
    
    def __contains__(self, times: np.ndarray):
        return np.any((times >= self.timestamps[0]) & times <= self.timestamps[-1])
    
    def valid_times(self, times: np.ndarray) -> np.ndarray:
        return (times >= self.timestamps[0]) & (times <= self.timestamps[-1])


class SequenceInterpolator(Interpolator):
    
        def __init__(self, root_folder: str) -> None:
            super().__init__(root_folder)
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


def get_npy_shape(file_path):
    with open(file_path, 'rb') as f:
        version = fmt.read_magic(f)
        fmt._check_version(version)
        shape, fortran_order, dtype = fmt._read_array_header(f, version)
    return shape


class ImageInterpolator(Interpolator):

    def __init__(self, root_folder: str) -> None:
        super().__init__(root_folder)

        # create mapping from image index to file index
        self._image_files = list(Path(os.path.join(root_folder, "data")).rglob("*.npy"))
        shape = get_npy_shape(self._image_files[0])
        self._image_size = shape[1:]

    def interpolate(self, times: np.ndarray) -> tuple:
        assert np.all(np.diff(times) > 0), "Times must be sorted"
        idx = np.searchsorted(self.timestamps, times) - 1 # convert times to image indices
        print(idx)
        
        # Go through files, load them and extract all images
        unique_img_idx = np.unique(idx)
        imgs = np.zeros([len(times)] + list(self._image_size))
        for u_idx in unique_img_idx:
            print(u_idx)
            image = np.load(self._image_files[u_idx])
            idx_for_this_img = np.where(idx == u_idx)
            imgs[idx_for_this_img] = np.repeat(image, len(idx_for_this_img), axis=0)

        return imgs, np.ones(len(times), dtype=bool)

