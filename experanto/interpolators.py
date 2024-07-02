
from pathlib import Path
import numpy as np
from abc import abstractmethod
import numpy.lib.format as fmt


class Interpolator:

    def __init__(self, root_folder: str) -> None:
        self.root_folder = Path(root_folder)
        self.timestamps = np.load(self.root_folder / "timestamps.npy")
    
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
        # create mapping from image index to file index
        self._data_files = list(Path(root_folder / "data").rglob("*.npy"))
        self._num_images = []
        self._image_file_idx = np.zeros([0])
        for i, f in enumerate(self._data_files):
            shape = get_npy_shape(f)
            assert len(shape) == 3, "Images must be 3D arrays"
            self._image_size = shape[:2]
            self._num_images.append(shape[2])
            self._image_file_idx.append(np.ones(shape[2]) * i)
        self._first_image_idx = np.cumsum([0] + self._num_images)

        # read timesteps
        self._timesteps = np.load(root_folder / "timesteps.npy")
        assert np.all(np.diff(self._timesteps) > 0), "Timesteps must be sorted"

    def interpolate(self, times: np.ndarray) -> tuple:
        assert np.all(np.diff(times) > 0), "Times must be sorted"
        idx = np.searchsorted(self._timesteps, times) - 1 # convert times to image indices
        assert(np.all(idx >= 0) and np.all(idx < len(self._timesteps)), "Times out of bounds")
        
        # Go through files, load them and extract all images
        unique_files = np.unique(self._image_file_idx[idx])
        imgs = np.array([len(times)] + self._image_size)
        for u_idx in unique_files:
            images = np.load(self._data_files[u_idx])
            idx_for_this_file = np.where(self._image_file_idx[idx] == u_idx)
            imgs[idx_for_this_file] = np.transpose(
                images[idx - self._first_image_idx[u_idx]], [2, 0, 1])

        return imgs, np.ones(len(times), dtype=bool)

