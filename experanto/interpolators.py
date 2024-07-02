
from pathlib import Path
import numpy as np
from abc import abstractmethod

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