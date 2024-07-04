from pathlib import Path
import numpy as np
from abc import abstractmethod
import yaml


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

        idx = np.round((valid_times[:, np.newaxis] - self._phase_shifts[np.newaxis, :] - self.start_time) / self.time_delta).astype(int)
        return np.take_along_axis(self._data, idx, axis=0), valid
