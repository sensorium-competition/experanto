
from abc import abstractmethod
from pathlib import Path
import yaml
import numpy as np

from .registry import INTERPOLATOR_SELECTORS, ensure_default_interpolators_registered

class Interpolator:
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
        ...
        # returns interpolated signal and boolean mask of valid samples

    def __contains__(self, times: np.ndarray):
        return np.any(self.valid_times(times))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def create(root_folder: str, cache_data: bool = False, **kwargs) -> "Interpolator":
        with open(Path(root_folder) / "meta.yml", "r") as file:
            meta_data = yaml.load(file, Loader=yaml.SafeLoader)
        modality = meta_data.get("modality")
        
        ensure_default_interpolators_registered()

        sorted_selectors = sorted(INTERPOLATOR_SELECTORS, key=lambda x: -x[0])  # highest priority first

        for priority, selector_fn, cls in sorted_selectors:
            if selector_fn(meta_data):
                print(f"[INFO] Using {cls.__name__} (priority={priority})")
                return cls(root_folder, cache_data, **kwargs)

        raise ValueError(f"No interpolator found for metadata={meta_data}.")

    def valid_times(self, times: np.ndarray) -> np.ndarray:
        return self.valid_interval.intersect(times)

    def close(self):
        ...
        # generally, nothing to do
        # can be overwritten to close any open files or resources