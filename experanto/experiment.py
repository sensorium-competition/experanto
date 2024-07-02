import numpy as np
from pathlib import Path
from collections import namedtuple
from collections.abc import Sequence, abstractmethod

Interval = namedtuple("Interval", ["start", "end"])


class Interpolator:

    def __init__(self, root_folder: str, sampling_rate: float, phase_shift: float = 0.0) -> None:
        self.root_folder = Path(root_folder)
        self.sampling_rate = sampling_rate
        self._phase_shift = phase_shift
        
        # initialization: load from file
        self.start_time = ...
        self.end_time = ...
        # self._sample_times = np.arange(self.start_time + phase_shift, self.end_time, 1.0 / self.sampling_rate) 
        self._modality = ... 
    
    @abstractmethod
    def interpolate(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...
        # returns interpolated signal and boolean mask of valid samples
    
    # @property
    # def sample_times(self) -> np.ndarray:
    #     return self._sample_times
        
    # def __getitem__(self, idx):
    #     return self.interpolate(self._sample_times[idx])

    # def __len__(self) -> int:
    #     return self._sample_times.size


class Experiment(Sequence):

    def __init__(self, root_folder: str, sampling_rate: float) -> None:
        self.root_folder = Path(root_folder)
        self.sampling_rate = sampling_rate

        self.start_time = ...
        self.end_time = ...
        self._sample_times = np.arange(self.start_time, self.end_time, 1.0 / self.sampling_rate) 

        self._blocks = defaultdict(list)
        self._devices = []
        self._load_blocks()

    def _load_blocks(self) -> None:
        # for block_folder in self.root_folder.iterdir():
        #     if block_folder.is_dir():
        #         self._blocks.append(DataBlock(block_folder, self.sampling_rate))
        
        # blocks need to be sorted by start time
        for b in blocks:
            self._blocks[b._modality].append(b)

    def __getitem__(self, idx) -> dict:
        # idx or slice
        return_value = {}
        
        assert(isinstance(idx, int) or (isinstance(idx, slice) and (idx.step is None or idx.step == 1)))
 
        for m in self._devices:
            # find block(s) that contain the data for the current modality
            return_value[m] = ... # create empty array
            for b in self._blocks[m]:
                if b.contains_sample_times(self._sample_times[idx]):
                    # check if you nedd values from the next block as well
                    values, valid = b.interpolate(self._sample_times[idx])
                    return_value[m][valid] = values

                    if valid[-1]: break # if the last time point is valid, we can stop here
        return return_value

    def __len__(self) -> int:
        ...


# class DataBlock(Block):

#     def __init__(self, root_folder: str, samplingrate: float, start_time=None) -> None: 
#         super().__init__(root_folder, samplingrate)
#         # load start and end times of stimuli
#         self.start_times = np.load(self.root_folder / "start_times.npy")
#         self.end_times = np.load(self.root_folder / "end_times.npy")

#         # extract time interval of the block
#         self.interval = Interval(self.start_times.min(), self.end_times.max())
#         self.reset_time_index(start_time or self.interval.start)

#     def reset_time_index(self, start_time) -> None:
#         # create time index
#         dt = 1.0 / self.samplingrate
#         self._time_idx = np.arange(start_time, self.interval.end + dt, dt)


    """
    experiment_config = {
        'screen1': {
            'images' : {
                'class': ImageInterpolator,
                'args': {...}   
            },
            'video' : {
                'class': VideoInterpolator,
                'args': {...}   
            }
        }   
        '2photon': {
            'rates': {
                'class': SequenceInterpolator,
                'args': {...}   
            },
        }   
    }
    """
