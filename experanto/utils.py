from typing import List, Union, Dict
import random
import torch
from itertools import cycle
from omegaconf import DictConfig
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any


def replace_nan_with_batch_mean(data: np.array) -> np.array:
    row, col = np.where(np.isnan(data))
    for i, j in zip(row, col):
        new_value = np.nanmean(data[:, j])
        data[i, j] = new_value if not np.isnan(new_value) else 0

    return data


def add_behavior_as_channels(data: dict[str, torch.Tensor]) -> dict:
    """
    Adds the behavior as additional channels to the datapoint of the datasets __getitem__.
    data = {
        'screen': torch.Tensor: (n_timepoints, channels, height, width)
        'eye_tracker': torch.Tensor: (n_timepoints, channels)
        'treadmill': torch.Tensor: (n_timepoints, channels)
    }
    The function apends the ey tracker and treadmill behavioral varialbes as entire channels to the screen data.
    data = {
        'screen': torch.Tensor,  # shape: (n_timepoints, channels+behavior_channels ,...)
        ...
    }
    """
    screen = data["screen"]
    h, w = screen.shape[-2:]
    eye_tracker = data["eye_tracker"]
    treadmill = data["treadmill"]

    # Add eye tracker and treadmill as channels
    eye_tracker = eye_tracker[..., None, None].repeat(1, 1, h, w)
    treadmill = treadmill[..., None, None,].repeat(1, 1, h, w)
    screen = torch.cat([screen, eye_tracker, treadmill], dim=1)

    data["screen"] = screen
    return data



def linear_interpolate_1d_sequence(row, times_old, times_new, keep_nans=False):
    """
    Interpolates columns in a NumPy array and replaces NaNs with interpolated values

    Args:
        array: The input NumPy array [Neurons, times]
        times: old time points [Neurons, times] or [times]
        times_new:  new time points [times2]
        keep_nans:  if we want to keep and return nans after interpolation

    Returns:
        The interpolated array with NaNs replaced (inplace).
    """
    if keep_nans:
        interpolated_array = np.interp(times_new, times_old, row)
    else:
        # Find indices of non-NaN values
        valid_indices = np.where(~np.isnan(row))[0]
        valid_times = times_old[valid_indices]
        # Interpolate the column using linear interpolation
        interpolated_array = np.interp(times_new, valid_times, row[valid_indices])
    return interpolated_array


def linear_interpolate_sequences(array, times, times_new, keep_nans=False):
    """
    Interpolates columns in a NumPy array and replaces NaNs with interpolated values

    Args:
        array: The input NumPy array [times, ch]
        times: old time points  [times]
        times_new:  new time points [times2]
        keep_nans:  if we want to keep and return nans after interpolation

    Returns:
        The interpolated array with NaNs replaced.
    """
    array = array.T
    if array.shape[0] == 1:
        return linear_interpolate_1d_sequence(
            array.T.flatten(), times, times_new, keep_nans=keep_nans
        )
    interpolated_array = np.full((array.shape[0], len(times_new)), np.nan)
    for row_idx, row in enumerate(array):
        interpolated_array[row_idx] = linear_interpolate_1d_sequence(
            row, times, times_new, keep_nans=keep_nans
        )
    return interpolated_array.T


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """ solves bug to keep all workers initialized across epochs.
    From https://discuss.pytorch.org/t/enumerate-dataloader-slow/87778
    and
    https://github.com/huggingface/pytorch-image-models/blob/d72ac0db259275233877be8c1d4872163954dfbb/timm/data/loader.py#L209-L238
    """
    def __init__(self, *args, shuffle_each_epoch=True, **kwargs, ):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()
        self.shuffle_each_epoch = shuffle_each_epoch

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        if self.shuffle_each_epoch and hasattr(self.dataset, "shuffle_valid_screen_times"):
            self.dataset.shuffle_valid_screen_times()
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


# borrowed with <3 from
# https://github.com/sinzlab/neuralpredictors/blob/main/neuralpredictors/training/cyclers.py
def cycle(iterable):
    # see https://github.com/pytorch/pytorch/issues/23900
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class Exhauster:
    """
    Given a dictionary of data loaders, mapping data_key into a data loader, steps through each data loader, moving onto the next data loader
    only upon exhausing the content of the current data loader.
    """

    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        for data_key, loader in self.loaders.items():
            for batch in loader:
                yield data_key, batch

    def __len__(self):
        return sum([len(loader) for loader in self.loaders])


class LongCycler:
    """
    Cycles through trainloaders until the loader with largest size is exhausted.
        Needed for dataloaders of unequal size (as in the monkey data).
    """

    def __init__(self, loaders):
        self.loaders = loaders
        self.max_batches = max([len(loader) for loader in self.loaders.values()])

    def __iter__(self):
        cycles = [cycle(loader) for loader in self.loaders.values()]
        for k, loader, _ in zip(
            cycle(self.loaders.keys()),
            (cycle(cycles)),
            range(len(self.loaders) * self.max_batches),
        ):
            yield k, next(loader)

    def __len__(self):
        return len(self.loaders) * self.max_batches


class ShuffledLongCycler:
    """
    Cycles through trainloaders until the loader with largest size is exhausted.
    Needed for dataloaders of unequal size (as in the monkey data).
    Can randomize the order of keys with a fixed random state for reproducibility.
    """

    def __init__(self, loaders, shuffle_keys=False, random_seed=None):
        self.loaders = loaders
        self.max_batches = max([len(loader) for loader in self.loaders.values()])
        self.shuffle_keys = shuffle_keys
        self.random_seed = random_seed
        self.rng = random.Random(random_seed) if random_seed is not None else random.Random()
        # Store initial state for reset capability
        self._initial_state = self.rng.getstate() if shuffle_keys else None

    def reset_rng(self):
        """Reset the random number generator to its initial state"""
        if self._initial_state is not None:
            self.rng.setstate(self._initial_state)

    def get_rng_state(self):
        """Get the current RNG state for checkpointing"""
        return self.rng.getstate() if self.shuffle_keys else None

    def set_rng_state(self, state):
        """Set the RNG state from a checkpoint"""
        if self.shuffle_keys and state is not None:
            self.rng.setstate(state)

    def __iter__(self):
        cycles = [cycle(loader) for loader in self.loaders.values()]
        cycles_dict = dict(zip(self.loaders.keys(), cycles))

        # Calculate total number of batches to yield
        total_batches = len(self.loaders) * self.max_batches
        batches_yielded = 0

        while batches_yielded < total_batches:
            # Get keys in either randomized or fixed order
            keys = list(self.loaders.keys())
            if self.shuffle_keys:
                # Use our seeded RNG for shuffling
                self.rng.shuffle(keys)

            # Yield each key and its corresponding batch
            for k in keys:
                if batches_yielded < total_batches:
                    yield k, next(cycles_dict[k])
                    batches_yielded += 1
                else:
                    break

    def __len__(self):
        return len(self.loaders) * self.max_batches


class ShortCycler:
    """
    Cycles through trainloaders until the loader with smallest size is exhausted.
        Needed for dataloaders of unequal size (as in the monkey data).
    """

    def __init__(self, loaders):
        self.loaders = loaders
        self.min_batches = min([len(loader) for loader in self.loaders.values()])

    def __iter__(self):
        cycles = [cycle(loader) for loader in self.loaders.values()]
        for k, loader, _ in zip(
            cycle(self.loaders.keys()),
            (cycle(cycles)),
            range(len(self.loaders) * self.min_batches),
        ):
            yield k, next(loader)

    def __len__(self):
        return len(self.loaders) * self.min_batches


class OrderedSubsetSampler(torch.utils.data.Sampler):
    """Samples elements sequentially from a given list of indices."""
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class StatefulDataLoader(DataLoader):
    """
    A DataLoader that maintains state for fault-tolerant training.
    Inherits from torch.utils.data.DataLoader for full compatibility.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0,
                 pin_memory=False, drop_last=False, seed: Optional[int] = 0,
                 **kwargs):
        # Initialize with our custom OrderedSubsetSampler
        self.indices = np.arange(len(dataset))
        self.base_sampler = OrderedSubsetSampler(self.indices)
        batch_sampler = torch.utils.data.BatchSampler(self.base_sampler, batch_size, drop_last)
        
        super().__init__(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs
        )
        
        # Rest of the initialization remains the same
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()
        
        self.shuffle = shuffle
        self.seed = seed
        self._rng = np.random.RandomState(seed) 
        self._prev_rng_state = None
        self.current_batch = 0
        self._init_indices()
            
    def _init_indices(self) -> None:
        """Initialize or reset the indices for iteration."""
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            self._prev_rng_state = self._rng.get_state()
            self._rng.shuffle(self.indices)

        # Cyclically shift indices by current_batch
        batch_size = self.batch_sampler.sampler.batch_size
        shift = self.current_batch * batch_size
        self.indices = np.roll(self.indices, -shift)

        # Update the sampler's indices
        self.base_sampler.indices = self.indices.tolist()

        # Reset iterator with new indices order
        self.iterator = super().__iter__()
            
    def get_state(self) -> Dict[str, Any]:
        """Return the current state of the dataloader."""
        return {
            'batch': self.current_batch,
            'prev_rng_state': self._prev_rng_state
        }
        
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the dataloader state."""

        self.current_batch = state['batch']
        self._rng.set_state(state['prev_rng_state'])
        

        self._init_indices()

    def __iter__(self):
        
        for _ in range(len(self) - self.current_batch):
            self.current_batch += 1
            yield next(self.iterator)
        
        # Reset batch counter when done
        self.current_batch = 0
        
        self._init_indices()

    def __len__(self):
        return len(self.batch_sampler.sampler)

class StatefulLongCycler:
    """
    Cycles through trainloaders until the loader with largest size is exhausted.
    Maintains state for fault-tolerant training.
    """

    def __init__(self, loaders, seed: Optional[int] = 0):
        self.loaders = loaders
        self.loader_cyclers = {key: cycle(loader) for key, loader in self.loaders.items()}
        self.max_batches = max([len(loader) for loader in self.loaders.values()])
        self._rng = np.random.RandomState(seed)
        self.cycle_order = None

        # State tracking
        self.current_cycle = 0
        self.current_cycle_position = 0
        self._prev_rng_state = None

        self._generate_cycle_order()


    def _generate_cycle_order(self):
        """Generate the random order of loaders for current cycle."""
        keys = sorted(list(self.loaders.keys()))
        self._prev_rng_state = self._rng.get_state()
        self._rng.shuffle(keys)
        self.cycle_order = keys
        
    def get_state(self) -> Dict[str, Any]:
        """Return the current state of the cycler and its dataloaders."""
        return {
            'current_cycle': self.current_cycle,
            'current_cycle_position': self.current_cycle_position,
            'prev_rng_state': self._prev_rng_state,
            'dataloader_states': {
                key: loader.get_state() 
                for key, loader in self.loaders.items()
            }
        }
        
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the cycler state and its dataloaders."""
        self.current_cycle = state['current_cycle']
        self.current_cycle_position = state['current_cycle_position']
        self._rng.set_state(state['prev_rng_state'])
        
        for key, loader_state in state['dataloader_states'].items():
            self.loaders[key].set_state(loader_state)

        self._generate_cycle_order()
            
    def __iter__(self):

        while self.current_cycle < self.max_batches:
            
            # Continue from saved position in current cycle
            while self.current_cycle_position < len(self.cycle_order):
                key = self.cycle_order[self.current_cycle_position]
                loader_cycler = self.loader_cyclers[key]
                self.current_cycle_position += 1
                yield key, next(loader_cycler)
            
            # Move to next cycle
            self.current_cycle += 1
            self.current_cycle_position = 0

            print(f"Cycle {self.current_cycle} completed")

            self._generate_cycle_order()
            
        # Reset state when done
        self.current_cycle = 0
        self.current_cycle_position = 0

    def __len__(self):
        return len(self.loaders) * self.max_batches