from typing import List, Union, Dict
import random
import torch
from itertools import cycle
from omegaconf import DictConfig
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List
import math
from .intervals import TimeInterval
import torch
from torch.utils.data import DataLoader, Sampler
import numpy as np
from typing import Dict, Any, Optional, List, Iterator
from itertools import cycle


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


class _RepeatSampler(object):
    """Sampler that repeats forever."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class StatefulDataLoader(torch.utils.data.DataLoader):
    """
    A highly optimized DataLoader that maintains state for fault-tolerant training.
    Based on MultiEpochsDataLoader but with state tracking capabilities.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0,
                 pin_memory=False, drop_last=False, seed: Optional[int] = 0,
                 **kwargs):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        )

        # Setup repeating sampler - key to efficiency
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

        # State tracking - lightweight
        self.seed = seed
        self._rng = np.random.RandomState(seed) if seed is not None else None
        self.current_batch = 0
        self.shuffle = shuffle

    def get_state(self) -> Dict[str, Any]:
        """Return the current state of the dataloader."""
        return {
            'batch': self.current_batch,
            'rng_state': self._rng.get_state() if self._rng is not None else None
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the dataloader state."""
        self.current_batch = state['batch']
        if self._rng is not None and state['rng_state'] is not None:
            self._rng.set_state(state['rng_state'])

        # Skip to correct position in the iterator
        # This is faster than recreating the iterator and is key to efficiency
        for _ in range(self.current_batch % len(self)):
            next(self.iterator)

    def __iter__(self):
        for i in range(len(self)):
            batch = next(self.iterator)
            self.current_batch += 1
            yield batch

    def __len__(self):
        return len(self.batch_sampler.sampler)


def cycle(iterable):
    """Efficient cycling through an iterable."""
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class StatefulLongCycler:
    """
    Cycles through trainloaders until the loader with largest size is exhausted.
    Maintains state for fault-tolerant training while keeping original efficiency.
    """

    def __init__(self, loaders, seed: Optional[int] = 0):
        self.loaders = loaders
        self.max_batches = max([len(loader) for loader in self.loaders.values()])
        self.keys = list(self.loaders.keys())

        # Lightweight state tracking
        self._rng = np.random.RandomState(seed) if seed is not None else None
        self.current_cycle = 0
        self.current_position = 0

        # Create efficient cyclers
        self._setup_cyclers()

    def _setup_cyclers(self):
        """Setup the cycling iterators efficiently."""
        self.loader_cyclers = {key: cycle(loader) for key, loader in self.loaders.items()}

        # Prepare shuffled keys if using RNG
        if self._rng is not None:
            self.keys = list(self.loaders.keys())
            self._rng.shuffle(self.keys)

        self.key_cycler = cycle(self.keys)

    def get_state(self) -> Dict[str, Any]:
        """Return the current state of the cycler and its dataloaders."""
        return {
            'current_cycle': self.current_cycle,
            'current_position': self.current_position,
            'rng_state': self._rng.get_state() if self._rng is not None else None,
            'dataloader_states': {
                key: loader.get_state() if hasattr(loader, 'get_state') else None
                for key, loader in self.loaders.items()
            }
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the cycler state and its dataloaders."""
        self.current_cycle = state['current_cycle']
        self.current_position = state['current_position']

        if self._rng is not None and state['rng_state'] is not None:
            self._rng.set_state(state['rng_state'])

        # Restore dataloader states
        for key, loader_state in state['dataloader_states'].items():
            if hasattr(self.loaders[key], 'set_state') and loader_state is not None:
                self.loaders[key].set_state(loader_state)

        # Re-setup cyclers (retaining efficiency of original implementation)
        self._setup_cyclers()

        # Advance to the current position
        position = (self.current_cycle * len(self.keys)) + self.current_position

        # Skip efficiently to the right place in the iteration
        for _ in range(position % (len(self.loaders) * self.max_batches)):
            next(self.key_cycler)
            next(self.loader_cyclers[next(self.key_cycler)])

    def __iter__(self):
        # Track total batches processed across all loaders
        total_processed = 0
        max_total = len(self.loaders) * self.max_batches

        # Using the highly efficient cyclers
        while total_processed < max_total:
            key = next(self.key_cycler)
            batch = next(self.loader_cyclers[key])

            # Update state (minimal overhead)
            total_processed += 1
            self.current_position = (self.current_position + 1) % len(self.keys)
            if self.current_position == 0:
                self.current_cycle += 1
                if self._rng is not None:
                    # Regenerate key order efficiently at cycle boundaries
                    self._rng.shuffle(self.keys)
                    self.key_cycler = cycle(self.keys)

            yield key, batch

        # Reset state when done
        self.current_cycle = 0
        self.current_position = 0

    def __len__(self):
        return len(self.loaders) * self.max_batches