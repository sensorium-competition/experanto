from typing import Dict, Any, Optional, List, Iterator, Union, Tuple

# inbuilt libraries
import os
import random
import math
import time
import threading
import multiprocessing
import queue
import warnings
from itertools import cycle
from functools import partial
from copy import deepcopy
import bisect

# third-party libraries
import numpy as np
from omegaconf import DictConfig
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader, Sampler

# local libraries
from .intervals import TimeInterval



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
    """Simple sampler that repeats indefinitely."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class SessionConcatDataset(Dataset):
    """
    A custom concatenated dataset that keeps track of which session each sample belongs to
    and returns data items in their original format.
    """

    def __init__(self, datasets, session_names=None):
        """
        Initialize the concatenated dataset with session tracking.

        Args:
            datasets: List of datasets to concatenate
            session_names: Optional list of session names corresponding to datasets
        """
        if not datasets:
            raise ValueError("datasets is empty")

        # Store datasets
        self.datasets = list(datasets)

        # Track session names
        if session_names is None:
            session_names = [f"session_{i}" for i in range(len(datasets))]
        self.session_names = session_names

        # Compute cumulative lengths - this is crucial for indexing
        self.cumulative_sizes = []
        current_size = 0
        for dataset in self.datasets:
            current_size += len(dataset)
            self.cumulative_sizes.append(current_size)

        # Create mapping from index to session - using a more efficient approach
        # Instead of storing a dictionary for every index, we'll compute it on demand
        self.session_ranges = []
        start_idx = 0
        for i, dataset in enumerate(datasets):
            session_size = len(dataset)
            self.session_ranges.append((start_idx, start_idx + session_size, session_names[i]))
            start_idx += session_size

        # Create session indices dictionary (for faster lookup)
        self.session_indices = {}
        start_idx = 0
        for i, dataset in enumerate(datasets):
            session_name = session_names[i]
            session_size = len(dataset)
            self.session_indices[session_name] = (start_idx, start_idx + session_size)
            start_idx += session_size

    def __len__(self):
        """Return total length of the concatenated dataset."""
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        """Get item from the appropriate dataset."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        # Find which dataset the index belongs to
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # Get the data from the dataset
        data = self.datasets[dataset_idx][sample_idx]

        # Return the data as-is (no session key attached)
        return data

    def get_session_for_idx(self, idx):
        """Get the session name for a given index."""
        for start, end, name in self.session_ranges:
            if start <= idx < end:
                return name
        return None

    def get_indices_for_session(self, session_name):
        """Get all indices belonging to a given session."""
        if session_name in self.session_indices:
            start, end = self.session_indices[session_name]
            return list(range(start, end))
        return []

    def get_sessions_count(self):
        """Get number of sessions and sample counts per session."""
        return {name: end - start for start, end, name in self.session_ranges}


class FastSessionAwareBatchSampler(Sampler):
    """
    A more efficient batch sampler that ensures all samples in a batch come from the same session.
    """

    def __init__(self, dataset, batch_size, drop_last=False, shuffle=False, seed=None):
        """
        Initialize a fast session-aware batch sampler.

        Args:
            dataset: SessionConcatDataset instance
            batch_size: Number of samples per batch
            drop_last: Whether to drop the last incomplete batch
            shuffle: Whether to shuffle samples within each session
            seed: Random seed for shuffling
        """
        if not isinstance(dataset, SessionConcatDataset):
            raise ValueError("FastSessionAwareBatchSampler requires a SessionConcatDataset")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed) if seed is not None else None

        # Generate initial session order
        self.session_names = list(dataset.session_indices.keys())
        if self.rng is not None and shuffle:
            self.rng.shuffle(self.session_names)

        # For faster initialization, store session indices and batches lazily
        self._session_batches = {}
        self._cached_batches = None
        self._session_to_batch_indices = {}

    def _get_session_batches(self, session_name):
        """Get batches for a specific session (with caching)."""
        if session_name in self._session_batches:
            return self._session_batches[session_name]

        # Get indices for this session
        start, end = self.dataset.session_indices[session_name]
        session_indices = list(range(start, end))

        # Shuffle indices if needed
        if self.shuffle and self.rng is not None:
            indices_copy = session_indices.copy()
            self.rng.shuffle(indices_copy)
            session_indices = indices_copy

        # Create batches
        batches = []
        for i in range(0, len(session_indices), self.batch_size):
            if i + self.batch_size <= len(session_indices) or not self.drop_last:
                batch = session_indices[i:i + self.batch_size]
                batches.append(batch)

        # Cache the result
        self._session_batches[session_name] = batches
        return batches

    def _get_all_batches(self):
        """Get all batches lazily."""
        if self._cached_batches is not None:
            return self._cached_batches

        all_batches = []
        for i, session_name in enumerate(self.session_names):
            batches = self._get_session_batches(session_name)

            # Store the mapping from session to batch indices
            start_idx = len(all_batches)
            end_idx = start_idx + len(batches)
            self._session_to_batch_indices[session_name] = list(range(start_idx, end_idx))

            # Add batches with session info
            for batch in batches:
                all_batches.append((session_name, batch))

        self._cached_batches = all_batches
        return all_batches

    def __iter__(self):
        # Return only the batch indices, not the session names
        all_batches = self._get_all_batches()
        for _, batch in all_batches:
            yield batch

    def __len__(self):
        return len(self._get_all_batches())

    def get_session_for_batch(self, batch_idx):
        """Get the session name for a given batch index."""
        all_batches = self._get_all_batches()
        if 0 <= batch_idx < len(all_batches):
            return all_batches[batch_idx][0]
        return None


class SimpleStatefulDataLoader(DataLoader):
    """
    Fast stateful dataloader that provides both data and session keys.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0,
                 pin_memory=False, drop_last=False, seed=None, **kwargs):

        # Verify we have a SessionConcatDataset
        if not isinstance(dataset, SessionConcatDataset):
            raise ValueError("SimpleStatefulDataLoader requires a SessionConcatDataset")

        # Store parameters for state tracking
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        # Initialize RNG if seed provided
        self._rng = np.random.RandomState(seed) if seed is not None else None
        self._prev_rng_state = None

        # Create a faster session-aware batch sampler
        start_time = time.time()
        self.session_batch_sampler = FastSessionAwareBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=seed
        )
        print(f"Batch sampler creation took {time.time() - start_time:.2f} seconds")

        # Initialize parent class with our custom batch sampler
        super().__init__(
            dataset=dataset,
            batch_sampler=self.session_batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs
        )

        # Setup repeating sampler with minimal overhead
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True

        # Create single iterator that persists throughout loader lifetime
        self.iterator = super().__iter__()

        # State tracking
        self.current_batch = 0
        self.total_length = len(self.session_batch_sampler)
        print(f"Created dataloader with {self.total_length} batches")

    def get_state(self) -> Dict[str, Any]:
        """Return the current state of the dataloader."""
        return {
            'batch': self.current_batch,
            'prev_rng_state': self._prev_rng_state if self._rng is not None else None
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the dataloader state."""
        # Restore batch counter
        self.current_batch = state['batch']

        # Restore RNG state if applicable
        if self._rng is not None and state['prev_rng_state'] is not None:
            self._prev_rng_state = state['prev_rng_state']
            self._rng.set_state(state['prev_rng_state'])

        # Skip to correct position in the iterator
        target_batch = self.current_batch % self.total_length
        for _ in range(target_batch):
            _ = next(self.iterator)

    def __iter__(self):
        """
        Iterate through dataset, providing (session_key, batch) tuples.
        This mimics the behavior of LongCycler which yielded (key, batch) pairs.
        """
        batch_idx = 0
        for _ in range(self.total_length):
            try:
                # Get data batch without session keys
                batch_data = next(self.iterator)
                self.current_batch += 1

                # Get session key for this batch
                session_key = self.session_batch_sampler.get_session_for_batch(batch_idx % self.total_length)
                batch_idx += 1

                if session_key is None:
                    session_key = "unknown"

                # Return the session key and batch data directly
                yield session_key, batch_data

            except Exception as e:
                # Provide more detailed error information
                warnings.warn(f"Error in DataLoader iteration: {str(e)}")
                import traceback
                warnings.warn(f"Traceback: {traceback.format_exc()}")
                raise

    def __len__(self):
        """Return the number of batches in an epoch."""
        return self.total_length
