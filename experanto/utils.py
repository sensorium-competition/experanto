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
import logging
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

log = logging.getLogger(__name__)

def replace_nan_with_batch_mean(data: np.array) -> np.array:
    row, col = np.where(np.isnan(data))
    for i, j in zip(row, col):
        new_value = np.nanmean(data[:, j])
        data[i, j] = new_value if not np.isnan(new_value) else 0
    return data


def add_behavior_as_channels(data: dict[str, torch.Tensor]) -> dict:
    """
    Adds behavioral data as additional channels to screen data.

    Input:
    data = {
        'screen': torch.Tensor: (c, t, h, w)
        'eye_tracker': torch.Tensor: (t, c_eye) or (t, h, w)
        'treadmill': torch.Tensor: (t, c_tread) or (t, h, w)
    }

    Output:
    data = {
        'screen': torch.Tensor: (c+behavior_channels, t, h, w) - contiguous
        ...
    }
    """
    screen = data["screen"]  # Already contiguous, shape (c, t, h, w)
    c, t, h, w = screen.shape
    eye_tracker = data["eye_tracker"]
    treadmill = data["treadmill"]

    # Process eye_tracker
    if len(eye_tracker.shape) == 2:  # (t, c_eye)
        c_eye = eye_tracker.shape[1]
        # Reshape to (c_eye, t, h, w)
        eye_tracker = eye_tracker.transpose(0, 1)  # (c_eye, t)
        eye_tracker = eye_tracker.unsqueeze(-1).unsqueeze(-1)  # (c_eye, t, 1, 1)
        eye_tracker = eye_tracker.expand(-1, -1, h, w).contiguous()  # (c_eye, t, h, w)
    else:  # (t, h, w)
        # Reshape to (1, t, h, w)
        eye_tracker = eye_tracker.unsqueeze(0).contiguous()  # (1, t, h, w)

    # Process treadmill
    if len(treadmill.shape) == 2:  # (t, c_tread)
        c_tread = treadmill.shape[1]
        # Reshape to (c_tread, t, h, w)
        treadmill = treadmill.transpose(0, 1)  # (c_tread, t)
        treadmill = treadmill.unsqueeze(-1).unsqueeze(-1)  # (c_tread, t, 1, 1)
        treadmill = treadmill.expand(-1, -1, h, w).contiguous()  # (c_tread, t, h, w)
    else:  # (t, h, w)
        # Reshape to (1, t, h, w)
        treadmill = treadmill.unsqueeze(0).contiguous()  # (1, t, h, w)

    # Concatenate along the channel dimension (dim=0) and ensure the result is contiguous
    result = torch.cat([screen, eye_tracker, treadmill], dim=0)

    # Ensure the result is contiguous
    if not result.is_contiguous():
        result = result.contiguous()

    data["screen"] = result

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

    def __len__(self):
        """Return the length of the original sampler."""
        return len(self.sampler)

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class SessionConcatDataset(Dataset):
    """Memory-efficient concatenated dataset that reliably tracks sessions."""

    def __init__(self, datasets, session_names=None):
        """Initialize the concatenated dataset with session tracking."""
        if not datasets:
            raise ValueError("datasets is empty")

        # Store datasets
        self.datasets = list(datasets)

        # Track session names
        if session_names is None:
            session_names = [f"session_{i}" for i in range(len(datasets))]
        self.session_names = session_names

        # Print dataset sizes for debugging
        for i, (name, dataset) in enumerate(zip(session_names, datasets)):
            print(f"Dataset {i}: {name}, length = {len(dataset)}")

        # Compute cumulative sizes for efficient indexing
        self.cumulative_sizes = []
        current_size = 0
        for dataset in self.datasets:
            current_size += len(dataset)
            self.cumulative_sizes.append(current_size)

        # Create session indices dictionary for fast lookup
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

        # Return the data along with session information to ensure alignment
        return data

    def get_session_for_idx(self, idx):
        """Get the session name for a given index."""
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        return self.session_names[dataset_idx]

    def get_indices_for_session(self, session_name):
        """Get all indices belonging to a given session."""
        if session_name in self.session_indices:
            start, end = self.session_indices[session_name]
            return list(range(start, end))
        return []


class SessionBatchSampler(Sampler):
    """
    A batch sampler that selects batches from sessions, ensuring alignment.
    """

    def __init__(self, dataset, batch_size, drop_last=False, shuffle=False, seed=None):
        """
        Initialize a session batch sampler.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        # Use simpler RNG approach
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

        # Get all session names
        self.session_names = list(dataset.session_indices.keys())
        print(f"Sessions: {self.session_names}")

        # Pre-generate batch information more simply
        self.batches = self._generate_batches()
        print(f"Generated {len(self.batches)} batches")

    def _generate_batches(self):
        """
        Generate batches ensuring that session alignment is maintained.
        """
        all_batches = []

        # Process each session
        for session_name in self.session_names:
            # Get all indices for this session
            session_indices = self.dataset.get_indices_for_session(session_name)

            # Skip empty sessions
            if not session_indices:
                continue

            # Shuffle indices if needed
            if self.shuffle:
                indices_copy = session_indices.copy()
                self.rng.shuffle(indices_copy)
                session_indices = indices_copy

            # Create batches for this session
            for i in range(0, len(session_indices), self.batch_size):
                batch_indices = session_indices[i:i + self.batch_size]

                # Skip last batch if needed
                if self.drop_last and len(batch_indices) < self.batch_size:
                    continue

                # Store both session name and indices to ensure alignment
                all_batches.append((session_name, batch_indices))

        return all_batches

    def __iter__(self):
        """Yield batch indices and keep track of session alignment."""
        # Optionally shuffle the order of batches
        batch_order = list(range(len(self.batches)))
        if self.shuffle:
            self.rng.shuffle(batch_order)

        # Yield batches in potentially shuffled order
        for i in batch_order:
            _, batch_indices = self.batches[i]
            yield batch_indices

    def __len__(self):
        """Return the number of batches."""
        return len(self.batches)

    def get_session_for_batch_idx(self, batch_idx):
        """Get the session name for a given batch index."""
        if 0 <= batch_idx < len(self.batches):
            return self.batches[batch_idx][0]
        return None


class _RepeatSampler:
    """Simple sampler that repeats indefinitely."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class SessionDataLoader(DataLoader):
    """
    Simplified dataloader that ensures session-batch alignment
    with basic state tracking.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0,
                 pin_memory=False, drop_last=False, seed=None, **kwargs):

        # Verify we have a SessionConcatDataset
        if not isinstance(dataset, SessionConcatDataset):
            raise ValueError("SessionDataLoader requires a SessionConcatDataset")

        # Create session batch sampler
        self.session_batch_sampler = SessionBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=seed
        )

        # Initialize parent class with our custom batch sampler
        super().__init__(
            dataset=dataset,
            batch_sampler=self.session_batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs
        )

        # Create persistent repeating sampler
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True

        # Store attributes for state tracking
        self.dataset = dataset
        self.batch_size = batch_size
        self.current_batch = 0
        self.total_batches = len(self.session_batch_sampler)
        self.rng_state = None
        if seed is not None:
            self.rng_state = np.random.RandomState(seed).get_state()

        # Create persistent iterator
        self.iterator = super().__iter__()

    def get_state(self):
        """Return the current state of the dataloader."""
        return {
            'batch': self.current_batch,
            'rng_state': self.rng_state
        }

    def set_state(self, state):
        """
        Restore the dataloader state.
        This is a simplified version that doesn't guarantee perfect restoration.
        """
        old_batch = self.current_batch

        # Restore batch counter
        self.current_batch = state.get('batch', 0)

        # Restore RNG state if available
        if 'rng_state' in state and state['rng_state'] is not None:
            self.rng_state = state['rng_state']

        # Reset iterator if we need to go backwards
        if self.current_batch < old_batch:
            print(f"Resetting iterator to reach batch {self.current_batch}")
            self.iterator = super().__iter__()

            # Skip forward to target position (approximate)
            target_batch = self.current_batch % self.total_batches
            for _ in range(target_batch):
                try:
                    next(self.iterator)
                except StopIteration:
                    self.iterator = super().__iter__()

    def __iter__(self):
        """
        Iterate through dataset, providing (session_key, batch) tuples
        with state tracking.
        """
        for batch_idx in range(self.total_batches):
            try:
                # Get appropriate session index accounting for possible wrapping
                current_idx = batch_idx % self.total_batches

                # Get session key for current batch
                session_key = self.session_batch_sampler.get_session_for_batch_idx(current_idx)

                if session_key is None:
                    session_key = "unknown"
                    print(f"Warning: Could not determine session key for batch {batch_idx}, using 'unknown'")

                # Get batch data from the iterator
                batch_data = next(self.iterator)
                self.current_batch += 1

                # Return session key and batch data as a tuple
                yield session_key, batch_data

            except StopIteration:
                # If iterator is exhausted, create a new one
                print(f"Iterator exhausted at batch {batch_idx}, restarting")
                self.iterator = super().__iter__()

                # Try again with new iterator
                batch_data = next(self.iterator)
                self.current_batch += 1

                # Return with the correct session key
                yield session_key, batch_data