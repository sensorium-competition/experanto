from typing import Dict, Any, Optional, List, Iterator, Union, Tuple, Sequence, Literal

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
from collections import defaultdict

# third-party libraries
import numpy as np
from omegaconf import DictConfig
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader, Sampler

# local libraries
from .intervals import TimeInterval

logger = logging.getLogger(__name__)


def count_batches(indices: Sequence[Any], batch_size: int, drop_last: bool) -> int:
    # Calculate number of batches
    if drop_last:
        num_batches = len(indices) // batch_size
    else:
        num_batches = (len(indices) + batch_size - 1) // batch_size
    return num_batches


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

    def __init__(self, *args, shuffle_each_epoch=False, **kwargs, ):
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

        # Return the data along with session information
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

    def get_sessions_count(self):
        """Get number of sessions and count of samples per session."""
        return {name: end - start for name, (start, end) in self.session_indices.items()}

    def reset_state(self):
        """Reset the state of the dataset."""
        for d in self.datasets:
            d.reset_state()

    def get_state(self):
        """Return the state of the dataset."""
        return {
            'dataset_states': {
                d.data_key: d.get_state() for d in self.datasets
            }
        }

    def set_state(self, state, strict=False):
        """Restore the state of the dataset."""
        data_keys = set([d.data_key for d in self.datasets])
        state_keys = set(state['dataset_states'].keys())
        if not state_keys == data_keys:
            msg = (f"SessionConcatDataset states keys do not match: "
                  f"{state_keys} != {data_keys}")
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
        for d in self.datasets:
            dkey = d.data_key
            if dkey in state['dataset_states']:
                d.set_state(state['dataset_states'][d.data_key])


class SessionBatchSampler(Sampler):
    """
    A batch sampler that cycles through sessions, ensuring each session
    appears exactly once before repeating any session.
    """

    def __init__(self, dataset, batch_size, drop_last=False, shuffle=False, seed=None):
        """
        Initialize session batch sampler.

        Args:
            dataset: The SessionConcatDataset to sample from
            batch_size: Number of samples per batch
            drop_last: Whether to drop the last batch if it's smaller than batch_size
            shuffle: Whether to shuffle samples within each session
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        # Use its own RNG instance based on the provided seed
        self.rng = np.random.RandomState(seed)
        self.prv_rng_state = self.rng.get_state()

        # Get sessions
        self.session_names = list(dataset.session_indices.keys())
        print(f"Sessions: {self.session_names}")

        self.consumed_sessions = []

        # Pre-process session indices
        self.session_indices = {}
        for session_name in self.session_names:
            indices = dataset.get_indices_for_session(session_name)
            if indices:
                self.session_indices[session_name] = indices

        # Calculate batches per session
        self.batches_per_session = {}
        total_batches = 0
        for session_name, indices in self.session_indices.items():
            num_batches = count_batches(indices, batch_size, drop_last)
            self.batches_per_session[session_name] = num_batches
            total_batches += num_batches

        print(f"Batches per session: {self.batches_per_session}")
        print(f"Total batches: {total_batches}")

    def __len__(self):
        """Return the total number of batches across all sessions."""
        return sum(self.batches_per_session.values())

    def get_session_cycle(self):
        """
        Generate one cycle of sessions, with each session appearing exactly once.
        Sessions are shuffled unless their appearance order needs to be controlled.
        """
        order = list(self.session_names)
        if self.shuffle:
            self.prv_rng_state = self.rng.get_state()
            self.rng.shuffle(order)

        # Remove consumed sessions from order
        for session_name in self.consumed_sessions:
            order.remove(session_name)

        return order
    
    def reset_state(self):
        """Reset the state of the sampler."""
        self.consumed_sessions = []

    def get_state(self):
        """Return the state of the sampler (including RNG state)."""
        return {
            'prv_rng_state': self.prv_rng_state,
            'consumed_sessions': self.consumed_sessions
        }

    def set_state(self, state):
        """Restore the state of the sampler (including RNG state)."""
        self.prv_rng_state = state.get('prv_rng_state')
        if self.prv_rng_state is not None:
            self.rng.set_state(self.prv_rng_state)
        self.consumed_sessions = state.get('consumed_sessions', [])


class FastSessionDataLoader:
    """
    An optimized dataloader that ensures:
    1. Each session appears exactly once before repeating
    2. The epoch ends when the longest session is exhausted
    3. Perfect alignment between sessions and batches is maintained
    4. State is properly tracked and can be restored
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        cycle_mode: Literal['active', 'balanced'] = 'active',
        seed=None,
        **kwargs,
    ):
        """
        Initialize optimized session dataloader.

        Args:
            dataset: The SessionConcatDataset to load from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle indices within sessions
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory in GPU
            drop_last: Whether to drop the last batch if smaller than batch_size
            seed: Random seed for reproducibility
        """
        # Store dataset and parameters
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.kwargs = kwargs
        assert cycle_mode in ['active', 'balanced'], \
            f"Invalid cycle mode: {cycle_mode}! Must be one of ['active', 'balanced']"
        self.cycle_mode = cycle_mode
        # Create batch sampler
        self.batch_sampler = SessionBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=seed,
        )

        # Store session info for faster access
        self.session_names = self.batch_sampler.session_names
        self.session_indices = self.batch_sampler.session_indices
        self.batches_per_session = self.batch_sampler.batches_per_session

        # Compute maximum batches per session (for epoch tracking)
        self.max_batches_per_session = max(self.batches_per_session.values()) if self.batches_per_session else 0

        # Prepare session data loaders to avoid recreating them for each batch
        self.session_dataloaders = {}
        for i, session_name in enumerate(self.session_names):
            indices = self.session_indices[session_name]
            # Derive a unique seed for each session sampler
            session_seed = None if seed is None else seed + i + 1
            # Create a specific sampler for this session
            session_sampler = SessionSpecificSampler(
                indices=indices,
                batch_size=batch_size,
                drop_last=drop_last,
                shuffle=shuffle,
                seed=session_seed,
            )

            # Create a DataLoader for this session
            self.session_dataloaders[session_name] = DataLoader(
                dataset=dataset,
                batch_sampler=session_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                **kwargs
            )

        # State tracking variables
        self.current_batch = 0
        self.position_in_epoch = 0
        self.session_positions = {name: 0 for name in self.session_names}
        # Tracks batches yielded per session in the current epoch iteration
        self.batches_from_session = defaultdict(int)  
        # Track active sessions
        self.active_sessions = set(self.session_names)

        print(
            f"Created FastSessionDataLoader with {len(self.session_names)} sessions and {len(self)} total batches")

    def __len__(self):
        """Return the total number of batches in an epoch."""
        if self.cycle_mode == 'active':
            return sum(self.batches_per_session.values())
        elif self.cycle_mode == 'balanced':
            return self.max_batches_per_session * len(self.session_names)
        else:
            raise ValueError(
                f"Invalid cycle mode: {self.cycle_mode}! "
                f"Must be one of ['active', 'balanced']"
            )

    def reset_state(self):
        """Reset the state of the dataloader."""
        self.current_batch = 0
        self.position_in_epoch = 0
        self.batches_from_session = defaultdict(int)
        self.active_sessions = set(self.session_names)
        for name in self.session_names:
            self.session_positions[name] = 0
            self.session_dataloaders[name].batch_sampler.reset_state()
        self.batch_sampler.reset_state()
        self.dataset.reset_state()

    def get_state(self):
        """Return the current state of the dataloader."""
        return {
            'current_batch': self.current_batch,
            'position_in_epoch': self.position_in_epoch,
            'session_positions': self.session_positions.copy(),
            'batches_from_session': self.batches_from_session.copy(),
            'active_sessions': list(self.active_sessions),  # Store as list for serialization
            'batch_sampler_state': self.batch_sampler.get_state(),
            'session_sampler_states': {
                name: dl.batch_sampler.get_state()
                for name, dl in self.session_dataloaders.items()
            },
            'dataset_state': self.dataset.get_state(),
        }

    def set_state(self, state, strict=False):
        """Restore the dataloader state."""
        if not state:
            return

        # Restore batch counter
        current_batch = state.get('current_batch')
        if current_batch is not None:
            self.current_batch = current_batch

        # Restore position in epoch
        position_in_epoch = state.get('position_in_epoch')
        if position_in_epoch is not None:
            self.position_in_epoch = position_in_epoch

        # Restore session positions
        session_positions = state.get('session_positions')
        if session_positions:
            self.session_positions = session_positions

        # Restore RNG state for the batch sampler
        batch_sampler_state = state.get('batch_sampler_state')
        if batch_sampler_state is not None and hasattr(self.batch_sampler, 'set_state'):
            self.batch_sampler.set_state(batch_sampler_state)

        # Restore batches_from_session state
        batches_from_session_state = state.get('batches_from_session')
        if batches_from_session_state is not None:
            self.batches_from_session = defaultdict(int)
            self.batches_from_session.update(batches_from_session_state)
        else:
            # For backward compatibility or if state doesn't have it
            self.batches_from_session = defaultdict(int)

        # Restore active sessions
        active_sessions_list = state.get('active_sessions')
        if active_sessions_list is not None:
            self.active_sessions = set(active_sessions_list)
        else:
            # Default to all sessions if not in state (for backward compatibility)
            self.active_sessions = set(self.session_names)

        # Reset session iterators with new positions
        for session_name, dataloader in self.session_dataloaders.items():
            # Get sampler and reset its position
            sampler = dataloader.batch_sampler
            # Restore RNG state for each session sampler
            session_sampler_states = state.get('session_sampler_states', {})
            sampler_state = session_sampler_states.get(session_name)
            if sampler_state is not None and hasattr(sampler, 'set_state'):
                sampler.set_state(sampler_state)
            # Restore position for each session sampler from `self.session_positions`.
            # NOTE: Required still for backwards compatibility (i.e. if `position` not
            #  in `sampler_state`).
            if hasattr(sampler, 'set_position'):
                position = self.session_positions.get(session_name)
                if position is not None:
                    sampler.set_position(position)
        
        # Restore dataset state
        dataset_state = state.get('dataset_state')
        if dataset_state is not None and hasattr(self.dataset, 'set_state'):
            self.dataset.set_state(dataset_state, strict=strict)

        print(f"Restored dataloader state to batch {self.current_batch}")

    def _get_next_batch(self, iterator: Iterator, session_name: str) -> Tuple[str, Any]:
        """Update the counts of the dataloader."""
        # Get the next batch from this session
        batch = next(iterator)
        # Update state tracking
        self.current_batch += 1
        self.session_positions[session_name] += 1
        self.batches_from_session[session_name] += 1  # Update local dictionary
        self.batch_sampler.consumed_sessions.append(session_name)
        # Return the batch and the session name
        return session_name, batch

    def __iter__(self):
        """
        Iterate through sessions, cycling through them until all are exhausted.

        The iteration scheme ensures:
        1. Each session appears exactly once in each cycle
        2. Samples within a session are properly batched and optionally shuffled
        3. The epoch ends when the longest session is exhausted
        """

        # Create iterators for each session.
        session_iterators = {}
        for s, dl in self.session_dataloaders.items():
            # NOTE: Calling `iter(dl)` actually increments sampler position by 2, so we
            #  manually re-set the position to the pre-iteration value!
            _pre_iter_position = dl.batch_sampler.position
            session_iterators[s] = iter(dl)
            dl.batch_sampler.set_position(_pre_iter_position)

        # Reset iterators with current positions
        for session_name, dataloader in self.session_dataloaders.items():
            # Reset sampler position
            sampler = dataloader.batch_sampler
            if hasattr(sampler, 'set_position'):
                sampler.set_position(self.session_positions[session_name])

        # Continue until we've gone through one full epoch
        # (i.e., until the longest session is exhausted)
        while self.position_in_epoch < self.max_batches_per_session:

            # Create a cycle order of sessions
            cycle_order = self.batch_sampler.get_session_cycle()

            # Process one batch from each active session in this cycle
            for session_name in cycle_order:
                # Skip if session is already exhausted
                # NOTE: Only required when continuing training (in `active` mode)
                #  because `SessionSpecificSampler` resets its state automatically.
                if session_name not in self.active_sessions:
                    continue

                # Get iterator for this session
                assert session_name in session_iterators, \
                    f"Session {session_name} not in `session_iterators`!"
                iterator = session_iterators[session_name]

                try:
                    yield self._get_next_batch(iterator, session_name)
                except StopIteration:
                    if self.cycle_mode == 'active':
                        # This session is exhausted for the current epoch
                        if session_name in self.active_sessions:
                            self.active_sessions.remove(session_name)
                    elif self.cycle_mode == 'balanced':
                        # Reset iterator and try again. NOTE: `SessionSpecificSampler`
                        #  resets its state automatically when `__iter__` expires.
                        iterator = iter(self.session_dataloaders[session_name])
                        # HACK: Undo the increment of the sampler position during call to `iter`
                        if hasattr(iterator._index_sampler, 'set_position'):
                            iterator._index_sampler.set_position(0)
                        session_iterators[session_name] = iterator
                        yield self._get_next_batch(iterator, session_name)
                    else:
                        raise ValueError(
                            f"Invalid cycle mode: {self.cycle_mode}! "
                            f"Must be one of ['active', 'balanced']"
                        )

            self.batch_sampler.consumed_sessions = []

            # If we've completed a full cycle, increment the position counter
            self.position_in_epoch += 1

        # End of epoch - reset state
        self.reset_state()


class SessionSpecificSampler(Sampler):
    """
    A batch sampler specific to a single session that efficiently
    generates batches from the session's indices.
    """

    def __init__(self, indices, batch_size, drop_last=False, shuffle=False, seed=None):
        """
        Initialize session-specific sampler.

        Args:
            indices: List of dataset indices belonging to this session
            batch_size: Number of samples per batch
            drop_last: Whether to drop the last batch if smaller than batch_size
            shuffle: Whether to shuffle indices
            seed: Random seed for reproducibility
        """
        self._original_indices = list(indices)  # Save a copy so we can shuffle from original state
        self.indices = deepcopy(self._original_indices)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
        self.prv_rng_state = self.rng.get_state()
        self.num_batches = count_batches(indices, batch_size, drop_last)
        # Track current position
        self.position = 0
        # Shuffle indices
        self.reset_state()

    def __len__(self):
        """Return the number of batches."""
        return self.num_batches

    def set_position(self, position: int):
        """Set the current batch position."""
        self.position = position % self.num_batches if self.num_batches > 0 else 0

    def shuffle_indices(self):
        """Shuffle the indices."""
        if self.shuffle:
            self.prv_rng_state = self.rng.get_state()
            # NOTE: We always shuffle from the original indices such that the shuffled result
            #  only depends on the current RNG state, not on the previous state of the indices.
            self.indices = deepcopy(self._original_indices)
            self.rng.shuffle(self.indices)

    def reset_state(self):
        """Reset the state of the sampler."""
        # Re-shuffle indices if needed
        self.shuffle_indices()
        # Reset position
        self.position = 0

    def get_state(self):
        """Return the state of the sampler (including RNG state)."""
        # NOTE: We don't save the indices! This requires that the reloaded sampler must be
        #  initialized with the same indices *in the same order* as the original one.
        return {
            'prv_rng_state': self.prv_rng_state,
            'position': self.position,
        }

    def set_state(self, state):
        """Restore the state of the sampler (including RNG state)."""
        self.prv_rng_state = state.get('prv_rng_state')
        if self.prv_rng_state is not None:
            self.rng.set_state(self.prv_rng_state)
            # shuffle indices with loaded RNG state
            self.shuffle_indices()
        position = state.get('position')
        if position is not None:
            self.position = position

    def __iter__(self):
        """
        Yield batches of indices starting from the current position.
        """
        # Start from current position
        start_idx = self.position * self.batch_size

        # Generate batches from start_idx to end
        for i in range(start_idx, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]

            # Skip last batch if needed
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            # Update position
            self.position += 1

            yield batch_indices
        
        # Reset state
        self.reset_state()