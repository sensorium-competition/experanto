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

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class OptimizedSessionConcatDataset(Dataset):
    """Memory-efficient concatenated dataset that keeps track of sessions."""

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
            log.info(f"Dataset {i}: {name}, length = {len(dataset)}")

        # Compute cumulative sizes for efficient indexing
        self.cumulative_sizes = []
        current_size = 0
        for dataset in self.datasets:
            current_size += len(dataset)
            self.cumulative_sizes.append(current_size)

        # Create session ranges more efficiently - store just boundaries
        self.session_ranges = []
        start_idx = 0
        for i, dataset in enumerate(datasets):
            session_size = len(dataset)
            self.session_ranges.append((start_idx, start_idx + session_size, session_names[i]))
            start_idx += session_size

        # Create session indices dictionary for fast lookup
        self.session_indices = {}
        start_idx = 0
        for i, dataset in enumerate(datasets):
            session_name = session_names[i]
            session_size = len(dataset)
            self.session_indices[session_name] = (start_idx, start_idx + session_size)
            start_idx += session_size

    def get_indices_for_session(self, session_name):
        """Get indices for a session - lazy generation to save memory."""
        if session_name in self.session_indices:
            start, end = self.session_indices[session_name]
            # Return a range object instead of a materialized list
            return range(start, end)
        return []

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

        # Return the data as-is
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


class RandomSessionBatchSampler(Sampler):
    """
    A batch sampler that randomly selects sessions and ensures all samples in a batch
    come from the same session.
    """

    def __init__(self, dataset, batch_size, drop_last=False, shuffle=False, seed=None, num_batches=None):
        """
        Initialize a random session batch sampler.
        """
        if not isinstance(dataset, OptimizedSessionConcatDataset):
            raise ValueError("RandomSessionBatchSampler requires a SessionConcatDataset")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed) if seed is not None else None

        # Get all session names
        self.session_names = list(dataset.session_indices.keys())
        log.info(f"Sessions: {self.session_names}")

        # Calculate approximate number of batches per session
        self.batches_per_session = {}
        total_possible_batches = 0

        for session_name in self.session_names:
            start, end = dataset.session_indices[session_name]
            session_size = end - start

            # Calculate batches for this session
            if drop_last:
                session_batches = session_size // batch_size
            else:
                session_batches = (session_size + batch_size - 1) // batch_size

            self.batches_per_session[session_name] = session_batches
            total_possible_batches += session_batches

        log.info(f"Batches per session: {self.batches_per_session}")
        log.info(f"Total possible batches: {total_possible_batches}")

        # Determine total number of batches to generate
        if num_batches is None:
            self.num_batches = total_possible_batches
        else:
            self.num_batches = num_batches

        # Pre-generate batch information (session and indices)
        self.batches = self._generate_batches()
        (log.info
         (f"Generated {len(self.batches)} batches"))

    def _generate_batches(self):
        """
        Generate batches ensuring:
        1. Each session appears exactly once before repeating any session
        2. Samples within each session are drawn randomly without repeating
        3. Empty sessions are included in the cycle but skipped for batch creation
        4. Cycling continues until the longest session is exhausted
        """
        all_batches = []

        # Create randomized indices for each session
        session_indices = {}
        remaining_indices = {}  # Track remaining indices for each session

        # Track which sessions have data
        sessions_with_data = set()

        for session_name in self.session_names:
            # Get all indices for this session
            all_session_indices = self.dataset.get_indices_for_session(session_name)

            # Convert to list if we received a range object or other iterable
            if not isinstance(all_session_indices, list):
                all_session_indices = list(all_session_indices)

            # Skip initialization for truly empty sessions
            if len(all_session_indices) == 0:
                session_indices[session_name] = []
                remaining_indices[session_name] = []
                continue

            # Create a copy for random sampling
            if self.shuffle and self.rng is not None:
                # Shuffle the indices for truly random sampling
                shuffled_indices = all_session_indices.copy()
                self.rng.shuffle(shuffled_indices)
                session_indices[session_name] = shuffled_indices
            else:
                session_indices[session_name] = all_session_indices.copy()

            # Initialize remaining indices
            remaining_indices[session_name] = session_indices[session_name].copy()

            # Mark this session as having data
            sessions_with_data.add(session_name)

        # Skip full batch generation if no batches needed
        if self.num_batches == 0 or not sessions_with_data:
            return []

        # Continue cycling until all sessions with data are exhausted or we reach num_batches
        batch_count = 0
        while sessions_with_data and batch_count < self.num_batches:
            # Create a shuffled order of all sessions for this cycle
            cycle_sessions = list(self.session_names)
            if self.shuffle and self.rng is not None:
                self.rng.shuffle(cycle_sessions)

            # Generate one batch from each session in this order
            for session_name in cycle_sessions:
                # Skip if we've reached the requested number of batches
                if batch_count >= self.num_batches:
                    break

                # Skip empty sessions but keep them in the cycle
                if session_name not in sessions_with_data:
                    continue

                # Get remaining indices for this session
                indices = remaining_indices[session_name]

                # If we've used all indices, regenerate
                if len(indices) < self.batch_size:
                    # If there aren't enough indices and we can't generate a valid batch
                    if len(session_indices[session_name]) < self.batch_size and self.drop_last:
                        # Remove this session from consideration
                        sessions_with_data.remove(session_name)
                        continue

                    # Reshuffle all indices for this session
                    if self.shuffle and self.rng is not None:
                        new_indices = session_indices[session_name].copy()
                        self.rng.shuffle(new_indices)
                        remaining_indices[session_name] = new_indices
                    else:
                        remaining_indices[session_name] = session_indices[session_name].copy()

                    indices = remaining_indices[session_name]

                # Take a batch of indices from the remaining ones
                batch_size = min(self.batch_size, len(indices))
                batch_indices = indices[:batch_size]

                # Skip if the batch is too small and we're dropping last
                if self.drop_last and batch_size < self.batch_size:
                    # Remove this session from consideration
                    sessions_with_data.remove(session_name)
                    continue

                # Remove used indices
                remaining_indices[session_name] = indices[batch_size:]

                # Add the batch
                all_batches.append((session_name, batch_indices))
                batch_count += 1

                # Check if this session is now empty and should be removed
                if len(remaining_indices[session_name]) == 0 and len(session_indices[session_name]) < self.batch_size:
                    sessions_with_data.remove(session_name)

        return all_batches

    def __iter__(self):
        """Yield batch indices."""
        for _, batch_indices in self.batches:
            yield batch_indices

    def __len__(self):
        """Return the number of batches."""
        return len(self.batches)

    def get_session_for_batch_idx(self, batch_idx):
        """Get the session name for a given batch index."""
        if 0 <= batch_idx < len(self.batches):
            return self.batches[batch_idx][0]
        return None


class SimpleStatefulDataLoader(DataLoader):
    """
    Fast stateful dataloader that provides both data and session keys.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0,
                 pin_memory=False, drop_last=False, seed=None, **kwargs):

        # Verify we have a SessionConcatDataset
        if not isinstance(dataset, OptimizedSessionConcatDataset):
            raise ValueError("SimpleStatefulDataLoader requires a SessionConcatDataset")

        # Store parameters for state tracking
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.num_workers = num_workers

        # Initialize RNG if seed provided
        self._rng = ThreadSafeRNG(seed) if seed is not None else None
        self._prev_rng_state = None if self._rng is None else self._rng.get_state()

        # Create a random session batch sampler
        start_time = time.time()
        self.session_batch_sampler = RandomSessionBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=seed
        )
        log.info(f"Batch sampler creation took {time.time() - start_time:.2f} seconds")

        # Set worker init function for proper per-worker seeding
        worker_init_fn = kwargs.get('worker_init_fn', None)
        if worker_init_fn is None and num_workers > 0 and seed is not None:
            kwargs['worker_init_fn'] = self._worker_init_fn

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
        log.info(f"Created dataloader with {self.total_length} batches")

    def _worker_init_fn(self, worker_id):
        """
        Initialize each worker with a unique but deterministic seed.
        This ensures each worker gets different but reproducible data.
        """
        # Get base seed from dataloader
        base_seed = self.seed if self.seed is not None else 0

        # Create unique seed per worker
        worker_seed = base_seed + worker_id

        # Set numpy seed
        np.random.seed(worker_seed)

        # Set torch seed
        torch.manual_seed(worker_seed)

        # Set random seed
        import random
        random.seed(worker_seed)

    def get_state(self) -> Dict[str, Any]:
        """Return the current state of the dataloader."""
        return {
            'batch': self.current_batch,
            'prev_rng_state': self._prev_rng_state if self._rng is not None else None
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the dataloader state with robust iterator repositioning."""
        # Store old batch for comparison
        old_batch = self.current_batch

        # Restore batch counter
        self.current_batch = state.get('batch', 0)

        # Restore RNG state if applicable
        if self._rng is not None and 'prev_rng_state' in state and state['prev_rng_state'] is not None:
            self._prev_rng_state = state['prev_rng_state']
            self._rng.set_state(state['prev_rng_state'])

        # Reset iterator if we're moving backward or to the beginning
        if self.current_batch < old_batch or self.current_batch == 0:
            log.info(f"Resetting iterator and moving to batch {self.current_batch}")
            self.iterator = super().__iter__()

            # Skip to correct position
            target_batch = self.current_batch % self.total_length

            # Skip forward to target position
            for _ in range(target_batch):
                try:
                    next(self.iterator)
                except StopIteration:
                    # If we reach the end, restart
                    self.iterator = super().__iter__()

    def __iter__(self):
        """
        Iterate through dataset, providing (session_key, batch) tuples.
        This mimics the behavior of LongCycler which yielded (key, batch) pairs.
        """
        for batch_idx in range(self.total_length):
            try:
                # Get session key BEFORE getting the batch - this is critical for alignment
                current_idx = batch_idx % len(self.session_batch_sampler)
                session_key = self.session_batch_sampler.get_session_for_batch_idx(current_idx)

                if session_key is None:
                    session_key = "unknown"
                    log.warning(f"Could not determine session key for batch {batch_idx}, using 'unknown'")

                # Get data batch
                batch_data = next(self.iterator)
                self.current_batch += 1

                # Verify batch belongs to correct session (extra validation)
                self._verify_batch_session(batch_data, session_key, batch_idx)

                # Return the session key and batch data
                yield session_key, batch_data

            except StopIteration:
                # If we reach the end, restart the iterator
                log.warning(f"Iterator exhausted at batch {batch_idx}, restarting")
                self.iterator = super().__iter__()

                # Try again with the new iterator
                batch_data = next(self.iterator)
                self.current_batch += 1

                # Return batch with the correct session key
                yield session_key, batch_data

            except Exception as e:
                # Provide more detailed error information
                log.error(f"Error in DataLoader iteration at batch {batch_idx}: {str(e)}")
                import traceback
                log.error(f"Traceback: {traceback.format_exc()}")
                raise

    def _verify_batch_session(self, batch_data, session_key, batch_idx):
        """
        Extra validation to verify batch corresponds to the expected session.
        Only runs in debug mode or if there have been previous errors.
        """
        # Only run this verification occasionally to avoid performance impact
        if batch_idx % 100 != 0:
            return

        try:
            # Try to extract indices from batch if available
            indices = None
            if hasattr(batch_data, 'indices'):
                indices = batch_data.indices
            elif isinstance(batch_data, tuple) and len(batch_data) > 0 and hasattr(batch_data[0], 'indices'):
                indices = batch_data[0].indices

            if indices is not None and len(indices) > 0:
                # Check first index to verify session
                first_idx = indices[0]
                actual_session = self.dataset.get_session_for_idx(first_idx)

                if actual_session != session_key and session_key != "unknown":
                    log.error(f"CRITICAL: Session mismatch at batch {batch_idx}! "
                              f"Expected: {session_key}, Actual: {actual_session}")
        except Exception as e:
            # Don't let validation errors break the dataloader
            log.debug(f"Batch validation error (non-critical): {e}")


class ThreadSafeRNG:
    """Thread-safe wrapper for numpy's RandomState."""

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self.lock = threading.Lock()

    def shuffle(self, array):
        """Thread-safe array shuffling."""
        with self.lock:
            return self.rng.shuffle(array)

    def choice(self, a, size=None, replace=True, p=None):
        """Thread-safe random choice."""
        with self.lock:
            return self.rng.choice(a, size, replace, p)

    def get_state(self):
        """Get the RNG state."""
        with self.lock:
            return self.rng.get_state()

    def set_state(self, state):
        """Set the RNG state."""
        with self.lock:
            self.rng.set_state(state)