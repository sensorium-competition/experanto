from typing import Dict, Any, Optional, List, Iterator

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
import numpy as np

# third-party libraries
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, Sampler

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
    """Sampler that repeats forever with no overhead."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class WorkerPoolManager:
    """
    Manages a shared pool of workers across multiple DataLoaders.
    This class ensures we don't exceed a total worker count across all loaders.
    """
    _instance = None

    @classmethod
    def get_instance(cls, max_workers=None):
        """Get the singleton instance of the worker pool manager."""
        if cls._instance is None:
            cls._instance = WorkerPoolManager(max_workers)
        return cls._instance

    def __init__(self, max_workers=None):
        """
        Initialize the worker pool manager.

        Args:
            max_workers: Maximum number of workers to create. If None, uses CPU count.
        """
        if max_workers is None:
            # Use 75% of available CPU cores as the default
            max_workers = max(1, int(mp.cpu_count() * 0.75))

        self.max_workers = max_workers
        self.available_workers = list(range(max_workers))
        self.worker_assignments = {}  # Maps dataloader ID to worker IDs
        self.lock = threading.Lock()  # Use threading.Lock() for better compatibility

        print(f"Worker Pool initialized with {max_workers} total workers")

    def allocate_workers(self, dataloader_id, requested_count):
        """
        Allocate worker IDs to a dataloader.

        Args:
            dataloader_id: Unique identifier for the dataloader
            requested_count: Number of workers requested

        Returns:
            List of worker IDs allocated
        """
        with self.lock:
            # If this dataloader already has workers, return them
            if dataloader_id in self.worker_assignments:
                return self.worker_assignments[dataloader_id]

            # Calculate how many workers we can actually allocate
            available_count = len(self.available_workers)
            actual_count = min(requested_count, available_count)

            if actual_count == 0:
                # No workers available, use a round-robin assignment from existing workers
                all_assigned = [w for workers in self.worker_assignments.values() for w in workers]
                if not all_assigned:
                    print(f"Warning: No workers available for dataloader {dataloader_id}")
                    return []  # No workers at all, will fall back to synchronous

                # Choose least loaded workers
                worker_counts = {}
                for w in range(self.max_workers):
                    worker_counts[w] = all_assigned.count(w)

                # Sort by usage count
                sorted_workers = sorted(worker_counts.items(), key=lambda x: x[1])
                actual_count = min(requested_count, len(sorted_workers))
                allocated = [w for w, _ in sorted_workers[:actual_count]]
                print(f"Dataloader {dataloader_id} reusing workers: {allocated}")
            else:
                # Allocate new workers from the available pool
                allocated = self.available_workers[:actual_count]
                self.available_workers = self.available_workers[actual_count:]
                print(f"Dataloader {dataloader_id} allocated {actual_count} workers: {allocated}")

            self.worker_assignments[dataloader_id] = allocated
            return allocated

    def release_workers(self, dataloader_id):
        """
        Release workers allocated to a dataloader.

        Args:
            dataloader_id: Unique identifier for the dataloader
        """
        with self.lock:
            if dataloader_id in self.worker_assignments:
                workers = self.worker_assignments.pop(dataloader_id)
                self.available_workers.extend(workers)
                print(f"Released workers for dataloader {dataloader_id}: {workers}")


class PooledDataLoader(torch.utils.data.DataLoader):
    """
    DataLoader that uses a shared worker pool to reduce the total number of processes.
    Based on your LightweightDataLoader with added pooling capabilities.
    """

    _id_counter = 0
    _pool_manager = None

    @classmethod
    def _get_next_id(cls):
        """Get a unique ID for each dataloader instance."""
        cls._id_counter += 1
        return cls._id_counter

    @classmethod
    def configure_pool(cls, max_workers=None):
        """
        Configure the worker pool for all PooledDataLoader instances.

        Args:
            max_workers: Maximum number of workers in the pool.
        """
        if cls._pool_manager is None:
            cls._pool_manager = WorkerPoolManager.get_instance(max_workers)
        return cls._pool_manager

    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0,
                 prefetch_factor=2, persistent_workers=True,
                 worker_restart_threshold=60, seed=None, **kwargs):
        """
        Initialize a DataLoader with pooled workers.

        Args:
            dataset: Dataset to load data from
            batch_size: Batch size
            shuffle: Whether to shuffle the dataset
            num_workers: Requested number of workers
            pin_memory: Whether to pin memory
            drop_last: Whether to drop the last incomplete batch
            timeout: Timeout for worker processes
            prefetch_factor: Number of samples to prefetch per worker
            persistent_workers: Whether to keep worker processes alive between iterations
            worker_restart_threshold: Time threshold for worker stall detection
            seed: Random seed for reproducibility
            **kwargs: Additional arguments to pass to DataLoader
        """
        # Initialize pool manager if not already done
        if self.__class__._pool_manager is None:
            self.__class__._pool_manager = WorkerPoolManager.get_instance()

        # Get a unique ID for this dataloader
        self.dataloader_id = self.__class__._get_next_id()

        # Store parameters for state tracking and worker recreation
        self.worker_restart_threshold = worker_restart_threshold
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.kwargs = kwargs
        self.seed = seed

        # Handle worker allocation
        self.requested_workers = num_workers
        allocated_workers = []
        actual_workers = 0

        if num_workers > 0:
            # Request workers from the pool
            allocated_workers = self.__class__._pool_manager.allocate_workers(
                self.dataloader_id, num_workers)
            actual_workers = len(allocated_workers)

            # If we got less workers than requested, adjust expectations
            if actual_workers < num_workers:
                warnings.warn(f"Requested {num_workers} workers, but only {actual_workers} available.")
                num_workers = actual_workers

            # Force persistent workers
            persistent_workers = True

            # Lower prefetch factor to reduce memory pressure
            prefetch_factor = max(2, min(prefetch_factor, 2))

            # Enable timeout detection but make it generous
            if timeout <= 0:
                timeout = 120  # 2 minutes should be enough for most batches

        # Store actual allocated worker count
        self.num_workers = num_workers

        # Define a worker_init_fn that sets the worker ID in a consistent way
        original_init_fn = kwargs.get('worker_init_fn', None)

        def pooled_worker_init_fn(worker_id):
            # Map local worker_id to global worker_id from our allocation
            global_worker_id = allocated_workers[worker_id] if allocated_workers else worker_id

            # Set worker seed based on the global ID for reproducibility
            if self.seed is not None:
                # Instead of modifying WorkerInfo directly, set the random seeds
                # The global seed is already used by PyTorch to set the initial seed
                # We just need to make it deterministic based on the worker_id
                worker_seed = self.seed + global_worker_id
                torch.manual_seed(worker_seed)
                random.seed(worker_seed)
                np.random.seed(worker_seed)

            # Set worker ID in environment for potential external tools
            os.environ['WORKER_ID'] = str(global_worker_id)

            # Call the user's worker_init_fn if provided
            if original_init_fn is not None:
                original_init_fn(worker_id)

        # Initialize the actual DataLoader with pooled workers
        kwargs['worker_init_fn'] = pooled_worker_init_fn if num_workers > 0 else None

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            **kwargs
        )

        # Setup repeating sampler with minimal overhead - just like your LightweightDataLoader
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True

        # Create single iterator that persists throughout loader lifetime
        self.iterator = super().__iter__()

        # State tracking
        self.current_batch = 0
        self.total_length = len(self.batch_sampler.sampler)

        # Track health of workers for debugging
        self._last_batch_time = time.time()
        self._stall_count = 0
        self._worker_restarted = False

    def __del__(self):
        """Release workers when the dataloader is deleted."""
        if hasattr(self, 'dataloader_id') and hasattr(self.__class__, '_pool_manager'):
            if self.__class__._pool_manager is not None:
                self.__class__._pool_manager.release_workers(self.dataloader_id)

    def get_state(self):
        """Simple state tracking for checkpointing."""
        return {'batch': self.current_batch}

    def set_state(self, state):
        """Restore state by skipping ahead to the correct batch."""
        target_batch = state['batch'] % self.total_length
        current = self.current_batch % self.total_length

        # Calculate how many batches to skip
        skip_count = (target_batch - current) % self.total_length

        # Skip batches efficiently
        for _ in range(skip_count):
            _ = next(self.iterator)
            self.current_batch += 1

    def _resuscitate_workers(self):
        """
        Attempt to restart workers by recreating the iterator.
        This is similar to what happens when you press Ctrl+C - a signal gets
        sent that can sometimes jolt workers back to life.
        """
        warnings.warn(f"Attempting to resuscitate workers for dataloader {self.dataloader_id}...")

        try:
            # Close the current iterator and its worker processes
            # This is the equivalent of the "kick" from Ctrl+C
            self._DataLoader__shutdown_workers()
        except Exception as e:
            warnings.warn(f"Error shutting down workers: {str(e)}")

        # Release and re-acquire workers
        if self.__class__._pool_manager is not None:
            self.__class__._pool_manager.release_workers(self.dataloader_id)
            allocated_workers = self.__class__._pool_manager.allocate_workers(
                self.dataloader_id, self.requested_workers)
            actual_workers = len(allocated_workers)

            # Update worker count
            self.num_workers = actual_workers

            # Define a new worker_init_fn for the restarted workers
            original_init_fn = self.kwargs.get('worker_init_fn', None)

            def pooled_worker_init_fn(worker_id):
                # Map local worker_id to global worker_id from our allocation
                global_worker_id = allocated_workers[worker_id] if allocated_workers else worker_id

                # Set worker seed based on the global ID for reproducibility
                if self.seed is not None:
                    torch.utils.data.get_worker_info().seed = self.seed + global_worker_id

                # Set worker ID in environment for potential external tools
                os.environ['WORKER_ID'] = str(global_worker_id)

                # Call the user's worker_init_fn if provided
                if original_init_fn is not None:
                    original_init_fn(worker_id)

            # Update worker_init_fn
            self.kwargs['worker_init_fn'] = pooled_worker_init_fn if self.num_workers > 0 else None

        # Create a fresh iterator
        self._worker_restarted = True
        self.iterator = super().__iter__()
        self._last_batch_time = time.time()
        self._stall_count = 0

        return True

    def check_worker_health(self):
        """
        Check if workers are stalled and attempt resuscitation if needed.
        Returns True if workers are healthy or were successfully restarted.
        """
        current_time = time.time()
        elapsed = current_time - self._last_batch_time

        # If we've gone too long without a batch, attempt worker resuscitation
        if elapsed > self.worker_restart_threshold and self.num_workers > 0:
            return self._resuscitate_workers()

        return elapsed < self.worker_restart_threshold

    def __iter__(self):
        """
        Efficient iteration with worker stall detection and resuscitation.
        """
        for _ in range(self.total_length):
            start_time = time.time()

            # Check if workers are stalled before attempting to get next batch
            elapsed_since_last = time.time() - self._last_batch_time
            if elapsed_since_last > self.worker_restart_threshold and self.num_workers > 0:
                self._resuscitate_workers()

            try:
                batch = next(self.iterator)
                self.current_batch += 1

                # Track batch timing for stall detection
                elapsed = time.time() - start_time
                if elapsed > 5.0:  # Consider a batch slow if >5 seconds
                    self._stall_count += 1
                    if self._stall_count >= 3:
                        warnings.warn(f"DataLoader {self.dataloader_id} experiencing slow batches ({elapsed:.1f}s).")
                else:
                    self._stall_count = max(0, self._stall_count - 1)  # Decrease counter for normal batches

                self._last_batch_time = time.time()
                yield batch

            except Exception as e:
                # Check if this was just after a worker restart
                if self._worker_restarted:
                    self._worker_restarted = False
                    warnings.warn(f"Error after worker restart: {str(e)}. Trying one more time...")

                    # Try one more time with a fresh restart
                    self._resuscitate_workers()
                    try:
                        batch = next(self.iterator)
                        self.current_batch += 1
                        self._last_batch_time = time.time()
                        yield batch
                        continue
                    except Exception as e2:
                        warnings.warn(f"Second worker restart failed: {str(e2)}")

                # Provide helpful diagnostic info for worker failures
                elapsed = time.time() - self._last_batch_time
                warnings.warn(f"DataLoader {self.dataloader_id} worker error after {elapsed:.1f}s: {str(e)}")
                raise

    def __len__(self):
        """Return the number of batches in an epoch."""
        return self.total_length


class PooledStatefulDataLoader(PooledDataLoader):
    """
    A drop-in replacement for StatefulDataLoader that uses a worker pool.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0,
                 pin_memory=False, drop_last=False, seed=None, **kwargs):

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            seed=seed,
            **kwargs
        )

        # Additional state tracking for StatefulDataLoader compatibility
        self._prev_rng_state = None
        self.shuffle = shuffle
        self.seed = seed
        self._rng = np.random.RandomState(seed) if seed is not None else None

    def get_state(self) -> Dict[str, Any]:
        """Return the current state of the dataloader."""
        return {
            'batch': self.current_batch,
            'prev_rng_state': self._prev_rng_state
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the dataloader state."""
        self.current_batch = state['batch']

        if state['prev_rng_state'] is not None and self._rng is not None:
            self._prev_rng_state = state['prev_rng_state']
            self._rng.set_state(state['prev_rng_state'])

        # Skip to correct position in the iterator
        target_batch = self.current_batch % self.total_length
        for _ in range(target_batch):
            _ = next(self.iterator)


class StatefulLongCycler:
    """
    Cycles through trainloaders until the loader with largest size is exhausted.
    Maintains state for fault-tolerant training while keeping original efficiency.
    Works with pooled dataloaders.
    """

    def __init__(self, loaders, seed: Optional[int] = 0):
        self.loaders = loaders
        self.max_batches = max([len(loader) for loader in self.loaders.values()])
        self.keys = list(self.loaders.keys())

        # Lightweight state tracking
        self._rng = np.random.RandomState(seed) if seed is not None else None
        self.current_cycle = 0
        self.current_position = 0
        self._prev_rng_state = None

        # Create efficient cyclers
        self._setup_cyclers()

    def _setup_cyclers(self):
        """Setup the cycling iterators efficiently."""
        self.loader_cyclers = {key: iter(loader) for key, loader in self.loaders.items()}

        # Prepare shuffled keys if using RNG
        if self._rng is not None:
            self._prev_rng_state = self._rng.get_state()
            self.keys = list(self.loaders.keys())
            self._rng.shuffle(self.keys)

        self.key_cycler = iter(self.keys)
        self.current_key_idx = 0

    def get_state(self):
        """Return the current state of the cycler and its dataloaders."""
        return {
            'current_cycle': self.current_cycle,
            'current_position': self.current_position,
            'prev_rng_state': self._prev_rng_state,
            'dataloader_states': {
                key: loader.get_state() if hasattr(loader, 'get_state') else None
                for key, loader in self.loaders.items()
            }
        }

    def set_state(self, state):
        """Restore the cycler state and its dataloaders."""
        self.current_cycle = state['current_cycle']
        self.current_position = state['current_position']

        if self._rng is not None and state['prev_rng_state'] is not None:
            self._prev_rng_state = state['prev_rng_state']
            self._rng.set_state(state['prev_rng_state'])

        # Restore dataloader states
        for key, loader_state in state['dataloader_states'].items():
            if key in self.loaders and hasattr(self.loaders[key], 'set_state') and loader_state is not None:
                self.loaders[key].set_state(loader_state)

        # Re-setup cyclers preserving the current state
        self._setup_cyclers()

        # Advance to current position
        for _ in range(self.current_position):
            try:
                next(self.key_cycler)
                self.current_key_idx = (self.current_key_idx + 1) % len(self.keys)
            except StopIteration:
                self.key_cycler = iter(self.keys)
                self.current_key_idx = 0

    def __iter__(self):
        # Track progress
        total_iterations = 0
        total_expected = len(self.keys) * self.max_batches

        while total_iterations < total_expected:
            try:
                key = next(self.key_cycler)
                self.current_key_idx = (self.current_key_idx + 1) % len(self.keys)
            except StopIteration:
                # Regenerate key iterator when exhausted
                if self._rng is not None:
                    self._prev_rng_state = self._rng.get_state()
                    self._rng.shuffle(self.keys)

                self.key_cycler = iter(self.keys)
                key = next(self.key_cycler)
                self.current_key_idx = 0
                self.current_cycle += 1
                print(f"Cycle {self.current_cycle} completed")

            # Get batch from this loader
            try:
                batch = next(self.loader_cyclers[key])
            except StopIteration:
                # Recreate this specific iterator only
                self.loader_cyclers[key] = iter(self.loaders[key])
                batch = next(self.loader_cyclers[key])

            # Update position
            self.current_position = self.current_key_idx
            total_iterations += 1

            yield key, batch

        # Reset state when done
        self.current_cycle = 0
        self.current_position = 0

        # Reset cyclers for next iteration
        self._setup_cyclers()

    def __len__(self):
        return len(self.loaders) * self.max_batches


class WorkerPoolMonitor:
    """
    Monitor for the worker pool to track and report on worker health and allocation.
    """

    def __init__(self, pool_manager=None):
        """
        Initialize the worker pool monitor.

        Args:
            pool_manager: The WorkerPoolManager instance to monitor
        """
        self.pool_manager = pool_manager
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        self.worker_stats = defaultdict(list)
        self.last_check_time = time.time()
        self.check_interval = 10  # seconds

    def get_worker_pool_status(self):
        """
        Get the current status of the worker pool.

        Returns:
            dict: Status information about the worker pool
        """
        if self.pool_manager is None:
            # Try to get pool manager from any PooledDataLoader class
            import importlib
            try:
                # Get the pool manager from any existing PooledDataLoader class
                for name, obj in globals().items():
                    if hasattr(obj, '_pool_manager') and obj._pool_manager is not None:
                        self.pool_manager = obj._pool_manager
                        break
            except:
                return {"error": "No pool manager found"}

        if self.pool_manager is None:
            return {"error": "No pool manager found"}

        with self.pool_manager.lock:
            # Get basic pool stats
            total_workers = self.pool_manager.max_workers
            assigned_workers = sum(len(workers) for workers in self.pool_manager.worker_assignments.values())
            free_workers = len(self.pool_manager.available_workers)

            # Get dataloader allocations
            loader_allocations = {
                loader_id: {
                    "workers": workers,
                    "count": len(workers)
                }
                for loader_id, workers in self.pool_manager.worker_assignments.items()
            }

            # Get per-worker information
            worker_info = {}
            for loader_id, workers in self.pool_manager.worker_assignments.items():
                for worker_id in workers:
                    if worker_id not in worker_info:
                        worker_info[worker_id] = {"loaders": []}
                    worker_info[worker_id]["loaders"].append(loader_id)

            return {
                "total_workers": total_workers,
                "assigned_workers": assigned_workers,
                "free_workers": free_workers,
                "loader_allocations": loader_allocations,
                "worker_info": worker_info
            }

    def get_worker_processes(self):
        """
        Identify PyTorch DataLoader worker processes.

        Returns:
            dict: Information about worker processes
        """
        try:
            # Get current process
            current_proc = psutil.Process()

            # Get all child processes
            children = current_proc.children(recursive=True)

            # Filter for likely worker processes
            worker_procs = []
            for proc in children:
                try:
                    # Check if process is likely a worker
                    cmd = " ".join(proc.cmdline()).lower()
                    if "python" in cmd and ("worker" in cmd or proc.name().startswith("python")):
                        # Get process info
                        worker_procs.append({
                            "pid": proc.pid,
                            "cpu_percent": proc.cpu_percent(),
                            "memory_percent": proc.memory_percent(),
                            "status": proc.status(),
                            "create_time": proc.create_time(),
                            "num_threads": proc.num_threads(),
                            "nice": proc.nice(),
                            "worker_id": os.environ.get('WORKER_ID', 'unknown') if hasattr(proc,
                                                                                           'environ') else 'unknown'
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return worker_procs
        except Exception as e:
            return {"error": str(e)}

    def start_monitoring(self, interval=10):
        """
        Start monitoring the worker pool in a background thread.

        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            print("Monitoring already active")
            return

        self.check_interval = interval
        self.stop_event.clear()

        def monitor_func():
            while not self.stop_event.is_set():
                try:
                    # Gather current stats
                    status = self.get_worker_pool_status()
                    if "error" not in status:
                        self.worker_stats["timestamp"].append(time.time())
                        self.worker_stats["assigned_workers"].append(status["assigned_workers"])
                        self.worker_stats["free_workers"].append(status["free_workers"])

                        # Get process stats
                        procs = self.get_worker_processes()
                        if isinstance(procs, list):
                            avg_cpu = np.mean([p["cpu_percent"] for p in procs]) if procs else 0
                            self.worker_stats["avg_cpu_percent"].append(avg_cpu)
                            self.worker_stats["worker_count"].append(len(procs))

                    self.last_check_time = time.time()
                except Exception as e:
                    print(f"Error in monitoring: {str(e)}")

                # Wait for next check interval
                self.stop_event.wait(self.check_interval)

        self.monitoring_thread = threading.Thread(target=monitor_func, daemon=True)
        self.monitoring_thread.start()
        print(f"Worker pool monitoring started (interval: {interval}s)")

    def stop_monitoring(self):
        """Stop the monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_event.set()
            self.monitoring_thread.join(timeout=1)
            print("Worker pool monitoring stopped")

    def get_monitoring_stats(self, as_dataframe=True):
        """
        Get collected monitoring statistics.

        Args:
            as_dataframe: If True, return as pandas DataFrame

        Returns:
            DataFrame or dict: Monitoring statistics
        """
        if as_dataframe:
            try:
                df = pd.DataFrame(self.worker_stats)
                if not df.empty:
                    df["time"] = pd.to_datetime(df["timestamp"], unit="s")
                    df = df.set_index("time")
                    df = df.drop("timestamp", axis=1)
                return df
            except ImportError:
                print("Pandas not available, returning dict")
                return self.worker_stats
        return self.worker_stats

    def print_pool_status(self):
        """Print a formatted report of the current worker pool status."""
        status = self.get_worker_pool_status()

        if "error" in status:
            print(f"Error: {status['error']}")
            return

        print("\n===== WORKER POOL STATUS =====")
        print(f"Total workers: {status['total_workers']}")
        print(f"Assigned workers: {status['assigned_workers']}")
        print(f"Free workers: {status['free_workers']}")

        print("\n----- LOADER ALLOCATIONS -----")
        for loader_id, info in status['loader_allocations'].items():
            print(f"Loader {loader_id}: {info['count']} workers {info['workers']}")

        print("\n----- WORKER ASSIGNMENTS -----")
        for worker_id, info in status['worker_info'].items():
            print(f"Worker {worker_id}: assigned to loaders {info['loaders']}")

        print("\n----- WORKER PROCESSES -----")
        processes = self.get_worker_processes()
        if isinstance(processes, list):
            for i, proc in enumerate(processes[:10]):  # Show only the first 10
                print(f"Process {i}: PID={proc['pid']}, CPU={proc['cpu_percent']:.1f}%, "
                      f"Memory={proc['memory_percent']:.1f}%, Status={proc['status']}")

            if len(processes) > 10:
                print(f"... and {len(processes) - 10} more")
        else:
            print(f"Error getting processes: {processes.get('error', 'Unknown error')}")

        print("\n==============================")


def monitor_worker_pool(pool_manager=None, auto_start=True, interval=10):
    """
    Create and return a worker pool monitor.

    Args:
        pool_manager: The WorkerPoolManager instance to monitor
        auto_start: Whether to automatically start monitoring
        interval: Monitoring interval in seconds

    Returns:
        WorkerPoolMonitor: Monitor instance
    """
    monitor = WorkerPoolMonitor(pool_manager)

    if auto_start:
        monitor.start_monitoring(interval=interval)

    return monitor


# Helper function to add monitoring to PooledDataLoader
def add_monitoring_to_pooled_loader(loader_class, check_interval=30):
    """
    Add monitoring capabilities to a PooledDataLoader class.

    Args:
        loader_class: The PooledDataLoader class to enhance
        check_interval: How often to check worker health

    Returns:
        function: Function to get the monitor
    """
    # Create monitor if pool manager exists
    if hasattr(loader_class, '_pool_manager') and loader_class._pool_manager is not None:
        monitor = WorkerPoolMonitor(loader_class._pool_manager)

        # Patch the loader class to include monitoring
        if not hasattr(loader_class, 'get_pool_monitor'):
            @classmethod
            def get_pool_monitor(cls, start=True):
                nonlocal monitor
                if start and not monitor.monitoring_thread:
                    monitor.start_monitoring(interval=check_interval)
                return monitor

            loader_class.get_pool_monitor = get_pool_monitor

            print(f"Monitoring added to {loader_class.__name__}")
            return monitor
        else:
            print(f"{loader_class.__name__} already has monitoring")
            return loader_class.get_pool_monitor()
    else:
        print(f"No pool manager found for {loader_class.__name__}")
        return None