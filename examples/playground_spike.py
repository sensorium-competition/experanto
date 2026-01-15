print("Working...")

# from experanto.interpolators import SpikesInterpolator as Interpolator
from experanto.interpolators import Interpolator
from experanto.intervals import TimeInterval

import numpy as np
import yaml
import tempfile
import shutil
from pathlib import Path
from numba import njit, prange
import time


import numpy as np
from numba import njit, prange
from pathlib import Path

# --- THE ENGINE ---
# This function is compiled to machine code the first time it runs.
# 'parallel=True' allows it to use all CPU cores.
@njit(parallel=True, fastmath=True)
def fast_count_spikes(all_spikes, indices, window_starts, window_ends, out_counts):
    """
    all_spikes: 1D array (1.1GB)
    indices: 1D array (38k) - start/end of each neuron in all_spikes
    window_starts: 1D array (BatchSize) - start times for the query
    window_ends: 1D array (BatchSize)
    out_counts: 2D array (BatchSize, N_Neurons) - Result placeholder
    """
    n_batch = len(window_starts)
    n_neurons = len(indices) - 1
    
    # We parallelize the OUTER loop (the batch). 
    # Or we can parallelize the NEURON loop. 
    # Since N_Neurons (38k) > Batch (e.g. 128), parallelizing neurons is better.
    
    for i in prange(n_neurons):
        # 1. Get the slice for this neuron
        # (This is zero-copy in Numba)
        idx_start = indices[i]
        idx_end = indices[i+1]
        neuron_spikes = all_spikes[idx_start:idx_end]
        
        # 2. Check all time windows for this neuron
        # Since spikes are sorted, we use binary search
        for b in range(n_batch):
            t0 = window_starts[b]
            t1 = window_ends[b]
            
            # Binary Search
            # np.searchsorted is supported natively in Numba
            # It finds where t0 and t1 would fit in the sorted array
            c_start = np.searchsorted(neuron_spikes, t0)
            c_end = np.searchsorted(neuron_spikes, t1)
            
            out_counts[b, i] = c_end - c_start

# --- THE CLASS ---
class SpikesInterpolator(Interpolator):
    def __init__(
            self, 
            root_folder: str,
            cache_data: bool = False,
            interpolation_window: float = 0.3,
            interpolation_align: str = "center",
            load_to_ram: bool = False,
            ):
        super().__init__(root_folder)

        meta = self.load_meta()

        self.start_time = meta.get("start_time", 0)
        self.end_time = meta.get("end_time", np.inf)
        self.valid_interval = TimeInterval(self.start_time, self.end_time)

        self.cache_trials = cache_data
        self.interpolation_window = interpolation_window
        self.interpolation_align = interpolation_align

        # Use self.root_folder, defined in the base class
        self.dat_path = self.root_folder / "spikes.npy"
        
        # Ensure indices are typed correctly for Numba
        self.indices = np.array(meta["spike_indices"]).astype(np.int64)
        self.n_signals = len(self.indices) - 1

        if load_to_ram:
            print("Loading spikes to RAM...")
            self.spikes = np.fromfile(self.dat_path, dtype='float64')
        else:
            self.spikes = np.memmap(self.dat_path, dtype='float64', mode='r')

    def interpolate(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # 1. Filter for valid times
        valid = self.valid_times(times)
        valid_times = times[valid]
        
        # Handle edge case where no times are valid
        if len(valid_times) == 0:
            return np.empty((0, self.n_signals)), valid

        valid_times += 1e-4

        # 2. Prepare boundaries
        if self.interpolation_align == "center":
            starts = valid_times - self.interpolation_window / 2
            ends   = valid_times + self.interpolation_window / 2
        elif self.interpolation_align == "left":
            starts = valid_times
            ends   = valid_times + self.interpolation_window
        elif self.interpolation_align == "right":
            starts = valid_times - self.interpolation_window
            ends   = valid_times
        else:
            raise ValueError(f"Unknown alignment mode: {self.interpolation_align}")

        # 3. Prepare Output
        # SIZE FIX: Only allocate for the VALID batch size
        batch_size = len(valid_times)
        counts = np.zeros((batch_size, self.n_signals), dtype=np.float64)
        
        # 4. Call Numba Engine
        fast_count_spikes(self.spikes, self.indices, starts, ends, counts)
        
        # SIGNATURE FIX: Return both data and the mask
        return counts, valid


print("Loaded SpikesInterpolator")

# ==========================================
# 2. DATA GENERATION & TESTING
# ==========================================

def create_dummy_data(folder_path, n_neurons=50, duration=100.0, rate=20):
    """Creates synthetic spike data and metadata."""
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    
    all_spikes = []
    indices = [0]
    
    for _ in range(n_neurons):
        n_spikes = int(duration * rate)
        # Random spikes sorted
        spikes = np.sort(np.random.uniform(0, duration, n_spikes))
        all_spikes.append(spikes)
        indices.append(indices[-1] + len(spikes))
        
    flat_spikes = np.concatenate(all_spikes)
    flat_spikes.tofile(folder / "spikes.npy")
    
    meta = {
        "modality": "spikes",
        "n_signals": n_neurons,
        "spike_indices": indices,
        "start_time": 0.0,
        "end_time": duration,
        "sampling_rate": 1000.0 
    }
    with open(folder / "meta.yml", "w") as f:
        yaml.dump(meta, f)
        
    return all_spikes, flat_spikes, indices


# ==========================================
# 3. TEST RUNNER (ALIGNMENT + SPEED)
# ==========================================

import sys


temp_dir = tempfile.mkdtemp()

try:
    duration=1000.0
    n_neurons=5000
    print(f"Test Environment: {temp_dir}")
    start = time.perf_counter()
    gt_spikes_list, gt_flat, gt_indices = create_dummy_data(temp_dir, n_neurons=n_neurons, duration=duration)
    print(gt_spikes_list[0][:10])
    print(gt_flat[:10])
    print(gt_indices[:10])
    end = time.perf_counter()
    print(f"Data creation time: {end - start:.2f} seconds")

    print(f"Size of gt_spikes_list in GB: {sys.getsizeof(gt_spikes_list)*1e-9:<10.2f}")
    print(f"Size of gt_flat in GB: {sys.getsizeof(gt_flat)*1e-9:<10.2f}")
    print(f"Size of gt_indices in GB: {sys.getsizeof(gt_indices)*1e-9:<10.2f}")
    
    # Define query times (randomly sampled)
    n_queries = 1000
    query_times = np.sort(np.random.uniform(1.0, duration, size=n_queries))
    window_size = 0.5
    
    alignments_to_test = ["center"]
    # alignments_to_test = ["center", "left", "right"]
    
    print(f"\n{'='*60}")
    print(f"STARTING TESTS: {n_queries} queries, Window={window_size}s")
    print(f"{'='*60}")

    for align in alignments_to_test:
        print(f"\n>>> Testing Alignment: {align.upper()}")
        
        # 1. Instantiate
        interpolator = SpikesInterpolator(
            temp_dir, 
            interpolation_window=window_size, 
            interpolation_align=align
        )
        
        # 2. Run & Time
        # Warmup (optional, to compile JIT)
        _ = interpolator.interpolate(query_times[:10])
        
        start_t = time.perf_counter()
        counts, valid = interpolator.interpolate(query_times)
        end_t = time.perf_counter()
        
        duration_sec = end_t - start_t
        speed_qps = n_queries / duration_sec
        
        print(f"Time: {duration_sec*1000:.2f} ms")
        print(f"Speed: {speed_qps:.0f} queries/sec")

        easdgs
        
        # 3. Verify Correctness
        print("Verifying accuracy...")
        errors = 0
        big_errors = 0
        total_checks = 0
        
        for t_idx, t in enumerate(query_times):
            # Adjust ground truth logic based on alignment
            if align == "center":
                t_start = t - window_size/2
                t_end   = t + window_size/2
            elif align == "left":
                t_start = t
                t_end   = t + window_size
            elif align == "right":
                t_start = t - window_size
                t_end   = t
                
            for n_idx in range(len(gt_spikes_list)):
                total_checks += 1
                neuron_spikes = gt_spikes_list[n_idx]
                # Ground truth count
                manual_count = np.sum((neuron_spikes >= t_start) & (neuron_spikes < t_end))
                
                numba_count = counts[t_idx, n_idx]
                
                if manual_count != numba_count:
                    # Print only the first error to avoid spamming
                    if errors%1000 == 0:
                        print(f"Mismatch at time {t:.2f}, neuron {n_idx}: Expected {manual_count}, got {numba_count}")
                    errors += 1
                    if abs(manual_count - numba_count) > 1:
                        big_errors += 1

        if errors == 0:
            print("SUCCESS: All counts match.")
        else:
            print(f"FAILED: {errors}/{total_checks} mismatches found. {errors/total_checks*100:.2f}%")
            print(f"Large Errors (>1 count difference): {big_errors}")

finally:
    shutil.rmtree(temp_dir)
    print(f"\n{'='*60}")
    print("Cleanup complete.")