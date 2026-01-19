import shutil
from contextlib import closing, contextmanager
from pathlib import Path

import numpy as np
import yaml

from experanto.interpolators import Interpolator

SPIKES_ROOT = Path("tests/spikes_data")


@contextmanager
def create_spikes_data(duration=100.0, n_neurons=50, rate=20.0):
    """
    Context manager to create temporary spike data for testing.
    
    Yields:
        list[np.ndarray]: A list containing the sorted spike times for each neuron.
                          This serves as the 'Ground Truth' for verification.
    """
    try:
        SPIKES_ROOT.mkdir(parents=True, exist_ok=True)
        
        all_spikes_list = []
        indices = [0]
        
        # Generate random sorted spikes for each neuron
        for _ in range(n_neurons):
            n_spikes = int(duration * rate)
            # Uniform distribution of spikes, sorted
            spikes = np.sort(np.random.uniform(0, duration, n_spikes))
            all_spikes_list.append(spikes)
            indices.append(indices[-1] + len(spikes))
            
        # Flatten and save to binary file (replicating real data structure)
        flat_spikes = np.concatenate(all_spikes_list)
        flat_spikes.tofile(SPIKES_ROOT / "spikes.npy")
        
        meta = {
            "modality": "spikes",
            "n_signals": n_neurons,
            "spike_indices": indices,
            "start_time": 0.0,
            "end_time": duration,
            "sampling_rate": 1000.0,  # Arbitrary for spikes, but required by some loaders
        }
        
        with open(SPIKES_ROOT / "meta.yml", "w") as f:
            yaml.dump(meta, f)
            
        yield all_spikes_list

    finally:
        if SPIKES_ROOT.exists():
            shutil.rmtree(SPIKES_ROOT)


@contextmanager
def spikes_data_and_interpolator(data_kwargs=None, interp_kwargs=None):
    """
    Context manager that yields both the ground truth data and an active Interpolator instance.
    """
    data_kwargs = data_kwargs or {}
    interp_kwargs = interp_kwargs or {}
    
    with create_spikes_data(**data_kwargs) as gt_spikes:
        # Interpolator.create detects 'modality: spikes' from meta.yml and returns SpikesInterpolator
        with closing(Interpolator.create("tests/spikes_data", **interp_kwargs)) as interp:
            yield gt_spikes, interp