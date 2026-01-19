import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from experanto.interpolators import SpikesInterpolator

from .create_spikes_data import spikes_data_and_interpolator


@pytest.mark.parametrize("align", ["center", "left", "right"])
def test_spikes_interpolation_accuracy(align):
    """
    Verifies that the Numba-optimized counting logic matches a simple Python
    implementation for all alignment modes.
    """
    duration = 10.0
    window = 0.5
    n_neurons = 10
    
    with spikes_data_and_interpolator(
        data_kwargs={"duration": duration, "n_neurons": n_neurons, "rate": 10.0},
        interp_kwargs={"interpolation_window": window, "interpolation_align": align}
    ) as (gt_spikes, interp):
        
        assert isinstance(interp, SpikesInterpolator)
        
        # Query random times within the duration (avoiding edges for simplicity)
        times = np.sort(np.random.uniform(window, duration - window, 20))
        
        counts, valid = interp.interpolate(times)
        
        # 1. Check Shapes
        # Note: valid is an array of INDICES (integers), not a boolean mask.
        assert len(valid) == len(times), "Expect all queried times to be valid indices"
        assert counts.shape == (len(times), n_neurons)
        
        # 2. Verify Exact Counts against Ground Truth
        for t_idx, t in enumerate(times):
            # Calculate expected window based on alignment
            if align == "center":
                t_start, t_end = t - window/2, t + window/2
            elif align == "left":
                t_start, t_end = t, t + window
            elif align == "right":
                t_start, t_end = t - window, t
            
            for n_idx, neuron_spikes in enumerate(gt_spikes):
                # Manual count: how many spikes fall in [t_start, t_end)
                expected_count = np.sum((neuron_spikes >= t_start) & (neuron_spikes < t_end))
                
                assert counts[t_idx, n_idx] == expected_count, \
                    f"Mismatch at time {t} for neuron {n_idx} with align={align}"


@pytest.mark.parametrize("sigma", [1.0, 3.0])
def test_spikes_smoothing(sigma):
    """
    Verifies that Gaussian smoothing is applied correctly along the time axis.
    """
    duration = 10.0
    window = 0.2
    n_neurons = 5
    
    with spikes_data_and_interpolator(
        data_kwargs={"duration": duration, "n_neurons": n_neurons, "rate": 50.0},
        interp_kwargs={
            "interpolation_window": window, 
            "smoothing_sigma": sigma,
            "interpolation_align": "center"
        }
    ) as (gt_spikes, interp):
        
        # Create a dense grid of times to see smoothing effects
        times = np.linspace(1.0, 9.0, 100)
        
        counts, _ = interp.interpolate(times)
        
        # Calculate raw counts manually
        raw_counts = np.zeros_like(counts)
        for t_idx, t in enumerate(times):
            t_start, t_end = t - window/2, t + window/2
            for n_idx, neuron_spikes in enumerate(gt_spikes):
                raw_counts[t_idx, n_idx] = np.sum(
                    (neuron_spikes >= t_start) & (neuron_spikes < t_end)
                )
        
        # Apply scipy gaussian filter to raw manual counts
        expected_smoothed = gaussian_filter1d(raw_counts, sigma=sigma, axis=0)
        
        # Check if the interpolator's output matches our manual smoothing
        # We use a small tolerance due to potential floating point differences
        assert np.allclose(counts, expected_smoothed, atol=1e-5), \
            "Smoothed output does not match expected Gaussian filtered counts"


def test_spikes_no_valid_times():
    """
    Test behavior when querying times outside the valid range.
    """
    with spikes_data_and_interpolator(
        data_kwargs={"duration": 5.0, "n_neurons": 2}
    ) as (_, interp):
        
        # Times completely outside [0, 5.0]
        times = np.array([-10.0, -5.0, 10.0, 20.0])
        
        counts, valid = interp.interpolate(times)
        
        # Should return empty result and valid indices array should be empty
        assert counts.shape == (0, 2)
        assert len(valid) == 0, "Expected zero valid indices for out-of-bounds times"


def test_spikes_load_to_ram():
    """
    Verify that loading data to RAM (vs memmap) works correctly.
    """
    with spikes_data_and_interpolator(
        data_kwargs={"duration": 5.0, "n_neurons": 2},
        interp_kwargs={"load_to_ram": True}
    ) as (gt_spikes, interp):
        
        assert isinstance(interp.spikes, np.ndarray)
        assert not isinstance(interp.spikes, np.memmap)
        
        times = np.array([2.5])
        counts, valid = interp.interpolate(times)
        
        assert len(valid) == 1, "Expected 1 valid index"
        assert valid[0] == 0, "Expected index 0 to be valid"
        assert counts.shape == (1, 2)