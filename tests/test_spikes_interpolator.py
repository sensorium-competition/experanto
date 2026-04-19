import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from experanto.interpolators import SpikeInterpolator

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
        interp_kwargs={"interpolation_window": window, "interpolation_align": align},
    ) as (gt_spikes, interp):

        assert isinstance(interp, SpikeInterpolator), f"Expected interp to be of type SpikeInterpolator, got {type(interp).__name__}"

        # Query random times within the duration (avoiding edges for simplicity)
        rng = np.random.default_rng(12345)
        times = np.sort(rng.uniform(window, duration - window, 20))

        counts, valid = interp.interpolate(times, return_valid=True)

        # 1. Check Shapes
        # Note: valid is an array of INDICES (integers), not a boolean mask.
        assert len(valid) == len(times), "Expect all queried times to be valid indices"
        assert counts.shape == (len(times), n_neurons)

        # 2. Verify Exact Counts against Ground Truth
        for t_idx, t in enumerate(times):
            # Calculate expected window based on alignment
            if align == "center":
                t_start, t_end = t - window / 2, t + window / 2
            elif align == "left":
                t_start, t_end = t, t + window
            elif align == "right":
                t_start, t_end = t - window, t

            for n_idx, neuron_spikes in enumerate(gt_spikes):
                # Manual count: how many spikes fall in [t_start, t_end)
                expected_count = np.sum(
                    (neuron_spikes >= t_start) & (neuron_spikes < t_end)
                )

                assert (
                    counts[t_idx, n_idx] == expected_count
                ), f"Mismatch at time {t} for neuron {n_idx} with align={align}"


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
            "interpolation_align": "center",
        },
    ) as (gt_spikes, interp):

        # Create a dense grid of times to see smoothing effects
        times = np.linspace(1.0, 9.0, 100)

        counts = interp.interpolate(times)

        # Calculate raw counts manually
        raw_counts = np.zeros_like(counts)
        for t_idx, t in enumerate(times):
            t_start, t_end = t - window / 2, t + window / 2
            for n_idx, neuron_spikes in enumerate(gt_spikes):
                raw_counts[t_idx, n_idx] = np.sum(
                    (neuron_spikes >= t_start) & (neuron_spikes < t_end)
                )

        # Apply scipy gaussian filter to raw manual counts
        expected_smoothed = gaussian_filter1d(raw_counts, sigma=sigma, axis=0)

        # Check if the interpolator's output matches our manual smoothing
        # We use a small tolerance due to potential floating point differences
        assert np.allclose(
            counts, expected_smoothed, atol=1e-5
        ), "Smoothed output does not match expected Gaussian filtered counts"


def test_spikes_no_valid_times():
    """
    Test behavior when querying times outside the valid range.
    """
    with spikes_data_and_interpolator(
        data_kwargs={"duration": 5.0, "n_neurons": 2}
    ) as (_, interp):

        # Times completely outside [0, 5.0]
        times = np.array([-10.0, -5.0, 10.0, 20.0])

        counts, valid = interp.interpolate(times, return_valid=True)

        # Should return empty result and valid indices array should be empty
        assert counts.shape == (0, 2)
        assert len(valid) == 0, "Expected zero valid indices for out-of-bounds times"


def test_spikes_cache_data():
    """
    Verify that loading data to RAM (vs memmap) works correctly.
    """
    with spikes_data_and_interpolator(
        data_kwargs={"duration": 5.0, "n_neurons": 2},
        interp_kwargs={"cache_data": True},
    ) as (gt_spikes, interp):
        assert isinstance(interp, SpikeInterpolator), f"Expected interp to be of type SpikeInterpolator, but got {type(interp).__name__}"
        assert isinstance(interp.spikes, np.ndarray), f"Expected spikes to be np.ndarray, got {type(interp.spikes).__name__}"
        assert not isinstance(interp.spikes, np.memmap), f"Expected spikes not to be np.memmap"

        times = np.array([2.5])
        counts, valid = interp.interpolate(times, return_valid=True)

        assert len(valid) == 1, "Expected 1 valid index"
        assert valid[0] == 0, "Expected index 0 to be valid"
        assert counts.shape == (1, 2)


def test_spikes_invalid_alignment():
    """
    Test behavior when an invalid alignment is provided.
    """
    with pytest.raises(
        ValueError,
        match="Unknown alignment mode: invalid, should be 'center', 'left' or 'right'",
    ):
        with spikes_data_and_interpolator(
            data_kwargs={"duration": 5.0, "n_neurons": 2},
            interp_kwargs={"interpolation_align": "invalid"},
        ):
            pass


def test_memmap_loading():
    """
    Verify that loading data from a memmap file works correctly.
    """
    duration = 10.0
    n_neurons = 5
    rate = 20.0

    # 1. Test lazy loading (memmap)
    with spikes_data_and_interpolator(
        data_kwargs={
            "duration": duration,
            "n_neurons": n_neurons,
            "rate": rate,
            "use_mem_mapped": True,
        },
        interp_kwargs={"cache_data": False},
    ) as (gt_spikes, interp):
        assert isinstance(interp, SpikeInterpolator), f"Expected SpikeInterpolator, got {type(interp).__name__}"
        assert isinstance(interp.spikes, np.memmap), "Expected a memmap object"

        # Verify content matches ground truth
        # Since 'gt_spikes' is a list of arrays, let's reconstruct flat array
        flat_gt = np.concatenate(gt_spikes)
        np.testing.assert_allclose(interp.spikes, flat_gt)

    # 2. Test eager loading (cache_data=True) from memmap source
    with spikes_data_and_interpolator(
        data_kwargs={
            "duration": duration,
            "n_neurons": n_neurons,
            "rate": rate,
            "use_mem_mapped": True,
        },
        interp_kwargs={"cache_data": True},
    ) as (gt_spikes, interp):
        assert isinstance(interp, SpikeInterpolator), f"Expected SpikeInterpolator, got {type(interp).__name__}"
        assert isinstance(
            interp.spikes, np.ndarray
        ), "Expected a numpy array (loaded into RAM)"
        assert not isinstance(interp.spikes, np.memmap), "Should not be a memmap"

        flat_gt = np.concatenate(gt_spikes)
        np.testing.assert_allclose(interp.spikes, flat_gt)


def test_spikes_neuron_indices_filtering():
    with spikes_data_and_interpolator(data_kwargs={"n_neurons": 5}) as (
        gt_spikes,
        interp,
    ):

        interp = SpikeInterpolator(interp.root_folder, neuron_indices=[1, 3])

        assert interp.n_signals == 2

        # verify spikes correspond to selected neurons
        selected = [gt_spikes[1], gt_spikes[3]]
        flat_selected = np.concatenate(selected)

        np.testing.assert_allclose(interp.spikes, flat_selected)


def test_spikes_neuron_ids_indices_mismatch():
    with spikes_data_and_interpolator(data_kwargs={"n_neurons": 5}) as (_, interp):

        meta_folder = interp.root_folder / "meta"
        meta_folder.mkdir(parents=True, exist_ok=True)
        np.save(meta_folder / "unit_ids.npy", np.arange(5))

        with pytest.raises(ValueError):
            SpikeInterpolator(
                interp.root_folder,
                neuron_ids=[0, 1],
                neuron_indices=[2, 3],
            )


def test_spikes_empty_selection():
    with spikes_data_and_interpolator(data_kwargs={"n_neurons": 5}) as (_, interp):

        interp = SpikeInterpolator(interp.root_folder, neuron_indices=[])

        assert interp.n_signals == 0
        assert interp.spikes.shape == (0,)
