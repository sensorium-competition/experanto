import pytest
import numpy as np

from create_mock_data import sequence_data_and_interpolator
from experanto.interpolators import SequenceInterpolator, PhaseShiftedSequenceInterpolator


@pytest.mark.parametrize("sampling_rate", [3.0, 10.0, 100.0])
@pytest.mark.parametrize("use_mem_mapped", [False, True])
def test_nearest_neighbor_interpolation(sampling_rate, use_mem_mapped):
    with sequence_data_and_interpolator(
        n_signals=10,
        use_mem_mapped=use_mem_mapped,
        t_end=5.0,
        sampling_rate=sampling_rate,
    ) as (timestamps, data, _, seq_interp):
        assert isinstance(
            seq_interp, SequenceInterpolator
        ), "Interpolation object is not a SequenceInterpolator"
        assert not isinstance(
            seq_interp, PhaseShiftedSequenceInterpolator
        ), "Interpolation object is a PhaseShiftedSequenceInterpolator"

        interp, valid = seq_interp.interpolate(
            times=timestamps[:10] + 1e-9
        )  # Add a small epsilon to avoid floating point errors
        assert np.all(valid), "All samples should be valid"
        assert np.allclose(interp, data[:10]), "Nearest neighbor interpolation does not match expected data"
        assert valid.shape == (10,), f"Expected valid.shape == (10,), got {valid.shape}"
        assert interp.shape == (10, 10), f"Expected interp.shape == (10, 10), got {interp.shape}"


@pytest.mark.parametrize("sampling_rate", [3.0, 10.0, 100.0])
def test_nearest_neighbor_interpolation_with_inbetween_times(sampling_rate):
    t_end = 5.0
    with sequence_data_and_interpolator(
        n_signals=10,
        use_mem_mapped=True,
        t_end=t_end,
        sampling_rate=sampling_rate,
    ) as (timestamps, data, _, seq_interp):
        assert isinstance(
            seq_interp, SequenceInterpolator
        ), "Interpolation object is not a SequenceInterpolator"

        delta_t = 1.0 / sampling_rate

        # timestamps multiplied by 0.8 should be floored to the same timestamp
        interp, valid = seq_interp.interpolate(
            times=timestamps[:10] + 0.8 * delta_t
        )
        assert np.all(valid), "All samples should be valid"
        assert np.allclose(interp, data[:10]), "Nearest neighbor interpolation does not match expected data"

        # timestamps multiplied by 1.2 should be floored to the next timestamp
        interp, valid = seq_interp.interpolate(
            times=timestamps[:10] + 1.2 * delta_t
        )
        assert np.all(valid), "All samples should be valid"
        assert np.allclose(interp, data[1:11]), "Nearest neighbor interpolation does not match expected data"


@pytest.mark.parametrize("sampling_rate", [3.0, 10.0, 100.0])
@pytest.mark.parametrize("use_mem_mapped", [False, True])
def test_nearest_neighbor_interpolation_with_phase_shifts(sampling_rate, use_mem_mapped):
    with sequence_data_and_interpolator(
        n_signals=10,
        use_mem_mapped=use_mem_mapped,
        t_end=5.0,
        sampling_rate=sampling_rate,
        shifts_per_signal=True,
    ) as (timestamps, data, shift, seq_interp):
        assert isinstance(
            seq_interp, PhaseShiftedSequenceInterpolator
        ), "Interpolation object is not a PhaseShiftedSequenceInterpolator"

        delta_t = 1.0 / sampling_rate
        idx = slice(1, 11)
        ret_idx = slice(
            0, 10
        )  # because we floor indices and all shifts are positive, we need to shift the data by one
        times = (
            timestamps[idx] + 1e-9
        )  # Add a small epsilon to avoid floating point errors

        interp, valid = seq_interp.interpolate(times=times)
        assert np.allclose(
            interp, data[ret_idx]
        ), "Data does not match original data"
        assert np.all(valid), "All samples should be valid"
        assert valid.shape == (10,), f"Expected valid.shape == (10,), got {valid.shape}"
        assert interp.shape == (10, 10), f"Expected interp.shape == (10, 10), got {interp.shape}"

        # Test phase shifts
        for i in range(data.shape[1]):
            for dt in np.linspace(0, 0.99) * delta_t:
                shifted_times = times + shift[i] + dt

                interp, valid = seq_interp.interpolate(times=shifted_times)
                assert np.allclose(
                    interp[:, i], data[1:11, i]
                ), f"Data at {dt} does not match original data"

            # Test phase shifts
            for dt in np.linspace(1.0, 1.99) * delta_t:
                shifted_times = times + shift[i] + dt

                interp, valid = seq_interp.interpolate(times=shifted_times)
                assert np.allclose(
                    interp[:, i], data[2:12, i]
                ), f"Data at {dt} does not match original data (use_mem_mapped={use_mem_mapped}, sampling_rate={sampling_rate}, shifts_per_signal={True})"


@pytest.mark.parametrize("sampling_rate", [3.0, 10.0, 100.0])
@pytest.mark.parametrize("use_mem_mapped", [False, True])
def test_linear_interpolation(sampling_rate, use_mem_mapped):
    with sequence_data_and_interpolator(
        n_signals=10,
        use_mem_mapped=use_mem_mapped,
        t_end=5.0,
        sampling_rate=sampling_rate,
    ) as (timestamps, data, _, seq_interp):
        assert isinstance(seq_interp, SequenceInterpolator), "Not a SequenceInterpolator"
        seq_interp.interpolation_mode = 'linear'

        delta_t = 1.0 / sampling_rate
        idx = [i for i in range(1, 11)]
        times = timestamps[idx] + 0.5 * delta_t

        t1, t2 = timestamps[idx], timestamps[[id+1 for id in idx]]
        y1, y2 = data[idx], data[[id+1 for id in idx]]
        expected = y1 + ((times - t1) / (t2 - t1)) * (y2 - y1)

        interp, valid = seq_interp.interpolate(times=times)

        assert np.all(valid), "All samples should be valid"
        assert np.allclose(interp, expected, atol=1e-6), "Linear interpolation does not match expected data"
        assert valid.shape == (10,), f"Expected valid.shape == (10,), got {valid.shape}"
        assert interp.shape == (10, 10), f"Expected interp.shape == (10, 10), got {interp.shape}"


@pytest.mark.parametrize("sampling_rate", [3.0, 10.0, 100.0])
@pytest.mark.parametrize("use_mem_mapped", [False, True])
def test_linear_interpolation_with_phase_shifts(sampling_rate, use_mem_mapped):
    with sequence_data_and_interpolator(
        n_signals=10,
        use_mem_mapped=use_mem_mapped,
        t_end=5.0,
        sampling_rate=sampling_rate,
        shifts_per_signal=True,
    ) as (timestamps, data, shift, seq_interp):
        assert isinstance(seq_interp, PhaseShiftedSequenceInterpolator), "Not a PhaseShiftedSequenceInterpolator"
        seq_interp.interpolation_mode = 'linear'

        delta_t = 1.0 / sampling_rate
        idx = slice(1, 11)
        times = timestamps[idx] + 0.5 * delta_t

        for sig_idx in range(data.shape[1]):
                shift_offset = shift[sig_idx]
                shifted_times = times + shift_offset

                expected = np.zeros(len(shifted_times))
                for i in range(len(shifted_times)):
                    t = shifted_times[i]

                    shifted_timestamps = timestamps + shift_offset
                    left_idx = np.searchsorted(shifted_timestamps, t, side='right') - 1
                    right_idx = left_idx + 1
                    
                    if left_idx < 0 or right_idx >= len(timestamps):
                        continue
                        
                    t1, t2 = shifted_timestamps[left_idx], shifted_timestamps[right_idx]
                    y1, y2 = data[left_idx, sig_idx], data[right_idx, sig_idx]
                    
                    expected[i] = y1 + ((t - t1) / (t2 - t1)) * (y2 - y1)

                interp, valid = seq_interp.interpolate(times=shifted_times)

                valid_indices = np.where(valid)[0]
                if len(valid_indices) > 0:
                    assert np.allclose(interp[valid_indices, sig_idx], 
                                    expected[valid_indices], 
                                    atol=1e-6), f"Linear interpolation mismatch for signal {sig_idx}"


@pytest.mark.parametrize("sampling_rate", [3.0, 10.0, 100.0])
@pytest.mark.parametrize("interpolation_mode", ["nearest_neighbor", "linear"])
def test_interpolation_for_invalid_times(sampling_rate, interpolation_mode):
    with sequence_data_and_interpolator(
        n_signals=10,
        use_mem_mapped=True,
        t_end=5.0,
        sampling_rate=sampling_rate,
    ) as (timestamps, _, _, seq_interp):
        assert isinstance(
            seq_interp, SequenceInterpolator
        ), "Interpolation object is not a SequenceInterpolator"
        seq_interp.interpolation_mode = interpolation_mode

        _, valid = seq_interp.interpolate(
            times=np.array([-0.1, 0.1, 4.9, 5.0, 5.1])
        )
        assert np.all(valid == np.array([False, True, True, False, False])), "Validity does not match expected values"


@pytest.mark.parametrize("interpolation_mode", ["nearest_neighbor", "linear"])
@pytest.mark.parametrize("phase_shifts", [True, False])
def test_interpolation_for_empty_times(interpolation_mode, phase_shifts):
    with sequence_data_and_interpolator(
        n_signals=10,
        use_mem_mapped=True,
        t_end=5.0,
        sampling_rate=10.0,
        shifts_per_signal=phase_shifts
    ) as (timestamps, _, _, seq_interp):
        assert isinstance(
            seq_interp, SequenceInterpolator
        ), "Interpolation object is not a SequenceInterpolator"
        seq_interp.interpolation_mode = interpolation_mode

        interp, valid = seq_interp.interpolate(
            times=np.array([])
        )
        assert interp.shape[0] == 0, 'No data expected'
        assert valid.shape[0] == 0, 'No data expected'
                    

def test_interpolation_mode_not_implemented():
    with sequence_data_and_interpolator() as (_, _, _, seq_interp):
        seq_interp.interpolation_mode = "unsupported_mode"
        with pytest.raises(NotImplementedError):
            seq_interp.interpolate(np.array([0.0, 1.0, 2.0]))


if __name__ == "__main__":
    print("Running tests")
    pytest.main([__file__])