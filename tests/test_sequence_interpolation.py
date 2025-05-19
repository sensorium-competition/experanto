import numpy as np
import pytest
from .create_sequence_data import create_sequence_data

from experanto.interpolators import Interpolator, SequenceInterpolator


# Parameterize across different test cases for both interpolation and data tests
@pytest.mark.parametrize("shifts_per_signal", [False, True])
@pytest.mark.parametrize("sampling_rate", [10.0, 100.0, 3.0])
@pytest.mark.parametrize("use_mem_mapped", [False, True])
def test_linear_sequence_interpolation(
    shifts_per_signal, sampling_rate, use_mem_mapped
):
    with create_sequence_data(
        n_signals=10,
        shifts_per_signal=shifts_per_signal,
        use_mem_mapped=use_mem_mapped,
        t_end=5.0,
        sampling_rate=sampling_rate,
    ) as (timestamps, data, shift):
        seq_interp = Interpolator.create("tests/sequence_data")
        assert isinstance(
            seq_interp, SequenceInterpolator
        ), "Not a SequenceInterpolator"
        seq_interp.interpolation_mode = "linear"

        delta_t = 1.0 / sampling_rate
        idx = slice(1, 11)
        times = timestamps[idx] + 0.5 * delta_t

        if not shifts_per_signal:
            expected = np.zeros((len(times), data.shape[1]))
            for i in range(len(times)):
                t = times[i]

                left_idx = i
                right_idx = i + 1

                t1, t2 = timestamps[left_idx + 1], timestamps[right_idx + 1]

                for sig_idx in range(data.shape[1]):
                    y1, y2 = data[left_idx + 1, sig_idx], data[right_idx + 1, sig_idx]
                    expected[i, sig_idx] = y1 + ((t - t1) / (t2 - t1)) * (y2 - y1)

            interp, valid = seq_interp.interpolate(times=times)

            assert np.all(valid), "All samples should be valid"
            assert np.allclose(
                interp, expected, atol=1e-6
            ), "Linear interpolation mismatch for no shift"

        else:
            for sig_idx in range(data.shape[1]):
                shift_offset = shift[sig_idx]
                shifted_times = times + shift_offset

                expected = np.zeros(len(shifted_times))
                for i in range(len(shifted_times)):
                    t = shifted_times[i]

                    shifted_timestamps = timestamps + shift_offset
                    left_idx = np.searchsorted(shifted_timestamps, t, side="right") - 1
                    right_idx = left_idx + 1

                    if left_idx < 0 or right_idx >= len(timestamps):
                        continue

                    t1, t2 = shifted_timestamps[left_idx], shifted_timestamps[right_idx]
                    y1, y2 = data[left_idx, sig_idx], data[right_idx, sig_idx]

                    expected[i] = y1 + ((t - t1) / (t2 - t1)) * (y2 - y1)

                interp, valid = seq_interp.interpolate(times=shifted_times)

                valid_indices = np.where(valid)[0]
                if len(valid_indices) > 0:
                    assert np.allclose(
                        interp[valid_indices, sig_idx],
                        expected[valid_indices],
                        atol=1e-6,
                    ), f"Linear interpolation mismatch for signal {sig_idx}"


@pytest.mark.parametrize("shifts_per_signal", [False, True])
@pytest.mark.parametrize("sampling_rate", [10.0, 100.0, 3.0])
@pytest.mark.parametrize("use_mem_mapped", [False, True])
def test_sequence_data(shifts_per_signal, sampling_rate, use_mem_mapped):
    with create_sequence_data(
        n_signals=10,
        shifts_per_signal=shifts_per_signal,
        use_mem_mapped=use_mem_mapped,
        t_end=5.0,
        sampling_rate=sampling_rate,
    ) as (timestamps, data, shift):
        seq_interp = Interpolator.create("tests/sequence_data")
        assert isinstance(
            seq_interp, SequenceInterpolator
        ), "Interpolation object is not a SequenceInterpolator"

        if not shifts_per_signal:
            interp, valid = seq_interp.interpolate(
                times=timestamps[:10] + 1e-9
            )  # Add a small epsilon to avoid floating point errors
            assert np.allclose(interp, data[:10]), "Data does not match original data"
            assert np.all(valid), "All samples should be valid"
        else:
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
                    ), f"Data at {dt} does not match original data (use_mem_mapped={use_mem_mapped}, sampling_rate={sampling_rate}, shifts_per_signal={shifts_per_signal})"


if __name__ == "__main__":
    print("Running tests")
    # Run both tests
    test_sequence_data(True, 11.0, True)
    test_linear_sequence_interpolation(True, 11.0, True)
