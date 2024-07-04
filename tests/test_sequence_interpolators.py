import pytest

from create_mock_data import create_sequence_data
from experanto.interpolators import Interpolator, SequenceInterpolator
import numpy as np


@pytest.mark.parametrize("shifts_per_signal", [False, True])
@pytest.mark.parametrize("sampling_rate", [10.0, 100., 3.])
@pytest.mark.parametrize("use_mem_mapped", [False, True])
def test_sequence_data(shifts_per_signal, sampling_rate, use_mem_mapped):
    timestamps, data, shift = create_sequence_data(
        n_signals=10,
        shifts_per_signal=shifts_per_signal,
        use_mem_mapped=use_mem_mapped,
        t_end=5.0,
        sampling_rate=sampling_rate,
    )

    seq_interp = Interpolator.create("tests/sequence_data")
    assert isinstance(
        seq_interp, SequenceInterpolator
    ), "Interpolation object is not a SequenceInterpolator"

    if not shifts_per_signal:
        interp, valid = seq_interp.interpolate(times=timestamps[:10] + 1e-9) # Add a small epsilon to avoid floating point errors
        assert np.allclose(interp, data[:10]), "Data does not match original data"
        assert np.all(valid), "All samples should be valid"
    else:
        delta_t = 1.0 / sampling_rate
        idx = slice(1, 11)
        ret_idx = slice(0, 10) # because we floor indices and all shifts are positive, we need to shift the data by one
        times = timestamps[idx] + 1e-9 # Add a small epsilon to avoid floating point errors

        interp, valid = seq_interp.interpolate(times=times)
        assert np.allclose(interp, data[ret_idx]), "Data does not match original data"
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

if __name__=="__main__":
    test_sequence_data(True, 10.0, False)