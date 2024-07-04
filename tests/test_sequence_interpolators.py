import pytest

from create_mock_data import create_sequence_data
from experanto.interpolators import Interpolator, SequenceInterpolator
import numpy as np



@pytest.mark.parametrize("sampling_rate", [10., 100., 3.])
@pytest.mark.parametrize("shifts_per_signal", [False, True])
def test_sequence_data(shifts_per_signal, sampling_rate):
    timestamps, data, shift = create_sequence_data(n_signals=10, shifts_per_signal=shifts_per_signal,
                                                   use_mem_mapped=False, t_end = 5.0, sampling_rate=sampling_rate)
    
    seq_interp = Interpolator.create("tests/sequence_data")
    assert isinstance(seq_interp, SequenceInterpolator), "Interpolation object is not a SequenceInterpolator"
    
    if not shifts_per_signal:
        interp, valid = seq_interp.interpolate(times=timestamps[:10])
        assert np.allclose(interp,  data[:10]), "Data does not match original data"
        assert np.all(valid), "All samples should be valid"
    else:
        delta_t = 1./sampling_rate
        times = timestamps[1:11] 

        interp, valid = seq_interp.interpolate(times=times)
        assert np.allclose(interp,  data[1:11]), "Data does not match original data"
        assert np.all(valid), "All samples should be valid"

        # Test phase shifts
        for i in range(data.shape[1]):
            for dt in np.linspace(-0.49, 0.49) * delta_t:
                shifted_times = times + shift[i] + dt
            
                interp, valid = seq_interp.interpolate(times=shifted_times)
                assert np.allclose(interp[:, i],  data[1:11, i]), f"Data at {dt} does not match original data"

            # Test phase shifts
            for dt in np.linspace(0.51, 1.49) * delta_t:
                shifted_times = times + shift[i] + dt
            
                interp, valid = seq_interp.interpolate(times=shifted_times)
                assert np.allclose(interp[:, i],  data[2:12, i]), f"Data at {dt} does not match original data"
