import pytest

from create_mock_data import create_sequence_data
from experanto.interpolators import Interpolator, SequenceInterpolator
import numpy as np




def test_sequence_data():
    timestamps, data, shift = create_sequence_data(n_signals=10, shifts_per_signal=False, use_mem_mapped=False, t_end = 5.0)
    
    seq_interp = Interpolator.create("tests/sequence_data")
    assert isinstance(seq_interp, SequenceInterpolator), "Interpolation object is not a SequenceInterpolator"
    
    interp, valid = seq_interp.interpolate(times=timestamps[:10])
    assert np.allclose(interp,  data[:10]), "Data does not match original data"
    assert np.all(valid), "All samples should be valid"