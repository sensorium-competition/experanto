import numpy as np
import pytest

from experanto.interpolators import Interpolator, TimeIntervalInterpolator

from .create_time_intervals_data import create_time_intervals_data


def test_time_interval_interpolation():
    with create_time_intervals_data() as timestamps:
        interp_obj = Interpolator.create("tests/time_interval_data")
        assert isinstance(interp_obj, TimeIntervalInterpolator)

        signal, valid = interp_obj.interpolate(timestamps)

        assert valid.shape == timestamps.shape
        assert signal.shape == (
            len(valid),
            3,
        )  # 3 labels: train, validation, test

        test_indices = [[400, 600]]
        for start, end in test_indices:
            assert signal[start:end, 0].all()

        train_indices = [[0, 200], [600, 800]]
        for start, end in train_indices:
            assert signal[start:end, 1].all()

        validation_indices = [[200, 400], [800, 1000]]
        for start, end in validation_indices:
            assert signal[start:end, 2].all()


def test_time_interval_interpolation_complement():
    with create_time_intervals_data() as timestamps:
        interp_obj = Interpolator.create("tests/time_interval_data")
        assert isinstance(interp_obj, TimeIntervalInterpolator)

        signal, valid = interp_obj.interpolate(timestamps)

        assert valid.shape == timestamps.shape
        assert signal.shape == (
            len(valid),
            3,
        )  # 3 labels: train, validation, test

        test_indices = [[0, 400], [600, 1000]]
        for start, end in test_indices:
            assert not signal[start:end, 0].any()

        train_indices = [[200, 600], [800, 1000]]
        for start, end in train_indices:
            assert not signal[start:end, 1].any()

        validation_indices = [[0, 200], [400, 800]]
        for start, end in validation_indices:
            assert not signal[start:end, 2].any()
