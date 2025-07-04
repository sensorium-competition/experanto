import numpy as np
import pytest

from experanto.interpolators import Interpolator, TimeIntervalInterpolator

from .create_time_intervals_data import create_time_intervals_data


def test_time_interval_interpolation():
    with create_time_intervals_data() as timestamps:
        interp_obj = Interpolator.create("tests/time_interval_data")
        assert isinstance(interp_obj, TimeIntervalInterpolator)

        signal = interp_obj.interpolate(timestamps)

        test_indices = [[0, 200]]
        for start, end in test_indices:
            assert signal[start:end, 0].all()

        train_indices = [[200, 400], [600, 800]]
        for start, end in train_indices:
            assert signal[start:end, 1].all()

        validation_indices = [[400, 600], [800, 1000]]
        for start, end in validation_indices:
            assert signal[start:end, 2].all()


def test_time_interval_interpolation_complement():
    with create_time_intervals_data() as timestamps:
        interp_obj = Interpolator.create("tests/time_interval_data")
        assert isinstance(interp_obj, TimeIntervalInterpolator)

        signal = interp_obj.interpolate(timestamps)

        test_indices = [[200, 1000]]
        for start, end in test_indices:
            assert not signal[start:end, 0].any()

        train_indices = [[0, 200], [400, 600], [800, 1000]]
        for start, end in train_indices:
            assert not signal[start:end, 1].any()

        validation_indices = [[0, 400], [600, 800]]
        for start, end in validation_indices:
            assert not signal[start:end, 2].any()


def test_time_interval_interpolation_overlap2():
    with create_time_intervals_data(type="overlap2") as timestamps:
        interp_obj = Interpolator.create("tests/time_interval_data")
        assert isinstance(interp_obj, TimeIntervalInterpolator)

        signal = interp_obj.interpolate(timestamps)

        test_indices = [[0, 250]]
        for start, end in test_indices:
            assert signal[start:end, 0].all()

        train_indices = [[200, 400], [600, 800]]
        for start, end in train_indices:
            assert signal[start:end, 1].all()

        validation_indices = [[400, 600], [800, 1000]]
        for start, end in validation_indices:
            assert signal[start:end, 2].all()


def test_time_interval_interpolation_overlap3():
    with create_time_intervals_data(type="overlap3") as timestamps:
        interp_obj = Interpolator.create("tests/time_interval_data")
        assert isinstance(interp_obj, TimeIntervalInterpolator)

        signal = interp_obj.interpolate(timestamps)

        test_indices = [[0, 250]]
        for start, end in test_indices:
            assert signal[start:end, 0].all()

        train_indices = [[178, 403], [557, 823]]
        for start, end in train_indices:
            assert signal[start:end, 1].all()

        validation_indices = [[375, 601], [789, 1000]]
        for start, end in validation_indices:
            assert signal[start:end, 2].all()


def test_time_interval_interpolation_gap():
    with create_time_intervals_data(type="gap") as timestamps:
        interp_obj = Interpolator.create("tests/time_interval_data")
        assert isinstance(interp_obj, TimeIntervalInterpolator)

        signal = interp_obj.interpolate(timestamps)

        test_indices = [[0, 180]]
        for start, end in test_indices:
            assert signal[start:end, 0].all()

        train_indices = [[200, 400], [632, 800]]
        for start, end in train_indices:
            assert signal[start:end, 1].all()

        validation_indices = [[420, 600], [827, 1000]]
        for start, end in validation_indices:
            assert signal[start:end, 2].all()


def test_time_interval_interpolation_gap_and_overlap():
    with create_time_intervals_data(type="gap_and_overlap") as timestamps:
        interp_obj = Interpolator.create("tests/time_interval_data")
        assert isinstance(interp_obj, TimeIntervalInterpolator)

        signal = interp_obj.interpolate(timestamps)

        test_indices = [[0, 250]]
        for start, end in test_indices:
            assert signal[start:end, 0].all()

        train_indices = [[200, 390], [600, 800]]
        for start, end in train_indices:
            assert signal[start:end, 1].all()

        validation_indices = [[400, 600], [800, 1000]]
        for start, end in validation_indices:
            assert signal[start:end, 2].all()


def test_time_interval_interpolation_nans():
    with create_time_intervals_data(type="nans") as timestamps:
        interp_obj = Interpolator.create("tests/time_interval_data")
        assert isinstance(interp_obj, TimeIntervalInterpolator)

        signal = interp_obj.interpolate(timestamps)

        test_indices = [[0, 100]]
        for start, end in test_indices:
            assert signal[start:end, 0].all()

        train_indices = [[100, 250], [450, 650]]
        for start, end in train_indices:
            assert signal[start:end, 1].all()

        validation_indices = [[250, 450], [650, 850]]
        for start, end in validation_indices:
            assert signal[start:end, 2].all()
