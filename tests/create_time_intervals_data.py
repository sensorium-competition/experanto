import shutil
from contextlib import closing, contextmanager
from pathlib import Path

import numpy as np
import yaml

from experanto.interpolators import Interpolator

TIME_INTERVAL_ROOT = Path("tests/time_interval_data")


@contextmanager
def create_time_interval_data(
    duration=10.0,
    sampling_rate=30.0,
    test_intervals=None,
    train_intervals=None,
    validation_intervals=None,
):
    """
    Create time interval test data with non-integer timestamps.

    Parameters
    ----------
    duration : float
        Total duration of the recording in seconds.
    sampling_rate : float
        Sampling rate in Hz for generating timestamps.
    test_intervals : list of [start, end], optional
        List of time ranges for test label. Defaults to [[0.0, 2.0]].
    train_intervals : list of [start, end], optional
        List of time ranges for train label. Defaults to [[2.0, 4.0], [6.0, 8.0]].
    validation_intervals : list of [start, end], optional
        List of time ranges for validation label. Defaults to [[4.0, 6.0], [8.0, 10.0]].

    Yields
    ------
    timestamps : np.ndarray
        Array of timestamp values.
    intervals_dict : dict
        Dictionary mapping label names to their interval arrays.
    """
    try:
        TIME_INTERVAL_ROOT.mkdir(parents=True, exist_ok=True)

        # Default intervals
        if test_intervals is None:
            test_intervals = [[0.0, 2.0]]
        if train_intervals is None:
            train_intervals = [[2.0, 4.0], [6.0, 8.0]]
        if validation_intervals is None:
            validation_intervals = [[4.0, 6.0], [8.0, duration]]

        # Generate non-integer timestamps
        n_samples = int(duration * sampling_rate)
        timestamps = np.linspace(0.0, duration, n_samples, endpoint=False)

        # Create metadata
        meta = {
            "labels": {
                "test": "test.npy",
                "train": "train.npy",
                "validation": "validation.npy",
            },
            "start_time": 0.0,
            "end_time": duration,
            "modality": "time_interval",
        }

        with open(TIME_INTERVAL_ROOT / "meta.yml", "w") as f:
            yaml.dump(meta, f)

        # Save interval files
        test_array = np.array(test_intervals, dtype=np.float64)
        train_array = np.array(train_intervals, dtype=np.float64)
        validation_array = np.array(validation_intervals, dtype=np.float64)

        np.save(TIME_INTERVAL_ROOT / "test.npy", test_array)
        np.save(TIME_INTERVAL_ROOT / "train.npy", train_array)
        np.save(TIME_INTERVAL_ROOT / "validation.npy", validation_array)

        intervals_dict = {
            "test": test_array,
            "train": train_array,
            "validation": validation_array,
        }

        yield timestamps, intervals_dict

    finally:
        shutil.rmtree(TIME_INTERVAL_ROOT, ignore_errors=True)


@contextmanager
def time_interval_data_and_interpolator(data_kwargs=None, interp_kwargs=None):
    """
    Create time interval test data and interpolator in one context manager.

    This follows the pattern used in sequence_data_and_interpolator for consistency.

    Parameters
    ----------
    data_kwargs : dict, optional
        Keyword arguments to pass to create_time_interval_data.
    interp_kwargs : dict, optional
        Keyword arguments to pass to Interpolator.create.

    Yields
    ------
    timestamps : np.ndarray
        Array of timestamp values.
    intervals_dict : dict
        Dictionary mapping label names to their interval arrays.
    interpolator : TimeIntervalInterpolator
        The interpolator object.
    """
    data_kwargs = data_kwargs or {}
    interp_kwargs = interp_kwargs or {}

    with create_time_interval_data(**data_kwargs) as (
        timestamps,
        intervals_dict,
    ):
        with closing(
            Interpolator.create("tests/time_interval_data", **interp_kwargs)
        ) as time_interp:
            yield timestamps, intervals_dict, time_interp
