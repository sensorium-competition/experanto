import numpy as np

from experanto.interpolators import TimeIntervalInterpolator

from .create_time_intervals_data import time_interval_data_and_interpolator


def assert_intervals_by_timestamps(
    signal,
    query_timestamps,
    expected_intervals,
    label_idx,
):
    """
    Assert that specific timestamp ranges are marked True/False in the signal.

    Parameters
    ----------
    signal : np.ndarray of bool, shape (n_times, n_labels)
        Boolean array output from TimeIntervalInterpolator.interpolate().
    query_timestamps : np.ndarray
        The actual timestamp values that were passed to interpolate().
    expected_intervals : list of [start, end]
        List of timestamp ranges [start, end) that should be True for this
        label. Timestamps outside these ranges should be False.
    label_idx : int
        Column index in signal corresponding to the label being tested.
    """
    # Create a mask for all timestamps that should be True
    expected_mask = np.zeros(len(query_timestamps), dtype=bool)
    for start_time, end_time in expected_intervals:
        interval_mask = (query_timestamps >= start_time) & (query_timestamps < end_time)
        expected_mask |= interval_mask

        # Assert that all timestamps in this interval are True
        assert signal[interval_mask, label_idx].all(), (
            f"Expected True for timestamps in [{start_time}, {end_time}) "
            f"for label {label_idx}"
        )

    # Assert that all timestamps outside the intervals are False
    outside_mask = ~expected_mask
    assert (~signal[outside_mask, label_idx]).all(), (
        f"Expected False for timestamps outside {expected_intervals} "
        f"for label {label_idx}"
    )


def run_interval_interpolation_test(timestamps, intervals_dict, time_interp):
    """
    Helper function to run standard time interval interpolation assertions.

    This reduces code duplication across test functions by encapsulating the
    common pattern of verifying interpolator type, shape, and interval matching.

    Parameters
    ----------
    timestamps : np.ndarray
        Array of timestamp values to interpolate.
    intervals_dict : dict
        Dictionary mapping label names ('test', 'train', 'validation') to
        their interval arrays.
    time_interp : TimeIntervalInterpolator
        The interpolator instance to test.
    """
    assert isinstance(time_interp, TimeIntervalInterpolator)

    signal = time_interp.interpolate(timestamps)

    assert signal.shape == (
        len(timestamps),
        3,
    ), f"Expected shape ({len(timestamps)}, 3), got {signal.shape}"

    assert_intervals_by_timestamps(
        signal, timestamps, intervals_dict["test"], label_idx=0
    )
    assert_intervals_by_timestamps(
        signal, timestamps, intervals_dict["train"], label_idx=1
    )
    assert_intervals_by_timestamps(
        signal, timestamps, intervals_dict["validation"], label_idx=2
    )


def test_time_interval_interpolation():
    """
    Test time interval interpolation with non-overlapping intervals.

    This test validates that TimeIntervalInterpolator correctly identifies which
    timestamps fall within labeled time intervals.

    The test creates three non-overlapping interval sets:
    - test: [0.0, 2.0)
    - train: [2.0, 4.0) and [6.0, 8.0)
    - validation: [4.0, 6.0) and [8.0, 10.0)

    Each timestamp should be marked True for exactly one label based on which
    interval it falls within.
    """
    with time_interval_data_and_interpolator(
        data_kwargs=dict(
            duration=10.0,
            sampling_rate=30.0,
        )
    ) as (timestamps, intervals_dict, time_interp):
        run_interval_interpolation_test(timestamps, intervals_dict, time_interp)


def test_time_interval_interpolation_overlap2():
    """
    Test time interval interpolation with overlapping intervals (case 2).

    This test validates handling of overlapping intervals where the test interval
    extends into the train interval range.

    Intervals:
    - test: [0.0, 2.5) - overlaps with train
    - train: [2.0, 4.0) and [6.0, 8.0)
    - validation: [4.0, 6.0) and [8.0, 10.0)

    The overlap region [2.0, 2.5) should be marked as True for both test and train.
    """
    with time_interval_data_and_interpolator(
        data_kwargs=dict(
            duration=10.0,
            sampling_rate=30.0,
            test_intervals=[[0.0, 2.5]],
            train_intervals=[[2.0, 4.0], [6.0, 8.0]],
            validation_intervals=[[4.0, 6.0], [8.0, 10.0]],
        )
    ) as (timestamps, intervals_dict, time_interp):
        run_interval_interpolation_test(timestamps, intervals_dict, time_interp)


def test_time_interval_interpolation_overlap3():
    """
    Test time interval interpolation with multiple overlapping intervals (case 3).

    This test validates handling of complex overlapping scenarios across all labels.

    Intervals:
    - test: [0.0, 2.5)
    - train: [1.78, 4.03) and [5.57, 8.23)
    - validation: [3.75, 6.01) and [7.89, 10.0)

    Multiple overlap regions exist between different label pairs.
    """
    with time_interval_data_and_interpolator(
        data_kwargs=dict(
            duration=10.0,
            sampling_rate=30.0,
            test_intervals=[[0.0, 2.5]],
            train_intervals=[[1.78, 4.03], [5.57, 8.23]],
            validation_intervals=[[3.75, 6.01], [7.89, 10.0]],
        )
    ) as (timestamps, intervals_dict, time_interp):
        run_interval_interpolation_test(timestamps, intervals_dict, time_interp)


def test_time_interval_interpolation_gap():
    """
    Test time interval interpolation with gaps between intervals.

    This test validates handling of gaps where some timestamps don't fall within
    any defined interval.

    Intervals:
    - test: [0.0, 1.8)
    - train: [2.0, 4.0) and [6.32, 8.0) - gap from [1.8, 2.0)
    - validation: [4.2, 6.0) and [8.27, 10.0) - gap from [4.0, 4.2)

    Timestamps in the gaps should be marked as False for all labels.
    """
    with time_interval_data_and_interpolator(
        data_kwargs=dict(
            duration=10.0,
            sampling_rate=30.0,
            test_intervals=[[0.0, 1.8]],
            train_intervals=[[2.0, 4.0], [6.32, 8.0]],
            validation_intervals=[[4.2, 6.0], [8.27, 10.0]],
        )
    ) as (timestamps, intervals_dict, time_interp):
        run_interval_interpolation_test(timestamps, intervals_dict, time_interp)


def test_time_interval_interpolation_gap_and_overlap():
    """
    Test time interval interpolation with both gaps and overlaps.

    This test validates handling of scenarios with both overlapping intervals
    and gaps between intervals.

    Intervals:
    - test: [0.0, 2.5) - overlaps with train
    - train: [2.0, 3.9) and [6.0, 8.0)
    - validation: [4.0, 6.0) and [8.0, 10.0)

    Gap from [3.9, 4.0) and overlap from [2.0, 2.5).
    """
    with time_interval_data_and_interpolator(
        data_kwargs=dict(
            duration=10.0,
            sampling_rate=30.0,
            test_intervals=[[0.0, 2.5]],
            train_intervals=[[2.0, 3.9], [6.0, 8.0]],
            validation_intervals=[[4.0, 6.0], [8.0, 10.0]],
        )
    ) as (timestamps, intervals_dict, time_interp):
        run_interval_interpolation_test(timestamps, intervals_dict, time_interp)


def test_time_interval_interpolation_nans():
    """
    Test handling of NaN values in timestamps.

    This test validates that the interpolator correctly filters out NaN timestamps
    and only returns results for valid timestamps. When timestamps contain NaNs,
    the interpolator should:
    1. Filter out NaN values via valid_times()
    2. Return a shorter output array containing only valid timestamps
    3. Correctly mark which valid timestamps fall within each interval

    The NaN values in timestamps represent missing or invalid data points that
    should be excluded. Example: if there's an issue with the eye tracker and we have NaNs from 1.0 to 2.5 seconds, those timestamps should be ignored. Consequently, if one of the test intervals overlap with this NaN region, e.g. it's an interval from 0.0 to 1.5 seconds, the valid timestamps from 0.0 to 1.0 second will be marked as True. If one of the train intervals overlap with the NaN region, e.g. from 2.0 to 4.0 seconds, then only the timestamps from 2.5 to 4.0 seconds will be marked as True. IMPORTANT: the interpolated signal will have fewer rows than the input timestamps due to NaN filtering.
    """
    duration = 10.0
    sampling_rate = 30.0
    n_samples = int(duration * sampling_rate)
    timestamps = np.linspace(0.0, duration, n_samples, endpoint=False)

    # Introduce NaNs in the middle portion (roughly 1.0 to 2.5 seconds)
    nan_start_idx = int(1.0 * sampling_rate)
    nan_end_idx = int(2.5 * sampling_rate)
    timestamps[nan_start_idx:nan_end_idx] = np.nan

    with time_interval_data_and_interpolator(
        data_kwargs=dict(
            duration=duration,
            sampling_rate=sampling_rate,
            test_intervals=[[0.0, 1.5]],
            train_intervals=[[2.0, 4.0], [6.0, 8.0]],
            validation_intervals=[[4.0, 6.0], [8.0, 10.0]],
        )
    ) as (_, intervals_dict, time_interp):
        assert isinstance(time_interp, TimeIntervalInterpolator)

        # Interpolate with NaN-containing timestamps
        signal = time_interp.interpolate(timestamps)

        # Should return fewer rows than input (NaNs filtered out)
        expected_valid_count = n_samples - (nan_end_idx - nan_start_idx)
        assert signal.shape == (
            expected_valid_count,
            3,
        ), f"Expected shape ({expected_valid_count}, 3), got {signal.shape}"

        # Get the valid (non-NaN) timestamps for assertion
        valid_timestamps = timestamps[~np.isnan(timestamps)]

        assert_intervals_by_timestamps(
            signal, valid_timestamps, intervals_dict["test"], label_idx=0
        )
        assert_intervals_by_timestamps(
            signal, valid_timestamps, intervals_dict["train"], label_idx=1
        )
        assert_intervals_by_timestamps(
            signal, valid_timestamps, intervals_dict["validation"], label_idx=2
        )


def test_time_interval_interpolation_zero_length():
    """
    Test handling of zero-length intervals (start == end).

    This test validates that zero-length intervals are properly handled. A
    zero-length interval [t, t) should not mark any timestamps as True since
    the interval is empty (no time passes).

    Intervals:
    - test: [1.0, 1.0) - zero-length interval, should match nothing
    - train: [2.0, 4.0) and [6.0, 8.0)
    - validation: [4.0, 6.0) and [8.0, 10.0)
    """
    with time_interval_data_and_interpolator(
        data_kwargs=dict(
            duration=10.0,
            sampling_rate=30.0,
            test_intervals=[[1.0, 1.0]],
            train_intervals=[[2.0, 4.0], [6.0, 8.0]],
            validation_intervals=[[4.0, 6.0], [8.0, 10.0]],
        )
    ) as (timestamps, intervals_dict, time_interp):
        assert isinstance(time_interp, TimeIntervalInterpolator)

        signal = time_interp.interpolate(timestamps)

        assert signal.shape == (
            len(timestamps),
            3,
        ), f"Expected shape ({len(timestamps)}, 3), got {signal.shape}"

        # Empty list means no timestamps should be True for test label
        assert_intervals_by_timestamps(signal, timestamps, [], label_idx=0)
        assert_intervals_by_timestamps(
            signal, timestamps, intervals_dict["train"], label_idx=1
        )
        assert_intervals_by_timestamps(
            signal, timestamps, intervals_dict["validation"], label_idx=2
        )


def test_time_interval_interpolation_multi_zero_length():
    """
    Test handling of multiple zero-length intervals mixed with normal intervals.

    This test validates that zero-length intervals are properly ignored when
    mixed with valid intervals.

    Intervals:
    - test: [1.0, 1.0) and [5.0, 6.0) - one zero-length, one normal
    - train: [4.0, 4.0) and [6.0, 6.0) - both zero-length
    - validation: [2.0, 2.0) and [8.0, 10.0) - one zero-length, one normal
    """
    with time_interval_data_and_interpolator(
        data_kwargs=dict(
            duration=10.0,
            sampling_rate=30.0,
            test_intervals=[[1.0, 1.0], [5.0, 6.0]],
            train_intervals=[[4.0, 4.0], [6.0, 6.0]],
            validation_intervals=[[2.0, 2.0], [8.0, 10.0]],
        )
    ) as (timestamps, intervals_dict, time_interp):
        assert isinstance(time_interp, TimeIntervalInterpolator)

        signal = time_interp.interpolate(timestamps)

        assert signal.shape == (
            len(timestamps),
            3,
        ), f"Expected shape ({len(timestamps)}, 3), got {signal.shape}"

        # Only the non-zero-length interval [5.0, 6.0) should match
        assert_intervals_by_timestamps(signal, timestamps, [[5.0, 6.0]], label_idx=0)
        # Both are zero-length, so nothing should match
        assert_intervals_by_timestamps(signal, timestamps, [], label_idx=1)
        # Only the non-zero-length interval [8.0, 10.0) should match
        assert_intervals_by_timestamps(signal, timestamps, [[8.0, 10.0]], label_idx=2)


def test_time_interval_interpolation_full_range():
    """
    Test handling of an interval covering the entire duration.

    This test validates that an interval spanning the full recording duration
    correctly marks all timestamps as True for that label.

    Intervals:
    - test: [0.0, 10.0) - covers entire duration
    - train: [2.0, 4.0) and [6.0, 8.0)
    - validation: [4.0, 6.0) and [8.0, 10.0)

    All timestamps should be marked True for the test label.
    """
    with time_interval_data_and_interpolator(
        data_kwargs=dict(
            duration=10.0,
            sampling_rate=30.0,
            test_intervals=[[0.0, 10.0]],
            train_intervals=[[2.0, 4.0], [6.0, 8.0]],
            validation_intervals=[[4.0, 6.0], [8.0, 10.0]],
        )
    ) as (timestamps, intervals_dict, time_interp):
        run_interval_interpolation_test(timestamps, intervals_dict, time_interp)


def test_time_interval_interpolation_inverted_interval():
    """
    Test handling of inverted intervals (start > end).

    This test validates that intervals where the start time is after the end time
    are properly detected as invalid and skipped. Such intervals are logically
    impossible and should not mark any timestamps as True.

    Intervals:
    - test: [5.0, 2.0) - inverted interval, should match nothing
    - train: [2.0, 4.0) and [6.0, 8.0)
    - validation: [4.0, 6.0) and [8.0, 10.0)

    Note: This test produces an expected warning about the invalid inverted
    interval for the test label.
    """
    with time_interval_data_and_interpolator(
        data_kwargs=dict(
            duration=10.0,
            sampling_rate=30.0,
            test_intervals=[[5.0, 2.0]],
            train_intervals=[[2.0, 4.0], [6.0, 8.0]],
            validation_intervals=[[4.0, 6.0], [8.0, 10.0]],
        )
    ) as (timestamps, intervals_dict, time_interp):
        assert isinstance(time_interp, TimeIntervalInterpolator)

        signal = time_interp.interpolate(timestamps)

        assert signal.shape == (
            len(timestamps),
            3,
        ), f"Expected shape ({len(timestamps)}, 3), got {signal.shape}"

        # Empty list means no timestamps should be True for test label
        assert_intervals_by_timestamps(signal, timestamps, [], label_idx=0)
        assert_intervals_by_timestamps(
            signal, timestamps, intervals_dict["train"], label_idx=1
        )
        assert_intervals_by_timestamps(
            signal, timestamps, intervals_dict["validation"], label_idx=2
        )


def test_time_interval_interpolation_multi_inverted():
    """
    Test handling of multiple inverted intervals mixed with normal intervals.

    This test validates that inverted intervals are properly ignored when mixed
    with valid intervals, similar to zero-length intervals.

    Intervals:
    - test: [8.0, 3.0) and [5.0, 6.0) - one inverted, one normal
    - train: [7.0, 2.0) and [9.0, 1.0) - both inverted
    - validation: [6.0, 4.0) and [8.0, 10.0) - one inverted, one normal

    Note: This test produces expected warnings about the invalid inverted
    intervals for test, train (two warnings), and validation labels.
    """
    with time_interval_data_and_interpolator(
        data_kwargs=dict(
            duration=10.0,
            sampling_rate=30.0,
            test_intervals=[[8.0, 3.0], [5.0, 6.0]],
            train_intervals=[[7.0, 2.0], [9.0, 1.0]],
            validation_intervals=[[6.0, 4.0], [8.0, 10.0]],
        )
    ) as (timestamps, intervals_dict, time_interp):
        assert isinstance(time_interp, TimeIntervalInterpolator)

        signal = time_interp.interpolate(timestamps)

        assert signal.shape == (
            len(timestamps),
            3,
        ), f"Expected shape ({len(timestamps)}, 3), got {signal.shape}"

        # Only the non-inverted interval [5.0, 6.0) should match
        assert_intervals_by_timestamps(signal, timestamps, [[5.0, 6.0]], label_idx=0)
        # Both are inverted, so nothing should match
        assert_intervals_by_timestamps(signal, timestamps, [], label_idx=1)
        # Only the non-inverted interval [8.0, 10.0) should match
        assert_intervals_by_timestamps(signal, timestamps, [[8.0, 10.0]], label_idx=2)
