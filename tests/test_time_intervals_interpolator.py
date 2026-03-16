import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import composite

from experanto.interpolators import TimeIntervalInterpolator
from experanto.intervals import (
    TimeInterval,
    find_complement_of_interval_array,
    find_intersection_across_arrays_of_intervals,
    find_intersection_between_two_interval_arrays,
    find_union_across_arrays_of_intervals,
    get_stats_for_valid_interval,
    uniquefy_interval_array,
)

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


# ============================================================================
# TimeInterval — __contains__ (closed interval [start, end])
# Note: The interpolator tests above use half-open [start, end) semantics.
# These tests cover the closed-interval __contains__ defined in intervals.py.
# ============================================================================


@pytest.mark.parametrize(
    "interval, time, expected",
    [
        (TimeInterval(1.0, 5.0), 3.0, True),  # inside
        (TimeInterval(1.0, 5.0), 1.0, True),  # start boundary (inclusive)
        (
            TimeInterval(1.0, 5.0),
            5.0,
            True,
        ),  # end boundary (inclusive, differs from interpolator)
        (TimeInterval(1.0, 5.0), 0.5, False),  # before interval
        (TimeInterval(1.0, 5.0), 5.5, False),  # after interval
    ],
)
def test_time_interval_contains(interval, time, expected):
    """Test closed-interval __contains__ semantics [start, end]."""
    assert (time in interval) == expected


# ============================================================================
# TimeInterval — intersect (closed interval [start, end] with numpy array)
# Note: Uses closed-interval semantics, unlike the interpolator's half-open.
# ============================================================================


@pytest.mark.parametrize(
    "interval, times, expected_indices",
    [
        (
            TimeInterval(2.0, 5.0),
            np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            np.array([2, 3, 4, 5]),
        ),  # partial match
        (
            TimeInterval(10.0, 20.0),
            np.array([0.0, 1.0, 2.0]),
            np.array([]),
        ),  # no match
        (
            TimeInterval(0.0, 10.0),
            np.array([1.0, 2.0, 3.0]),
            np.array([0, 1, 2]),
        ),  # all match
    ],
)
def test_time_interval_intersect(interval, times, expected_indices):
    """Test closed-interval intersect() with numpy arrays."""
    indices = interval.intersect(times)
    np.testing.assert_array_equal(indices, expected_indices)


# ============================================================================
# TimeInterval — find_intersection_between_two_intervals
# ============================================================================


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (
            TimeInterval(1.0, 5.0),
            TimeInterval(3.0, 7.0),
            TimeInterval(3.0, 5.0),
        ),  # overlap
        (TimeInterval(1.0, 3.0), TimeInterval(5.0, 7.0), None),  # disjoint
        (
            TimeInterval(1.0, 3.0),
            TimeInterval(3.0, 5.0),
            TimeInterval(3.0, 3.0),
        ),  # touching
        (
            TimeInterval(1.0, 10.0),
            TimeInterval(3.0, 5.0),
            TimeInterval(3.0, 5.0),
        ),  # contained
        (
            TimeInterval(2.0, 6.0),
            TimeInterval(2.0, 6.0),
            TimeInterval(2.0, 6.0),
        ),  # identical
    ],
)
def test_two_interval_intersection(a, b, expected):
    """Test pairwise intersection of two TimeInterval objects."""
    assert a.find_intersection_between_two_intervals(b) == expected


# ============================================================================
# uniquefy_interval_array
# ============================================================================


@pytest.mark.parametrize(
    "intervals, expected",
    [
        ([], []),  # empty
        ([TimeInterval(1.0, 3.0)], [TimeInterval(1.0, 3.0)]),  # single
        (
            [TimeInterval(1.0, 3.0), TimeInterval(5.0, 7.0)],
            [TimeInterval(1.0, 3.0), TimeInterval(5.0, 7.0)],
        ),  # non-overlapping
        (
            [TimeInterval(1.0, 5.0), TimeInterval(3.0, 7.0)],
            [TimeInterval(1.0, 7.0)],
        ),  # overlapping merge
        (
            [TimeInterval(1.0, 3.0), TimeInterval(3.0, 5.0)],
            [TimeInterval(1.0, 5.0)],
        ),  # adjacent merge
        (
            [TimeInterval(1.0, 10.0), TimeInterval(3.0, 5.0)],
            [TimeInterval(1.0, 10.0)],
        ),  # contained
        (
            [TimeInterval(5.0, 7.0), TimeInterval(1.0, 3.0)],
            [TimeInterval(1.0, 3.0), TimeInterval(5.0, 7.0)],
        ),  # unsorted input non-overlapping
        (
            [TimeInterval(3.0, 7.0), TimeInterval(1.0, 3.0)],
            [TimeInterval(1.0, 7.0)],
        ),  # unsorted input overlapping
        (
            [TimeInterval(1.0, 4.0), TimeInterval(3.0, 6.0), TimeInterval(5.0, 8.0)],
            [TimeInterval(1.0, 8.0)],
        ),  # chain merge
        (
            [TimeInterval(1.0, 3.0), TimeInterval(2.0, 5.0), TimeInterval(8.0, 10.0)],
            [TimeInterval(1.0, 5.0), TimeInterval(8.0, 10.0)],
        ),  # mix of overlapping and disjoint
    ],
)
def test_uniquefy(intervals, expected):
    """Test merging of overlapping/adjacent intervals."""
    assert uniquefy_interval_array(intervals) == expected


# ============================================================================
# find_intersection_between_two_interval_arrays
# ============================================================================


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ([TimeInterval(1.0, 3.0)], [TimeInterval(5.0, 7.0)], []),  # no overlap
        (
            [TimeInterval(1.0, 5.0)],
            [TimeInterval(1.0, 5.0)],
            [TimeInterval(1.0, 5.0)],
        ),  # full overlap
        (
            [TimeInterval(1.0, 5.0)],
            [TimeInterval(3.0, 7.0)],
            [TimeInterval(3.0, 5.0)],
        ),  # partial
        (
            [TimeInterval(1.0, 3.0)],
            [TimeInterval(3.0, 5.0)],
            [TimeInterval(3.0, 3.0)],
        ),  # touching
        (
            [TimeInterval(1.0, 4.0), TimeInterval(6.0, 9.0)],
            [TimeInterval(2.0, 7.0)],
            [TimeInterval(2.0, 4.0), TimeInterval(6.0, 7.0)],
        ),  # multiple intersections
        ([], [TimeInterval(1.0, 5.0)], []),  # empty first
        ([TimeInterval(1.0, 5.0)], [], []),  # empty second
        ([], [], []),  # both empty
    ],
)
def test_array_intersection(a, b, expected):
    """Test intersection of two interval arrays."""
    assert find_intersection_between_two_interval_arrays(a, b) == expected


# ============================================================================
# find_intersection_across_arrays_of_intervals
# ============================================================================


@pytest.mark.parametrize(
    "arrays, expected",
    [
        (
            [[TimeInterval(1.0, 5.0)], [TimeInterval(3.0, 7.0)]],
            [TimeInterval(3.0, 5.0)],
        ),  # two arrays
        (
            [
                [TimeInterval(1.0, 6.0)],
                [TimeInterval(3.0, 8.0)],
                [TimeInterval(4.0, 10.0)],
            ],
            [TimeInterval(4.0, 6.0)],
        ),  # three arrays
        (
            [[TimeInterval(1.0, 3.0)], [TimeInterval(5.0, 7.0)]],
            [],
        ),  # no common overlap
        (
            [[TimeInterval(1.0, 3.0), TimeInterval(5.0, 7.0)]],
            [TimeInterval(1.0, 3.0), TimeInterval(5.0, 7.0)],
        ),  # single array returns itself
    ],
)
def test_across_intersection(arrays, expected):
    """Test intersection across multiple interval arrays."""
    assert find_intersection_across_arrays_of_intervals(arrays) == expected


# ============================================================================
# find_union_across_arrays_of_intervals
# ============================================================================


@pytest.mark.parametrize(
    "arrays, expected",
    [
        (
            [[TimeInterval(1.0, 3.0)], [TimeInterval(5.0, 7.0)]],
            [TimeInterval(1.0, 3.0), TimeInterval(5.0, 7.0)],
        ),  # non-overlapping
        (
            [[TimeInterval(1.0, 5.0)], [TimeInterval(3.0, 7.0)]],
            [TimeInterval(1.0, 7.0)],
        ),  # overlapping merges
        (
            [
                [TimeInterval(1.0, 3.0)],
                [TimeInterval(2.0, 5.0)],
                [TimeInterval(8.0, 10.0)],
            ],
            [TimeInterval(1.0, 5.0), TimeInterval(8.0, 10.0)],
        ),  # multiple arrays
        ([[], []], []),  # empty arrays
    ],
)
def test_union(arrays, expected):
    """Test union across multiple interval arrays."""
    assert find_union_across_arrays_of_intervals(arrays) == expected


# ============================================================================
# find_complement_of_interval_array
# ============================================================================


@pytest.mark.parametrize(
    "intervals, start, end, expected",
    [
        ([], 0.0, 10.0, [TimeInterval(0.0, 10.0)]),  # empty → full range
        ([TimeInterval(0.0, 10.0)], 0.0, 10.0, []),  # full coverage → empty
        (
            [TimeInterval(3.0, 10.0)],
            0.0,
            10.0,
            [TimeInterval(0.0, 3.0)],
        ),  # gap at start
        ([TimeInterval(0.0, 7.0)], 0.0, 10.0, [TimeInterval(7.0, 10.0)]),  # gap at end
        (
            [TimeInterval(0.0, 3.0), TimeInterval(7.0, 10.0)],
            0.0,
            10.0,
            [TimeInterval(3.0, 7.0)],
        ),  # middle gap
        (
            [TimeInterval(1.0, 3.0), TimeInterval(5.0, 7.0)],
            0.0,
            10.0,
            [TimeInterval(0.0, 1.0), TimeInterval(3.0, 5.0), TimeInterval(7.0, 10.0)],
        ),  # multiple gaps
        (
            [TimeInterval(1.0, 5.0), TimeInterval(3.0, 7.0)],
            0.0,
            10.0,
            [TimeInterval(0.0, 1.0), TimeInterval(7.0, 10.0)],
        ),  # overlapping input
    ],
)
def test_complement(intervals, start, end, expected):
    """Test complement of interval array within a given range."""
    assert find_complement_of_interval_array(start, end, intervals) == expected


# ============================================================================
# get_stats_for_valid_interval
# ============================================================================


@pytest.mark.parametrize(
    "intervals, start, end, expected_substring",
    [
        ([], 10.0, 5.0, "Error"),  # invalid range
        ([], 5.0, 5.0, "Error"),  # zero duration
        ([TimeInterval(0.0, 10.0)], 0.0, 10.0, "100.00%"),  # full coverage
        ([], 0.0, 10.0, "0.00%"),  # no valid intervals
        ([TimeInterval(0.0, 5.0)], 0.0, 10.0, "50.00%"),  # half coverage
        ([TimeInterval(-5.0, 15.0)], 0.0, 10.0, "100.00%"),  # clamped to range
        (
            [TimeInterval(0.0, 3.0), TimeInterval(7.0, 10.0)],
            0.0,
            10.0,
            "60.00%",
        ),  # multiple intervals
        ([TimeInterval(0.0, 5.0)], 0.0, 10.0, "Valid Intervals"),  # has valid section
        (
            [TimeInterval(0.0, 5.0)],
            0.0,
            10.0,
            "Invalid Intervals",
        ),  # has invalid section
        (
            [TimeInterval(0.0, 3.0), TimeInterval(7.0, 10.0)],
            0.0,
            10.0,
            "Valid Intervals (2)",
        ),  # interval count
    ],
)
def test_stats(intervals, start, end, expected_substring):
    """Test statistics computation for valid intervals."""
    result = get_stats_for_valid_interval(intervals, start, end)
    assert expected_substring in result


# ============================================================================
# Property-based tests using Hypothesis
# These test mathematical invariants rather than specific examples.
# ============================================================================


@composite
def time_intervals(draw, min_value=0.0, max_value=100.0):
    """Strategy to generate valid TimeInterval objects with start <= end."""
    start = draw(
        st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    end = draw(
        st.floats(
            min_value=start, max_value=max_value, allow_nan=False, allow_infinity=False
        )
    )
    return TimeInterval(start, end)


@given(a=time_intervals(), b=time_intervals())
def test_intersection_commutative_property(a, b):
    """Intersection must be commutative: a ∩ b == b ∩ a for all intervals."""
    assert a.find_intersection_between_two_intervals(
        b
    ) == b.find_intersection_between_two_intervals(a)


@given(a=time_intervals(), b=time_intervals())
def test_intersection_contained_in_both_property(a, b):
    """If a ∩ b exists, it must be contained within both a and b."""
    result = a.find_intersection_between_two_intervals(b)
    if result is not None:
        assert result.start >= a.start and result.end <= a.end
        assert result.start >= b.start and result.end <= b.end


@given(intervals=st.lists(time_intervals(), min_size=0, max_size=10))
def test_uniquefy_idempotent_property(intervals):
    """Running uniquefy twice must give the same result as running it once."""
    once = uniquefy_interval_array(intervals)
    twice = uniquefy_interval_array(once)
    assert once == twice


@given(intervals=st.lists(time_intervals(), min_size=0, max_size=10))
def test_uniquefy_preserves_coverage_property(intervals):
    """Uniquefy must not lose any covered point — merged result covers same range."""
    merged = uniquefy_interval_array(intervals)
    for iv in intervals:
        # Every point at the midpoint of an original interval must still be covered
        if iv.start < iv.end:
            mid = (iv.start + iv.end) / 2.0
            assert any(m.start <= mid <= m.end for m in merged)


@given(
    intervals=st.lists(
        time_intervals(min_value=0.0, max_value=10.0), min_size=0, max_size=5
    )
)
def test_complement_and_original_cover_full_range_property(intervals):
    """The union of intervals and their complement must cover the entire range."""
    start, end = 0.0, 10.0
    complement = find_complement_of_interval_array(start, end, intervals)
    merged_original = uniquefy_interval_array(intervals)
    # Clamp original intervals to [start, end]
    clamped = []
    for iv in merged_original:
        s = max(iv.start, start)
        e = min(iv.end, end)
        if s <= e:
            clamped.append(TimeInterval(s, e))
    all_intervals = clamped + complement
    full_coverage = uniquefy_interval_array(all_intervals)
    # The merged result should cover [start, end] completely
    if full_coverage:
        assert full_coverage[0].start <= start
        assert full_coverage[-1].end >= end
        # No gaps between consecutive intervals
        for i in range(len(full_coverage) - 1):
            assert full_coverage[i].end >= full_coverage[i + 1].start
