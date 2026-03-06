import numpy as np
import pytest

from experanto.intervals import (
    TimeInterval,
    uniquefy_interval_array,
    find_intersection_between_two_interval_arrays,
    find_intersection_across_arrays_of_intervals,
    find_union_across_arrays_of_intervals,
    find_complement_of_interval_array,
    get_stats_for_valid_interval,
)


# ============================================================================
# TimeInterval — find_intersection_between_two_intervals
# ============================================================================


def test_intersection_two_intervals_overlap():
    """Overlapping intervals should return the overlap."""
    a = TimeInterval(1.0, 5.0)
    b = TimeInterval(3.0, 7.0)
    result = a.find_intersection_between_two_intervals(b)
    assert result == TimeInterval(3.0, 5.0)


def test_intersection_two_intervals_no_overlap():
    """Disjoint intervals should return None."""
    a = TimeInterval(1.0, 3.0)
    b = TimeInterval(5.0, 7.0)
    result = a.find_intersection_between_two_intervals(b)
    assert result is None


def test_intersection_two_intervals_touching():
    """Touching at a single point (end == start) returns a zero-width interval."""
    a = TimeInterval(1.0, 3.0)
    b = TimeInterval(3.0, 5.0)
    result = a.find_intersection_between_two_intervals(b)
    assert result == TimeInterval(3.0, 3.0)


def test_intersection_two_intervals_contained():
    """When one interval is fully inside the other, return the smaller one."""
    a = TimeInterval(1.0, 10.0)
    b = TimeInterval(3.0, 5.0)
    result = a.find_intersection_between_two_intervals(b)
    assert result == TimeInterval(3.0, 5.0)


def test_intersection_two_intervals_identical():
    """Identical intervals should return the same interval."""
    a = TimeInterval(2.0, 6.0)
    result = a.find_intersection_between_two_intervals(a)
    assert result == TimeInterval(2.0, 6.0)


def test_intersection_two_intervals_commutative():
    """a ∩ b should equal b ∩ a."""
    a = TimeInterval(1.0, 5.0)
    b = TimeInterval(3.0, 7.0)
    assert (
        a.find_intersection_between_two_intervals(b)
        == b.find_intersection_between_two_intervals(a)
    )


# ============================================================================
# TimeInterval — __repr__
# ============================================================================


def test_time_interval_repr():
    interval = TimeInterval(1.5, 3.5)
    assert repr(interval) == "TimeInterval(start=1.5, end=3.5)"


# ============================================================================
# uniquefy_interval_array
# ============================================================================


def test_uniquefy_empty():
    assert uniquefy_interval_array([]) == []


def test_uniquefy_single():
    result = uniquefy_interval_array([TimeInterval(1.0, 3.0)])
    assert result == [TimeInterval(1.0, 3.0)]


def test_uniquefy_non_overlapping():
    """Non-overlapping intervals should remain unchanged."""
    intervals = [TimeInterval(1.0, 3.0), TimeInterval(5.0, 7.0)]
    result = uniquefy_interval_array(intervals)
    assert result == [TimeInterval(1.0, 3.0), TimeInterval(5.0, 7.0)]


def test_uniquefy_overlapping_merge():
    """Overlapping intervals should merge into one."""
    intervals = [TimeInterval(1.0, 5.0), TimeInterval(3.0, 7.0)]
    result = uniquefy_interval_array(intervals)
    assert result == [TimeInterval(1.0, 7.0)]


def test_uniquefy_adjacent_merge():
    """Adjacent intervals (end == start of next) should merge."""
    intervals = [TimeInterval(1.0, 3.0), TimeInterval(3.0, 5.0)]
    result = uniquefy_interval_array(intervals)
    assert result == [TimeInterval(1.0, 5.0)]


def test_uniquefy_contained():
    """An interval fully inside another should be absorbed."""
    intervals = [TimeInterval(1.0, 10.0), TimeInterval(3.0, 5.0)]
    result = uniquefy_interval_array(intervals)
    assert result == [TimeInterval(1.0, 10.0)]


def test_uniquefy_unsorted_input():
    """Should handle unsorted input correctly."""
    intervals = [TimeInterval(5.0, 7.0), TimeInterval(1.0, 3.0)]
    result = uniquefy_interval_array(intervals)
    assert result == [TimeInterval(1.0, 3.0), TimeInterval(5.0, 7.0)]


def test_uniquefy_chain_merge():
    """Chain of overlapping intervals should merge into one."""
    intervals = [
        TimeInterval(1.0, 4.0),
        TimeInterval(3.0, 6.0),
        TimeInterval(5.0, 8.0),
    ]
    result = uniquefy_interval_array(intervals)
    assert result == [TimeInterval(1.0, 8.0)]


def test_uniquefy_mixed():
    """Mix of overlapping and disjoint should merge only overlapping ones."""
    intervals = [
        TimeInterval(1.0, 3.0),
        TimeInterval(2.0, 5.0),
        TimeInterval(8.0, 10.0),
    ]
    result = uniquefy_interval_array(intervals)
    assert result == [TimeInterval(1.0, 5.0), TimeInterval(8.0, 10.0)]


# ============================================================================
# find_intersection_between_two_interval_arrays
# ============================================================================


def test_array_intersection_no_overlap():
    a = [TimeInterval(1.0, 3.0)]
    b = [TimeInterval(5.0, 7.0)]
    result = find_intersection_between_two_interval_arrays(a, b)
    assert result == []


def test_array_intersection_full_overlap():
    a = [TimeInterval(1.0, 5.0)]
    b = [TimeInterval(1.0, 5.0)]
    result = find_intersection_between_two_interval_arrays(a, b)
    assert result == [TimeInterval(1.0, 5.0)]


def test_array_intersection_partial_overlap():
    a = [TimeInterval(1.0, 5.0)]
    b = [TimeInterval(3.0, 7.0)]
    result = find_intersection_between_two_interval_arrays(a, b)
    assert result == [TimeInterval(3.0, 5.0)]


def test_array_intersection_multiple():
    """One interval in b overlaps with two in a."""
    a = [TimeInterval(1.0, 4.0), TimeInterval(6.0, 9.0)]
    b = [TimeInterval(2.0, 7.0)]
    result = find_intersection_between_two_interval_arrays(a, b)
    assert result == [TimeInterval(2.0, 4.0), TimeInterval(6.0, 7.0)]


def test_array_intersection_empty_first():
    result = find_intersection_between_two_interval_arrays(
        [], [TimeInterval(1.0, 5.0)]
    )
    assert result == []


def test_array_intersection_empty_second():
    result = find_intersection_between_two_interval_arrays(
        [TimeInterval(1.0, 5.0)], []
    )
    assert result == []


def test_array_intersection_both_empty():
    result = find_intersection_between_two_interval_arrays([], [])
    assert result == []


# ============================================================================
# find_intersection_across_arrays_of_intervals
# ============================================================================


def test_across_intersection_two_arrays():
    arrays = [
        [TimeInterval(1.0, 5.0)],
        [TimeInterval(3.0, 7.0)],
    ]
    result = find_intersection_across_arrays_of_intervals(arrays)
    assert result == [TimeInterval(3.0, 5.0)]


def test_across_intersection_three_arrays():
    arrays = [
        [TimeInterval(1.0, 6.0)],
        [TimeInterval(3.0, 8.0)],
        [TimeInterval(4.0, 10.0)],
    ]
    result = find_intersection_across_arrays_of_intervals(arrays)
    assert result == [TimeInterval(4.0, 6.0)]


def test_across_intersection_no_common():
    arrays = [
        [TimeInterval(1.0, 3.0)],
        [TimeInterval(5.0, 7.0)],
    ]
    result = find_intersection_across_arrays_of_intervals(arrays)
    assert result == []


def test_across_intersection_single_array():
    """Single array should return itself (uniquefied)."""
    arrays = [[TimeInterval(1.0, 3.0), TimeInterval(5.0, 7.0)]]
    result = find_intersection_across_arrays_of_intervals(arrays)
    assert result == [TimeInterval(1.0, 3.0), TimeInterval(5.0, 7.0)]


# ============================================================================
# find_union_across_arrays_of_intervals
# ============================================================================


def test_union_non_overlapping():
    arrays = [
        [TimeInterval(1.0, 3.0)],
        [TimeInterval(5.0, 7.0)],
    ]
    result = find_union_across_arrays_of_intervals(arrays)
    assert result == [TimeInterval(1.0, 3.0), TimeInterval(5.0, 7.0)]


def test_union_overlapping_merges():
    arrays = [
        [TimeInterval(1.0, 5.0)],
        [TimeInterval(3.0, 7.0)],
    ]
    result = find_union_across_arrays_of_intervals(arrays)
    assert result == [TimeInterval(1.0, 7.0)]


def test_union_multiple_arrays():
    arrays = [
        [TimeInterval(1.0, 3.0)],
        [TimeInterval(2.0, 5.0)],
        [TimeInterval(8.0, 10.0)],
    ]
    result = find_union_across_arrays_of_intervals(arrays)
    assert result == [TimeInterval(1.0, 5.0), TimeInterval(8.0, 10.0)]


def test_union_empty_arrays():
    result = find_union_across_arrays_of_intervals([[], []])
    assert result == []


# ============================================================================
# find_complement_of_interval_array
# ============================================================================


def test_complement_of_empty():
    """Complement of nothing = the entire range."""
    result = find_complement_of_interval_array(0.0, 10.0, [])
    assert result == [TimeInterval(0.0, 10.0)]


def test_complement_of_full_coverage():
    """Full coverage has no complement."""
    result = find_complement_of_interval_array(
        0.0, 10.0, [TimeInterval(0.0, 10.0)]
    )
    assert result == []


def test_complement_gap_in_middle():
    intervals = [TimeInterval(0.0, 3.0), TimeInterval(7.0, 10.0)]
    result = find_complement_of_interval_array(0.0, 10.0, intervals)
    assert result == [TimeInterval(3.0, 7.0)]


def test_complement_gap_at_start():
    intervals = [TimeInterval(3.0, 10.0)]
    result = find_complement_of_interval_array(0.0, 10.0, intervals)
    assert result == [TimeInterval(0.0, 3.0)]


def test_complement_gap_at_end():
    intervals = [TimeInterval(0.0, 7.0)]
    result = find_complement_of_interval_array(0.0, 10.0, intervals)
    assert result == [TimeInterval(7.0, 10.0)]


def test_complement_multiple_gaps():
    intervals = [TimeInterval(1.0, 3.0), TimeInterval(5.0, 7.0)]
    result = find_complement_of_interval_array(0.0, 10.0, intervals)
    assert result == [
        TimeInterval(0.0, 1.0),
        TimeInterval(3.0, 5.0),
        TimeInterval(7.0, 10.0),
    ]


def test_complement_with_overlapping_input():
    """Overlapping input intervals — complement should still be correct."""
    intervals = [TimeInterval(1.0, 5.0), TimeInterval(3.0, 7.0)]
    result = find_complement_of_interval_array(0.0, 10.0, intervals)
    assert result == [TimeInterval(0.0, 1.0), TimeInterval(7.0, 10.0)]


# ============================================================================
# get_stats_for_valid_interval
# ============================================================================


def test_stats_returns_string():
    intervals = [TimeInterval(0.0, 5.0)]
    result = get_stats_for_valid_interval(intervals, 0.0, 10.0)
    assert isinstance(result, str)


def test_stats_invalid_time_range():
    """end_time <= start_time should return error message."""
    result = get_stats_for_valid_interval([], 10.0, 5.0)
    assert "Error" in result


def test_stats_zero_duration():
    """end_time == start_time should return error message."""
    result = get_stats_for_valid_interval([], 5.0, 5.0)
    assert "Error" in result


def test_stats_full_coverage():
    intervals = [TimeInterval(0.0, 10.0)]
    result = get_stats_for_valid_interval(intervals, 0.0, 10.0)
    assert "100.00%" in result


def test_stats_no_valid_intervals():
    result = get_stats_for_valid_interval([], 0.0, 10.0)
    assert "0.00%" in result


def test_stats_half_coverage():
    intervals = [TimeInterval(0.0, 5.0)]
    result = get_stats_for_valid_interval(intervals, 0.0, 10.0)
    assert "50.00%" in result


def test_stats_contains_valid_and_invalid_sections():
    intervals = [TimeInterval(0.0, 5.0)]
    result = get_stats_for_valid_interval(intervals, 0.0, 10.0)
    assert "Valid Intervals" in result
    assert "Invalid Intervals" in result


def test_stats_intervals_clamped_to_range():
    """Intervals outside [start, end] should be clamped."""
    intervals = [TimeInterval(-5.0, 15.0)]
    result = get_stats_for_valid_interval(intervals, 0.0, 10.0)
    assert "100.00%" in result


def test_stats_multiple_intervals():
    intervals = [TimeInterval(0.0, 3.0), TimeInterval(7.0, 10.0)]
    result = get_stats_for_valid_interval(intervals, 0.0, 10.0)
    assert "60.00%" in result
    assert "Valid Intervals (2)" in result
