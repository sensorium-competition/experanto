import typing
from typing import List

import numpy as np


class TimeInterval(typing.NamedTuple):
    start: float
    end: float

    def __contains__(self, time):
        return self.start <= time <= self.end

    def find_intersection_between_two_intervals(
        self, other_interval: "TimeInterval"
    ) -> "TimeInterval":
        start = max(self.start, other_interval.start)
        end = min(self.end, other_interval.end)
        if start <= end:
            return TimeInterval(start, end)
        else:
            return None

    def __repr__(self) -> str:
        return f"TimeInterval(start={self.start}, end={self.end})"

    def intersect(self, times: np.ndarray) -> np.ndarray:
        return np.where((times >= self.start) & (times <= self.end))[0]


def uniquefy_interval_array(interval_array: List[TimeInterval]) -> List[TimeInterval]:
    """
    Takes an array of TimeIntervals and returns a new array where no intervals overlap.
    If intervals overlap or are adjacent, they are merged into a single interval.

    Args:
        interval_array: List of TimeInterval objects

    Returns:
        List of non-overlapping TimeInterval objects sorted by start time
    """
    if not interval_array:
        return []

    # Sort intervals by start time
    sorted_intervals = sorted(interval_array, key=lambda x: x.start)

    unique_intervals = [sorted_intervals[0]]

    for current in sorted_intervals[1:]:
        previous = unique_intervals[-1]

        # If current interval overlaps or is adjacent to previous
        if current.start <= previous.end or current.start == previous.end:
            # Merge by creating new interval with max end time
            merged = TimeInterval(previous.start, max(previous.end, current.end))
            unique_intervals[-1] = merged
        else:
            unique_intervals.append(current)

    return unique_intervals


def find_intersection_between_two_interval_arrays(
    interval_array_1: List[TimeInterval], interval_array_2: List[TimeInterval]
) -> List[TimeInterval]:
    # Sort both arrays by start time
    sorted_1 = sorted(interval_array_1, key=lambda x: x.start)
    sorted_2 = sorted(interval_array_2, key=lambda x: x.start)

    intersection_array = []
    i = j = 0

    while i < len(sorted_1) and j < len(sorted_2):
        interval_1 = sorted_1[i]
        interval_2 = sorted_2[j]

        # Check for intersection
        intersection = interval_1.find_intersection_between_two_intervals(interval_2)
        if intersection is not None:
            intersection_array.append(intersection)

        # Advance the interval with the earlier end point
        if interval_1.end < interval_2.end:
            i += 1
        else:
            j += 1

    return intersection_array


def find_intersection_across_arrays_of_intervals(
    intervals_array: List[List[TimeInterval]],
) -> TimeInterval:

    common_interval_array = intervals_array[0]

    for interval_array in intervals_array[1:]:
        common_interval_array = find_intersection_between_two_interval_arrays(
            common_interval_array, interval_array
        )

    return uniquefy_interval_array(common_interval_array)


def find_union_across_arrays_of_intervals(
    intervals_array: List[List[TimeInterval]],
) -> TimeInterval:
    union_array = []
    for interval_array in intervals_array:
        union_array.extend(interval_array)
    return uniquefy_interval_array(union_array)


def find_complement_of_interval_array(
    start: float, end: float, interval_array: List[TimeInterval]
) -> List[TimeInterval]:
    """
    Finds the complement of an interval array within a given range [start, end].
    Returns intervals that represent the gaps not covered by the input intervals.

    Args:
        start: Start time of the range
        end: End time of the range
        interval_array: List of TimeInterval objects

    Returns:
        List of TimeInterval objects representing the complement
    """
    if not interval_array:
        return [TimeInterval(start, end)]

    # Sort intervals by start time
    sorted_intervals = sorted(interval_array, key=lambda x: x.start)

    complement_intervals = []
    current_time = start

    for interval in sorted_intervals:
        # If there's a gap before current interval, add it
        if current_time < interval.start:
            complement_intervals.append(TimeInterval(current_time, interval.start))
        # Update current_time to the rightmost point we've seen
        current_time = max(current_time, interval.end)

    # Add final interval if there's space after the last interval
    if current_time < end:
        complement_intervals.append(TimeInterval(current_time, end))

    return complement_intervals


# stat information about valid intervals


def get_stats_for_valid_interval(
    intervals: List[TimeInterval], start_time: float, end_time: float
) -> str:
    """
    Calculates and returns statistics about valid and invalid time intervals within a given range.

    Args:
        intervals: List of TimeInterval objects representing the valid periods.
        start_time: The beginning of the total time range to consider.
        end_time: The end of the total time range to consider.

    Returns:
        A string summarizing the statistics of valid and invalid intervals.
    """
    total_duration = end_time - start_time
    if total_duration <= 0:
        return f"Error: Invalid time range (end_time <= start_time). Total duration must be positive."

    # Ensure intervals are unique and sorted, then clamp them to the analysis window
    unique_intervals = uniquefy_interval_array(intervals)
    valid_intervals_in_range = []
    for interval in unique_intervals:
        clamped_start = max(interval.start, start_time)
        clamped_end = min(interval.end, end_time)
        if (
            clamped_end > clamped_start
        ):  # Only consider intervals with positive duration within the range
            valid_intervals_in_range.append(TimeInterval(clamped_start, clamped_end))

    # Calculate stats for valid intervals
    if valid_intervals_in_range:
        valid_durations = np.array(
            [inv.end - inv.start for inv in valid_intervals_in_range]
        )
        total_valid_duration = np.sum(valid_durations)
        mean_valid_duration = np.mean(valid_durations)
        std_valid_duration = np.std(valid_durations)
        num_valid_intervals = len(valid_intervals_in_range)
    else:
        total_valid_duration = 0.0
        mean_valid_duration = 0.0
        std_valid_duration = 0.0
        num_valid_intervals = 0

    valid_percentage = (total_valid_duration / total_duration) * 100

    # Calculate stats for invalid intervals (complement)
    invalid_intervals = find_complement_of_interval_array(
        start_time, end_time, valid_intervals_in_range
    )

    if invalid_intervals:
        invalid_durations = np.array([inv.end - inv.start for inv in invalid_intervals])
        total_invalid_duration = np.sum(invalid_durations)
        mean_invalid_duration = np.mean(invalid_durations)
        std_invalid_duration = np.std(invalid_durations)
        num_invalid_intervals = len(invalid_intervals)
    else:
        total_invalid_duration = 0.0
        mean_invalid_duration = 0.0
        std_invalid_duration = 0.0
        num_invalid_intervals = 0

    invalid_percentage = (total_invalid_duration / total_duration) * 100

    # Format the results into a string
    stats_string = (
        f"Interval Statistics, Total Duration: {total_duration:.3f}s):\n"
        f"  Valid Intervals ({num_valid_intervals}):\n"
        f"    - Total Duration: {total_valid_duration:.3f}s ({valid_percentage:.2f}%)\n"
        f"    - Mean Duration:  {mean_valid_duration:.3f}s\n"
        f"    - Std Dev:        {std_valid_duration:.3f}s\n"
        f"  Invalid Intervals ({num_invalid_intervals}):\n"
        f"    - Total Duration: {total_invalid_duration:.3f}s ({invalid_percentage:.2f}%)\n"
        f"    - Mean Duration:  {mean_invalid_duration:.3f}s\n"
        f"    - Std Dev:        {std_invalid_duration:.3f}s\n"
    )

    return stats_string
