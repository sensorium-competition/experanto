import numpy as np

from experanto.interpolators import SequenceInterpolator
from experanto.intervals import (
    TimeInterval,
    find_complement_of_interval_array,
    uniquefy_interval_array,
)


def nan_filter(vicinity=0.05):
    """Create a filter that excludes time regions around NaN values.

    Returns a closure that, given a :class:`~experanto.interpolators.SequenceInterpolator`,
    identifies all time points containing NaN in any channel and marks a
    symmetric window of ``vicinity`` seconds around each as invalid.

    Parameters
    ----------
    vicinity : float, optional
        Half-width of the exclusion window in seconds around each NaN
        time point. Default is 0.05.

    Returns
    -------
    callable
        A function that takes a
        :class:`~experanto.interpolators.SequenceInterpolator` and returns
        a list of :class:`~experanto.intervals.TimeInterval` representing
        the valid (NaN-free) portions of the recording.
    """

    def implementation(device_: SequenceInterpolator):
        time_delta = device_.time_delta
        start_time = device_.start_time
        end_time = device_.end_time
        data = device_._data

        nan_mask = np.any(np.isnan(data), axis=1)
        nan_indices = np.where(nan_mask)[0]

        invalid_intervals = []
        for idx in nan_indices:
            time_point = start_time + idx * time_delta
            interval_start = max(start_time, time_point - vicinity)
            interval_end = min(end_time, time_point + vicinity)
            invalid_intervals.append(TimeInterval(interval_start, interval_end))

        invalid_intervals = uniquefy_interval_array(invalid_intervals)
        valid_intervals = find_complement_of_interval_array(
            start_time, end_time, invalid_intervals
        )

        return valid_intervals

    return implementation
