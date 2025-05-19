import numpy as np

from experanto.interpolators import Interpolator
from experanto.intervals import (TimeInterval,
                                 find_complement_of_interval_array,
                                 uniquefy_interval_array)


def nan_filter(vicinity=0.05):

    def implementation(device_: Interpolator):
        time_delta = device_.time_delta
        start_time = device_.start_time
        end_time = device_.end_time
        data = device_._data  # (T, n_neurons)

        # detect nans
        nan_mask = np.isnan(data)  # (T, n_neurons)
        nan_mask = np.any(nan_mask, axis=1)  # (T,)

        # Find indices where nan_mask is True
        nan_indices = np.where(nan_mask)[0]

        # Create invalid TimeIntervals around each nan point
        invalid_intervals = []
        vicinity_seconds = vicinity  # vicinity is already in seconds
        for idx in nan_indices:
            time_point = start_time + idx * time_delta
            interval_start = max(start_time, time_point - vicinity_seconds)
            interval_end = min(end_time, time_point + vicinity_seconds)
            invalid_intervals.append(TimeInterval(interval_start, interval_end))

        # Merge overlapping invalid intervals
        invalid_intervals = uniquefy_interval_array(invalid_intervals)

        # Find the complement of invalid intervals to get valid intervals
        valid_intervals = find_complement_of_interval_array(
            start_time, end_time, invalid_intervals
        )

        return valid_intervals

    return implementation
