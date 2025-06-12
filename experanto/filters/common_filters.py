from experanto.interpolators import Interpolator
import numpy as np
from experanto.intervals import TimeInterval, uniquefy_interval_array, find_complement_of_interval_array

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
        # We need to be more conservative with the vicinity to account for:
        # 1. Potential timing misalignments between device's native sampling and interpolation requests
        # 2. Nearest neighbor interpolation that might pick up a NaN value from a nearby sample
        # 3. Different sampling rates between devices

        # Use a larger buffer: at least 2 sample periods or 10% of vicinity, whichever is larger
        buffer = max(2 * time_delta, 0.1 * vicinity_seconds)
        effective_vicinity = vicinity_seconds + buffer

        for idx in nan_indices:
            time_point = start_time + idx * time_delta
            interval_start = max(start_time, time_point - vicinity_seconds)
            interval_end = min(end_time, time_point + vicinity_seconds)
            interval_start = max(start_time, time_point - effective_vicinity)
            interval_end = min(end_time, time_point + effective_vicinity)
            invalid_intervals.append(TimeInterval(interval_start, interval_end))

        # Also check for any potential edge cases at the boundaries
        # If there are NaNs very close to start or end, extend the invalid regions
        if len(nan_indices) > 0:
            first_nan_time = start_time + nan_indices[0] * time_delta
            last_nan_time = start_time + nan_indices[-1] * time_delta

            # If NaN is within vicinity of start/end boundaries, extend invalid region
            if first_nan_time - start_time < effective_vicinity:
                invalid_intervals.append(TimeInterval(start_time, first_nan_time + effective_vicinity))
            if end_time - last_nan_time < effective_vicinity:
                invalid_intervals.append(TimeInterval(last_nan_time - effective_vicinity, end_time))
        # Merge overlapping invalid intervals
        invalid_intervals = uniquefy_interval_array(invalid_intervals)
        
        # Find the complement of invalid intervals to get valid intervals
        valid_intervals = find_complement_of_interval_array(start_time, end_time, invalid_intervals)
        
        return valid_intervals

    return implementation