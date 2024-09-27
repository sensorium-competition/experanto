import numba
import numpy as np


@numba.jit
def linear_interpolate_1d_sequence(row, times_old, times_new, keep_nans=False):
    """
    Interpolates columns in a NumPy array and replaces NaNs with interpolated values

    Args:
        array: The input NumPy array [Neurons, times]
        times: old time points [Neurons, times] or [times]
        times_new:  new time points [times2]
        keep_nans:  if we want to keep and return nans after interpolation

    Returns:
        The interpolated array with NaNs replaced (inplace).
    """
    if keep_nans:
        interpolated_array = np.interp(times_new, times_old, row)
    else:
        # Find indices of non-NaN values
        valid_indices = ~np.isnan(row)
        # Interpolate the column using linear interpolation
        interpolated_array = np.interp(
            times_new, times_old[valid_indices], row[valid_indices]
        )
    return interpolated_array


def linear_interpolate_sequences(array, times, times_new, keep_nans=False):
    """
    Interpolates columns in a NumPy array and replaces NaNs with interpolated values

    Args:
        array: The input NumPy array [times, ch]
        times: old time points  [times]
        times_new:  new time points [times2]
        keep_nans:  if we want to keep and return nans after interpolation

    Returns:
        The interpolated array with NaNs replaced.
    """
    array = array.T
    if array.shape[0] == 1:
        return linear_interpolate_1d_sequence(
            array.T.flatten(), times, times_new, keep_nans=keep_nans
        )
    interpolated_array = np.array(
        [
            linear_interpolate_1d_sequence(row, times, times_new, keep_nans=False)
            for row in array
        ]
    )
    return interpolated_array.T
