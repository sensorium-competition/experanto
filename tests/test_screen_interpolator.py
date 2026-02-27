from pathlib import Path

import numpy as np
import pytest

from experanto.interpolators import Interpolator, ScreenInterpolator

from .create_screen_data import create_screen_data


@pytest.mark.parametrize("duration", [10, 20])
@pytest.mark.parametrize("fps", [10.0, 30])
@pytest.mark.parametrize("image_frame_count", [10, 100])
@pytest.mark.parametrize("num_videos", [1, 10])
def test_nearest_neighbor_interpolation(duration, fps, image_frame_count, num_videos):

    with create_screen_data(
        duration=duration,
        frame_shape=(32, 32),
        fps=fps,
        image_frame_count=image_frame_count,
        num_videos=num_videos,
    ) as timestamps:

        interp_obj = Interpolator.create("tests/screen_data")
        assert isinstance(interp_obj, ScreenInterpolator), "Expected ScreenInterpolator"

        delta_t = 1.0 / fps
        idx = slice(0, len(timestamps) - 1)
        times = timestamps[idx] + 0.4 * delta_t

        root = Path("tests/screen_data")
        data_dir = root / "data"

        npy_files = [
            f
            for f in sorted(root.glob("*.npy")) + sorted(data_dir.glob("*.npy"))
            if f.name != "timestamps.npy"
        ]

        frames = []

        for fpath in npy_files:
            arr = np.load(fpath)
            if arr.ndim == 3:
                for i in range(arr.shape[0]):
                    frames.append(arr[i])
            elif arr.ndim == 2:
                frames.append(arr)
            else:
                raise ValueError(f"Unexpected array shape {arr.shape} in {fpath}")

        raw_array = np.stack(frames)

        expected_indices = np.round((times - timestamps[0]) * fps).astype(int)
        expected_frames = raw_array[expected_indices]

        interp, valid = interp_obj.interpolate(times=times, return_valid=True)

        assert times.shape == valid.shape, "All interpolated frames should be valid"
        assert np.allclose(
            interp, expected_frames, atol=1e-5
        ), "Nearest neighbor interpolation mismatch"


def test_nearest_neighbor_interpolation_return_valid_false():
    with create_screen_data(
        duration=10,
        frame_shape=(32, 32),
        fps=10.0,
        image_frame_count=10,
        num_videos=1,
    ) as timestamps:
        interp_obj = Interpolator.create("tests/screen_data")
        assert isinstance(interp_obj, ScreenInterpolator), "Expected ScreenInterpolator"

        delta_t = 1.0 / 10.0
        times = timestamps[:-1] + 0.4 * delta_t

        result = interp_obj.interpolate(times=times, return_valid=False)
        assert isinstance(result, np.ndarray), "Expected np.ndarray, not a tuple"

        interp, _ = interp_obj.interpolate(times=times, return_valid=True)
        assert np.array_equal(
            result, interp
        ), "Data from return_valid=False should match data from return_valid=True"


def test_nearest_neighbor_interpolation_default_return_valid():
    with create_screen_data(
        duration=10,
        frame_shape=(32, 32),
        fps=10.0,
        image_frame_count=10,
        num_videos=1,
    ) as timestamps:
        interp_obj = Interpolator.create("tests/screen_data")
        assert isinstance(interp_obj, ScreenInterpolator), "Expected ScreenInterpolator"

        delta_t = 1.0 / 10.0
        times = timestamps[:-1] + 0.4 * delta_t

        result = interp_obj.interpolate(times=times)
        assert isinstance(result, np.ndarray), "Expected np.ndarray, not a tuple"

        interp, _ = interp_obj.interpolate(times=times, return_valid=True)
        assert np.array_equal(
            result, interp
        ), "Data from default (no return_valid) should match data from return_valid=True"
