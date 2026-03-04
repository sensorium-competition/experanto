from unittest.mock import MagicMock

import numpy as np
import pytest

from experanto.experiment import Experiment
from experanto.interpolators import Interpolator

from .create_experiment import create_experiment, get_default_config


class TestInterpolator(Interpolator):
    """Small concrete interpolator used for testing Experiment routing logic."""

    def __init__(self):
        self.start_time = 0.0
        self.end_time = 100.0
        self.valid_interval = (self.start_time, self.end_time)
        self.interpolate = MagicMock(return_value=np.array([1, 2, 3]))


@pytest.fixture
def mock_interpolator():
    """Shared interpolator instance to isolate Experiment logic from interpolation math."""
    return TestInterpolator()


def test_experiment_initialization_and_device_loading(tmp_path, mock_interpolator):
    """Verify that only devices defined in modality_config are initialized."""
    (tmp_path / "screen").mkdir()
    (tmp_path / "eye_tracker").mkdir()
    (tmp_path / "ignored_device").mkdir()

    config = {
        "screen": {"interpolation": mock_interpolator},
        "eye_tracker": {"interpolation": mock_interpolator},
    }

    exp = Experiment(root_folder=str(tmp_path), modality_config=config)

    assert "screen" in exp.devices
    assert "eye_tracker" in exp.devices
    assert "ignored_device" not in exp.devices
    assert set(exp.device_names) == {"screen", "eye_tracker"}


def test_experiment_interpolate_routing(tmp_path, mock_interpolator):
    """Check if Experiment correctly delegates calls to underlying interpolators."""
    (tmp_path / "screen").mkdir()
    config = {"screen": {"interpolation": mock_interpolator}}
    exp = Experiment(root_folder=str(tmp_path), modality_config=config)
    test_times = np.array([10.0, 20.0])

    # Single device routing
    res = exp.interpolate(test_times, device="screen")
    mock_interpolator.interpolate.assert_called_once_with(
        test_times, return_valid=False
    )
    np.testing.assert_array_equal(res, np.array([1, 2, 3]))

    # Bulk routing (device=None)
    res_dict = exp.interpolate(test_times, device=None)
    assert "screen" in res_dict
    np.testing.assert_array_equal(res_dict["screen"], np.array([1, 2, 3]))


@pytest.mark.parametrize(
    "device_name, start_t, end_t", [("device_0", 0.0, 10.0), ("device_1", 0.0, 20.0)]
)
def test_get_valid_range_all_devices(device_name, start_t, end_t):
    """Integration test for valid_interval propagation from disk to object."""
    with create_experiment(
        devices_kwargs=[{"t_end": 10.0}, {"t_end": 20.0}],
    ) as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path,
            modality_config=get_default_config(),
        )

        valid_range = experiment.get_valid_range(device_name)
        assert valid_range == (start_t, end_t)
        assert isinstance(valid_range, tuple)


def test_get_valid_range_raises_for_invalid_device():
    with create_experiment() as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path,
            modality_config=get_default_config(),
        )
        with pytest.raises(KeyError):
            experiment.get_valid_range("device_does_not_exist")


def test_experiment_with_non_zero_start_time():
    """Test boundary conditions for data not starting at t=0."""
    start_offset, duration = 1.5, 10.0

    with create_experiment(
        devices_kwargs=[{"t_end": start_offset + duration, "start_time": start_offset}]
    ) as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path,
            modality_config=get_default_config(),
        )

        # Confirm interpolation doesn't crash at the offset start
        res = experiment.interpolate(np.array([start_offset]), device="device_0")
        assert res is not None


def test_experiment_numeric_precision_offset():
    """Stress test using non-integer rates and offsets to catch float drift."""
    start_offset = 0.123456789
    sampling_rate = 33.3333333
    duration = 1.0

    with create_experiment(
        devices_kwargs=[
            {
                "start_time": start_offset,
                "t_end": start_offset + duration,
                "sampling_rate": sampling_rate,
            }
        ]
    ) as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path,
            modality_config=get_default_config(),
        )

        valid_range = experiment.get_valid_range("device_0")
        assert valid_range[0] == pytest.approx(start_offset)
        assert valid_range[1] == pytest.approx(start_offset + duration)

        res = experiment.interpolate(np.array([start_offset]), device="device_0")
        assert res is not None


def test_experiment_irregular_timestamps():
    """Verify interpolation stability with jittered (non-linear) time steps."""
    with create_experiment(
        devices_kwargs=[{"irregular": True, "sampling_rate": 10.0}]
    ) as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path,
            modality_config=get_default_config(),
        )

        valid_range = experiment.get_valid_range("device_0")
        mid_point = (valid_range[0] + valid_range[1]) / 2

        res = experiment.interpolate(np.array([mid_point]), device="device_0")
        assert res is not None
        assert res.shape == (1, 10)


def test_experiment_multi_device_interpolation():
    """Check data consistency when interpolating across multiple modalities."""
    with create_experiment(n_devices=2) as experiment_path:
        exp = Experiment(
            root_folder=experiment_path, modality_config=get_default_config()
        )

        times = np.array([1.0, 2.0])
        results = exp.interpolate(times, device=None)

        assert isinstance(results, dict)
        assert "device_0" in results and "device_1" in results
        assert results["device_0"].shape == (2, 10)
