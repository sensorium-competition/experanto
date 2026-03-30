import logging
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from experanto.experiment import Experiment
from experanto.interpolators import Interpolator

from .create_experiment import (
    make_modality_config,
    setup_test_experiment,
)

# --- Test Data and Mocks ---

DEVICE_TIME_RANGE_CASES = [
    ([(2.0, 9.0)], 2.0, 9.0),
    ([(1.0, 8.0), (0.0, 10.0)], 0.0, 10.0),
    ([(0.0, 10.0), (1.0, 8.0), (2.0, 9.0)], 0.0, 10.0),
    ([(0.0, 3.0), (7.0, 8.0)], 0.0, 8.0),
    ([(1.0, 5.0), (1.0, 5.0)], 1.0, 5.0),
    ([(1e9, 1e9 + 100), (1e9 - 50, 1e9 + 50)], 1e9 - 50, 1e9 + 100),
]

DEVICE_TIME_RANGE_IDS = [
    "single_device",
    "two_devices_different_ranges",
    "three_devices_different_ranges",
    "non_overlapping_ranges",
    "identical_ranges",
    "large_time_stamps",
]

INVALID_META_CASES = [
    {"start_time": None, "end_time": None},
    {"start_time": None, "end_time": 10.0},
    {"start_time": 0.0, "end_time": None},
    {"start_time": float("inf"), "end_time": 10.0},
    {"start_time": 0.0, "end_time": float("inf")},
    {"start_time": float("-inf"), "end_time": 10.0},
    {"start_time": 0.0, "end_time": float("-inf")},
    {"start_time": float("nan"), "end_time": 10.0},
    {"start_time": 0.0, "end_time": float("nan")},
]

INVALID_META_IDS = [
    "both_missing",
    "missing_start_time",
    "missing_end_time",
    "infinite_start_time",
    "infinite_end_time",
    "negative_infinite_start_time",
    "negative_infinite_end_time",
    "nan_start_time",
    "nan_end_time",
]


class DummyInterpolator(Interpolator):
    """Small concrete interpolator used for testing Experiment routing logic."""

    def __init__(self):
        self.start_time = 0.0
        self.end_time = 100.0
        self.valid_interval = (self.start_time, self.end_time)
        self.interpolate = MagicMock(return_value=np.array([1, 2, 3]))


@pytest.fixture
def mock_interpolator():
    """Shared interpolator instance to isolate Experiment logic from interpolation math."""
    return DummyInterpolator()


# --- Tests ---


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
    res = exp.interpolate(test_times, device="screen")
    mock_interpolator.interpolate.assert_called_once_with(
        test_times, return_valid=False
    )
    np.testing.assert_array_equal(res, np.array([1, 2, 3]))
    res_dict = exp.interpolate(test_times, device=None)
    assert isinstance(res_dict, dict)
    np.testing.assert_array_equal(res_dict["screen"], np.array([1, 2, 3]))


@pytest.mark.parametrize(
    "device_name, start_t, end_t",
    [("device_0", 0.0, 10.0), ("device_1", 0.0, 20.0), ("device_2", 5.0, 15.0)],
)
def test_get_valid_range_all_devices(tmp_path, device_name, start_t, end_t):
    """Integration test for valid_interval propagation from disk to object."""
    with setup_test_experiment(
        tmp_path,
        n_devices=3,
        devices_kwargs=[
            {"t_end": 10.0},
            {"t_end": 20.0},
            {"start_time": 5.0, "t_end": 15.0},
        ],
    ) as experiment_path:
        config = make_modality_config("device_0", "device_1", "device_2")
        config["device_2"] = {
            "sampling_rate": 1.0,
            "chunk_size": 40,
            "interpolation": {"interpolation_mode": "nearest_neighbor"},
        }
        experiment = Experiment(
            root_folder=str(experiment_path), modality_config=config
        )
        valid_range = experiment.get_valid_range(device_name)
        assert valid_range == (start_t, end_t)


def test_get_valid_range_raises_for_invalid_device(tmp_path):
    with setup_test_experiment(tmp_path) as experiment_path:
        experiment = Experiment(
            root_folder=str(experiment_path),
            modality_config=make_modality_config("device_0", "device_1"),
        )
        with pytest.raises(KeyError):
            experiment.get_valid_range("device_does_not_exist")


def test_experiment_with_non_zero_start_time(tmp_path):
    """Test boundary conditions for data not starting at t=0."""
    start_offset, duration = 1.5, 10.0
    with setup_test_experiment(
        tmp_path,
        n_devices=1,
        devices_kwargs=[{"t_end": start_offset + duration, "start_time": start_offset}],
    ) as experiment_path:
        experiment = Experiment(
            root_folder=str(experiment_path),
            modality_config=make_modality_config("device_0"),
        )
        res = experiment.interpolate(np.array([start_offset + 1.0]), device="device_0")
        assert res is not None
        with pytest.warns(UserWarning, match="no valid times queried"):
            experiment.interpolate(np.array([start_offset - 1.0]), device="device_0")


@given(
    start_offset=st.floats(min_value=0.0, max_value=100.0),
    sampling_rate=st.floats(min_value=0.1, max_value=100.0),
    duration=st.floats(min_value=0.0, max_value=100.0),
)
@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_experiment_numeric_precision_offset(
    tmp_path, start_offset, sampling_rate, duration
):
    """Stress test using non-integer rates and offsets to catch float drift."""
    with setup_test_experiment(
        tmp_path,
        n_devices=1,
        devices_kwargs=[
            {
                "start_time": start_offset,
                "t_end": start_offset + duration,
                "sampling_rate": sampling_rate,
            }
        ],
    ) as experiment_path:
        experiment = Experiment(
            root_folder=str(experiment_path),
            modality_config=make_modality_config("device_0"),
        )
        valid_range = experiment.get_valid_range("device_0")
        # Keep pytest.approx here as it's dealing with floats generated by Hypothesis
        assert valid_range[0] == pytest.approx(start_offset)
        assert valid_range[1] == pytest.approx(start_offset + duration)
        res = experiment.interpolate(np.array([start_offset]), device="device_0")
        assert res is not None


@pytest.mark.parametrize("return_valid", [False, True])
@pytest.mark.parametrize("device", [None, "device_0"])
def test_experiment_multi_device_interpolation(tmp_path, return_valid, device):
    """Check data consistency when interpolating across multiple modalities."""
    with setup_test_experiment(tmp_path, n_devices=2) as experiment_path:
        exp = Experiment(
            root_folder=str(experiment_path),
            modality_config=make_modality_config("device_0", "device_1"),
        )
        times = np.array([1.0, 2.0])
        results = exp.interpolate(times, device=device, return_valid=return_valid)
        if return_valid:
            data, valid_idx = results
            assert (
                data["device_0"].shape == (2, 10)
                if device is None
                else data.shape == (2, 10)
            )
        else:
            assert (
                results["device_0"].shape == (2, 10)
                if device is None
                else results.shape == (2, 10)
            )


@pytest.mark.parametrize("n_signals", [5, 20])
@pytest.mark.parametrize(
    "device_ranges, expected_start, expected_end",
    DEVICE_TIME_RANGE_CASES,
    ids=DEVICE_TIME_RANGE_IDS,
)
def test_experiment_start_end_time_reflects_union(
    tmp_path, device_ranges, expected_start, expected_end, n_signals
):
    """Experiment.start_time and end_time should reflect the union of all device time ranges."""
    devices_kwargs = [
        {
            "start_time": start,
            "t_end": end,
            "n_signals": n_signals,
            "sampling_rate": float(np.random.randint(5, 30)),
        }
        for start, end in device_ranges
    ]

    with setup_test_experiment(
        tmp_path, n_devices=len(device_ranges), devices_kwargs=devices_kwargs
    ) as experiment_path:
        # Dynamically build the config using make_modality_config
        device_names = [f"device_{i}" for i in range(len(device_ranges))]
        offsets = [float(np.random.rand()) for _ in device_ranges]
        config = make_modality_config(*device_names, offsets=offsets)

        experiment = Experiment(
            root_folder=str(experiment_path), modality_config=config
        )

    # Removed pytest.approx here
    assert experiment.start_time == expected_start
    assert experiment.end_time == expected_end


@pytest.mark.parametrize("override_meta", INVALID_META_CASES, ids=INVALID_META_IDS)
def test_experiment_invalid_metadata(tmp_path, override_meta):
    with setup_test_experiment(
        tmp_path, n_devices=1, devices_kwargs=[{"start_time": 0.0, "t_end": 10.0}]
    ) as experiment_path:
        # Explicitly corrupt the generated metadata file
        meta_file = experiment_path / "device_0" / "meta.yml"
        with open(meta_file, "r") as f:
            meta = yaml.safe_load(f)
        meta.update(override_meta)
        with open(meta_file, "w") as f:
            yaml.safe_dump(meta, f)

        config = make_modality_config("device_0")

        with pytest.raises(
            ValueError, match="Experiment time range could not be determined"
        ):
            Experiment(root_folder=str(experiment_path), modality_config=config)


@pytest.mark.parametrize("override_meta", INVALID_META_CASES, ids=INVALID_META_IDS)
def test_experiment_skips_invalid_devices(tmp_path, override_meta, caplog):
    start_val, duration_val = (
        np.random.lognormal(0.0, 1.0),
        np.random.lognormal(0.0, 1.0),
    )
    end_val = start_val + duration_val

    devices_kwargs = [
        {"start_time": start_val, "t_end": end_val},  # valid device
        {"start_time": 0.0, "t_end": 10.0},  # invalid device
    ]

    with setup_test_experiment(
        tmp_path, n_devices=2, devices_kwargs=devices_kwargs
    ) as experiment_path:
        # Rename the folders to match what the old test expected
        (experiment_path / "device_0").rename(experiment_path / "valid_device")
        (experiment_path / "device_1").rename(experiment_path / "invalid_device")

        # Explicitly corrupt the metadata file for the invalid device
        meta_file = experiment_path / "invalid_device" / "meta.yml"
        with open(meta_file, "r") as f:
            meta = yaml.safe_load(f)
        meta.update(override_meta)
        with open(meta_file, "w") as f:
            yaml.safe_dump(meta, f)

        config = make_modality_config("valid_device", "invalid_device")

        with caplog.at_level(logging.WARNING, logger="experanto.experiment"):
            experiment = Experiment(
                root_folder=str(experiment_path), modality_config=config
            )

    assert "valid_device" in experiment.devices
    assert "invalid_device" not in experiment.devices
    # Removed pytest.approx here
    assert experiment.start_time == start_val
    assert experiment.end_time == end_val
