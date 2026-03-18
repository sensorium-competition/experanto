import logging
from contextlib import ExitStack

import pytest

from experanto.experiment import Experiment

from .create_experiment import make_modality_config, make_sequence_device

DEVICE_TIME_RANGE_CASES = [
    # Single device: start and end should match that device's range
    ([(2.0, 9.0)], 2.0, 9.0),
    # Two devices with different ranges: start should be min, end should be max
    ([(1.0, 8.0), (0.0, 10.0)], 0.0, 10.0),
    # Three devices with different ranges: start should be min, end should be max
    ([(0.0, 10.0), (1.0, 8.0), (2.0, 9.0)], 0.0, 10.0),
    # Devices with non-overlapping ranges: start should be min, end should be max
    ([(0.0, 3.0), (7.0, 8.0)], 0.0, 8.0),
    # Devices with identical ranges: start and end should match that range
    ([(1.0, 5.0), (1.0, 5.0)], 1.0, 5.0),
    # Large time stamps: start should be min, end should be max
    ([(1e9, 1e9 + 100), (1e9 - 50, 1e9 + 50)], 1e9 - 50, 1e9 + 100),
]

DEVICE_TIME_RANGE_IDS = [
    "single_device",
    "two_devices_different_ranges",
    "three_devices_different_ranges",
    "non_overlapping_ranges",
    "identical_ranges",
    "large_time_stamps"
]

# Inverted range is intentionally separate from INVALID_META_CASES —
# None/NaN/inf are caught per-device before being added to self.devices,
# whereas start > end is only caught after all devices are loaded.
INVALID_META_CASES = [
    {"start_time": None, "end_time": None},  # Both missing
    {"start_time": None, "end_time": 10.0},  # Missing start_time
    {"start_time": 0.0, "end_time": None},  # Missing end_time
    {"start_time": float("inf"), "end_time": 10.0},  # Infinite start_time
    {"start_time": 0.0, "end_time": float("inf")},  # Infinite end_time
    {"start_time": float("-inf"), "end_time": 10.0},  # Negative Infinite start_time
    {"start_time": 0.0, "end_time": float("-inf")},  # Negative Infinite end_time
    {"start_time": float("nan"), "end_time": 10.0},  # NaN start_time
    {"start_time": 0.0, "end_time": float("nan")},  # NaN end_time
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
    "nan_end_time"
]

# Test for union of device time ranges
@pytest.mark.parametrize("n_signals", [5, 20])
@pytest.mark.parametrize(
    "device_ranges, expected_start, expected_end", 
    DEVICE_TIME_RANGE_CASES,
    ids=DEVICE_TIME_RANGE_IDS,
)
def test_experiment_start_end_time_reflects_union(
    tmp_path, device_ranges, expected_start, expected_end, n_signals
):
    """
    Experiment.start_time and end_time should reflect the union of all
    device time ranges — earliest start and latest end across all devices.
    """
    device_names = [f"device_{i}" for i in range(len(device_ranges))]

    with ExitStack() as stack:
        for name, (start, end) in zip(device_names, device_ranges):
            stack.enter_context(
                make_sequence_device(
                    tmp_path, name, start=start, end=end, n_signals=n_signals
                )
            )

        experiment = Experiment(
            root_folder=tmp_path,
            modality_config=make_modality_config(*device_names),
        )

    assert experiment.start_time == pytest.approx(
        expected_start
    ), f"Expected start_time={expected_start}, got {experiment.start_time}"
    assert experiment.end_time == pytest.approx(
        expected_end
    ), f"Expected end_time={expected_end}, got {experiment.end_time}"


# Safety check
@pytest.mark.parametrize("override_meta", INVALID_META_CASES, ids=INVALID_META_IDS)
def test_experiment_invalid_metadata(tmp_path, override_meta):
    """
    Experiment should raise an error when initialized with invalid metadata.
    Covers cases where start_time or end_time is None, NaN, or infinite.
    """
    with make_sequence_device(
        tmp_path,
        "device_0",
        start=0.0,
        end=10.0,
        override_meta=override_meta,
    ):
        with pytest.raises(
            ValueError, match="Experiment time range could not be determined"
        ):
            Experiment(
                root_folder=tmp_path,
                modality_config=make_modality_config("device_0"),
            )

def test_experiment_inverted_time_range_raises(tmp_path):
    """
    Experiment should raise ValueError when start_time > end_time.
    This is a separate guard from invalid metadata (None/NaN/inf) because it 
    only becomes apparent after all devices are loaded and the overall time range is computed.
    """
    with make_sequence_device(
        tmp_path,
        "device_0",
        start=0.0,
        end=10.0, 
        override_meta={"start_time": 5.0, "end_time": 2.0},
    ):
        with pytest.raises(ValueError, match="Experiment time range could not be determined"):
            Experiment(
                root_folder=tmp_path,
                modality_config=make_modality_config("device_0"),
            )

@pytest.mark.parametrize("override_meta", INVALID_META_CASES, ids=INVALID_META_IDS)
def test_experiment_skips_invalid_devices(tmp_path, override_meta, caplog):
    """
    Experiment should skip devices with invalid start_time or end_time and
    log a warning, but still initialize successfully if at least one valid
    device is present. The experiment time range should reflect only the
    valid device.
    """
    with ExitStack() as stack:
        # Valid device with proper metadata
        stack.enter_context(
            make_sequence_device(
                tmp_path,
                "valid_device",
                start=0.0,
                end=10.0,
            )
        )
        # Invalid device with missing start_time and end_time
        stack.enter_context(
            make_sequence_device(
                tmp_path,
                "invalid_device",
                start=0.0,
                end=10.0,
                override_meta=override_meta,
            )
        )

        with caplog.at_level(logging.WARNING, logger="experanto.experiment"):
            experiment = Experiment(
                root_folder=tmp_path,
                modality_config=make_modality_config("valid_device", "invalid_device"),
            )

    assert "valid_device" in experiment.devices
    assert "invalid_device" not in experiment.devices

    assert experiment.start_time == pytest.approx(0.0), (
        f"Expected start_time=0.0, got {experiment.start_time}"
    )
    assert experiment.end_time == pytest.approx(10.0), (
        f"Expected end_time=10.0, got {experiment.end_time}"
    )
    assert any("invalid_device" in message for message in caplog.messages),(
        "Expected warning about invalid_device was skipped"
    )