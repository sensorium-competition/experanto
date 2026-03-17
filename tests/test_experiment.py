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
]

INVALID_META_CASES = [
    {"start_time": None, "end_time": None},  # Both missing
    {"start_time": None, "end_time": 10.0},  # Missing start_time
    {"start_time": 0.0, "end_time": None},  # Missing end_time
    {"start_time": float("inf"), "end_time": 10.0},  # Infinite start_time
    {"start_time": 0.0, "end_time": float("inf")},  # Infinite end_time
    {"start_time": 5.0, "end_time": 2.0},  # start_time > end_time
]


@pytest.mark.parametrize("n_signals", [5, 20])
@pytest.mark.parametrize(
    "device_ranges, expected_start, expected_end", DEVICE_TIME_RANGE_CASES
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


@pytest.mark.parametrize("override_meta", INVALID_META_CASES)
def test_experiment_invalid_metadata(tmp_path, override_meta):
    """
    Experiment should raise an error when initialized with invalid metadata.
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
