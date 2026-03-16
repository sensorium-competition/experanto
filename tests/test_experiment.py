import pytest

from experanto.experiment import Experiment

from .create_experiment import make_modality_config, make_sequence_device


def test_experiment_start_end_time_reflects_union(tmp_path):
    """
    Experiment.start_time and end_time should reflect the union of all
    device time ranges — earliest start and latest end across all devices.
    """
    make_sequence_device(tmp_path, "device_0", start=1.0, end=8.0)
    make_sequence_device(tmp_path, "device_1", start=0.0, end=10.0)

    experiment = Experiment(
        root_folder=tmp_path,
        modality_config=make_modality_config("device_0", "device_1"),
    )

    # Union: start = min(1.0, 0.0) = 0.0, end = max(8.0, 10.0) = 10.0
    assert experiment.start_time == pytest.approx(
        0.0
    ), f"Expected start_time=0.0, got {experiment.start_time}"
    assert experiment.end_time == pytest.approx(
        10.0
    ), f"Expected end_time=10.0, got {experiment.end_time}"


def test_experiment_single_device_time_range(tmp_path):
    """With a single device, start_time and end_time should match that device's range."""
    make_sequence_device(tmp_path, "device_0", start=2.0, end=9.0)

    experiment = Experiment(
        root_folder=tmp_path,
        modality_config=make_modality_config("device_0"),
    )

    assert experiment.start_time == pytest.approx(2.0)
    assert experiment.end_time == pytest.approx(9.0)


def test_experiment_start_end_time_three_devices(tmp_path):
    """With three devices, start_time and end_time should reflect the union of all three."""
    make_sequence_device(tmp_path, "device_0", start=0.0, end=10.0)
    make_sequence_device(tmp_path, "device_1", start=1.0, end=8.0)
    make_sequence_device(tmp_path, "device_2", start=2.0, end=9.0)

    experiment = Experiment(
        root_folder=tmp_path,
        modality_config=make_modality_config("device_0", "device_1", "device_2"),
    )

    # Union: start = min(0.0, 1.0, 2.0) = 0.0, end = max(10.0, 8.0, 9.0) = 10.0
    assert experiment.start_time == pytest.approx(0.0)
    assert experiment.end_time == pytest.approx(10.0)
