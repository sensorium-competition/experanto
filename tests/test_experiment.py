import numpy as np
import pytest
import yaml

from experanto.experiment import Experiment


def make_sequence_device(root, name, start, end, sampling_rate=10.0):
    """Create a single sequence device folder under root."""
    device_root = root / name
    (device_root / "meta").mkdir(parents=True, exist_ok=True)

    n_samples = int((end - start) * sampling_rate) + 1
    timestamps = np.linspace(start, end, n_samples)
    data = np.random.rand(n_samples, 5)

    np.save(device_root / "timestamps.npy", timestamps)
    np.save(device_root / "data.npy", data)

    meta = {
        "start_time": start,
        "end_time": end,
        "modality": "sequence",
        "sampling_rate": sampling_rate,
        "phase_shift_per_signal": False,
        "is_mem_mapped": False,
        "n_signals": 5,
        "n_timestamps": n_samples,
        "dtype": "float64",
    }
    with open(device_root / "meta.yml", "w") as f:
        yaml.safe_dump(meta, f)


def make_config(*device_names):
    return {
        name: {"interpolation": {"sampling_rate": 10.0, "offset": 0.0}}
        for name in device_names
    }


def test_experiment_start_end_time_reflects_union(tmp_path):
    """
    Experiment.start_time and end_time should reflect the union of all
    device time ranges, earliest start and latest end across all devices.
    """
    make_sequence_device(tmp_path, "device_0", start=1.0, end=8.0)
    make_sequence_device(tmp_path, "device_1", start=0.0, end=10.0)

    experiment = Experiment(
        root_folder=tmp_path,
        modality_config=make_config("device_0", "device_1"),
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
        modality_config=make_config("device_0"),
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
        modality_config=make_config("device_0", "device_1", "device_2"),
    )

    # Union: start = min(0.0, 1.0, 2.0) = 0.0, end = max(10.0, 8.0, 9.0) = 10.0
    assert experiment.start_time == pytest.approx(0.0)
    assert experiment.end_time == pytest.approx(10.0)
