import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import yaml

from experanto.experiment import Experiment

EXPERIMENT_ROOT = Path("tests/experiment_data")


@contextmanager
def create_two_device_experiment(
    device0_start=0.0,
    device0_end=10.0,
    device1_start=1.0,
    device1_end=8.0,
    sampling_rate=10.0,
):
    """Create a temporary experiment with two sequence devices with different time ranges."""
    try:
        for device_name, start, end in [
            ("device_0", device0_start, device0_end),
            ("device_1", device1_start, device1_end),
        ]:
            device_root = EXPERIMENT_ROOT / device_name
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

        yield EXPERIMENT_ROOT

    finally:
        shutil.rmtree(EXPERIMENT_ROOT)


@contextmanager
def create_three_device_experiment():
    """Create a temporary experiment with three sequence devices with different time ranges."""
    try:
        for device_name, start, end in [
            ("device_0", 0.0, 10.0),
            ("device_1", 1.0, 8.0),
            ("device_2", 2.0, 9.0),
        ]:
            device_root = EXPERIMENT_ROOT / device_name
            (device_root / "meta").mkdir(parents=True, exist_ok=True)

            sampling_rate = 10.0
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

        yield EXPERIMENT_ROOT

    finally:
        shutil.rmtree(EXPERIMENT_ROOT)


def get_two_device_config():
    return {
        "device_0": {"interpolation": {"sampling_rate": 10.0, "offset": 0.0}},
        "device_1": {"interpolation": {"sampling_rate": 10.0, "offset": 0.0}},
    }


def test_experiment_start_end_time_reflects_union():
    """
    Experiment.start_time and end_time should reflect the union
    of all device time ranges — the earliest start and latest end
    across all devices, not just the last loaded device.
    """
    with create_two_device_experiment(
        device0_start=1.0,
        device0_end=8.0,
        device1_start=0.0,
        device1_end=10.0,
    ) as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path,
            modality_config=get_two_device_config(),
        )

        # Union: start = min(1.0, 0.0) = 0.0, end = max(8.0, 10.0) = 10.0
        assert experiment.start_time == pytest.approx(0.0), (
            f"Expected start_time=0.0, got {experiment.start_time}"
        )
        assert experiment.end_time == pytest.approx(10.0), (
            f"Expected end_time=10.0, got {experiment.end_time}"
        )


def test_experiment_single_device_time_range():
    """With a single device, start_time and end_time should match that device's range."""
    with create_two_device_experiment(
        device0_start=2.0,
        device0_end=9.0,
    ) as experiment_path:
        config = {
            "device_0": {"interpolation": {"sampling_rate": 10.0, "offset": 0.0}}
        }
        experiment = Experiment(
            root_folder=experiment_path,
            modality_config=config,
        )

        assert experiment.start_time == pytest.approx(2.0)
        assert experiment.end_time == pytest.approx(9.0)


def test_experiment_start_end_time_three_devices():
    """With three devices, start_time and end_time should reflect the union of all three —
    earliest start and latest end across all devices."""
    with create_three_device_experiment() as experiment_path:
        config = {
            "device_0": {"interpolation": {"sampling_rate": 10.0, "offset": 0.0}},
            "device_1": {"interpolation": {"sampling_rate": 10.0, "offset": 0.0}},
            "device_2": {"interpolation": {"sampling_rate": 10.0, "offset": 0.0}},
        }
        experiment = Experiment(
            root_folder=experiment_path,
            modality_config=config,
        )

        # Union: start = min(0.0, 1.0, 2.0) = 0.0, end = max(10.0, 8.0, 9.0) = 10.0
        assert experiment.start_time == pytest.approx(0.0)
        assert experiment.end_time == pytest.approx(10.0)