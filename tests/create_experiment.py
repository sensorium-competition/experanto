import copy
import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

from .create_sequence_data import _generate_sequence_data

DEFAULT_CONFIG = {
    "device_0": {
        "sampling_rate": 1.0,
        "chunk_size": 40,
        "interpolation": {
            "interpolation_mode": "nearest_neighbor",
        },
    },
    "device_1": {
        "sampling_rate": 1.0,
        "chunk_size": 60,
        "interpolation": {
            "interpolation_mode": "linear",
        },
    },
}


def get_default_config():
    """Return a fresh copy of the default modality configuration."""
    return copy.deepcopy(DEFAULT_CONFIG)


@contextmanager
def setup_test_experiment(
    tmp_path,
    n_devices=2,
    devices_kwargs=None,
    default_sampling_rate=1.0,
):
    devices_kwargs = devices_kwargs or [{}] * n_devices
    default_params = {"sampling_rate": default_sampling_rate}

    devices_kwargs = [default_params | kwargs for kwargs in devices_kwargs]

    try:
        tmp_path.mkdir(parents=True, exist_ok=True)
        for device_id, device_kwargs in enumerate(devices_kwargs):
            device_path = tmp_path / f"device_{device_id}"
            _generate_sequence_data(device_path, **device_kwargs)
        yield tmp_path
    finally:
        if tmp_path.exists():
            shutil.rmtree(tmp_path)


@contextmanager
def make_sequence_device(
    root, name, start, end, sampling_rate=10.0, n_signals=5, override_meta=None
):
    """Create a single sequence device folder under root."""
    device_root = root / name
    try:
        (device_root / "meta").mkdir(parents=True, exist_ok=True)
        n_samples = int((end - start) * sampling_rate) + 1
        timestamps = np.linspace(start, end, n_samples)
        data = np.random.rand(n_samples, n_signals)

        np.save(device_root / "timestamps.npy", timestamps)
        np.save(device_root / "data.npy", data)

        meta = {
            "start_time": start,
            "end_time": end,
            "modality": "sequence",
            "sampling_rate": sampling_rate,
            "phase_shift_per_signal": False,
            "is_mem_mapped": False,
            "n_signals": n_signals,
            "n_timestamps": n_samples,
            "dtype": "float64",
        }
        if override_meta:
            meta.update(override_meta)
        with open(device_root / "meta.yml", "w") as f:
            yaml.safe_dump(meta, f)
        yield device_root
    finally:
        if device_root.exists():
            shutil.rmtree(device_root)


def make_modality_config(*device_names, sampling_rates=None, offsets=None):
    if sampling_rates is None:
        sampling_rates = [10.0] * len(device_names)
    elif isinstance(sampling_rates, (int, float)):
        sampling_rates = [sampling_rates] * len(device_names)

    if offsets is None:
        offsets = [0.0] * len(device_names)
    elif isinstance(offsets, (int, float)):
        offsets = [offsets] * len(device_names)

    assert len(device_names) == len(sampling_rates), (
        f"sampling_rates length {len(sampling_rates)} does not match device_names length {len(device_names)}"
    )
    assert len(device_names) == len(offsets), (
        f"offsets length {len(offsets)} does not match device_names length {len(device_names)}"
    )

    return {
        name: {"interpolation": {"sampling_rate": sr, "offset": off}}
        for name, sr, off in zip(device_names, sampling_rates, offsets)
    }
