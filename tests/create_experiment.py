import copy
import shutil
from contextlib import contextmanager
from pathlib import Path

import yaml

from .create_sequence_data import _generate_sequence_data

EXPERIMENT_ROOT = Path("tests/experiment")

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
    return copy.deepcopy(DEFAULT_CONFIG)


@contextmanager
def create_experiment(
    n_devices=2,
    devices_kwargs=None,
):
    devices_kwargs = devices_kwargs or [{}] * n_devices
    default_devices_kwargs = {
        "sampling_rate": 1.0,
    }
    devices_kwargs = [default_devices_kwargs | kwargs for kwargs in devices_kwargs]
    try:
        EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)

        for device_id, device_kwargs in enumerate(devices_kwargs):
            device_path = EXPERIMENT_ROOT / f"device_{device_id}"
            _generate_sequence_data(device_path, **device_kwargs)

        yield EXPERIMENT_ROOT
    finally:
        shutil.rmtree(EXPERIMENT_ROOT)
