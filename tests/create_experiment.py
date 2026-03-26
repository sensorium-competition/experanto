import copy
import shutil
from contextlib import contextmanager

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
