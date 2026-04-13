import shutil
from contextlib import contextmanager

from .create_sequence_data import _generate_sequence_data


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


def make_modality_config(*device_names, sampling_rates=None, offsets=None):
    if sampling_rates is None:
        sampling_rates = [10.0] * len(device_names)
    elif isinstance(sampling_rates, (int, float)):
        sampling_rates = [sampling_rates] * len(device_names)

    if offsets is None:
        offsets = [0.0] * len(device_names)
    elif isinstance(offsets, (int, float)):
        offsets = [offsets] * len(device_names)

    assert len(device_names) == len(
        sampling_rates
    ), f"sampling_rates length {len(sampling_rates)} does not match device_names length {len(device_names)}"
    assert len(device_names) == len(
        offsets
    ), f"offsets length {len(offsets)} does not match device_names length {len(device_names)}"

    return {
        name: {"interpolation": {"sampling_rate": sr, "offset": off}}
        for name, sr, off in zip(device_names, sampling_rates, offsets)
    }
