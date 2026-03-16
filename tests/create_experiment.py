import numpy as np
import yaml


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


def make_modality_config(*device_names):
    return {
        name: {"interpolation": {"sampling_rate": 10.0, "offset": 0.0}}
        for name in device_names
    }
