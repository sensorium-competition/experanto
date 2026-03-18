import shutil
from contextlib import contextmanager

import numpy as np
import yaml


@contextmanager
def make_sequence_device(root, name, start, end, sampling_rate=10.0, n_signals=5, override_meta=None):
    """Create a single sequence device folder under root."""
    device_root = root / name
    try:
        (device_root / "meta").mkdir(parents=True, exist_ok=True)

        n_samples = int((end - start) * sampling_rate) + 1 # +1 to include both start and end as sample points
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
        shutil.rmtree(device_root)


def make_modality_config(*device_names, sampling_rate=10.0, offset=0.0):
    return {
        name: {"interpolation": {"sampling_rate": sampling_rate, "offset": offset}}
        for name in device_names
    }
