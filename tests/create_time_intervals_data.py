import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

TIME_INTERVAL_ROOT = Path("tests/time_interval_data")


@contextmanager
def create_time_intervals_data():
    try:
        TIME_INTERVAL_ROOT.mkdir(parents=True, exist_ok=True)

        meta = {
            "labels": {
                "test": "test.npy",
                "train": "train.npy",
                "validation": "validation.npy",
            },
            "start_time": 0,
            "end_time": 1000,
            "modality": "time_interval",
        }

        with open(TIME_INTERVAL_ROOT / "meta.yml", "w") as f:
            yaml.dump(meta, f)

        np.save(TIME_INTERVAL_ROOT / "test.npy", np.array([[400, 600]]))
        np.save(
            TIME_INTERVAL_ROOT / "train.npy",
            np.array([[0, 200], [600, 800]]),
        )
        np.save(
            TIME_INTERVAL_ROOT / "validation.npy",
            np.array([[200, 400], [800, 1000]]),
        )

        timestamps = np.arange(1000)
        yield timestamps

    finally:
        shutil.rmtree(TIME_INTERVAL_ROOT, ignore_errors=True)
