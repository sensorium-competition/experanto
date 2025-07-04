import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

TIME_INTERVAL_ROOT = Path("tests/time_interval_data")


@contextmanager
def create_time_intervals_data(type="simple", t_end=1000.0):
    try:
        TIME_INTERVAL_ROOT.mkdir(parents=True, exist_ok=True)

        meta = {
            "labels": {
                "test": "test.npy",
                "train": "train.npy",
                "validation": "validation.npy",
            },
            "start_time": 0.0,
            "end_time": t_end,
            "modality": "time_interval",
        }
        timestamps = np.arange(t_end)

        with open(TIME_INTERVAL_ROOT / "meta.yml", "w") as f:
            yaml.dump(meta, f)

        if type == "simple":
            np.save(TIME_INTERVAL_ROOT / "test.npy", np.array([[0, 200]]))
            np.save(
                TIME_INTERVAL_ROOT / "train.npy",
                np.array([[200, 400], [600, 800]]),
            )
            np.save(
                TIME_INTERVAL_ROOT / "validation.npy",
                np.array([[400, 600], [800, t_end]]),
            )
        elif type == "overlap2":
            np.save(TIME_INTERVAL_ROOT / "test.npy", np.array([[0, 250]]))
            np.save(
                TIME_INTERVAL_ROOT / "train.npy",
                np.array([[200, 400], [600, 800]]),
            )
            np.save(
                TIME_INTERVAL_ROOT / "validation.npy",
                np.array([[400, 600], [800, t_end]]),
            )
        elif type == "overlap3":
            np.save(TIME_INTERVAL_ROOT / "test.npy", np.array([[0, 250]]))
            np.save(
                TIME_INTERVAL_ROOT / "train.npy",
                np.array([[178, 403], [557, 823]]),
            )
            np.save(
                TIME_INTERVAL_ROOT / "validation.npy",
                np.array([[375, 601], [789, t_end]]),
            )
        elif type == "gap":
            np.save(TIME_INTERVAL_ROOT / "test.npy", np.array([[0, 180]]))
            np.save(
                TIME_INTERVAL_ROOT / "train.npy",
                np.array([[200, 400], [632, 800]]),
            )
            np.save(
                TIME_INTERVAL_ROOT / "validation.npy",
                np.array([[420, 600], [827, t_end]]),
            )
        elif type == "gap_and_overlap":
            np.save(TIME_INTERVAL_ROOT / "test.npy", np.array([[0, 250]]))
            np.save(
                TIME_INTERVAL_ROOT / "train.npy",
                np.array([[200, 390], [600, 800]]),
            )
            np.save(
                TIME_INTERVAL_ROOT / "validation.npy",
                np.array([[400, 600], [800, t_end]]),
            )
        elif type == "nans":
            np.save(TIME_INTERVAL_ROOT / "test.npy", np.array([[0, 150]]))
            np.save(
                TIME_INTERVAL_ROOT / "train.npy",
                np.array([[200, 400], [600, 800]]),
            )
            np.save(
                TIME_INTERVAL_ROOT / "validation.npy",
                np.array([[400, 600], [800, t_end]]),
            )
            timestamps[100:250] = np.nan

        yield timestamps

    finally:
        shutil.rmtree(TIME_INTERVAL_ROOT, ignore_errors=True)
