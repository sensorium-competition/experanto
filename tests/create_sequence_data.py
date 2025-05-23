import shutil
from contextlib import closing, contextmanager
from pathlib import Path

import numpy as np
import yaml

from experanto.interpolators import Interpolator

SEQUENCE_ROOT = Path("tests/sequence_data")


@contextmanager
def create_sequence_data(
    n_signals=10,
    shifts_per_signal=False,
    use_mem_mapped=False,
    t_end=10.0,
    sampling_rate=10.0,
    contain_nans=False,
):
    try:
        SEQUENCE_ROOT.mkdir(parents=True, exist_ok=True)
        (SEQUENCE_ROOT / "meta").mkdir(parents=True, exist_ok=True)

        meta = {
            "start_time": 0,
            "end_time": t_end,
            "modality": "sequence",
            "sampling_rate": sampling_rate,
            "phase_shift_per_signal": shifts_per_signal,
            "is_mem_mapped": use_mem_mapped,
            "n_signals": n_signals,
        }

        timestamps = np.linspace(
            meta["start_time"],
            meta["end_time"],
            int((meta["end_time"] - meta["start_time"]) * meta["sampling_rate"]) + 1,
        )
        np.save(SEQUENCE_ROOT / "timestamps.npy", timestamps)
        meta["n_timestamps"] = len(timestamps)

        data = np.random.rand(len(timestamps), n_signals)

        if contain_nans:
            nan_indices = np.random.choice(
                data.size, size=int(0.1 * data.size), replace=False
            )
            data.flat[nan_indices] = np.nan

        if not use_mem_mapped:
            np.save(SEQUENCE_ROOT / "data.npy", data)
        else:
            filename = SEQUENCE_ROOT / "data.mem"

            fp = np.memmap(filename, dtype=data.dtype, mode="w+", shape=data.shape)
            fp[:] = data[:]
            fp.flush()  # Ensure data is written to disk
            del fp
        meta["dtype"] = str(data.dtype)

        if shifts_per_signal:
            shifts = np.random.rand(n_signals) / meta["sampling_rate"] * 0.9
            np.save(SEQUENCE_ROOT / "meta" / "phase_shifts.npy", shifts)
        else:
            shifts = np.zeros(n_signals)

        with open(SEQUENCE_ROOT / "meta.yml", "w") as f:
            yaml.dump(meta, f)

        yield timestamps, data, shifts
    finally:
        shutil.rmtree(SEQUENCE_ROOT)


@contextmanager
def sequence_data_and_interpolator(data_kwargs=None, interp_kwargs=None):
    data_kwargs = data_kwargs or {}
    interp_kwargs = interp_kwargs or {}
    with create_sequence_data(**data_kwargs) as (timestamps, data, shifts):
        with closing(
            Interpolator.create("tests/sequence_data", **interp_kwargs)
        ) as seq_interp:
            yield timestamps, data, shifts, seq_interp
