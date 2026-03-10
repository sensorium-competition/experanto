import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

SEQUENCE_ROOT = Path("tests/sequence_data")


def _generate_sequence_data(
    sequence_root,
    n_signals=10,
    shifts_per_signal=False,
    use_mem_mapped=False,
    start_time=0.0,
    t_end=10.0,
    sampling_rate=10.0,
    contain_nans=False,
    irregular=False,
):
    """Generates synthetic sequence data folders for testing interpolator logic."""
    from pathlib import Path

    sequence_root = Path(sequence_root)
    sequence_root.mkdir(parents=True, exist_ok=True)
    (sequence_root / "meta").mkdir(parents=True, exist_ok=True)

    meta = {
        "start_time": start_time,
        "end_time": t_end,
        "modality": "sequence",
        "sampling_rate": sampling_rate,
        "phase_shift_per_signal": shifts_per_signal,
        "is_mem_mapped": use_mem_mapped,
        "n_signals": n_signals,
    }

    # Determine number of samples based on duration and rate
    duration = meta["end_time"] - meta["start_time"]
    n_samples = int(round(duration * meta["sampling_rate"])) + 1

    if not irregular:
        timestamps = np.linspace(meta["start_time"], meta["end_time"], n_samples)
    else:
        steps = np.random.uniform(0.5, 1.5, size=n_samples - 1) / sampling_rate
        timestamps = np.zeros(n_samples)
        timestamps[0] = meta["start_time"]
        timestamps[1:] = meta["start_time"] + np.cumsum(steps)
        meta["end_time"] = float(timestamps[-1])

    data = np.random.rand(len(timestamps), n_signals)

    if contain_nans:
        nan_indices = np.random.choice(
            data.size, size=int(0.1 * data.size), replace=False
        )
        data.flat[nan_indices] = np.nan

    if not use_mem_mapped:
        np.save(sequence_root / "data.npy", data)
    else:
        filename = sequence_root / "data.mem"
        fp = np.memmap(filename, dtype=data.dtype, mode="w+", shape=data.shape)
        fp[:] = data[:]
        fp.flush()
        del fp

    np.save(sequence_root / "timestamps.npy", timestamps)
    meta["n_timestamps"] = len(timestamps)
    meta["dtype"] = str(data.dtype)

    # Handle per-signal phase shifts if required by the test case
    shifts = None
    if shifts_per_signal:
        shifts = np.random.rand(n_signals) / meta["sampling_rate"] * 0.9
        np.save(sequence_root / "meta" / "phase_shifts.npy", shifts)

    with open(sequence_root / "meta.yml", "w") as f:
        yaml.safe_dump(meta, f)

    return timestamps, data, shifts


@contextmanager
def create_sequence_data(
    n_signals=10,
    shifts_per_signal=False,
    use_mem_mapped=False,
    t_end=10.0,
    sampling_rate=10.0,
    contain_nans=False,
    start_time=0.0,
):
    """Context manager for temporary sequence data creation and cleanup."""
    try:
        yield _generate_sequence_data(
            sequence_root=SEQUENCE_ROOT,
            n_signals=n_signals,
            shifts_per_signal=shifts_per_signal,
            use_mem_mapped=use_mem_mapped,
            t_end=t_end,
            sampling_rate=sampling_rate,
            contain_nans=contain_nans,
            start_time=start_time,
        )
    finally:
        if SEQUENCE_ROOT.exists():
            shutil.rmtree(SEQUENCE_ROOT)


@contextmanager
def sequence_data_and_interpolator(data_kwargs=None, interp_kwargs=None):
    data_kwargs = data_kwargs or {}
    interp_kwargs = interp_kwargs or {}
    with create_sequence_data(**data_kwargs) as (timestamps, data, shifts):
        # Restore the helper expected by the rest of the test suite
        from experanto.interpolators import Interpolator

        seq_interp = Interpolator.create(str(SEQUENCE_ROOT), **interp_kwargs)
        try:
            yield timestamps, data, shifts, seq_interp
        finally:
            seq_interp.close()
