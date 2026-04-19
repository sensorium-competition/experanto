import numpy as np
import pytest

from experanto.interpolators import (
    PhaseShiftedSequenceInterpolator,
    SequenceInterpolator,
)

from .create_sequence_data import sequence_data_and_interpolator

DEFAULT_SEQUENCE_LENGTH = 10


@pytest.mark.parametrize("n_signals", [0, 1, 10, 50])
@pytest.mark.parametrize("sampling_rate", [3.0, 10.0, 100.0])
@pytest.mark.parametrize("use_mem_mapped", [False, True])
def test_nearest_neighbor_interpolation(n_signals, sampling_rate, use_mem_mapped):
    with sequence_data_and_interpolator(
        data_kwargs={
            "n_signals": n_signals,
            "use_mem_mapped": use_mem_mapped,
            "t_end": 5.0,
            "sampling_rate": sampling_rate,
        }
    ) as (timestamps, data, _, seq_interp):
        assert isinstance(
            seq_interp, SequenceInterpolator
        ), "Interpolation object is not a SequenceInterpolator"
        assert not isinstance(
            seq_interp, PhaseShiftedSequenceInterpolator
        ), "Interpolation object is a PhaseShiftedSequenceInterpolator"

        times = timestamps[:DEFAULT_SEQUENCE_LENGTH] + 1e-9
        interp, valid = seq_interp.interpolate(
            times=times, return_valid=True
        )  # Add a small epsilon to avoid floating point errors
        assert times.shape == valid.shape, "All samples should be valid"
        assert (
            interp == data[:DEFAULT_SEQUENCE_LENGTH]
        ).all(), "Nearest neighbor interpolation does not match expected data"
        assert valid.shape == (
            DEFAULT_SEQUENCE_LENGTH,
        ), f"Expected valid.shape == ({DEFAULT_SEQUENCE_LENGTH},), got {valid.shape}"
        assert interp.shape == (
            DEFAULT_SEQUENCE_LENGTH,
            n_signals,
        ), f"Expected interp.shape == ({DEFAULT_SEQUENCE_LENGTH}, {n_signals}), got {interp.shape}"


@pytest.mark.parametrize("n_signals", [0, 1, 10, 50])
@pytest.mark.parametrize("keep_nans", [False, True])
def test_nearest_neighbor_interpolation_handles_nans(n_signals, keep_nans):
    with sequence_data_and_interpolator(
        data_kwargs={
            "n_signals": n_signals,
            "use_mem_mapped": True,
            "t_end": 5.0,
            "sampling_rate": 10.0,
            "contain_nans": True,
        },
        interp_kwargs={"keep_nans": keep_nans},
    ) as (timestamps, data, _, seq_interp):
        assert isinstance(
            seq_interp, SequenceInterpolator
        ), "Interpolation object is not a SequenceInterpolator"

        times = timestamps[:DEFAULT_SEQUENCE_LENGTH] + 1e-9
        interp, valid = seq_interp.interpolate(
            times=times, return_valid=True
        )  # Add a small epsilon to avoid floating point errors
        assert times.shape == valid.shape, "All samples should be valid"
        assert np.array_equal(
            interp, data[:DEFAULT_SEQUENCE_LENGTH], equal_nan=True
        ), "Nearest neighbor interpolation does not match expected data"
        assert valid.shape == (
            DEFAULT_SEQUENCE_LENGTH,
        ), f"Expected valid.shape == ({DEFAULT_SEQUENCE_LENGTH},), got {valid.shape}"
        assert interp.shape == (
            DEFAULT_SEQUENCE_LENGTH,
            n_signals,
        ), f"Expected interp.shape == ({DEFAULT_SEQUENCE_LENGTH}, {n_signals}), got {interp.shape}"


@pytest.mark.parametrize("n_signals", [0, 1, 10, 50])
@pytest.mark.parametrize("sampling_rate", [3.0, 10.0, 100.0])
def test_nearest_neighbor_interpolation_with_inbetween_times(n_signals, sampling_rate):
    with sequence_data_and_interpolator(
        data_kwargs={
            "n_signals": n_signals,
            "use_mem_mapped": True,
            "t_end": 5.0,
            "sampling_rate": sampling_rate,
        }
    ) as (timestamps, data, _, seq_interp):
        assert isinstance(
            seq_interp, SequenceInterpolator
        ), "Interpolation object is not a SequenceInterpolator"

        delta_t = 1.0 / sampling_rate

        # timestamps multiplied by 0.8 should be floored to the same timestamp
        times = timestamps[:DEFAULT_SEQUENCE_LENGTH] + 0.8 * delta_t
        interp, valid = seq_interp.interpolate(times=times, return_valid=True)
        assert times.shape == valid.shape, "All samples should be valid"
        assert (
            interp == data[:DEFAULT_SEQUENCE_LENGTH]
        ).all(), "Nearest neighbor interpolation does not match expected data"

        # timestamps multiplied by 1.2 should be floored to the next timestamp
        times = timestamps[:DEFAULT_SEQUENCE_LENGTH] + 1.2 * delta_t
        interp, valid = seq_interp.interpolate(times=times, return_valid=True)
        assert times.shape == valid.shape, "All samples should be valid"
        assert (
            interp == data[1 : DEFAULT_SEQUENCE_LENGTH + 1]
        ).all(), "Nearest neighbor interpolation does not match expected data"


@pytest.mark.parametrize("n_signals", [0, 1, 10, 50])
@pytest.mark.parametrize("sampling_rate", [3.0, 10.0, 100.0])
@pytest.mark.parametrize("use_mem_mapped", [False, True])
def test_nearest_neighbor_interpolation_with_phase_shifts(
    n_signals, sampling_rate, use_mem_mapped
):
    with sequence_data_and_interpolator(
        data_kwargs={
            "n_signals": n_signals,
            "use_mem_mapped": use_mem_mapped,
            "t_end": 5.0,
            "sampling_rate": sampling_rate,
            "shifts_per_signal": True,
        }
    ) as (timestamps, data, shift, seq_interp):
        assert shift is not None, f"Expected shift to be not None, received {shift}"
        assert isinstance(
            seq_interp, PhaseShiftedSequenceInterpolator
        ), "Interpolation object is not a PhaseShiftedSequenceInterpolator"

        delta_t = 1.0 / sampling_rate
        times = (
            timestamps[1 : DEFAULT_SEQUENCE_LENGTH + 1] + 1e-9
        )  # Add a small epsilon to avoid floating point errors
        interp, valid = seq_interp.interpolate(times=times, return_valid=True)
        assert times.shape == valid.shape, "All samples should be valid"
        assert (
            interp == data[0:DEFAULT_SEQUENCE_LENGTH]
        ).all(), "Nearest neighbor interpolation does not match expected data"
        assert valid.shape == (
            DEFAULT_SEQUENCE_LENGTH,
        ), f"Expected valid.shape == ({DEFAULT_SEQUENCE_LENGTH},), got {valid.shape}"
        assert interp.shape == (
            DEFAULT_SEQUENCE_LENGTH,
            n_signals,
        ), f"Expected interp.shape == ({DEFAULT_SEQUENCE_LENGTH}, {n_signals}), got {interp.shape}"

        # Test phase shifts
        for i in range(data.shape[1]):
            for dt in np.linspace(0, 0.99) * delta_t:
                shifted_times = times + shift[i] + dt

                interp, valid = seq_interp.interpolate(
                    times=shifted_times, return_valid=True
                )
                assert (
                    interp[:, i] == data[1 : DEFAULT_SEQUENCE_LENGTH + 1, i]
                ).all(), f"Data at {dt} does not match original data (use_mem_mapped={use_mem_mapped}, sampling_rate={sampling_rate}, shifts_per_signal={True})"

            for dt in np.linspace(1.0, 1.99) * delta_t:
                shifted_times = times + shift[i] + dt

                interp, valid = seq_interp.interpolate(
                    times=shifted_times, return_valid=True
                )
                assert (
                    interp[:, i] == data[2 : DEFAULT_SEQUENCE_LENGTH + 2, i]
                ).all(), f"Data at {dt} does not match original data (use_mem_mapped={use_mem_mapped}, sampling_rate={sampling_rate}, shifts_per_signal={True})"


@pytest.mark.parametrize("n_signals", [0, 1, 10, 50])
@pytest.mark.parametrize("keep_nans", [False, True])
def test_nearest_neighbor_interpolation_with_phase_shifts_handles_nans(
    n_signals, keep_nans
):
    with sequence_data_and_interpolator(
        data_kwargs={
            "n_signals": n_signals,
            "use_mem_mapped": True,
            "t_end": 5.0,
            "sampling_rate": 10.0,
            "shifts_per_signal": True,
            "contain_nans": True,
        },
        interp_kwargs={"keep_nans": keep_nans},
    ) as (timestamps, data, _, seq_interp):
        assert isinstance(
            seq_interp, PhaseShiftedSequenceInterpolator
        ), "Interpolation object is not a PhaseShiftedSequenceInterpolator"

        times = (
            timestamps[1 : DEFAULT_SEQUENCE_LENGTH + 1] + 1e-9
        )  # Add a small epsilon to avoid floating point errors
        interp, valid = seq_interp.interpolate(times=times, return_valid=True)
        assert times.shape == valid.shape, "All samples should be valid"
        assert np.array_equal(
            interp, data[0:DEFAULT_SEQUENCE_LENGTH], equal_nan=True
        ), "Nearest neighbor interpolation does not match expected data"
        assert valid.shape == (
            DEFAULT_SEQUENCE_LENGTH,
        ), f"Expected valid.shape == ({DEFAULT_SEQUENCE_LENGTH},), got {valid.shape}"
        assert interp.shape == (
            DEFAULT_SEQUENCE_LENGTH,
            n_signals,
        ), f"Expected interp.shape == ({DEFAULT_SEQUENCE_LENGTH}, {n_signals}), got {interp.shape}"


@pytest.mark.parametrize("n_signals", [0, 1, 10, 50])
@pytest.mark.parametrize("sampling_rate", [3.0, 10.0, 100.0])
@pytest.mark.parametrize("use_mem_mapped", [False, True])
@pytest.mark.parametrize("contain_nans", [False, True])
@pytest.mark.parametrize("keep_nans", [False, True])
def test_linear_interpolation(
    n_signals, sampling_rate, use_mem_mapped, contain_nans, keep_nans
):
    with sequence_data_and_interpolator(
        data_kwargs={
            "n_signals": n_signals,
            "use_mem_mapped": use_mem_mapped,
            "t_end": 5.0,
            "sampling_rate": sampling_rate,
            "contain_nans": contain_nans,
        },
        interp_kwargs={"keep_nans": keep_nans},
    ) as (timestamps, data, _, seq_interp):
        assert isinstance(
            seq_interp, SequenceInterpolator
        ), "Not a SequenceInterpolator"
        seq_interp.interpolation_mode = "linear"

        delta_t = 1.0 / sampling_rate
        idx = list(range(1, DEFAULT_SEQUENCE_LENGTH + 1))
        times = timestamps[idx] + 0.5 * delta_t

        t1, t2 = (
            timestamps[idx][:, np.newaxis],
            timestamps[[id + 1 for id in idx]][:, np.newaxis],
        )
        y1, y2 = data[idx], data[[id + 1 for id in idx]]
        expected = y1 + ((times[:, np.newaxis] - t1) / (t2 - t1)) * (y2 - y1)
        if not keep_nans:
            np.copyto(expected, np.nanmean(expected, axis=0), where=np.isnan(expected))
        interp, valid = seq_interp.interpolate(times=times, return_valid=True)

        assert times.shape == valid.shape, "All samples should be valid"
        assert np.allclose(
            interp, expected, atol=1e-6, equal_nan=True
        ), "Linear interpolation does not match expected data"
        assert valid.shape == (
            DEFAULT_SEQUENCE_LENGTH,
        ), f"Expected valid.shape == ({DEFAULT_SEQUENCE_LENGTH},), got {valid.shape}"
        assert interp.shape == (
            DEFAULT_SEQUENCE_LENGTH,
            n_signals,
        ), f"Expected interp.shape == ({DEFAULT_SEQUENCE_LENGTH}, {n_signals}), got {interp.shape}"
        if not keep_nans:
            assert (
                np.isnan(interp).sum() == 0
            ), "Interpolated data should not contain NaNs"


@pytest.mark.parametrize("n_signals", [0, 1, 10, 50])
@pytest.mark.parametrize("sampling_rate", [3.0, 10.0, 100.0])
@pytest.mark.parametrize("use_mem_mapped", [False, True])
@pytest.mark.parametrize("keep_nans", [False, True])
def test_linear_interpolation_with_phase_shifts(
    n_signals, sampling_rate, use_mem_mapped, keep_nans
):
    with sequence_data_and_interpolator(
        data_kwargs={
            "n_signals": n_signals,
            "use_mem_mapped": use_mem_mapped,
            "t_end": 5.0,
            "sampling_rate": sampling_rate,
            "shifts_per_signal": True,
        },
        interp_kwargs={"keep_nans": keep_nans},
    ) as (timestamps, data, shift, seq_interp):
        assert shift is not None, f"Expected shift to be not None, received {shift}"
        assert isinstance(
            seq_interp, PhaseShiftedSequenceInterpolator
        ), "Not a PhaseShiftedSequenceInterpolator"
        seq_interp.interpolation_mode = "linear"

        delta_t = 1.0 / sampling_rate
        idx = slice(1, DEFAULT_SEQUENCE_LENGTH + 1)
        times = timestamps[idx] + 0.5 * delta_t

        for sig_idx in range(data.shape[1]):
            shift_offset = shift[sig_idx]
            shifted_times = times + shift_offset

            expected = np.zeros(len(shifted_times))
            for i in range(len(shifted_times)):
                t = shifted_times[i]

                shifted_timestamps = timestamps + shift_offset
                left_idx = np.searchsorted(shifted_timestamps, t, side="right") - 1
                right_idx = left_idx + 1

                if left_idx < 0 or right_idx >= len(timestamps):
                    continue

                t1, t2 = shifted_timestamps[left_idx], shifted_timestamps[right_idx]
                y1, y2 = data[left_idx, sig_idx], data[right_idx, sig_idx]

                expected[i] = y1 + ((t - t1) / (t2 - t1)) * (y2 - y1)
            if not keep_nans:
                np.copyto(
                    expected, np.nanmean(expected, axis=0), where=np.isnan(expected)
                )

            interp, valid = seq_interp.interpolate(
                times=shifted_times, return_valid=True
            )

            valid_indices = np.where(valid)[0]
            if len(valid_indices) > 0:
                assert np.allclose(
                    interp[valid_indices, sig_idx], expected[valid_indices], atol=1e-6
                ), f"Linear interpolation mismatch for signal {sig_idx}"
            if not keep_nans:
                assert (
                    np.isnan(interp).sum() == 0
                ), "Interpolated data should not contain NaNs"


@pytest.mark.filterwarnings(
    "ignore:Interpolation returns empty array, no valid times queried.:UserWarning"
)
@pytest.mark.parametrize("interpolation_mode", ["nearest_neighbor", "linear"])
@pytest.mark.parametrize("end_time", [0.05, 1.0, 5.0, 12.0])
@pytest.mark.parametrize("keep_nans", [False, True])
def test_interpolation_for_invalid_times(interpolation_mode, end_time, keep_nans):
    n_signals = 10
    with sequence_data_and_interpolator(
        data_kwargs={
            "n_signals": n_signals,
            "use_mem_mapped": True,
            "t_end": end_time,
            "sampling_rate": 10.0,
        },
        interp_kwargs={"keep_nans": keep_nans},
    ) as (_, _, _, seq_interp):
        assert isinstance(
            seq_interp, SequenceInterpolator
        ), "Interpolation object is not a SequenceInterpolator"
        seq_interp.interpolation_mode = interpolation_mode

        times = np.array([-5.0, -0.1, 0.1, 4.9, 5.0, 5.1, 10.0])
        interp, valid = seq_interp.interpolate(times=times, return_valid=True)
        expected_valid = (
            np.where((times >= 0.0) & (times <= end_time))[0]
            if interpolation_mode == "nearest_neighbor"
            else np.where((times >= 0.0) & (times < end_time))[0]
        )
        assert (
            expected_valid == valid
        ).all(), "Valid times does not match expected values"
        expected_nr_valid = len(valid)
        assert interp.shape == (
            expected_nr_valid,
            n_signals,
        ), f"Expected interp.shape == ({expected_nr_valid}, {n_signals}), got {interp.shape}"


@pytest.mark.filterwarnings(
    "ignore:Interpolation returns empty array, no valid times queried.:UserWarning"
)
@pytest.mark.parametrize("interpolation_mode", ["nearest_neighbor", "linear"])
@pytest.mark.parametrize("end_time", [0.05, 1.0, 5.0, 12.0])
@pytest.mark.parametrize("keep_nans", [False, True])
def test_interpolation_with_phase_shifts_for_invalid_times(
    interpolation_mode, end_time, keep_nans
):
    n_signals = 10
    with sequence_data_and_interpolator(
        data_kwargs={
            "n_signals": n_signals,
            "use_mem_mapped": True,
            "t_end": end_time,
            "sampling_rate": 10.0,
            "shifts_per_signal": True,
        },
        interp_kwargs={"keep_nans": keep_nans},
    ) as (_, _, phase_shifts, seq_interp):
        assert phase_shifts is not None, f"Expected phase_shifts to be not None, received {phase_shifts}"
        assert isinstance(
            seq_interp, PhaseShiftedSequenceInterpolator
        ), "Interpolation object is not a PhaseShiftedSequenceInterpolator"
        seq_interp.interpolation_mode = interpolation_mode

        times = np.array([-5.0, -0.1, 0.1, 4.9, 4.9999999, 5.0, 5.0000001, 5.1, 10.0])
        interp, valid = seq_interp.interpolate(times=times, return_valid=True)
        assert (
            np.where(
                (times >= np.min(phase_shifts))
                & (times <= end_time + np.max(phase_shifts))
            )[0]
            == valid
        ).all(), "Valid times does not match expected values"
        expected_nr_valid = len(valid)
        assert interp.shape == (
            expected_nr_valid,
            n_signals,
        ), f"Expected interp.shape == ({expected_nr_valid}, {n_signals}), got {interp.shape}"


@pytest.mark.parametrize("interpolation_mode", ["nearest_neighbor", "linear"])
@pytest.mark.parametrize("phase_shifts", [True, False])
def test_interpolation_for_empty_times(interpolation_mode, phase_shifts):
    with sequence_data_and_interpolator(
        data_kwargs={
            "n_signals": 10,
            "use_mem_mapped": True,
            "t_end": 5.0,
            "sampling_rate": 10.0,
            "shifts_per_signal": phase_shifts,
        }
    ) as (_, _, _, seq_interp):
        assert isinstance(
            seq_interp, SequenceInterpolator
        ), "Interpolation object is not a SequenceInterpolator"
        seq_interp.interpolation_mode = interpolation_mode

        with pytest.warns(
            UserWarning,
            match="Interpolation returns empty array, no valid times queried.",
        ):
            interp, valid = seq_interp.interpolate(
                times=np.array([]), return_valid=True
            )
        assert interp.shape[0] == 0, "No data expected"
        assert valid.shape[0] == 0, "No data expected"


def test_nearest_neighbor_interpolation_return_valid_false():
    with sequence_data_and_interpolator(
        data_kwargs={
            "n_signals": 10,
            "use_mem_mapped": False,
            "t_end": 5.0,
            "sampling_rate": 10.0,
        }
    ) as (timestamps, data, _, seq_interp):
        times = timestamps[:DEFAULT_SEQUENCE_LENGTH] + 1e-9

        result = seq_interp.interpolate(times=times, return_valid=False)
        assert isinstance(result, np.ndarray), "Expected np.ndarray, not a tuple"
        assert result.shape == (
            DEFAULT_SEQUENCE_LENGTH,
            10,
        ), f"Expected shape ({DEFAULT_SEQUENCE_LENGTH}, 10), got {result.shape}"

        interp, _ = seq_interp.interpolate(times=times, return_valid=True)
        assert np.array_equal(
            result, interp
        ), "Data from return_valid=False should match data from return_valid=True"


def test_nearest_neighbor_interpolation_default_return_valid():
    with sequence_data_and_interpolator(
        data_kwargs={
            "n_signals": 10,
            "use_mem_mapped": False,
            "t_end": 5.0,
            "sampling_rate": 10.0,
        }
    ) as (timestamps, data, _, seq_interp):
        times = timestamps[:DEFAULT_SEQUENCE_LENGTH] + 1e-9

        result = seq_interp.interpolate(times=times)
        assert isinstance(result, np.ndarray), "Expected np.ndarray, not a tuple"
        assert result.shape == (
            DEFAULT_SEQUENCE_LENGTH,
            10,
        ), f"Expected shape ({DEFAULT_SEQUENCE_LENGTH}, 10), got {result.shape}"

        interp, _ = seq_interp.interpolate(times=times, return_valid=True)
        assert np.array_equal(
            result, interp
        ), "Data from default (no return_valid) should match data from return_valid=True"


def test_linear_interpolation_return_valid_false():
    with sequence_data_and_interpolator(
        data_kwargs={
            "n_signals": 10,
            "use_mem_mapped": False,
            "t_end": 5.0,
            "sampling_rate": 10.0,
        }
    ) as (timestamps, data, _, seq_interp):
        seq_interp.interpolation_mode = "linear"

        delta_t = 1.0 / 10.0
        times = timestamps[1 : DEFAULT_SEQUENCE_LENGTH + 1] + 0.5 * delta_t

        result = seq_interp.interpolate(times=times, return_valid=False)
        assert isinstance(result, np.ndarray), "Expected np.ndarray, not a tuple"
        assert result.shape == (
            DEFAULT_SEQUENCE_LENGTH,
            10,
        ), f"Expected shape ({DEFAULT_SEQUENCE_LENGTH}, 10), got {result.shape}"

        interp, _ = seq_interp.interpolate(times=times, return_valid=True)
        assert np.allclose(
            result, interp, equal_nan=True
        ), "Data from return_valid=False should match data from return_valid=True"


def test_linear_interpolation_default_return_valid():
    with sequence_data_and_interpolator(
        data_kwargs={
            "n_signals": 10,
            "use_mem_mapped": False,
            "t_end": 5.0,
            "sampling_rate": 10.0,
        }
    ) as (timestamps, data, _, seq_interp):
        seq_interp.interpolation_mode = "linear"

        delta_t = 1.0 / 10.0
        times = timestamps[1 : DEFAULT_SEQUENCE_LENGTH + 1] + 0.5 * delta_t

        result = seq_interp.interpolate(times=times)
        assert isinstance(result, np.ndarray), "Expected np.ndarray, not a tuple"
        assert result.shape == (
            DEFAULT_SEQUENCE_LENGTH,
            10,
        ), f"Expected shape ({DEFAULT_SEQUENCE_LENGTH}, 10), got {result.shape}"

        interp, _ = seq_interp.interpolate(times=times, return_valid=True)
        assert np.allclose(
            result, interp, equal_nan=True
        ), "Data from default (no return_valid) should match data from return_valid=True"


def test_interpolation_mode_not_implemented():
    with sequence_data_and_interpolator() as (_, _, _, seq_interp):
        seq_interp.interpolation_mode = "unsupported_mode"
        with pytest.raises(NotImplementedError):
            seq_interp.interpolate(np.array([0.0, 1.0, 2.0]), return_valid=True)


def test_sequence_interpolator_indexes_selection():
    with sequence_data_and_interpolator(
        data_kwargs={"n_signals": 10, "use_mem_mapped": False}
    ) as (_, data, _, seq_interp):

        seq_interp = SequenceInterpolator(
            seq_interp.root_folder, neuron_indices=[1, 3, 5]
        )

        assert seq_interp.n_signals == 3
        assert seq_interp._data.shape[1] == 10


def test_sequence_interpolator_neuron_ids_selection(tmp_path):
    with sequence_data_and_interpolator(data_kwargs={"n_signals": 4}) as (
        _,
        _,
        _,
        seq_interp,
    ):

        meta_folder = seq_interp.root_folder / "meta"
        meta_folder.mkdir(exist_ok=True)

        unit_ids = np.array([10, 20, 30, 40])
        np.save(meta_folder / "unit_ids.npy", unit_ids)

        interp = SequenceInterpolator(seq_interp.root_folder, neuron_ids=[20, 40])

        assert interp.n_signals == 2


def test_sequence_interpolator_neuron_ids_indexes_mismatch():
    with sequence_data_and_interpolator(data_kwargs={"n_signals": 5}) as (
        _,
        _,
        _,
        seq_interp,
    ):
        meta_folder = seq_interp.root_folder / "meta"
        meta_folder.mkdir(exist_ok=True)
        np.save(meta_folder / "unit_ids.npy", np.arange(5))

        with pytest.raises(ValueError):
            SequenceInterpolator(
                seq_interp.root_folder, neuron_ids=[1], neuron_indices=[2]
            )


def test_phase_shift_interpolator_indexes_filtering():
    with sequence_data_and_interpolator(
        data_kwargs={"n_signals": 6, "shifts_per_signal": True}
    ) as (_, _, phase_shifts, seq_interp):

        interp = PhaseShiftedSequenceInterpolator(
            seq_interp.root_folder, neuron_indices=[0, 2, 4]
        )

        assert len(interp._phase_shifts) == 3


if __name__ == "__main__":
    pytest.main([__file__])
