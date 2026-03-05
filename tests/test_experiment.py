import numpy as np
import pytest

from experanto.experiment import Experiment
from experanto.intervals import TimeInterval

from .create_experiment import create_experiment, get_default_config


@pytest.mark.parametrize("device_name", ["device_0", "device_1"])
def test_get_valid_range_all_devices(device_name):
    with create_experiment(
        devices_kwargs=[{"t_end": 10.0}, {"t_end": 20.0}],
    ) as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path,
            modality_config=get_default_config(),
        )

        valid_range = experiment.get_valid_range(device_name)

        assert isinstance(valid_range, tuple), "Result should be a tuple"
        assert len(valid_range) == 2, "Valid interval should have a start and end"

        start, end = valid_range
        assert start < end, "Start of valid interval should be less than end"

        assert valid_range == (0.0, 10.0 if device_name == "device_0" else 20.0)


def test_get_valid_range_raises_for_invalid_device():
    with create_experiment() as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path,
            modality_config=get_default_config(),
        )
        with pytest.raises(KeyError):
            experiment.get_valid_range("device_does_not_exist")


def test_get_data_for_interval_basic():
    with create_experiment(
        n_devices=2,
        devices_kwargs=[
            {"sampling_rate": 1.0, "n_signals": 10},
            {"sampling_rate": 2.0, "n_signals": 5},
        ],
    ) as experiment_path:
        modality_config = get_default_config()
        modality_config["device_1"]["sampling_rate"] = 2.0
        experiment = Experiment(
            root_folder=experiment_path, modality_config=modality_config
        )

        interval = TimeInterval(start=0.0, end=5.0)
        result = experiment.get_data_for_interval(interval)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"device_0", "device_1"}

        assert isinstance(result["device_0"], np.ndarray)
        assert result["device_0"].shape == (5, 10)

        assert isinstance(result["device_1"], np.ndarray)
        assert result["device_1"].shape == (10, 5)


def test_get_data_for_interval_single_device():
    with create_experiment(
        n_devices=2,
        devices_kwargs=[
            {"sampling_rate": 1.0, "n_signals": 10},
        ],
    ) as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path, modality_config=get_default_config()
        )
        interval = TimeInterval(0.0, 2.0)

        result = experiment.get_data_for_interval(interval, devices="device_0")

        assert list(result.keys()) == ["device_0"]
        assert isinstance(result["device_0"], np.ndarray)
        assert result["device_0"].shape == (2, 10)


def test_get_data_for_interval_custom_sampling_rate():
    with create_experiment(
        n_devices=2,
        devices_kwargs=[
            {"sampling_rate": 1.0, "n_signals": 10},
            {"sampling_rate": 2.0, "n_signals": 10},
        ],
    ) as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path, modality_config=get_default_config()
        )
        interval = TimeInterval(0.0, 1.0)

        result = experiment.get_data_for_interval(interval, target_sampling_rates=5.0)

        assert isinstance(result["device_0"], np.ndarray)
        assert result["device_0"].shape == (5, 10)

        assert isinstance(result["device_1"], np.ndarray)
        assert result["device_1"].shape == (5, 10)


def test_get_data_for_interval_per_device_sampling_rate():
    with create_experiment(
        n_devices=2,
        devices_kwargs=[
            {"sampling_rate": 1.0, "n_signals": 10},
            {"sampling_rate": 2.0, "n_signals": 10},
        ],
    ) as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path, modality_config=get_default_config()
        )
        interval = TimeInterval(0.0, 1.0)

        result = experiment.get_data_for_interval(
            interval, target_sampling_rates={"device_0": 2.0, "device_1": 10.0}
        )

        assert isinstance(result["device_0"], np.ndarray)
        assert result["device_0"].shape == (2, 10)

        assert isinstance(result["device_1"], np.ndarray)
        assert result["device_1"].shape == (10, 10)


def test_get_data_for_interval_invalid_device_raises():
    with create_experiment() as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path, modality_config=get_default_config()
        )
        interval = TimeInterval(0.0, 1.0)

        with pytest.raises(AssertionError):
            experiment.get_data_for_interval(interval, devices="unknown_device")


def test_get_data_for_interval_missing_sampling_rate_raises():
    with create_experiment() as experiment_path:
        modality_config = get_default_config()
        # remove sampling rate for one device
        modality_config["device_0"].pop("sampling_rate", None)
        experiment = Experiment(
            root_folder=experiment_path, modality_config=modality_config
        )
        interval = TimeInterval(0.0, 1.0)

        with pytest.raises(ValueError):
            experiment.get_data_for_interval(interval, devices="device_0")


def test_get_data_for_chunks_basic():
    with create_experiment(
        n_devices=2,
        devices_kwargs=[
            {"t_end": 100.0, "n_signals": 10},
            {"t_end": 100.0, "n_signals": 5},
        ],
    ) as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path, modality_config=get_default_config()
        )

        data, timestamps = experiment.get_data_for_chunks(start_time=0.0)

        assert isinstance(data, dict)
        assert isinstance(timestamps, dict)
        assert list(data.keys()) == list(timestamps.keys()) == ["device_0", "device_1"]

        assert isinstance(data["device_0"], np.ndarray)
        assert data["device_0"].shape == (40, 10)
        assert isinstance(timestamps["device_0"], np.ndarray)
        assert timestamps["device_0"].shape == (40,)
        assert np.all(np.diff(timestamps["device_0"]) > 0)

        assert isinstance(data["device_1"], np.ndarray)
        assert data["device_1"].shape == (60, 5)
        assert isinstance(timestamps["device_1"], np.ndarray)
        assert timestamps["device_1"].shape == (60,)
        assert np.all(np.diff(timestamps["device_1"]) > 0)


def test_get_data_for_chunks_single_device():
    with create_experiment(
        n_devices=2,
        devices_kwargs=[
            {"t_end": 100.0, "n_signals": 10},
            {"t_end": 100.0, "n_signals": 5},
        ],
    ) as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path, modality_config=get_default_config()
        )

        data, timestamps = experiment.get_data_for_chunks(
            start_time=0.0, devices="device_0"
        )

        assert list(data.keys()) == list(timestamps.keys()) == ["device_0"]
        assert isinstance(data["device_0"], np.ndarray)
        assert data["device_0"].shape == (40, 10)
        assert isinstance(timestamps["device_0"], np.ndarray)
        assert timestamps["device_0"].shape == (40,)
        assert np.all(np.diff(timestamps["device_0"]) > 0)


def test_get_data_for_chunks_custom_chunk_size():
    with create_experiment(
        n_devices=2,
        devices_kwargs=[
            {"t_end": 100.0, "n_signals": 10},
            {"t_end": 100.0, "n_signals": 5},
        ],
    ) as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path, modality_config=get_default_config()
        )

        chunk_size = 50
        data, timestamps = experiment.get_data_for_chunks(
            start_time=0.0, chunk_sizes=chunk_size
        )

        assert data["device_0"].shape == (50, 10)
        assert timestamps["device_0"].shape == (50,)
        assert data["device_1"].shape == (50, 5)
        assert timestamps["device_1"].shape == (50,)


def test_get_data_for_chunks_per_device_chunk_size():
    with create_experiment(
        n_devices=2,
        devices_kwargs=[
            {"t_end": 100.0, "n_signals": 10},
            {"t_end": 100.0, "n_signals": 5},
        ],
    ) as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path, modality_config=get_default_config()
        )

        chunk_sizes = {"device_0": 10, "device_1": 20}
        data, timestamps = experiment.get_data_for_chunks(
            start_time=0.0, chunk_sizes=chunk_sizes
        )

        assert data["device_0"].shape == (10, 10)
        assert timestamps["device_0"].shape == (10,)
        assert data["device_1"].shape == (20, 5)
        assert timestamps["device_1"].shape == (20,)


def test_get_data_for_chunks_custom_sampling_rate():
    with create_experiment() as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path, modality_config=get_default_config()
        )

        sr = 5.0
        data, timestamps = experiment.get_data_for_chunks(
            start_time=0.0, target_sampling_rates=sr
        )

        for ts in timestamps.values():
            deltas = np.diff(ts)
            # Should be roughly 1/sr spacing
            assert np.allclose(deltas, 1.0 / sr, rtol=0.1)


def test_get_data_for_chunks_per_device_sampling_rates():
    with create_experiment() as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path, modality_config=get_default_config()
        )

        srs = {"device_0": 2.0, "device_1": 10.0}
        data, timestamps = experiment.get_data_for_chunks(
            start_time=0.0, target_sampling_rates=srs
        )

        # ~0.5 sec steps for 2 Hz
        assert np.isclose(np.diff(timestamps["device_0"]).mean(), 0.5, rtol=0.1)
        # ~0.1 sec steps for 10 Hz
        assert np.isclose(np.diff(timestamps["device_1"]).mean(), 0.1, rtol=0.1)


def test_get_data_for_chunks_missing_sampling_rate_raises():
    with create_experiment() as experiment_path:
        modality_config = get_default_config()
        # Remove sampling rate for one device to force ValueError
        modality_config["device_0"].pop("sampling_rate", None)

        experiment = Experiment(
            root_folder=experiment_path, modality_config=modality_config
        )

        with pytest.raises(ValueError):
            experiment.get_data_for_chunks(start_time=0.0, devices="device_0")


def test_resolve_devices_none_returns_all():
    with create_experiment(
        n_devices=2,
    ) as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path, modality_config=get_default_config()
        )

        result = experiment._resolve_devices(None)

        assert result == ["device_0", "device_1"]
        assert isinstance(result, list)
        assert all(isinstance(d, str) for d in result)


def test_resolve_devices_single_string():
    with create_experiment() as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path, modality_config=get_default_config()
        )

        result = experiment._resolve_devices("device_0")
        assert result == ["device_0"]


def test_resolve_devices_valid_list():
    with create_experiment() as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path, modality_config=get_default_config()
        )

        result = experiment._resolve_devices(["device_0", "device_1"])
        assert result == ["device_0", "device_1"]


def test_resolve_devices_invalid_device_raises():
    with create_experiment() as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path, modality_config=get_default_config()
        )

        with pytest.raises(AssertionError) as excinfo:
            experiment._resolve_devices(["non_existent_device"])
        assert "Unknown device" in str(excinfo.value)


def test_resolve_sampling_rates_from_config():
    with create_experiment(
        n_devices=2,
    ) as experiment_path:
        modality_config = get_default_config()
        modality_config["device_1"]["sampling_rate"] = 2.0
        experiment = Experiment(
            root_folder=experiment_path, modality_config=modality_config
        )

        devices = ["device_0", "device_1"]
        result = experiment._resolve_sampling_rates(devices, rates=None)

        assert list(result.keys()) == ["device_0", "device_1"]
        assert isinstance(result, dict)

        for d in devices:
            assert d in result
            # Should come from modality_config (may be None if not set)
            assert result[d] == modality_config[d].get("sampling_rate")


def test_resolve_sampling_rates_with_int():
    with create_experiment() as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path, modality_config=get_default_config()
        )

        devices = ["device_0", "device_1"]
        result = experiment._resolve_sampling_rates(devices, rates=42)
        assert result == {"device_0": 42, "device_1": 42}


def test_resolve_sampling_rates_with_float():
    with create_experiment() as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path, modality_config=get_default_config()
        )

        devices = ["device_0", "device_1"]
        result = experiment._resolve_sampling_rates(devices, rates=42.0)
        assert result == {"device_0": 42.0, "device_1": 42.0}


def test_resolve_sampling_rates_with_dict():
    with create_experiment() as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path, modality_config=get_default_config()
        )

        custom = {"device_0": 10.0, "device_1": 20.0}
        result = experiment._resolve_sampling_rates(list(custom.keys()), rates=custom)
        assert result == custom


def test_resolve_chunk_sizes_from_config():
    with create_experiment(
        n_devices=2,
    ) as experiment_path:
        modality_config = get_default_config()
        experiment = Experiment(
            root_folder=experiment_path, modality_config=modality_config
        )

        devices = ["device_0", "device_1"]
        result = experiment._resolve_chunk_sizes(devices, chunk_sizes=None)

        assert list(result.keys()) == ["device_0", "device_1"]
        assert isinstance(result, dict)

        for d in devices:
            assert d in result
            # Should come from modality_config (may be None if not set)
            assert result[d] == modality_config[d].get("chunk_size")


def test_resolve_chunk_sizes_with_int():
    with create_experiment() as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path, modality_config=get_default_config()
        )

        devices = ["device_0", "device_1"]
        result = experiment._resolve_chunk_sizes(devices, chunk_sizes=42)
        assert result == {"device_0": 42, "device_1": 42}


def test_resolve_chunk_sizes_with_dict():
    with create_experiment() as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path, modality_config=get_default_config()
        )

        custom = {"device_0": 10.0, "device_1": 20.0}
        result = experiment._resolve_chunk_sizes(
            list(custom.keys()), chunk_sizes=custom
        )
        assert result == custom


def test_get_device_offset_defaults_to_zero():
    with create_experiment() as experiment_path:
        experiment = Experiment(
            root_folder=experiment_path, modality_config=get_default_config()
        )

        offset = experiment._get_device_offset("device_0")
        # Should be 0 if no offset is defined in modality_config
        assert offset == 0


def test_get_device_offset_applies_scale_precision():
    with create_experiment() as experiment_path:
        modality_config = get_default_config()
        modality_config["device_0"]["offset"] = 0.1234
        experiment = Experiment(
            root_folder=experiment_path,
            modality_config=modality_config,
            interpolate_precision=3,
        )

        offset = experiment._get_device_offset("device_0")
        assert offset == 123
