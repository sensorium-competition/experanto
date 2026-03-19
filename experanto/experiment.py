from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from .configs import DEFAULT_MODALITY_CONFIG
from .interpolators import Interpolator

logger = logging.getLogger(__name__)


class Experiment:
    """High-level interface for loading and querying neuroscience experiments.

    An Experiment represents a single recording session containing multiple
    modalities (e.g., visual stimuli, neural responses, behavioral data).
    Each modality is loaded as an Interpolator, allowing data to be queried
    at arbitrary time points.

    Parameters
    ----------
    root_folder : str
        Path to the experiment directory. Should contain subdirectories
        for each modality (e.g., ``screen/``, ``responses/``, ``eye_tracker/``).
    modality_config : dict, optional
        Configuration dictionary specifying interpolation and processing
        settings for each modality. See :mod:`experanto.configs` for the
        default configuration structure.
    cache_data : bool, default=False
        If True, loads all trial data into memory for faster access.
        Useful for smaller datasets or when memory is not a constraint.

    Attributes
    ----------
    devices : dict
        Dictionary mapping device names to their :class:`Interpolator` instances.
    start_time : float
        Earliest valid timestamp across all devices.
    end_time : float
        Latest valid timestamp across all devices.

    See Also
    --------
    ChunkDataset : Higher-level interface for ML training.
    Interpolator : Base class for modality-specific interpolators.

    Examples
    --------
    >>> from experanto.experiment import Experiment
    >>> exp = Experiment('/path/to/experiment')
    >>> exp.device_names
    ('screen', 'responses', 'eye_tracker')
    >>> times = np.linspace(0, 10, 100)
    >>> data = exp.interpolate(times, device='responses')
    """

    def __init__(
        self,
        root_folder: str,
        modality_config: dict = DEFAULT_MODALITY_CONFIG,
        cache_data: bool = False,
    ) -> None:
        self.root_folder = Path(root_folder)
        self.devices = {}
        self.start_time = np.inf
        self.end_time = -np.inf
        self.modality_config = modality_config
        self.cache_data = cache_data
        self._load_devices()

    def _load_devices(self) -> None:
        # Populate devices by going through subfolders
        # Assumption: blocks are sorted by start time
        device_folders = [d for d in self.root_folder.iterdir() if (d.is_dir())]

        for d in device_folders:
            if d.name not in self.modality_config:
                logger.info("Skipping %s data", d.name)
                continue
            logger.info("Parsing %s data", d.name)

            # Get interpolation config for this device
            interp_conf = self.modality_config[d.name]["interpolation"]

            if (
                isinstance(interp_conf, (dict, DictConfig))
                and "_target_" in interp_conf
            ):
                # Custom interpolator (Hydra instantiates it)
                dev = instantiate(
                    interp_conf, root_folder=d, cache_data=self.cache_data
                )
                # Check if instantiated object is proper Interpolator
                if not isinstance(dev, Interpolator):
                    raise ValueError(
                        "Please provide an Interpolator which inherits from experantos Interpolator class."
                    )

            elif isinstance(interp_conf, Interpolator):
                # Already instantiated Interpolator
                dev = interp_conf

            else:
                # Default back to original logic
                warnings.warn(
                    "Falling back to original Interpolator creation logic.",
                    UserWarning,
                    stacklevel=2,
                )
                dev = Interpolator.create(
                    d,
                    cache_data=self.cache_data,
                    **interp_conf,  # type: ignore[arg-type]
                )

            if (
                dev.start_time is None
                or dev.end_time is None
                or not np.isfinite(dev.start_time)
                or not np.isfinite(dev.end_time)
            ):
                logger.warning(
                    "Device %s has undefined start_time or end_time and will be "
                    "excluded from the experiment-wide time range.",
                    d.name,
                )
            else:
                self.start_time = min(self.start_time, dev.start_time)
                self.end_time = max(self.end_time, dev.end_time)
                self.devices[d.name] = dev
            logger.info("Parsing finished")

        if not self.devices:
            raise ValueError(
                "Experiment time range could not be determined: no devices with valid start_time and end_time were found."
            )
        elif self.start_time > self.end_time:
            raise ValueError(
                "Experiment time range could not be determined: at least one device "
                "must define finite start_time and end_time."
            )

    @property
    def device_names(self):
        return tuple(self.devices.keys())

    def interpolate(
        self,
        times: np.ndarray,
        device: str | Interpolator | None = None,
        return_valid: bool = False,
    ) -> tuple[dict, dict] | dict | tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Interpolate data from one or all devices at specified time points.

        Parameters
        ----------
        times : array_like
            1D array of time points (in seconds) at which to interpolate.
        device : str, optional
            Name of a specific device to interpolate. If None, interpolates
            all devices and returns dictionaries.

        Returns
        -------
        values : numpy.ndarray or dict
            If ``device`` is specified, returns the interpolated data array
            for the valid time points only (shape is modality-dependent, see
            below). Otherwise, returns a dict mapping device names to their
            data arrays.
        valid : numpy.ndarray or dict, optional
            Only present when ``return_valid=True``. Integer index array(s)
            into ``times`` indicating which entries were used to produce
            ``values``. ``values[i]`` corresponds to ``times[valid[i]]``, and
            ``len(valid) == values.shape[0]``. When a dict is returned,
            ``valid`` is a dict with the same keys and ``len(valid[d])`` may
            differ across devices because each modality has its own valid
            range.

        Notes
        -----
        Output shapes per modality:

        * Sequence modalities (``responses``, ``eye_tracker``, ``treadmill``):
          ``(n_valid, n_signals)``
        * Screen modality: ``(n_valid, H, W)`` for grayscale,
          ``(n_valid, H, W, C)`` for colour.

        Examples
        --------
        Interpolate a single device:

        >>> data, valid = exp.interpolate(times, device='responses', return_valid=True)
        >>> data.shape
        (n_valid, 500)  # n_valid <= len(times), 500 neurons
        >>> times[valid].shape == (data.shape[0],)
        True

        Interpolate all devices:

        >>> data = exp.interpolate(times)
        >>> data.keys()
        dict_keys(['screen', 'responses', 'eye_tracker'])
        """
        if device is None:
            values = {}
            valid = {}
            for d, interp in self.devices.items():
                res = interp.interpolate(times, return_valid=return_valid)
                if return_valid:
                    vals, vlds = res
                    values[d] = vals
                    valid[d] = vlds
                else:
                    values[d] = res
            if return_valid:
                return values, valid
            else:
                return values
        elif isinstance(device, str):
            assert device in self.devices, f"Unknown device '{device}'"
            res = self.devices[device].interpolate(times, return_valid=return_valid)
            return res
        else:
            raise ValueError(f"Unsupported device type: {type(device)}")

    def get_valid_range(self, device_name: str) -> tuple[float, float]:
        """Get the valid time range for a specific device.

        Parameters
        ----------
        device_name : str
            Name of the device (e.g., 'screen', 'responses').

        Returns
        -------
        tuple
            A tuple ``(start_time, end_time)`` representing the valid
            time interval in seconds.
        """
        return tuple(self.devices[device_name].valid_interval)
