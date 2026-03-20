"""Default configuration for Experanto datasets and dataloaders.

This module loads the default configuration from ``configs/default.yaml``
and exposes it as module-level constants. These defaults can be used
directly or customized for specific use cases.

Attributes
----------
DEFAULT_CONFIG : OmegaConf
    The complete default configuration including dataset and dataloader settings.
DEFAULT_DATASET_CONFIG : OmegaConf
    Dataset-specific configuration (sampling rates, chunk sizes, transforms).
DEFAULT_MODALITY_CONFIG : OmegaConf
    Per-modality settings (screen, responses, eye_tracker, treadmill).
DEFAULT_DATALOADER_CONFIG : OmegaConf
    DataLoader settings (batch_size, num_workers, etc.).

Examples
--------
Using default config directly:

>>> from experanto.configs import DEFAULT_CONFIG
>>> from omegaconf import OmegaConf
>>> print(OmegaConf.to_yaml(DEFAULT_CONFIG))

Customizing configuration:

>>> from experanto.configs import DEFAULT_MODALITY_CONFIG
>>> cfg = DEFAULT_MODALITY_CONFIG.copy()
>>> cfg.screen.sampling_rate = 60
>>> cfg.responses.chunk_size = 32

See Also
--------
ChunkDataset : Uses these configurations for data loading.
"""

from pathlib import Path

from omegaconf import OmegaConf

# get config relative to this file
script_dir = Path(__file__).parent
config_path = script_dir / ".." / "configs" / "default.yaml"
config_path = config_path.resolve()
cfg = OmegaConf.load(config_path)

DEFAULT_CONFIG = cfg
DEFAULT_DATASET_CONFIG = cfg.dataset
DEFAULT_MODALITY_CONFIG = cfg.dataset.modality_config
DEFAULT_DATALOADER_CONFIG = cfg.dataloader
