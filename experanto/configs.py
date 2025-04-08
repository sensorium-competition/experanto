from pathlib import Path
from omegaconf import OmegaConf, open_dict
from hydra import compose, initialize, initialize_config_dir


with initialize(version_base=None, config_path="../configs/", ):
    cfg = compose(config_name="default", )

DEFAULT_CONFIG = cfg
DEFAULT_DATASET_CONFIG = cfg.dataset
DEFAULT_MODALITY_CONFIG = cfg.dataset.modality_config
DEFAULT_DATALOADER_CONFIG = cfg.dataloader
