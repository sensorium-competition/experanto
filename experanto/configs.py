from hydra import compose, initialize, initialize_config_dir


with initialize(version_base=None, config_path="../configs/", ):
    cfg = compose(config_name="default", )


with initialize(version_base=None, config_path="../configs/", ):
    cfg1 = compose(config_name="monkeyFV", )

with initialize(version_base=None, config_path="../configs/", ):
    cfg2 = compose(config_name="monkeyFixation", )

with initialize(version_base=None, config_path="../configs/", ):
    cfg3 = compose(config_name="monkeyFixationOldFormat", )


DEFAULT_CONFIG = cfg
DEFAULT_DATASET_CONFIG = cfg.dataset
DEFAULT_MODALITY_CONFIG = cfg.dataset.modality_config
DEFAULT_DATALOADER_CONFIG = cfg.dataloader

MONKEY_FV_CONFIG = cfg1
MONKEY_FIX = cfg2
MONKEY_FIX_OLD = cfg3

