from pathlib import Path

import yaml

from . import (  # make sure default interpolators are registered
    screen_interpolator,
    sequence_interpolator,
)
from .base import Interpolator
from .registry import INTERPOLATOR_SELECTORS


def create_interpolator(
    root_folder: str, cache_data: bool = False, **kwargs
) -> Interpolator:
    with open(Path(root_folder) / "meta.yml", "r") as file:
        meta_data = yaml.load(file, Loader=yaml.SafeLoader)

    sorted_selectors = sorted(
        INTERPOLATOR_SELECTORS, key=lambda x: -x[0]
    )  # highest priority first

    for priority, selector_fn, cls in sorted_selectors:
        if selector_fn(meta_data):
            return cls(root_folder, cache_data, **kwargs)

    raise ValueError(f"No interpolator found for metadata={meta_data}.")
