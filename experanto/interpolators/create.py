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
    """
    Factory method to instantiate the appropriate interpolator based on metadata.

    This function reads `meta.yml` in the given root folder, then selects and
    instantiates the highest-priority interpolator whose selector function matches
    the metadata.

    Interpolators are registered via the `@register_interpolator` decorator.

    Args:
        root_folder (str): Path to the folder containing the `meta.yml` file.
        cache_data (bool): Whether the interpolator should cache precomputed data.
        **kwargs: Additional arguments passed to the interpolator constructor.

    Returns:
        Interpolator: An instance of the matching interpolator subclass.

    Raises:
        ValueError: If no matching interpolator is found for the given metadata.
    """
    with open(Path(root_folder) / "meta.yml", "r") as file:
        meta_data = yaml.load(file, Loader=yaml.SafeLoader)

    sorted_selectors = sorted(
        INTERPOLATOR_SELECTORS, key=lambda x: -x[0]
    )  # highest priority first

    for priority, selector_fn, cls in sorted_selectors:
        if selector_fn(meta_data):
            return cls(root_folder, cache_data, **kwargs)

    raise ValueError(f"No interpolator found for metadata={meta_data}.")
