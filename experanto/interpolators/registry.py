"""
Global registry of all available interpolators.
Each entry is a tuple of the form (priority, selector_fn, interpolator_class).
Interpolators must be registered using the `@register_interpolator` decorator.
"""

INTERPOLATOR_SELECTORS = []


def register_interpolator(selector_fn, *, priority=10):
    """
    Decorator to register a new interpolator class with the global registry.

    Each interpolator is associated with a selector function that inspects
    metadata and returns True if the interpolator should handle that case.

    Interpolators are sorted by descending priority; higher priority wins.

    Args:
        selector_fn (Callable[[dict], bool]): A function that takes metadata and
            returns True if this interpolator should be selected.
        priority (int, optional): Priority of the interpolator. Higher values take
            precedence over lower ones. Defaults to 10.

    Returns:
        Callable: The class decorator that registers the interpolator.
    """

    def decorator(cls):
        INTERPOLATOR_SELECTORS.append((priority, selector_fn, cls))
        return cls

    return decorator
