INTERPOLATOR_SELECTORS = []

def register_interpolator(selector_fn, *, priority=10):
    """Register a new interpolator class with an associated selector and priority."""
    def decorator(cls):
        INTERPOLATOR_SELECTORS.append((priority, selector_fn, cls))
        return cls
    return decorator

def ensure_default_interpolators_registered():
    from . import sequence_interpolator
    from . import screen_interpolator