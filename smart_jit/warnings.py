from numba.core.errors import NumbaWarning


class NumbaInterpreterModeWarning(NumbaWarning):
    """
    Emit a warning in case jit falls back into interpreter mode
    """
