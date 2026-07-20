from __future__ import annotations

from .autodiff import backward as backward

# ``aot_kernel`` and friends moved out of ``helion.experimental`` into
# ``helion.autotuner`` (the decorator is also exposed as ``helion.aot_kernel``).
# The old ``helion.experimental.aot_*`` paths still work as deprecated aliases.
_DEPRECATED_AOT = {"aot_kernel", "aot_key", "extract_shape_features", "make_aot_key"}

# Emit the move notice at most once per process, no matter how many of the
# deprecated aot_* modules/names are touched.
_aot_deprecation_warned = False


def _warn_aot_moved(stacklevel: int = 2) -> None:
    global _aot_deprecation_warned
    if _aot_deprecation_warned:
        return
    _aot_deprecation_warned = True
    import warnings

    warnings.warn(
        "helion.experimental.aot_* has moved to helion.autotuner "
        "(the decorator is also available as helion.aot_kernel). Update your "
        "imports; these aliases will be removed in a future release.",
        DeprecationWarning,
        stacklevel=stacklevel + 1,
    )


def __getattr__(name: str) -> object:
    if name in _DEPRECATED_AOT:
        _warn_aot_moved()
        import importlib

        _aot_mod = importlib.import_module("helion.autotuner.aot_kernel")
        return getattr(_aot_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
