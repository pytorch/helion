from __future__ import annotations

from .autodiff import backward as backward

# ``aot_kernel`` moved to ``helion.autotuner`` (exposed as ``helion.aot_kernel``).
# Keep ``helion.experimental.aot_kernel`` working as a deprecated alias, resolved
# lazily so the DeprecationWarning fires on access rather than at
# ``import helion.experimental`` time.
_DEPRECATED_AOT = {"aot_kernel", "aot_key", "extract_shape_features", "make_aot_key"}


def __getattr__(name: str) -> object:
    if name in _DEPRECATED_AOT:
        import warnings

        warnings.warn(
            f"helion.experimental.{name} has moved to helion.autotuner.aot_kernel "
            "(the decorator is also available as helion.aot_kernel). Update your "
            "imports; this alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        import importlib

        _aot_mod = importlib.import_module("helion.autotuner.aot_kernel")
        return getattr(_aot_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
