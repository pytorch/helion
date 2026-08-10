"""Deprecated alias for :mod:`helion.autotuner.aot_kernel`.

``aot_kernel`` moved out of ``helion.experimental`` into ``helion.autotuner``
(and is exposed as :func:`helion.aot_kernel`). Importing from here still works
but emits a :class:`DeprecationWarning`.
"""

from __future__ import annotations

from ..autotuner.aot_kernel import *  # noqa: F403  (re-export public API)

# Re-export names that ``import *`` skips (leading underscore / not in __all__)
# but that existing callers and tests import explicitly.
from ..autotuner.aot_kernel import BatchedSpec as BatchedSpec
from ..autotuner.aot_kernel import HeuristicKeyFunction as HeuristicKeyFunction
from ..autotuner.aot_kernel import KeyFunction as KeyFunction
from ..autotuner.aot_kernel import _flatten_key_value as _flatten_key_value
from ..autotuner.aot_kernel import aot_kernel as aot_kernel
from ..autotuner.aot_kernel import aot_key as aot_key
from ..autotuner.aot_kernel import extract_key_features as extract_key_features
from ..autotuner.aot_kernel import extract_shape_features as extract_shape_features
from ..autotuner.aot_kernel import make_aot_key as make_aot_key
from . import _warn_aot_moved

_warn_aot_moved()

# Public surface for `from helion.experimental.aot_kernel import *` consumers.
__all__ = [
    "HeuristicKeyFunction",
    "aot_kernel",
    "aot_key",
    "extract_key_features",
    "extract_shape_features",
    "make_aot_key",
]
