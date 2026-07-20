"""Deprecated alias for :mod:`helion.autotuner.aot_compile`.

Moved out of ``helion.experimental`` into ``helion.autotuner``. Importing from
here still works but emits a :class:`DeprecationWarning`.
"""

from __future__ import annotations

import warnings

from ..autotuner.aot_compile import *  # noqa: F403
from ..autotuner.aot_compile import generate_standalone_file as generate_standalone_file

warnings.warn(
    "helion.experimental.aot_compile has moved to helion.autotuner.aot_compile. "
    "Update your imports; this alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
