"""Deprecated alias for :mod:`helion.autotuner.aot_runner`.

Moved out of ``helion.experimental`` into ``helion.autotuner``. Both importing
from here and running ``python -m helion.experimental.aot_runner`` still work
but emit a :class:`DeprecationWarning`; use ``helion.autotuner.aot_runner``.
"""

from __future__ import annotations

import warnings

from ..autotuner.aot_runner import *  # noqa: F403
from ..autotuner.aot_runner import main as main

warnings.warn(
    "helion.experimental.aot_runner has moved to helion.autotuner.aot_runner. "
    "Use `python -m helion.autotuner.aot_runner`; this alias will be removed in "
    "a future release.",
    DeprecationWarning,
    stacklevel=2,
)


if __name__ == "__main__":
    main()
