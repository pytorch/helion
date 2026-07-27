"""Deprecated alias for :mod:`helion.autotuner.aot_runner`.

Moved out of ``helion.experimental`` into ``helion.autotuner``. Both importing
from here and running ``python -m helion.experimental.aot_runner`` still work
but emit a :class:`DeprecationWarning`; use ``helion.autotuner.aot_runner``.
"""

from __future__ import annotations

from ..autotuner.aot_runner import *  # noqa: F403
from ..autotuner.aot_runner import main as main
from . import _warn_aot_moved

_warn_aot_moved()


if __name__ == "__main__":
    main()
