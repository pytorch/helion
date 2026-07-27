"""Deprecated alias for :mod:`helion.autotuner.aot_compile`.

Moved out of ``helion.experimental`` into ``helion.autotuner``. Importing from
here still works but emits a :class:`DeprecationWarning`.
"""

from __future__ import annotations

from ..autotuner.aot_compile import *  # noqa: F403
from ..autotuner.aot_compile import generate_standalone_file as generate_standalone_file
from . import _warn_aot_moved

_warn_aot_moved()
