from __future__ import annotations

from triton import cdiv
from triton import next_power_of_2

from . import _logging
from . import exc
from . import language
from . import runtime
from .runtime import Config
from .runtime import Kernel
from .runtime import kernel
from .runtime import kernel as jit  # alias
from helion.runtime.settings import Settings
from helion.runtime.settings import set_default_settings

__all__ = [
    "Config",
    "Kernel",
    "Settings",
    "cdiv",
    "exc",
    "jit",
    "kernel",
    "language",
    "next_power_of_2",
    "runtime",
    "set_default_settings",
]

_logging.init_logs()
