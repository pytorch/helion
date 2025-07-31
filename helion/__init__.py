from __future__ import annotations

from triton import cdiv

from . import _logging
from . import exc
from . import language
from . import runtime
from .language.tunable_ops import next_power_of_2
from .runtime import Config
from .runtime import Kernel
from .runtime import kernel
from .runtime import kernel as jit  # alias
from .runtime.settings import RefMode
from .runtime.settings import Settings
from .runtime.settings import set_default_settings

__all__ = [
    "Config",
    "Kernel",
    "RefMode",
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
