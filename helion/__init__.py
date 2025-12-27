from __future__ import annotations

from triton import cdiv
from triton import next_power_of_2

from . import _compat as _compat_module  # noqa: F401  # side-effect import
from . import _logging
from . import exc
from . import language
from . import runtime
from .runtime import Config
from .runtime import Kernel
from .runtime import kernel
from .runtime import kernel as jit  # alias
from .runtime.settings import RefMode
from .runtime.settings import Settings

# Import _dynamo AFTER Kernel to avoid circular import during registration
from . import _dynamo as _dynamo_module  # noqa: F401,E402  # side-effect import

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
]

_logging.init_logs()
