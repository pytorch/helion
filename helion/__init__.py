from __future__ import annotations

from triton import cdiv
from triton import next_power_of_2

from . import _compat as _compat_module  # noqa: F401  # side-effect import
from . import _logging
from . import exc
from . import language
from . import runtime
from .runtime import AOTKeyFunction
from .runtime import Config
from .runtime import Kernel
from .runtime import aot_kernel
from .runtime import aot_key
from .runtime import extract_shape_features
from .runtime import kernel
from .runtime import kernel as jit  # alias
from .runtime import make_aot_key
from .runtime.settings import RefMode
from .runtime.settings import Settings

__all__ = [
    "AOTKeyFunction",
    "Config",
    "Kernel",
    "RefMode",
    "Settings",
    "aot_kernel",
    "aot_key",
    "cdiv",
    "exc",
    "extract_shape_features",
    "jit",
    "kernel",
    "language",
    "make_aot_key",
    "next_power_of_2",
    "runtime",
]

_logging.init_logs()
