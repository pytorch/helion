from __future__ import annotations

from triton import cdiv
from triton import next_power_of_2

from . import _compat as _compat_module  # noqa: F401  # side-effect import
from . import _logging
from . import exc
from . import language
from . import runtime
from .autotuner.config_space import ConfigSpace
from .runtime import Config
from .runtime import Kernel
from .runtime import kernel
from .runtime import kernel as jit  # alias
from .runtime.settings import RefMode
from .runtime.settings import Settings

__all__ = [
    "Config",
    "ConfigSpace",
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

# Register with Dynamo after all modules are fully loaded
from ._compiler._dynamo.variables import register_dynamo_variable  # noqa: E402

register_dynamo_variable()
