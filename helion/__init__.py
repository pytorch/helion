from __future__ import annotations

import os

from . import _logging
from . import exc
from . import language
from . import runtime
from ._utils import cdiv
from ._utils import next_power_of_2
from .runtime import Config
from .runtime import Kernel
from .runtime import kernel
from .runtime import kernel as jit  # alias
from .runtime.settings import RefMode
from .runtime.settings import Settings

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

# Register with Dynamo after all modules are fully loaded (only when torch.compile
# fusion is enabled, to avoid pulling in torch._dynamo at import time which triggers
# CUDA initialization and breaks pytest-xdist workers).
if os.environ.get("_WIP_DEV_ONLY_HELION_TORCH_COMPILE_FUSION", "0") == "1":
    from ._compiler._dynamo.variables import register_dynamo_variable

    register_dynamo_variable()
