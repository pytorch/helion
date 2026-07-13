from __future__ import annotations

from . import _compat as _compat_module  # noqa: F401  # side-effect import
from . import _logging
from . import exc
from . import language
from . import runtime
from ._utils import cdiv
from ._utils import next_power_of_2
from .autotuner import from_cache
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
    "from_cache",
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

# Register the NPU (Ascend) Inductor backend + Dynamo device interface when NPU
# is available, so inductor lowerings (e.g. aten.sum) can resolve a scheduling
# constructor for the npu device instead of hitting ``assert scheduling_ctor``.
import torch as _torch  # noqa: E402

if hasattr(_torch, "npu") and _torch.npu.is_available():
    try:
        _compat_module.register_npu_backend()
        _compat_module._register_interface_for_device()
    except Exception:  # pragma: no cover  # noqa: BLE001
        import logging as _logging

        _logging.getLogger(__name__).debug(
            "NPU Inductor backend registration failed; NPU lowerings may be limited.",
            exc_info=True,
        )
