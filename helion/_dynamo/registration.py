"""Registration of Helion with PyTorch Dynamo."""
from __future__ import annotations
import functools
from typing import Any

_registered = False

def _is_helion_kernel(value: Any) -> bool:
    try: from helion.runtime.kernel import Kernel; return isinstance(value, Kernel)
    except ImportError: return False

def register_with_dynamo() -> None:
    """Register Helion kernel handling with PyTorch Dynamo."""
    global _registered
    if _registered: return
    _registered = True
    try: from torch._dynamo.variables.builder import VariableBuilder, GuardBuilder
    except ImportError: return
    original_wrap = VariableBuilder._wrap
    @functools.wraps(original_wrap)
    def patched_wrap(self, value):
        if _is_helion_kernel(value):
            from helion._dynamo.variables import HelionKernelVariable
            self.install_guards(GuardBuilder.ID_MATCH)
            return HelionKernelVariable(value, None, source=self.source)
        return original_wrap(self, value)
    VariableBuilder._wrap = patched_wrap

def unregister_from_dynamo() -> None:
    global _registered; _registered = False
