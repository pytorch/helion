"""Registration of Helion with PyTorch Dynamo."""

from __future__ import annotations

from typing import TYPE_CHECKING

_registered = False


if TYPE_CHECKING:
    from typing import Any

    from torch._dynamo.variables.base import VariableTracker
    from torch._dynamo.variables.builder import VariableBuilder


def register_with_dynamo() -> None:
    """Register Helion kernel handling with PyTorch Dynamo.

    This registers Helion Kernel type with VariableBuilder's type dispatch table,
    so that Kernels are wrapped with HelionKernelVariable during tracing.
    """
    global _registered
    if _registered:
        return
    _registered = True

    try:
        from torch._dynamo.variables.builder import GuardBuilder
        from torch._dynamo.variables.builder import VariableBuilder
    except ImportError:
        return

    # Import Kernel - safe now because _dynamo is imported after Kernel in helion/__init__.py
    from helion.runtime.kernel import Kernel

    from helion._dynamo.variables import HelionKernelVariable

    def wrap_helion_kernel(
        self: VariableBuilder, value: Kernel
    ) -> HelionKernelVariable:
        """Wrap a Helion Kernel as a HelionKernelVariable."""
        self.install_guards(GuardBuilder.ID_MATCH)
        return HelionKernelVariable(value, None, source=self.source)

    # Get the type dispatch dictionary and add our handler
    # The dictionary is cached by functools.cache, so modifying it persists
    type_dispatch = VariableBuilder._type_dispatch()
    type_dispatch[Kernel] = wrap_helion_kernel
