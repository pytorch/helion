"""Registration of Helion with PyTorch Dynamo."""

from __future__ import annotations

from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.builder import GuardBuilder
from torch._dynamo.variables.builder import VariableBuilder
from torch._dynamo.variables.user_defined import UserDefinedObjectVariable

from helion._compiler._dynamo.variables import HelionKernelVariable


def register_dynamo_variable() -> None:
    """Register Helion kernel handling with PyTorch Dynamo.

    This registers Helion Kernel type with VariableBuilder's type dispatch table,
    so that Kernels are wrapped with HelionKernelVariable during tracing.
    """
    from helion.runtime.kernel import Kernel

    def wrap_helion_kernel(
        self: VariableBuilder, value: Kernel
    ) -> VariableTracker:
        """Wrap a Helion Kernel as a HelionKernelVariable.

        If fusion is disabled, returns UserDefinedObjectVariable for default handling.
        """
        self.install_guards(GuardBuilder.ID_MATCH)
        if not value.settings.allow_fuse_with_inductor_ops:
            return UserDefinedObjectVariable(value, source=self.source)
        return HelionKernelVariable(value, None, source=self.source)

    # Get the type dispatch dictionary and add our handler
    # The dictionary is cached by functools.cache, so modifying it persists
    type_dispatch = VariableBuilder._type_dispatch()
    type_dispatch[Kernel] = wrap_helion_kernel
