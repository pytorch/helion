from __future__ import annotations

from typing import TYPE_CHECKING

from torch._dynamo.variables.builder import GuardBuilder
from torch._dynamo.variables.builder import VariableBuilder

from helion._compiler._dynamo.variables import HelionKernelVariable

if TYPE_CHECKING:
    from torch._dynamo.variables.base import VariableTracker


def register_dynamo_variable() -> None:
    from helion.runtime.kernel import Kernel

    def wrap_helion_kernel(self: VariableBuilder, value: Kernel) -> VariableTracker:
        if value.settings._wip_experimental_allow_torch_compile_fusion:
            # Use HOP-based torch.compile integration with HelionKernelVariable.
            # This enables fusion of Helion kernels with surrounding Inductor ops.
            self.install_guards(GuardBuilder.ID_MATCH)
            return HelionKernelVariable(value, None, source=self.source)
        # Fall back to existing triton_kernel_wrapper_mutation HOP mechanism.
        return self.wrap_user_defined(value)

    type_dispatch = VariableBuilder._type_dispatch()
    type_dispatch[Kernel] = wrap_helion_kernel
