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
        self.install_guards(GuardBuilder.ID_MATCH)
        # Always use HelionKernelVariable for HOP-based torch.compile integration
        return HelionKernelVariable(value, None, source=self.source)

    type_dispatch = VariableBuilder._type_dispatch()
    type_dispatch[Kernel] = wrap_helion_kernel
