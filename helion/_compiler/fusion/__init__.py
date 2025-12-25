"""Helion-Inductor fusion support."""
from .convert import (
    EpilogueSpec, PrologueSpec,
    inject_closure_params, rename_function, FusionError,
    get_torch_op, fx_to_pytorch_epilogue,
    EpilogueOp, PrologueInfo, EpilogueTracingState,
    epilogue_tracing_context, get_epilogue_tracing_state, extract_prologue_ops,
    create_helion_traceable_epilogue,
)
from .inductor import (
    HelionTemplateBuffer, lower_helion_kernel,
    partition_multi_output,
)

__all__ = [
    "EpilogueSpec", "PrologueSpec",
    "inject_closure_params", "rename_function", "FusionError",
    "get_torch_op", "fx_to_pytorch_epilogue",
    "EpilogueOp", "PrologueInfo", "EpilogueTracingState",
    "epilogue_tracing_context", "get_epilogue_tracing_state",
    "HelionTemplateBuffer", "lower_helion_kernel",
    "partition_multi_output",
]
