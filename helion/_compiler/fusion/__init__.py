"""Helion-Inductor fusion support."""
from .convert import (
    EpilogueSpec, PrologueSpec, FusionContext, fusion_context, get_current_context,
    FusionFXConverter, apply_epilogue, apply_prologue, generate_epilogue_store,
    inject_closure_params, rename_function, FusionError,
)
from .inductor import (
    HelionTemplateBuffer, lower_helion_kernel,
    safe_get_name, is_multi_output_node, partition_multi_output,
)

# Backwards compatibility aliases
HelionTemplateKernel = HelionTemplateBuffer
HelionCodegen = HelionTemplateBuffer  # Deprecated: methods now on HelionTemplateBuffer

__all__ = [
    "EpilogueSpec", "PrologueSpec", "FusionContext", "fusion_context", "get_current_context",
    "HelionCodegen", "HelionTemplateKernel",
    "apply_epilogue", "apply_prologue", "generate_epilogue_store",
    "inject_closure_params", "rename_function", "FusionError",
    "FusionFXConverter", "HelionTemplateBuffer", "lower_helion_kernel",
    "safe_get_name", "is_multi_output_node", "partition_multi_output",
]
