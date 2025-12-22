"""Helion-Inductor fusion support."""
from .context import EpilogueSpec, PrologueSpec, FusionContext, fusion_context, get_current_context
from .codegen import HelionCodegen, HelionRender, apply_fusion_ast, inject_closure_params, rename_function, FusionError
from .ops_handler import FusionOpsHandler
from .template import HelionTemplate, lower_helion_kernel, register_helion_lowerings
from .helpers import safe_get_name, is_multi_output_node, partition_multi_output, get_node_buffer_names

__all__ = [
    "EpilogueSpec", "PrologueSpec", "FusionContext", "fusion_context", "get_current_context",
    "HelionCodegen", "HelionRender", "apply_fusion_ast", "inject_closure_params", "rename_function", "FusionError",
    "FusionOpsHandler", "HelionTemplate", "lower_helion_kernel", "register_helion_lowerings",
    "safe_get_name", "is_multi_output_node", "partition_multi_output", "get_node_buffer_names",
]
