"""Fusion eligibility checks for prologue and epilogue operations.

This module provides a unified validation pipeline for determining whether
scheduler nodes can be fused into Helion templates.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import sympy

if TYPE_CHECKING:
    from torch._inductor.scheduler import SchedulerNode

log = logging.getLogger(__name__)

# Ops supported in FusionOpsHandler (safe for fusion)
SUPPORTED_FUSION_OPS = frozenset({
    # Core ops
    "load", "store", "store_reduction", "constant", "index_expr", "to_dtype",
    # Math ops
    "exp", "log", "sqrt", "sin", "cos", "sigmoid", "tanh", "relu", "maximum", "minimum", "reciprocal",
    # Binary ops
    "add", "sub", "mul", "truediv", "floordiv", "mod", "neg",
    # Comparison ops
    "eq", "ne", "lt", "gt", "le", "ge", "where",
    # Logic ops
    "logical_and", "logical_or", "logical_not", "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not",
    # Other safe ops
    "abs", "pow", "floor", "ceil", "trunc", "round", "exp2", "log2", "log10", "log1p", "expm1",
    "rsqrt", "asin", "acos", "atan", "atan2", "sinh", "cosh", "asinh", "acosh", "atanh",
    "erf", "erfc", "sign", "signbit", "isinf", "isnan", "isfinite", "fma",
})

# Safe symbol prefixes for fusion indexing (static/predictable)
_SAFE_PREFIXES = ("s", "i", "x", "y", "z", "r0_", "r1_", "r", "idx", "ps", "h", "view", "u", "c", "d")
_UNSAFE_PREFIXES = ("tmp", "indirect")


# =============================================================================
# Core Validation Functions (shared by prologue and epilogue)
# =============================================================================

def _is_pointwise(node) -> bool:
    """Check if node is a simple pointwise operation."""
    from torch._inductor.ir import ComputedBuffer, Pointwise
    if isinstance(node, Pointwise):
        return True
    if isinstance(node, ComputedBuffer):
        return isinstance(node.data, Pointwise)
    return False


def _get_size(node):
    """Get the size/shape of a node."""
    if hasattr(node, "get_size"):
        return node.get_size()
    if hasattr(node, "layout") and hasattr(node.layout, "size"):
        return node.layout.size
    raise ValueError(f"Cannot get size from {type(node)}")


def _get_stride(node):
    """Get the stride of a node."""
    if hasattr(node, "get_stride"):
        return node.get_stride()
    if hasattr(node, "layout") and hasattr(node.layout, "stride"):
        return node.layout.stride
    raise ValueError(f"Cannot get stride from {type(node)}")


def _check_index_symbols(node) -> tuple[bool, Optional[str]]:
    """Check if all index symbols in node are safe for fusion."""
    rw = getattr(node, "read_writes", None) or getattr(getattr(node, "node", None), "read_writes", None)
    if rw is None:
        return True, None

    for dep_list, dep_type in [(getattr(rw, "reads", []), "read"), (getattr(rw, "writes", []), "write")]:
        for dep in dep_list:
            index = getattr(dep, "index", None)
            if index is None:
                continue
            try:
                free_syms = index.free_symbols if hasattr(index, "free_symbols") else set()
            except Exception:
                continue
            for sym in free_syms:
                if not isinstance(sym, sympy.Symbol):
                    continue
                name = sym.name.lower()
                if any(name.startswith(p) for p in _UNSAFE_PREFIXES):
                    return False, f"unsafe index symbol '{sym.name}' in {dep_type}"
                if not any(name.startswith(p) for p in _SAFE_PREFIXES):
                    return False, f"unknown index symbol '{sym.name}' in {dep_type}"
    return True, None


def _check_supported_ops(node) -> tuple[bool, Optional[str]]:
    """Check if all ops used by node are in the supported whitelist."""
    inner_fn = getattr(node, "inner_fn", None) or getattr(getattr(node, "data", None), "inner_fn", None)
    if inner_fn is None:
        return True, None

    class OpCollector:
        def __init__(self):
            self.ops = set()
        def __getattr__(self, name):
            self.ops.add(name)
            return lambda *a, **kw: f"<{name}>"

    try:
        collector = OpCollector()
        ranges = node.get_ranges() if hasattr(node, "get_ranges") else [[1]]
        num_dims = len(ranges[0]) if ranges else 1
        inner_fn(collector, *[sympy.Symbol(f"i{i}") for i in range(num_dims)])
        unsupported = collector.ops - SUPPORTED_FUSION_OPS
        if unsupported:
            return False, f"unsupported ops: {unsupported}"
    except Exception as e:
        log.debug(f"Op extraction failed (allowing): {e}")
    return True, None


# =============================================================================
# Unified Fusion Check
# =============================================================================

def _can_fuse_node(
    sched_node: "SchedulerNode",
    ref_buffer,
    *,
    check_shape: bool = True,
    check_stride: bool = True,
) -> tuple[bool, Optional[str]]:
    """Unified check for whether a node can be fused.

    Args:
        sched_node: The scheduler node to check
        ref_buffer: Reference buffer for shape/stride comparison (for prologues)
        check_shape: Whether to check shape compatibility
        check_stride: Whether to check stride compatibility

    Returns:
        (can_fuse, reason) tuple
    """
    inner_node = sched_node.node if hasattr(sched_node, "node") else sched_node

    # Must be pointwise
    if not _is_pointwise(inner_node):
        return False, "not pointwise"

    # Shape check (for prologues)
    if check_shape and ref_buffer is not None:
        try:
            if _get_size(inner_node) != _get_size(ref_buffer):
                return False, "shape mismatch"
        except Exception as e:
            return False, f"cannot determine shapes: {e}"

    # Stride check (for prologues)
    if check_stride and ref_buffer is not None:
        try:
            if _get_stride(inner_node) != _get_stride(ref_buffer):
                return False, "stride mismatch"
        except Exception as e:
            return False, f"cannot determine strides: {e}"

    # Index symbol check
    ok, reason = _check_index_symbols(sched_node)
    if not ok:
        return False, reason

    # Supported ops check
    ok, reason = _check_supported_ops(inner_node)
    if not ok:
        return False, reason

    return True, None


# =============================================================================
# Public API: Prologue and Epilogue Checks
# =============================================================================

def can_fuse_prologue(
    prologue_node: "SchedulerNode",
    template_input_buffer,
) -> tuple[bool, Optional[str]]:
    """Check if prologue can be fused into template."""
    return _can_fuse_node(prologue_node, template_input_buffer, check_shape=True, check_stride=True)


def can_fuse_epilogue(
    epilogue_node: "SchedulerNode",
    template_outputs: set[str],
) -> tuple[bool, Optional[str]]:
    """Check if epilogue can be fused into template."""
    from .helpers import is_multi_output_node

    if not (hasattr(epilogue_node, "node") and hasattr(epilogue_node, "read_writes")):
        return False, "missing required attributes"

    # MultiOutput nodes are always allowed (handled by template)
    if is_multi_output_node(epilogue_node):
        return True, None

    # Check for circular dependency (reading multiple outputs)
    if epilogue_node.read_writes:
        output_reads = {
            dep.name for dep in epilogue_node.read_writes.reads
            if hasattr(dep, "name") and dep.name in template_outputs
        }
        if len(output_reads) > 1:
            return False, f"reads from multiple outputs: {output_reads}"

    # Use unified check (no shape/stride check for epilogues)
    return _can_fuse_node(epilogue_node, None, check_shape=False, check_stride=False)


# =============================================================================
# Batch Preparation Functions
# =============================================================================

def prepare_prologues_for_template(
    template,
    buf_name_to_prologue_group: dict,
) -> tuple[dict, list]:
    """Separate prologues into fusable and non-fusable groups."""
    from .helpers import safe_get_name

    fusable, non_fusable = {}, []
    input_by_name = {
        name: inp for inp in template.inputs
        if (name := safe_get_name(inp)) is not None
    }

    for buf_name, prologue_nodes in buf_name_to_prologue_group.items():
        template_input = input_by_name.get(buf_name)
        if template_input is None:
            log.debug(f"Prologue {buf_name} has no matching template input")
            non_fusable.extend(prologue_nodes)
            continue

        all_ok = all(can_fuse_prologue(node, template_input)[0] for node in prologue_nodes)
        if all_ok:
            fusable[buf_name] = prologue_nodes
        else:
            non_fusable.extend(prologue_nodes)

    return fusable, non_fusable


def prepare_epilogues_for_template(
    template,
    epilogue_nodes: list,
) -> tuple[dict[str, list], list]:
    """Separate epilogues into fusable and non-fusable groups."""
    from .helpers import safe_get_name, is_multi_output_node

    if not epilogue_nodes:
        return {}, []

    template_outputs = {template.get_name()}
    for attr in ("outputs", "multi_output_nodes"):
        for out in getattr(template, attr, []):
            name = safe_get_name(out)
            if name:
                template_outputs.add(name)

    fusable_by_output: dict[str, list] = {}
    multi_outputs, non_fusable = [], []

    for ep in epilogue_nodes:
        can_fuse, reason = can_fuse_epilogue(ep, template_outputs)
        if not can_fuse:
            log.debug(f"Epilogue {getattr(ep, 'get_name', lambda: str(ep))()} not fusable: {reason}")
            non_fusable.append(ep)
        elif is_multi_output_node(ep):
            multi_outputs.append(ep)
        elif hasattr(ep, "read_writes") and ep.read_writes:
            for dep in ep.read_writes.reads:
                if hasattr(dep, "name") and dep.name in template_outputs:
                    fusable_by_output.setdefault(dep.name, []).append(ep)
                    break

    if multi_outputs:
        fusable_by_output["__multi_output__"] = multi_outputs

    return fusable_by_output, non_fusable
