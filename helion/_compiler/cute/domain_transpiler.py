"""Generic domain-based Helion → CuTe DSL transpiler.

Converts Helion FX graphs to CuTe DSL code by classifying every node into
one of 4 data domains (GLOBAL, SHARED, FRAGMENT, SCALAR) and emitting
CuTe DSL code at domain transitions.  Handles arbitrary op chains —
single MMA (GEMM), multi-MMA (attention), pure pointwise, or mixed —
through the same set of domain propagation rules.  No pattern detection.

Architecture:
  FX Graph → [Classify Nodes] → [Backward Pass] → [Forward Pass] → [Emit CuTe DSL]

Domains:
  GLOBAL   — HBM (kernel inputs/outputs)
  SHARED   — SMEM (only for MMA operand tiles)
  FRAGMENT — MMA registers (multi-element per thread, layout-dependent)
  SCALAR   — Per-thread register (single value, layout-independent)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import torch
from torch.fx.node import Node

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Domain & NodeKind enums
# ---------------------------------------------------------------------------


class Domain(enum.Enum):
    """Where a value lives at runtime."""

    GLOBAL = "global"
    SHARED = "shared"  # SMEM — only for MMA operand tiles
    FRAGMENT = "fragment"  # MMA registers — multi-element, layout-dependent
    SCALAR = "scalar"  # Per-thread register — layout-independent


class NodeKind(enum.Enum):
    """Classification of an FX node's operation type."""

    MMA = "mma"
    POINTWISE = "pointwise"
    REDUCTION = "reduction"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    CAST = "cast"
    CONSTANT = "constant"
    PLACEHOLDER = "placeholder"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DomainInfo:
    """Domain assignment for an FX node."""

    domain: Domain
    var: str | None = None  # variable name in generated code
    needs_staging: bool = False  # True when FRAGMENT must go through SMEM
    shape: tuple[int, ...] | None = None
    dtype: torch.dtype | None = None


@dataclass
class MmaContext:
    """Tracks state for one MMA operation in the loop."""

    tiled_mma: str  # variable name for tiled_mma
    thr_mma: str  # variable name for thr_mma
    acc_frag: str  # accumulator fragment variable
    smem_a: str  # shared memory A buffer
    smem_b: str  # shared memory B buffer
    bm: int
    bn: int
    bk: int
    is_loop_carried: bool  # True for o_frag (persists), False for s_frag
    input_dtype_str: str
    acc_dtype_str: str = "cutlass.Float32"


@dataclass
class TranspilerState:
    """Complete state for the domain transpiler during codegen."""

    node_domains: dict[Node, DomainInfo] = field(default_factory=dict)
    fragment_vars: dict[Node, str] = field(default_factory=dict)
    scalar_vars: dict[Node, str] = field(default_factory=dict)
    smem_buffers: dict[str, str] = field(default_factory=dict)
    mma_contexts: list[MmaContext] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Node Classification — fixed registry, no pattern detection
# ---------------------------------------------------------------------------

# Populated lazily to avoid import-time side effects (torch.ops resolution)
_NODE_KIND_MAP: dict[object, NodeKind] | None = None


def _get_node_kind_map() -> dict[object, NodeKind]:
    global _NODE_KIND_MAP
    if _NODE_KIND_MAP is not None:
        return _NODE_KIND_MAP

    from ...language import memory_ops

    _NODE_KIND_MAP = {
        # MMA operations
        torch.ops.aten.mm.default: NodeKind.MMA,
        torch.ops.aten.bmm.default: NodeKind.MMA,
        torch.ops.aten.addmm.default: NodeKind.MMA,
        torch.ops.aten.baddbmm.default: NodeKind.MMA,
        # Pointwise operations
        torch.ops.aten.add.Tensor: NodeKind.POINTWISE,
        torch.ops.aten.mul.Tensor: NodeKind.POINTWISE,
        torch.ops.aten.sub.Tensor: NodeKind.POINTWISE,
        torch.ops.aten.div.Tensor: NodeKind.POINTWISE,
        torch.ops.aten.neg.default: NodeKind.POINTWISE,
        torch.ops.aten.exp2.default: NodeKind.POINTWISE,
        torch.ops.aten.maximum.default: NodeKind.POINTWISE,
        torch.ops.aten.minimum.default: NodeKind.POINTWISE,
        torch.ops.aten.where.self: NodeKind.POINTWISE,
        torch.ops.aten.clamp_min.default: NodeKind.POINTWISE,
        torch.ops.aten.clamp_max.default: NodeKind.POINTWISE,
        # Reductions
        torch.ops.aten.sum.dim_IntList: NodeKind.REDUCTION,
        torch.ops.aten.amax.default: NodeKind.REDUCTION,
        torch.ops.aten.max.dim: NodeKind.REDUCTION,
        # Casts
        torch.ops.prims.convert_element_type.default: NodeKind.CAST,
        # Memory operations
        memory_ops.load: NodeKind.MEMORY_READ,
        memory_ops.store: NodeKind.MEMORY_WRITE,
    }
    return _NODE_KIND_MAP


def classify_node(node: Node) -> NodeKind:
    """Classify an FX node by its operation type."""
    if node.op == "placeholder":
        return NodeKind.PLACEHOLDER
    if node.op == "get_attr":
        return NodeKind.CONSTANT
    if node.op == "output":
        return NodeKind.OTHER
    if node.op != "call_function":
        return NodeKind.OTHER

    kind_map = _get_node_kind_map()
    kind = kind_map.get(node.target)
    if kind is not None:
        return kind

    # Check for scalar constants
    val = node.meta.get("val")
    if val is not None and not isinstance(val, torch.Tensor):
        return NodeKind.CONSTANT

    return NodeKind.OTHER


# ---------------------------------------------------------------------------
# MMA operand targets (for backward pass)
# ---------------------------------------------------------------------------

_MMA_TARGETS: set[object] | None = None


def _get_mma_targets() -> set[object]:
    global _MMA_TARGETS
    if _MMA_TARGETS is not None:
        return _MMA_TARGETS
    _MMA_TARGETS = {
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.baddbmm.default,
    }
    return _MMA_TARGETS


# Ops that are transparent for backward tracing (data-preserving)
_TRACE_THROUGH_TARGETS = {
    torch.ops.prims.convert_element_type.default,
}


def _trace_backward_to_loads(node: Node) -> set[Node]:
    """Trace backward from a node through casts to find load/placeholder nodes."""
    from ...language import memory_ops

    result: set[Node] = set()
    stack = [node]
    visited: set[Node] = set()
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)

        if current.op == "placeholder":
            result.add(current)
            continue
        if current.op == "call_function" and current.target is memory_ops.load:
            result.add(current)
            continue
        if current.op == "call_function" and current.target in _TRACE_THROUGH_TARGETS:
            for arg in current.args:
                if isinstance(arg, Node):
                    stack.append(arg)
            continue
        # For non-traceable ops, treat the node itself as a boundary
        result.add(current)
    return result


# ---------------------------------------------------------------------------
# Domain Propagation — Two-Pass
# ---------------------------------------------------------------------------


def _get_mma_operand_sources(graph: torch.fx.Graph) -> set[Node]:
    """Pass 1 (backward): Find all nodes that feed MMA operands via SMEM.

    For each MMA node, trace its matrix operand inputs backward through
    casts/permutes to find the load/placeholder nodes.  These loads need
    cooperative SMEM loading (Domain.SHARED).
    """
    mma_targets = _get_mma_targets()
    shared_nodes: set[Node] = set()

    for node in graph.nodes:
        if node.op != "call_function" or node.target not in mma_targets:
            continue

        # Get the matrix operand nodes (not the accumulator)
        operand_nodes: list[Node] = []
        if node.target in (
            torch.ops.aten.mm.default,
            torch.ops.aten.bmm.default,
        ):
            # mm/bmm: args = (lhs, rhs)
            for arg in node.args[:2]:
                if isinstance(arg, Node):
                    operand_nodes.append(arg)
        elif node.target in (
            torch.ops.aten.addmm.default,
            torch.ops.aten.baddbmm.default,
        ):
            # addmm/baddbmm: args = (acc, lhs, rhs)
            for arg in node.args[1:3]:
                if isinstance(arg, Node):
                    operand_nodes.append(arg)

        for operand in operand_nodes:
            sources = _trace_backward_to_loads(operand)
            shared_nodes.update(sources)

    return shared_nodes


def propagate_domains(graph: torch.fx.Graph) -> dict[Node, DomainInfo]:
    """Assign a domain to every node in the graph.

    Two-pass algorithm:
      Pass 1 (backward): Mark load/placeholder nodes that feed MMA operands
              as SHARED.
      Pass 2 (forward): Walk topologically, applying domain rules.
    """
    mma_targets = _get_mma_targets()

    # Pass 1: backward from MMA nodes to find SHARED sources
    shared_sources = _get_mma_operand_sources(graph)

    # Pass 2: forward propagation
    domains: dict[Node, DomainInfo] = {}

    # Also track which FRAGMENT nodes need SMEM staging (Rule 7)
    # A FRAGMENT node needs staging if it feeds an MMA as a matrix operand
    mma_operand_users: set[Node] = set()
    for node in graph.nodes:
        if node.op != "call_function" or node.target not in mma_targets:
            continue
        if node.target in (
            torch.ops.aten.mm.default,
            torch.ops.aten.bmm.default,
        ):
            for arg in node.args[:2]:
                if isinstance(arg, Node):
                    mma_operand_users.add(arg)
        elif node.target in (
            torch.ops.aten.addmm.default,
            torch.ops.aten.baddbmm.default,
        ):
            for arg in node.args[1:3]:
                if isinstance(arg, Node):
                    mma_operand_users.add(arg)

    for node in graph.nodes:
        kind = classify_node(node)
        val = node.meta.get("val")
        shape = tuple(val.shape) if isinstance(val, torch.Tensor) else None
        dtype = val.dtype if isinstance(val, torch.Tensor) else None

        if kind == NodeKind.PLACEHOLDER:
            # Placeholders: SHARED if they feed MMA, else SCALAR
            if node in shared_sources:
                domains[node] = DomainInfo(Domain.SHARED, shape=shape, dtype=dtype)
            else:
                domains[node] = DomainInfo(Domain.SCALAR, shape=shape, dtype=dtype)

        elif kind == NodeKind.MMA:
            # Rule 1: MMA output → FRAGMENT
            domains[node] = DomainInfo(Domain.FRAGMENT, shape=shape, dtype=dtype)

        elif kind == NodeKind.MEMORY_READ:
            # Rule 8 vs 9: SHARED if feeds MMA, else SCALAR
            if node in shared_sources:
                domains[node] = DomainInfo(Domain.SHARED, shape=shape, dtype=dtype)
            else:
                domains[node] = DomainInfo(Domain.SCALAR, shape=shape, dtype=dtype)

        elif kind == NodeKind.POINTWISE:
            # Rules 2, 3, 4: depends on input domains
            input_domains = _get_input_domains(node, domains)
            has_fragment = Domain.FRAGMENT in input_domains

            if has_fragment:
                # Rule 2 (all FRAGMENT) or Rule 3 (FRAGMENT + SCALAR)
                domains[node] = DomainInfo(Domain.FRAGMENT, shape=shape, dtype=dtype)
            else:
                # Rule 4: all SCALAR (or no tensor inputs)
                domains[node] = DomainInfo(Domain.SCALAR, shape=shape, dtype=dtype)

        elif kind == NodeKind.REDUCTION:
            # Rule 5 vs 6: depends on input domain
            input_domains = _get_input_domains(node, domains)
            if Domain.FRAGMENT in input_domains:
                # Rule 5: REDUCTION(FRAGMENT) → SCALAR
                domains[node] = DomainInfo(Domain.SCALAR, shape=shape, dtype=dtype)
            else:
                # Rule 6: REDUCTION(SCALAR) → SCALAR
                domains[node] = DomainInfo(Domain.SCALAR, shape=shape, dtype=dtype)

        elif kind == NodeKind.CAST:
            # Rules 12, 13: inherit domain from input
            input_domains = _get_input_domains(node, domains)
            if Domain.FRAGMENT in input_domains:
                domains[node] = DomainInfo(Domain.FRAGMENT, shape=shape, dtype=dtype)
            else:
                domains[node] = DomainInfo(Domain.SCALAR, shape=shape, dtype=dtype)

        elif kind == NodeKind.MEMORY_WRITE:
            # Rules 10, 11: destination is GLOBAL
            domains[node] = DomainInfo(Domain.GLOBAL, shape=shape, dtype=dtype)

        elif kind == NodeKind.CONSTANT:
            domains[node] = DomainInfo(Domain.SCALAR, shape=shape, dtype=dtype)

        else:
            # Default to SCALAR for unknown ops
            domains[node] = DomainInfo(Domain.SCALAR, shape=shape, dtype=dtype)

    # Mark FRAGMENT nodes that need SMEM staging (Rule 7)
    for node in mma_operand_users:
        info = domains.get(node)
        if info is not None and info.domain == Domain.FRAGMENT:
            info.needs_staging = True

    return domains


def _get_input_domains(
    node: Node, domains: dict[Node, DomainInfo]
) -> set[Domain]:
    """Get the set of domains of a node's tensor inputs."""
    result: set[Domain] = set()
    for arg in node.args:
        if isinstance(arg, Node):
            info = domains.get(arg)
            if info is not None:
                # Only count tensor inputs (not scalar constants like dim args)
                arg_val = arg.meta.get("val")
                if isinstance(arg_val, torch.Tensor):
                    result.add(info.domain)
    return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def has_mma_nodes(graph: torch.fx.Graph) -> bool:
    """Check if a graph contains any MMA-compatible operations."""
    mma_targets = _get_mma_targets()
    for node in graph.nodes:
        if node.op == "call_function" and node.target in mma_targets:
            return True
    return False


# ---------------------------------------------------------------------------
# CuTe DSL Emitters
# ---------------------------------------------------------------------------

# Maps aten ops to CuTe DSL operator strings for fragment-level codegen
_CUTE_BINARY_OPS: dict[object, str] = {
    torch.ops.aten.add.Tensor: "+",
    torch.ops.aten.sub.Tensor: "-",
    torch.ops.aten.mul.Tensor: "*",
    torch.ops.aten.div.Tensor: "/",
    torch.ops.aten.maximum.default: "max",
    torch.ops.aten.minimum.default: "min",
}

_CUTE_UNARY_OPS: dict[object, str] = {
    torch.ops.aten.neg.default: "-",
    torch.ops.aten.exp2.default: "cute.exp2",
}


def emit_fragment_pointwise(
    target: object,
    result_var: str,
    input_vars: list[str],
    input_domains: list[Domain],
) -> list[str]:
    """Emit element-wise fragment loop for Rules 2 and 3.

    Returns a list of CuTe DSL code lines.
    """
    lines: list[str] = []

    if target in _CUTE_UNARY_OPS:
        op = _CUTE_UNARY_OPS[target]
        src = input_vars[0]
        if op.startswith("cute."):
            # Function-call form: cute.exp2(x)
            lines.append(f"for _i in range(cute.size({result_var})):")
            lines.append(f"    {result_var}[_i] = {op}({src}[_i])")
        else:
            # Prefix unary: -x
            lines.append(f"for _i in range(cute.size({result_var})):")
            lines.append(f"    {result_var}[_i] = {op}{src}[_i]")
        return lines

    if target in _CUTE_BINARY_OPS:
        op = _CUTE_BINARY_OPS[target]
        lhs, rhs = input_vars[0], input_vars[1]
        lhs_dom, rhs_dom = input_domains[0], input_domains[1]

        # Determine which operands need indexing
        lhs_expr = f"{lhs}[_i]" if lhs_dom == Domain.FRAGMENT else lhs
        rhs_expr = f"{rhs}[_i]" if rhs_dom == Domain.FRAGMENT else rhs

        # Determine loop source (the FRAGMENT operand)
        loop_src = lhs if lhs_dom == Domain.FRAGMENT else rhs

        if op in ("max", "min"):
            lines.append(f"for _i in range(cute.size({loop_src})):")
            lines.append(
                f"    {result_var}[_i] = {op}({lhs_expr}, {rhs_expr})"
            )
        else:
            lines.append(f"for _i in range(cute.size({loop_src})):")
            lines.append(
                f"    {result_var}[_i] = {lhs_expr} {op} {rhs_expr}"
            )
        return lines

    # Fallback: treat as unary identity
    lines.append(f"for _i in range(cute.size({result_var})):")
    lines.append(f"    {result_var}[_i] = {input_vars[0]}[_i]")
    return lines


def emit_scalar_pointwise(
    target: object,
    result_var: str,
    input_vars: list[str],
) -> list[str]:
    """Emit plain register operation for Rule 4."""
    if target in _CUTE_UNARY_OPS:
        op = _CUTE_UNARY_OPS[target]
        if op.startswith("cute."):
            return [f"{result_var} = {op}({input_vars[0]})"]
        return [f"{result_var} = {op}{input_vars[0]}"]

    if target in _CUTE_BINARY_OPS:
        op = _CUTE_BINARY_OPS[target]
        if op in ("max", "min"):
            return [f"{result_var} = {op}({input_vars[0]}, {input_vars[1]})"]
        return [f"{result_var} = {input_vars[0]} {op} {input_vars[1]}"]

    return [f"{result_var} = {input_vars[0]}"]


def emit_fragment_reduction(
    target: object,
    frag_var: str,
    result_var: str,
    threads_in_group: int | str,
) -> list[str]:
    """Emit within-thread reduce + warp shuffle for Rule 5."""
    lines: list[str] = []

    if target in (torch.ops.aten.amax.default, torch.ops.aten.max.dim):
        lines.append(f"{result_var} = cutlass.Float32(float('-inf'))")
        lines.append(f"for _i in range(cute.size({frag_var})):")
        lines.append(f"    {result_var} = max({result_var}, {frag_var}[_i])")
        lines.append(
            f"{result_var} = cute.arch.warp_reduction_max("
            f"{result_var}, threads_in_group={threads_in_group})"
        )
    elif target == torch.ops.aten.sum.dim_IntList:
        lines.append(f"{result_var} = cutlass.Float32(0.0)")
        lines.append(f"for _i in range(cute.size({frag_var})):")
        lines.append(f"    {result_var} = {result_var} + {frag_var}[_i]")
        lines.append(
            f"{result_var} = cute.arch.warp_reduction_sum("
            f"{result_var}, threads_in_group={threads_in_group})"
        )
    else:
        # Generic fallback: sum
        lines.append(f"{result_var} = cutlass.Float32(0.0)")
        lines.append(f"for _i in range(cute.size({frag_var})):")
        lines.append(f"    {result_var} = {result_var} + {frag_var}[_i]")

    return lines


def emit_fragment_to_shared(
    frag_var: str,
    smem_var: str,
    thr_mma_var: str,
    partition_var: str,
) -> list[str]:
    """Emit FRAGMENT → SHARED transition (Rule 7)."""
    return [
        f"{partition_var} = {thr_mma_var}.partition_C({smem_var})",
        f"for _i in range(cute.size({partition_var})):",
        f"    {partition_var}[_i] = {frag_var}[_i]",
        "cute.arch.sync_threads()",
    ]


def emit_fragment_cast(
    frag_var: str,
    result_var: str,
    dtype_str: str,
) -> list[str]:
    """Emit CAST on FRAGMENT (Rule 12)."""
    return [
        f"for _i in range(cute.size({result_var})):",
        f"    {result_var}[_i] = {dtype_str}({frag_var}[_i])",
    ]


def emit_mma_setup_universal(
    tiled_mma_var: str,
    thr_mma_var: str,
    acc_frag_var: str,
    thread_linear_expr: str,
    acc_dtype_str: str,
    bm: int,
    bn: int,
) -> list[str]:
    """Emit universal MMA tiled_mma + thr_mma + accumulator setup."""
    return [
        f"{tiled_mma_var} = cute.make_tiled_mma("
        f"cute.nvgpu.MmaUniversalOp(abacc_dtype={acc_dtype_str}), "
        f"atom_layout_mnk=({bm}, {bn}, 1))",
        f"{thr_mma_var} = {tiled_mma_var}.get_slice({thread_linear_expr})",
        f"{acc_frag_var} = cute.make_fragment("
        f"{thr_mma_var}.partition_shape_C(({bm}, {bn})), {acc_dtype_str})",
        f"for _i in range(cute.size({acc_frag_var})):",
        f"    {acc_frag_var}[_i] = {acc_dtype_str}(0.0)",
    ]


def emit_smem_alloc(
    smem_ptr_var: str,
    smem_var: str,
    dtype_str: str,
    rows: int,
    cols: int,
) -> list[str]:
    """Emit shared memory allocation with row-major layout."""
    return [
        f"{smem_ptr_var} = cute.arch.alloc_smem({dtype_str}, {rows * cols})",
        f"{smem_var} = cute.make_tensor("
        f"{smem_ptr_var}, cute.make_layout(({rows}, {cols}), stride=({cols}, 1)))",
    ]


def emit_gemm(
    tiled_mma_var: str,
    acc_var: str,
    thr_mma_var: str,
    smem_a_var: str,
    smem_b_var: str,
    acc_dtype_str: str,
) -> list[str]:
    """Emit cute.gemm with partition_A/B and register staging."""
    tAsA = "_tAsA"
    tBsB = "_tBsB"
    rA = "_rA"
    rB = "_rB"
    return [
        f"{tAsA} = {thr_mma_var}.partition_A({smem_a_var})",
        f"{tBsB} = {thr_mma_var}.partition_B({smem_b_var})",
        f"{rA} = cute.make_fragment_like({tAsA}, {acc_dtype_str})",
        f"{rB} = cute.make_fragment_like({tBsB}, {acc_dtype_str})",
        f"for _mma_i in range(cute.size({rA})):",
        f"    {rA}[_mma_i] = {acc_dtype_str}({tAsA}[_mma_i])",
        f"for _mma_i in range(cute.size({rB})):",
        f"    {rB}[_mma_i] = {acc_dtype_str}({tBsB}[_mma_i])",
        f"cute.gemm({tiled_mma_var}, {acc_var}, {rA}, {rB}, {acc_var})",
    ]


def emit_fragment_readback(
    acc_var: str,
    thr_mma_var: str,
    smem_c_var: str,
    result_var: str,
    m_local: str,
    n_local: str,
) -> list[str]:
    """Emit fragment → SMEM → scalar readback (for final store)."""
    tCsC = "_tCsC"
    return [
        f"{tCsC} = {thr_mma_var}.partition_C({smem_c_var})",
        f"for _mma_i in range(cute.size({tCsC})):",
        f"    {tCsC}[_mma_i] = {acc_var}[_mma_i]",
        "cute.arch.sync_threads()",
        f"{result_var} = {smem_c_var}[{m_local}, {n_local}]",
    ]
