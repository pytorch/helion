"""Whole-root warp MMA lowering for contractions with computed operands.

Each contraction has its own logical coordinates and shared-memory operands.
The bridge and final epilogue reuse the ordinary pointwise lowerings, preserving
explicit conversions, broadcasting, indexing, and masks.  No expression,
function name, or workload shape is used to select this schedule.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._inductor.virtualized import V
from torch.fx import Node
from torch.fx.node import map_arg

from ... import exc
from ...language import _tracing_ops
from ...language import creation_ops
from ...language import memory_ops
from ...language import scan_ops
from ...language import tile_index
from ...language import tile_ops
from ...language import view_ops
from ...language.matmul_ops import dot
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from ..inductor_lowering import PointwiseLowering
from .chained_execution import ChainedExecution
from .chained_workspace import plan_contraction_workspace
from .contraction_region import collect_contraction_region
from .cute_reshape import _get_tile_shape
from .fragment_epilogue import _has_fresh_output_allocation
from .fx_matcher import _GeneratedCodeTemplate
from .mma_support import get_cute_mma_support
from .tcgen05_config import CuteTcgen05Config

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from .chained_contraction_groups import ContractionGroup
    from .chained_fragment_epilogue import WarpOperandEpilogue
    from .chained_initialized_accumulator import InitializedAccumulator
    from .chained_k_schedule import KSchedule
    from .chained_late_rhs import LateRhsArenaPlan
    from .chained_loop import ChainedLoopPlan
    from .chained_pointwise_residency import PointwiseCachePlan
    from .chained_preparation_pipeline import PreparationPipeline
    from .chained_prepared_image_emission import BoundRawWidenings
    from .chained_scan_export import ScanExport
    from .chained_scan_export import ScanExportStores
    from .contraction_region import ContractionRegion
    from .warp_specialized_plan import SharedMemoryLayoutPlan


class _UnsupportedChain(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class ChainedMatmulPlan:
    root_graph_id: int
    dots: tuple[Node, ...]
    store: Node
    axes: tuple[tuple[int, int, int], ...]
    shapes: tuple[tuple[int, int, int], ...]
    dtype: torch.dtype
    threads: int
    scans: tuple[Node, ...] = ()
    tensor_aliases: dict[str, str] = dataclasses.field(
        default_factory=dict, compare=False
    )
    strategy: str = "warp"
    scan_exports: tuple[ScanExport, ...] = ()
    initialized_accumulator: InitializedAccumulator | None = None
    late_rhs_reuse: LateRhsArenaPlan | None = None
    direct_output: bool = False
    k_schedule: KSchedule | None = None
    region: ContractionRegion | None = None
    loop: ChainedLoopPlan | None = None
    contraction_groups: tuple[ContractionGroup, ...] | None = None
    pointwise_cache: PointwiseCachePlan | None = None
    warp_mma_stages: frozenset[int] = frozenset()
    loop_workspace: SharedMemoryLayoutPlan | None = None
    preparation_pipeline: PreparationPipeline | None = None
    prepared_widenings: BoundRawWidenings | None = None

    def operand_dtype(self, stage: int) -> torch.dtype:
        """MMA storage belongs to each contraction, not the whole region."""
        if self.region is not None:
            return self.region.contractions[stage].operand_dtypes[0]
        return cast("Node", self.dots[stage].args[0]).meta["val"].dtype


@dataclasses.dataclass(frozen=True)
class _StagedInput:
    source: Node
    operand: Node
    shared: str
    role: str
    shape: tuple[int, int]
    indices: tuple[sympy.Expr, ...]
    coordinates: tuple[sympy.Symbol, sympy.Symbol]
    inverse_axes: tuple[tuple[int, int], tuple[int, int]]


@dataclasses.dataclass(frozen=True)
class _ScanInput:
    source: Node
    operand: Node
    shared: str
    extent: int
    indices: tuple[sympy.Expr, ...]
    coordinate: sympy.Symbol
    inverse_axis: tuple[int, int]


_SCALAR_BINARY = {
    operator.add: "+",
    operator.sub: "-",
    operator.mul: "*",
    operator.floordiv: "//",
    operator.mod: "%",
    operator.truediv: "/",
}
_REMAINDER_TARGETS = {
    torch.ops.aten.remainder.Tensor,
    torch.ops.aten.remainder.Scalar,
    torch.ops.aten.remainder.Scalar_Tensor,
}


def _signed_remainder_adjustment(remainder: str, divisor: str) -> str:
    # CuTe integer % is C remainder (sign of dividend); Torch remainder has
    # the sign of the divisor. The correction cannot overflow: |r| < |b|.
    return (
        f"(({remainder}) + ({divisor}) if (({remainder}) != 0) & "
        f"((({remainder}) < 0) != (({divisor}) < 0)) else ({remainder}))"
    )


def _signed_floor_adjustment(quotient: str, remainder: str, divisor: str) -> str:
    return (
        f"(({quotient}) - 1 if (({remainder}) != 0) & "
        f"((({remainder}) < 0) != (({divisor}) < 0)) else ({quotient}))"
    )


def _power_of_two_divisor_shift(divisor: object, dtype: torch.dtype) -> int | None:
    """Recognize a positive literal after the original signed operand cast."""
    if (
        type(divisor) is not int
        or divisor <= 0
        or divisor & (divisor - 1)
        or divisor > torch.iinfo(dtype).max
    ):
        return None
    return divisor.bit_length() - 1


def _is_floor_divide(node: Node) -> bool:
    return node.target in (
        torch.ops.aten.floor_divide.default,
        torch.ops.aten.floor_divide.Scalar,
    ) or (
        node.target in (torch.ops.aten.div.Tensor_mode, torch.ops.aten.div.Scalar_mode)
        and node.kwargs.get("rounding_mode") == "floor"
    )


_VIEWS = {
    _tracing_ops._new_var,
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.t.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.expand.default,
    view_ops.subscript,
}


def _shape(node: Node) -> tuple[int, ...]:
    from ..device_function import DeviceFunction

    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return ()
    return tuple(
        _get_tile_shape(
            value, CompileEnvironment.current(), DeviceFunction.current().config
        )
    )


def _resolved_extent(size: int | torch.SymInt) -> int:
    """Resolve a proven static extent, never a symbolic size hint."""
    from ..device_function import DeviceFunction

    if isinstance(size, int):
        return size
    env = CompileEnvironment.current()
    expression = env.specialize_expr(cast("sympy.Expr", size.node.expr))
    replacements = {}
    for symbol in expression.free_symbols:
        block = env.get_block_id(symbol)
        if block is None:
            raise _UnsupportedChain("dynamic host extent")
        resolved = DeviceFunction.current().resolved_block_size(
            env.canonical_block_id(block)
        )
        if not isinstance(resolved, int):
            raise _UnsupportedChain("dynamic host extent")
        replacements[symbol] = sympy.Integer(resolved)
    expression = expression.xreplace(replacements)
    if not isinstance(expression, sympy.Integer):
        raise _UnsupportedChain("dynamic host extent")
    return int(expression)


def _host_shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(_resolved_extent(size) for size in tensor.shape)


def _shape_domain(
    node: Node, coordinates: tuple[str, ...], plan: ChainedMatmulPlan
) -> list[str]:
    """Logical tensor bounds, distinct from the padded MMA tile dimensions.

    Masking loads alone is insufficient: exp, addition, and other pointwise
    functions can turn a padded zero into a nonzero contraction operand.
    """
    env = CompileEnvironment.current()
    axes = {axis: (extent, block) for axis, extent, block in plan.axes}
    bounds: list[str] = []
    for size, padded, coordinate in zip(
        node.meta["val"].shape, _shape(node), coordinates, strict=True
    ):
        block_id = env.get_block_id(size)
        if block_id is not None:
            block_id = env.canonical_block_id(block_id)
            if plan.loop is not None and block_id == plan.loop.block_id:
                bounds.append(f"chain_loop_index + ({coordinate}) < chain_loop_end")
                continue
            if block_id in axes:
                extent, block = axes[block_id]
                if extent % block:
                    bounds.append(
                        f"chain_origin_{block_id} + ({coordinate}) < {extent}"
                    )
                continue
            size = env.block_sizes[block_id].size
        if isinstance(size, torch.SymInt):
            size = env.specialize_expr(cast("sympy.Expr", size._sympy_()))
        if not isinstance(size, (int, sympy.Integer)):
            raise _UnsupportedChain("unproven contraction operand domain")
        if int(size) < padded:
            bounds.append(f"({coordinate}) < {int(size)}")
    return bounds


def _masked_operand(value: str, dtype: str, bounds: list[str]) -> str:
    if not bounds:
        return f"{dtype}({value})"
    predicate = " & ".join(f"({bound})" for bound in bounds)
    return f"({dtype}({value}) if {predicate} else {dtype}(0))"


def _ancestors(node: Node) -> set[Node]:
    result: set[Node] = set()
    pending = [node]
    while pending:
        current = pending.pop()
        if current not in result:
            result.add(current)
            pending.extend(current.all_input_nodes)
    return result


def _scan_cache_candidates(
    scan: Node, scans: tuple[Node, ...], store: Node
) -> set[Node]:
    """Find vector leaves also consumed after the scan prelude.

    Stop traversal at scans: an input used only to produce a scan result has
    no independent later read to cache. Exact coordinate proofs happen during
    codegen; this superset also provides conservative shared-memory accounting.
    """
    later: set[Node] = set()
    pending = [cast("Node", store.args[2])]
    while pending:
        node = pending.pop()
        if node in later or node in scans:
            continue
        later.add(node)
        pending.extend(node.all_input_nodes)
    sources = {
        node.args[0]
        for node in later
        if node.target is memory_ops.load
        and _direct_operand(node)
        and isinstance(node.args[0], Node)
        and node.args[0].target is _tracing_ops._host_tensor
    }
    return {
        node
        for node in _ancestors(cast("Node", scan.args[1]))
        if node.target is memory_ops.load
        and _direct_operand(node)
        and node.args[0] in sources
        and len(_shape(node)) == 1
        and node.meta["val"].dtype in (torch.float16, torch.bfloat16, torch.float32)
    }


def _supported(node: Node) -> bool:
    from .chained_collectives import classify_collective

    if node.op == "output":
        return True
    if node.op != "call_function":
        return False
    if classify_collective(node) is not None:
        return True
    if node.target in {
        _tracing_ops._host_tensor,
        _tracing_ops._get_symnode,
        _tracing_ops._mask_to,
        tile_ops.tile_begin,
        tile_index,
        torch.ops.prims.iota.default,
        torch.ops.aten.scalar_tensor.default,
        creation_ops.full,
        memory_ops.load,
        torch.ops.aten.where.self,
        dot,
        scan_ops._associative_scan,
        *_VIEWS,
        *_SCALAR_BINARY,
    }:
        return True
    if node.target is memory_ops.store:
        return len(node.args) <= 3 or node.args[3] is None
    return _pointwise_inputs(node) is not None


def _pointwise_inputs(node: Node) -> tuple[Node, ...] | None:
    lowering = node.meta.get("lowering")
    if not isinstance(lowering, PointwiseLowering):
        return None
    inputs: list[Node] = []

    def visit(value: Node) -> Node:
        inputs.append(value)
        return value

    map_arg((node.args, {**node.kwargs, "_extra_deps": None}), visit)
    if len(inputs) != len(lowering.input_names):
        return None
    return tuple(inputs)


def _direct_operand(node: Node) -> bool:
    if node.target in _VIEWS:
        return _direct_operand(cast("Node", node.args[0]))
    return (
        node.target is memory_ops.load
        and (len(node.args) < 3 or node.args[2] is None)
        and (len(node.args) < 4 or node.args[3] is None)
        and isinstance(source := node.args[0], Node)
        and source.target is _tracing_ops._host_tensor
    )


def _ordinary_mma_supported(node: Node) -> bool:
    from ..host_function import HostFunction
    from .cute_mma import analyze_cute_mma_node

    return (
        analyze_cute_mma_node(node, device_ir=HostFunction.current().device_ir)
        is not None
    )


def _root_graph(graphs: Sequence[GraphInfo]) -> GraphInfo | None:
    from ..device_ir import HelperFunctionGraphInfo
    from ..device_ir import RootGraphInfo

    roots = [graph for graph in graphs if isinstance(graph, RootGraphInfo)]
    if len(roots) != 1 or any(
        not isinstance(graph, (RootGraphInfo, HelperFunctionGraphInfo))
        for graph in graphs
    ):
        return None
    return roots[0]


def _additive_scan(node: Node) -> bool:
    from ..host_function import HostFunction

    if len(node.args) != 5 or node.args[3:] != (False, False):
        return False
    source = node.args[1]
    if (
        not isinstance(source, Node)
        or source.meta["val"].ndim not in (1, 2)
        or type(node.args[2]) is not int
        or not -source.meta["val"].ndim <= node.args[2] < source.meta["val"].ndim
    ):
        return False
    if source.meta["val"].dtype != torch.float32:
        return False
    graph = HostFunction.current().device_ir.graphs[cast("int", node.args[0])].graph
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    calls = [n for n in graph.nodes if n.op == "call_function"]
    outputs = [n for n in graph.nodes if n.op == "output"]
    return (
        len(placeholders) == 2
        and len(calls) == 1
        and len(outputs) == 1
        and calls[0].target in (operator.add, torch.add, torch.ops.aten.add.Tensor)
        and calls[0].args == tuple(placeholders)
        and not calls[0].kwargs
        and outputs[0].args == (calls[0],)
    )


@dataclasses.dataclass(frozen=True)
class _ChainedGraph:
    root: GraphInfo
    nodes: tuple[Node, ...]
    dots: tuple[Node, ...]
    scans: tuple[Node, ...]
    store: Node
    exports: ScanExportStores | None
    region: ContractionRegion
    loop: ChainedLoopPlan | None = None


def _classify_chained_graph(graphs: Sequence[GraphInfo]) -> _ChainedGraph | None:
    """Shared structural admission; physical layout/config proofs come later.

    Recompute for the graph supplied by each caller. Discovery and lowering
    can observe different graph revisions, so these facts are not cached.
    """
    from .chained_loop import discover_chained_loop
    from .chained_loop import register_loop_storage

    loop = discover_chained_loop(graphs)
    root = loop.root if loop is not None else _root_graph(graphs)
    if root is None:
        return None
    region = loop.region if loop is not None else collect_contraction_region(root)
    if region is None:
        return None
    nodes = region.nodes
    dots = tuple(spec.node for spec in region.contractions)
    scans, stores = region.scans, region.stores
    from .chained_scan_export import classify_scan_exports

    exports = (
        classify_scan_exports(nodes) if loop is None and len(stores) != 1 else None
    )
    allowed = exports.permitted_nodes if exports is not None else frozenset()
    if (
        not dots
        or not stores
        or (loop is None and len(stores) != 1 and exports is None)
        or not all(
            _supported(node)
            or node in allowed
            or loop is not None
            and _supported_loop_node(node)
            for node in nodes
        )
    ):
        return None
    if not all(map(_additive_scan, scans)):
        return None
    store = exports.primary if exports is not None else stores[0]
    if not isinstance(store.args[0], Node) or (
        loop is None and not _has_fresh_output_allocation(store.args[0])
    ):
        return None
    if not isinstance(store.args[2], Node) or store.args[2].meta["val"].ndim != 2:
        return None
    ancestors = _ancestors(store.args[2])
    if loop is not None:
        if not register_loop_storage(loop):
            return None
        for value in (
            *region.live_outs,
            *(cast("Node", node.args[2]) for node in stores),
        ):
            ancestors.update(_ancestors(value))
        if any(
            not _supported(node)
            and not _supported_loop_node(node)
            and node.target
            not in (_tracing_ops._for_loop, _tracing_ops._phi, operator.getitem)
            for node in loop.root.graph.nodes
        ):
            return None
    for spec in region.contractions:
        left_dtype, right_dtype = spec.operand_dtypes
        if (
            left_dtype != right_dtype
            or left_dtype not in (torch.bfloat16, torch.float16)
            or spec.result_dtype != torch.float32
            or spec.accumulator_dtype not in (None, torch.float32)
            or spec.requested_out_dtype not in (None, torch.float32)
            or spec.node not in ancestors
        ):
            return None
    if (
        loop is None
        and len(dots) == 1
        and all(_direct_operand(cast("Node", arg)) for arg in dots[0].args[:2])
        and _ordinary_mma_supported(dots[0])
    ):
        return None
    return _ChainedGraph(root, nodes, dots, scans, store, exports, region, loop)


def _supported_loop_node(node: Node) -> bool:
    return node.op == "placeholder" or node.target in (
        memory_ops.load,
        memory_ops.store,
        tile_ops.tile_id,
        torch.ops.aten.sym_size.int,
    )


def detect_chained_matmul_search(graphs: Sequence[GraphInfo]) -> bool:
    """Config-independent admission for the computed-contraction search family."""
    return _classify_chained_graph(graphs) is not None


def plan_chained_matmul(graphs: Sequence[GraphInfo]) -> ChainedMatmulPlan | None:
    from ..device_function import DeviceFunction
    from ..host_function import HostFunction
    from .chained_scan_export import valid_scan_exports

    env = CompileEnvironment.current()
    df = DeviceFunction.current()
    tcgen = df.config.config.get("cute_chained_mma_schedule") == "tcgen05_tmem"
    if env.backend.name != "cute" or df.config.pid_type != "flat":
        return None
    support = get_cute_mma_support()
    if not (support.tcgen05_f16bf16 if tcgen else support.warp_f16bf16):
        return None
    graph = _classify_chained_graph(graphs)
    if graph is None:
        return None
    from .chained_preparation_cohorts import supports_twenty_warp_preparation

    compact_twenty = graph.loop is not None and supports_twenty_warp_preparation(
        df.config.config
    )
    if (
        tcgen
        and df.config.num_warps
        not in ((4, 8, 16, 32) if graph.loop is not None else (4,))
        and not compact_twenty
    ):
        return None
    if (
        graph.loop is not None
        and df.config.num_warps not in (1, 2, 4, 8, 16, 32)
        and not compact_twenty
    ):
        return None
    if graph.loop is not None:
        from .chained_loop import loop_storage_is_proven

        if (
            "cute_chained_mma_schedule" not in df.config.config
            or not loop_storage_is_proven(graph.loop)
        ):
            return None
    nodes, dots, scans, store = graph.nodes, graph.dots, graph.scans, graph.store
    # Structural admission has established that every dot operand is a Node.
    first_input = cast("Node", dots[0].args[0]).meta["val"]
    dtype = first_input.dtype
    shapes: list[tuple[int, int, int]] = []
    for node in dots:
        lhs, rhs = (cast("Node", arg) for arg in node.args[:2])
        left, right = _shape(lhs), _shape(rhs)
        if (
            len(left) != 2
            or len(right) != 2
            or left[1] != right[0]
            or node.meta["val"].dtype != torch.float32
        ):
            return None
        m, k, n = *left, right[1]
        if isinstance(node.args[2], Node) and _shape(node.args[2]) != (m, n):
            return None
        if m % 16 or n % 8 or k % 16 or min(m, n, k) <= 0:
            return None
        shapes.append((m, n, k))
    # Bound the resident workspace; larger contractions use other schedules.
    cache_inputs = (
        df.config.config.get("cute_chained_mma_schedule")
        == "cp_async_register_reuse_scan"
    )
    general_collectives = (
        bool(graph.region.reductions)
        or graph.loop is not None
        and bool(scans)
        or any(node.meta["val"].ndim != 1 for node in scans)
        or any(
            node.target is dot
            for scan in scans
            for node in _ancestors(cast("Node", scan.args[1]))
        )
    )
    collective_allocations = (
        [4 * math.prod(_shape(node)) for node in (*scans, *graph.region.reductions)]
        if general_collectives
        else [4 * (_shape(scan)[0] + min(df.config.num_warps, 8)) for scan in scans]
    )
    shared_allocations = [
        2 * max(m * k + 8 * max(m, k) for m, _, k in shapes),
        2 * max(n * k + 8 * max(n, k) for _, n, k in shapes),
        *(4 * m * (n + 4) for m, n, _ in shapes),
        *collective_allocations,
        *(
            _shape(scan)[0] * leaf.meta["val"].element_size()
            for scan in scans
            for leaf in (
                _scan_cache_candidates(scan, scans, store) if cache_inputs else ()
            )
        ),
    ]
    # Every allocation has 128-byte alignment, including the short scan warp
    # totals. Round each allocation rather than only the aggregate footprint.
    smem_bytes = sum((size + 127) // 128 * 128 for size in shared_allocations)
    if graph.loop is not None:
        from .chained_loop import carry_shared_bytes

        smem_bytes += carry_shared_bytes(graph.loop)
    if not tcgen and smem_bytes > CuteTcgen05Config.per_cta_smem_capacity_bytes(
        first_input.device
    ):
        return None
    output = store.args[0]
    if not isinstance(output, Node) or output.target is not _tracing_ops._host_tensor:
        return None
    output_value = output.meta["val"]
    if output_value.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return None
    for node in nodes:
        if node.target is memory_ops.load:
            source = cast("Node", node.args[0])
            if source is output or source.meta.get("val") is output_value:
                return None
    ir = HostFunction.current().device_ir
    if len(ir.grid_block_ids) != 1 or len(ir.task_families) != 1:
        return None
    family = ir.task_families[0]
    axis_ids = tuple(ir.grid_block_ids[0])
    if family.logical_axis_order != axis_ids:
        return None
    axes: list[tuple[int, int, int]] = []
    for axis_id in axis_ids:
        axis = family.axis(axis_id)
        block_size = df.resolved_block_size(axis_id)
        if axis is None or not axis.canonical_origin or not isinstance(block_size, int):
            return None
        if not isinstance(axis.extent, sympy.Expr):
            return None
        extent = env.specialize_expr(axis.extent)
        if not isinstance(extent, sympy.Integer) or int(extent) <= 0:
            return None
        axes.append((axis_id, int(extent), block_size))
    if math.prod((size + block - 1) // block for _, size, block in axes) > 2**31 - 1:
        return None
    plan = ChainedMatmulPlan(
        graph.root.graph_id,
        dots,
        store,
        tuple(axes),
        tuple(shapes),
        dtype,
        32
        * (
            df.config.num_warps
            if graph.loop is not None
            else min(df.config.num_warps, 8)
        ),
        scans,
        strategy="tcgen05_tmem" if tcgen else "warp",
        scan_exports=graph.exports.exports if graph.exports is not None else (),
        direct_output=bool(df.config.config.get("cute_chained_direct_output", False)),
        region=graph.region,
        loop=graph.loop,
    )
    if not valid_scan_exports(plan):
        return None
    if df.config.config.get("cute_chained_initialized_accumulator"):
        from .chained_initialized_accumulator import classify_initialized_accumulator

        initialized = classify_initialized_accumulator(nodes)
        if not tcgen or initialized is None or plan.scan_exports:
            return None
        plan = dataclasses.replace(plan, initialized_accumulator=initialized)
    if df.config.config.get("cute_chained_late_rhs_reuse"):
        from .chained_late_rhs import resolve_late_rhs

        arena = resolve_late_rhs(plan)
        if not tcgen or arena is None:
            return None
        plan = dataclasses.replace(plan, late_rhs_reuse=arena)
    k_mode = df.config.config.get("cute_chained_k_schedule", "full")
    if k_mode != "full":
        from .chained_k_schedule import resolve_k_schedule

        schedule = resolve_k_schedule(plan, cast("str", k_mode))
        if schedule is None:
            return None
        plan = dataclasses.replace(plan, k_schedule=schedule)
    workspace = smem_bytes
    if tcgen and graph.loop is not None:
        from .chained_loop import carry_shared_bytes
        from .chained_tcgen_stage import shared_bytes
        from .chained_tcgen_stage import stage_geometry

        geometries = tuple(stage_geometry(shape) for shape in shapes)
        if any(geometry is None for geometry in geometries):
            return None
        from .chained_tcgen_stage import StageGeometry

        if df.config.config.get("cute_chained_group_contractions"):
            from .chained_contraction_groups import contraction_groups

            groups = contraction_groups(
                plan, cast("tuple[StageGeometry, ...]", geometries)
            )
            if all(len(group.stages) == 1 for group in groups):
                return None
            plan = dataclasses.replace(plan, contraction_groups=groups)
        warp_rows = df.config.config.get("cute_chained_warp_mma_rows", 0)
        if warp_rows:
            from .chained_mma_selection import select_warp_mma_stages

            selected = select_warp_mma_stages(
                plan,
                cast("tuple[StageGeometry, ...]", geometries),
                cast("int", warp_rows),
            )
            if not selected or not support.warp_f16bf16:
                return None
            plan = dataclasses.replace(plan, warp_mma_stages=selected)
        contraction_workspace = plan_contraction_workspace(plan)
        carry_bytes = carry_shared_bytes(graph.loop)
        workspace = (
            shared_bytes(
                cast("tuple[StageGeometry, ...]", geometries),
                plan.contraction_groups,
                contraction_workspace,
            )
            + carry_bytes
            + sum((size + 127) // 128 * 128 for size in collective_allocations)
        )
        if (
            plan.direct_output
            or plan.initialized_accumulator is not None
            or plan.late_rhs_reuse is not None
            or plan.k_schedule is not None
        ):
            return None
    cache_budget = df.config.config.get("cute_chained_pointwise_cache_bytes", 0)
    if cache_budget:
        from .chained_pointwise_residency import plan_pointwise_cache

        cache_entries = cast(
            "int", df.config.config.get("cute_chained_pointwise_cache_entries", 1)
        )
        cache_nested = cast(
            "bool", df.config.config.get("cute_chained_pointwise_cache_nested", False)
        )
        cache = plan_pointwise_cache(
            plan,
            cast("int", cache_budget),
            max_entries=cache_entries,
            nested=cache_nested,
        )
        if (
            not cache.entries
            or (cache_entries > 1 and len(cache.entries) < 2)
            or (
                cache_nested
                and not any(
                    dependency in {item.node for item in cache.entries}
                    for entry in cache.entries
                    for dependency in entry.dependencies
                )
            )
            or (
                plan.initialized_accumulator is not None
                and any(plan.dots[0] in entry.dependencies for entry in cache.entries)
            )
        ):
            # Initialized-accumulator fusion omits the first shared C boundary.
            # A cache may not extend a dependency whose storage was elided.
            return None
        plan = dataclasses.replace(plan, pointwise_cache=cache)
        workspace += cache.shared_bytes
    if (
        tcgen
        and graph.loop is not None
        and workspace
        > CuteTcgen05Config.per_cta_smem_capacity_bytes(first_input.device)
    ):
        from .chained_loop import carry_shared_bytes
        from .chained_loop_workspace import plan_loop_workspace

        # Preserve separate carry/C storage whenever the complete allocation,
        # including the actual typed cache, fits. Packing is a capacity rescue,
        # not a performance preference. The final capacity check still applies.
        loop_workspace = plan_loop_workspace(
            plan,
            tuple(
                cast("tuple[int, int]", _shape(carry.input))
                for carry in graph.loop.region.carries
            ),
        )
        separate_bytes = plan_contraction_workspace(
            plan
        ).allocated_bytes + carry_shared_bytes(graph.loop)
        if (
            loop_workspace is not None
            and loop_workspace.allocated_bytes < separate_bytes
        ):
            # End-of-body writeback snapshots every next carry before touching
            # any current slot, including mutually dependent carries.
            plan = dataclasses.replace(plan, loop_workspace=loop_workspace)
            workspace -= separate_bytes - loop_workspace.allocated_bytes
    if df.config.config.get("cute_chained_preparation_pipeline", False):
        from .chained_preparation_cut import plan_preparation_cut
        from .chained_preparation_pipeline import plan_preparation_pipeline

        cut = plan_preparation_cut(graphs)
        if cut is None:
            return None
        pipeline = plan_preparation_pipeline(
            plan,
            cut,
            {
                node: _shape(node)
                for node in cut.region.nodes
                if isinstance(node.meta.get("val"), torch.Tensor)
            },
            CuteTcgen05Config.per_cta_smem_capacity_bytes(first_input.device),
            consumer_warps=cast(
                "int", df.config.config.get("cute_chained_pipeline_consumer_warps", 4)
            ),
            cohort_count=cast(
                "int", df.config.config.get("cute_chained_preparation_cohorts", 1)
            ),
            has_tma=df.config.config.get("cute_chained_leaf_pipeline")
            == "rectangular_tma",
        )
        if pipeline is None:
            return None
        plan = dataclasses.replace(plan, preparation_pipeline=pipeline)
        workspace = pipeline.shared_bytes
    if tcgen and graph.loop is None:
        from .chained_tcgen05 import supported_plan

        if not supported_plan(plan):
            return None
    elif workspace > CuteTcgen05Config.per_cta_smem_capacity_bytes(
        first_input.device
    ) and not (
        plan.preparation_pipeline is not None
        and plan.preparation_pipeline.cohorts is not None
    ):
        return None
    return plan


def _names(node: ast.AST) -> frozenset[str]:
    return frozenset(item.id for item in ast.walk(node) if isinstance(item, ast.Name))


@dataclasses.dataclass
class _Statement:
    code: str
    target: str | None
    inputs: frozenset[str]


@dataclasses.dataclass(frozen=True)
class _ReadAccess:
    """A typed, completely masked read and its explicit scalar binding."""

    expression: ast.expr
    dtype: torch.dtype
    statement: _Statement

    @property
    def inputs(self) -> frozenset[str]:
        return self.statement.inputs


def _materialized_value(
    name: str,
    shape: tuple[int, ...],
    coordinates: tuple[str, ...],
    dtype: str,
    *,
    storage_value: str | None = None,
) -> str:
    """Read an exact typed boundary, with scalar storage in a one-element view.

    Boolean boundaries use one byte per element, not CuTe's packed i1 pointer
    representation. Restore the logical predicate type before evaluating users.
    """
    index = ", ".join(coordinates) if shape else "0"
    value = f"{name}[{index}]" if storage_value is None else storage_value
    if dtype == "cutlass.Boolean":
        value = f"cutlass.Boolean({value})"
    if not shape:
        return value
    bounds = " and ".join(
        f"0 <= ({coordinate}) < {size}"
        for coordinate, size in zip(coordinates, shape, strict=True)
    )
    return f"({value} if {bounds} else {dtype}(0))"


class _Expression:
    """Evaluate an admitted FX expression at arbitrary logical coordinates."""

    def __init__(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
    ) -> None:
        self.cg = cg
        self.plan = plan
        self.prepared_widenings = plan.prepared_widenings
        self.boundaries = boundaries
        self.statements: list[_Statement] = []
        self.memo: dict[tuple[Node, tuple[str, ...]], str] = {}
        self.definitions: dict[str, str] = {}
        self.definition_inputs: dict[str, frozenset[str]] = {}
        self.reads: dict[str, _ReadAccess] = {}
        self.bind_scan_reads = False
        self.accesses: list[tuple[list[str], tuple[int, ...]]] = []
        self.global_accesses: list[tuple[Node, list[str]]] = []
        self.loaded_inputs: list[tuple[Node, tuple[str, ...], list[str], str]] = []
        self.fragments: dict[Node, tuple[tuple[str, ...], str]] = {}
        self.fragment_images: dict[Node, dict[tuple[str, ...], str]] = {}
        self.staged_inputs: list[_StagedInput] = []
        self.scan_inputs: list[_ScanInput] = []
        self.coordinate_names: set[str] = set()
        self.origins = {
            axis_id: f"chain_origin_{axis_id}" for axis_id, _, _ in plan.axes
        }
        if plan.loop is not None:
            self.boundaries = {**plan.loop.boundaries(), **boundaries}
            self.fragments.update(
                (node, ((), name)) for node, name in plan.loop.captures().items()
            )
            self.origins[plan.loop.block_id] = "chain_loop_index"

    def block_size(self, block_id: int) -> int:
        size = self.cg.device_function.resolved_block_size(block_id)
        if size is None:
            raise _UnsupportedChain("unresolved contraction block size")
        return _resolved_extent(size)

    @property
    def lines(self) -> list[str]:
        return [statement.code for statement in self.statements]

    def record_statement(
        self, statement: ast.AST, code: str | None = None
    ) -> _Statement:
        target = None
        inputs = frozenset()
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            target = statement.targets[0].id
            inputs = _names(statement.value)
            self.definitions[target] = ast.unparse(statement.value)
            self.definition_inputs[target] = inputs
        result = _Statement(
            ast.unparse(statement) if code is None else code, target, inputs
        )
        self.statements.append(result)
        return result

    def bind(self, expression: str) -> str:
        name = self.cg.device_function.new_var("chain_value")
        self.record_statement(
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=cast("ast.expr", expr_from_string(expression)),
            ),
            f"{name} = {expression}",
        )
        # Preserve spelling used by the existing symbolic index proofs.
        self.definitions[name] = expression
        return name

    def bind_read(self, node: Node, expression: str) -> str:
        value = self.bind(expression)
        self.reads[value] = _ReadAccess(
            ast.parse(expression, mode="eval").body,
            node.meta["val"].dtype,
            self.statements[-1],
        )
        return value

    def replace_read(self, read: _ReadAccess, value: str) -> None:
        """Redirect exactly this read binding; no emitted AST search is needed."""
        statement = read.statement
        assert statement.target is not None
        statement.code = f"{statement.target} = {value}"
        statement.inputs = _names(expr_from_string(value))
        self.definitions[statement.target] = value
        self.definition_inputs[statement.target] = statement.inputs

    def tensor_name(self, node: Node) -> str:
        df = self.cg.device_function
        name = df.tensor_arg(
            node.meta["val"], prefer_name=cast("str", node.args[0])
        ).name
        if name not in self.plan.tensor_aliases:
            self.plan.tensor_aliases[name] = df.new_var("input_tensor")
        return self.plan.tensor_aliases[name]

    def dependencies(self, expression: str) -> set[str]:
        return self.expand_dependencies(_names(expr_from_string(expression)))

    def expand_dependencies(self, inputs: frozenset[str]) -> set[str]:
        names = set(inputs)
        pending = list(names)
        while pending:
            name = pending.pop()
            for dependency in self.definition_inputs.get(name, ()):
                if dependency not in names:
                    names.add(dependency)
                    pending.append(dependency)
        return names

    def block_id(self, node: Node) -> int:
        value = node.meta.get("val")
        axis = CompileEnvironment.current().resolve_block_id(value)
        if axis is None:
            raise _UnsupportedChain(f"non-tile index {node.name}")
        return CompileEnvironment.current().canonical_block_id(axis)

    def scalar_range(self, arg: object) -> tuple[int, int] | None:
        """Conservative signed-index bounds; reject any possible intermediate wrap."""
        bounds: tuple[int, int]
        if isinstance(arg, int):
            bounds = (arg, arg)
        elif isinstance(arg, Node) and arg.target is tile_ops.tile_begin:
            axis = self.block_id(cast("Node", arg.args[0]))
            if axis not in {item[0] for item in self.plan.axes}:
                return None
            _, extent, block = next(a for a in self.plan.axes if a[0] == axis)
            bounds = (0, max(0, (extent - 1) // block * block))
        elif isinstance(arg, Node) and arg.target is _tracing_ops._get_symnode:
            axis = CompileEnvironment.current().resolve_block_id(arg.meta.get("val"))
            if axis is None:
                return None
            block = self.block_size(axis)
            bounds = (block, block)
        elif isinstance(arg, Node) and arg.target in _SCALAR_BINARY:
            left, right = (self.scalar_range(value) for value in arg.args[:2])
            if left is None or right is None:
                return None
            if arg.target is operator.add:
                bounds = (left[0] + right[0], left[1] + right[1])
            elif arg.target is operator.sub:
                bounds = (left[0] - right[1], left[1] - right[0])
            elif arg.target is operator.mul:
                products = [a * b for a in left for b in right]
                bounds = (min(products), max(products))
            elif right[0] == right[1] and right[0] != 0:
                if arg.target is operator.floordiv:
                    quotients = [value // right[0] for value in left]
                    bounds = (min(quotients), max(quotients))
                elif arg.target is operator.mod:
                    bounds = (0, right[0] - 1) if right[0] > 0 else (right[0] + 1, 0)
                else:
                    return None
            else:
                return None
        else:
            return None
        limits = torch.iinfo(CompileEnvironment.current().index_dtype)
        return bounds if limits.min <= bounds[0] <= bounds[1] <= limits.max else None

    def scalar(self, arg: object) -> str:
        if isinstance(arg, (int, float, bool)):
            return repr(arg)
        if not isinstance(arg, Node):
            raise _UnsupportedChain(f"scalar {arg}")
        if arg.target in (tile_ops.tile_begin, tile_ops.tile_id):
            axis = self.block_id(cast("Node", arg.args[0]))
            origin = self.origins[axis]
            return (
                origin
                if arg.target is tile_ops.tile_begin
                else f"({origin} // {self.block_size(axis)})"
            )
        if arg.target in _SCALAR_BINARY:
            left, right = (self.scalar(v) for v in arg.args[:2])
            if arg.target in (operator.mod, operator.floordiv):
                left_range, right_range = (
                    self.scalar_range(value) for value in arg.args[:2]
                )
                same_sign = (
                    left_range is not None
                    and right_range is not None
                    and (
                        (left_range[0] >= 0 and right_range[0] > 0)
                        or (left_range[1] <= 0 and right_range[1] < 0)
                    )
                )
                positive = (
                    left_range is not None
                    and right_range is not None
                    and left_range[0] >= 0
                    and right_range[0] > 0
                )
                if (
                    arg.target is operator.floordiv
                    and isinstance(arg.meta.get("val"), (int, torch.SymInt))
                    and not positive
                ):
                    return self.signed_floor(left, right)
                if arg.target is operator.mod and not same_sign:
                    remainder = self.bind(f"({left} % {right})")
                    return self.bind(_signed_remainder_adjustment(remainder, right))
            return f"({left} {_SCALAR_BINARY[arg.target]} {right})"
        if arg.target in (_tracing_ops._get_symnode, torch.ops.aten.sym_size.int):
            value = arg.meta.get("val")
            axis = CompileEnvironment.current().resolve_block_id(value)
            if axis is not None:
                return str(self.block_size(axis))
            if isinstance(value, (int, float, torch.SymInt, torch.SymFloat)):
                return self.cg.device_function.literal_expr(value)
        if isinstance(arg.meta.get("val"), torch.Tensor) and not _shape(arg):
            return self.value(arg, ())
        raise _UnsupportedChain(f"scalar node {arg.name}: {arg.target}")

    def signed_floor(self, left: str, right: str) -> str:
        # CuTe's floordivsi lowering is unreliable for opposite signs. Divide
        # an exactly divisible numerator, then correct C's truncated quotient.
        # C remainder has the dividend's sign, so left - remainder cannot
        # overflow. Native zero-divisor/minimum/-1 behavior is unchanged.
        remainder = self.bind(f"({left} % {right})")
        quotient = self.bind(f"(({left} - {remainder}) // {right})")
        return self.bind(_signed_floor_adjustment(quotient, remainder, right))

    def _index(self, index: object, coordinate: str | None) -> str:
        if index == slice(None):
            assert coordinate is not None
            return coordinate
        if isinstance(index, Node):
            if index.target in (_tracing_ops._get_symnode, torch.ops.aten.sym_size.int):
                assert coordinate is not None
                return f"({self.origins[self.block_id(index)]} + {coordinate})"
            if isinstance(index.meta.get("val"), torch.Tensor) and _shape(index):
                assert coordinate is not None
                return self.value(index, (coordinate,))
        return self.scalar(index)

    def indices(self, node: Node, coordinates: tuple[str, ...]) -> list[str]:
        result: list[str] = []
        dim = 0
        selectors = cast("Sequence[object]", node.args[1])
        advanced = [
            index
            for index in selectors
            if isinstance(index, Node)
            and isinstance(index.meta.get("val"), torch.Tensor)
            and len(_shape(index)) > 1
        ]
        if advanced:
            # Explicit broadcast indices share one output domain. They are not
            # separate one-dimensional tile selectors (e.g. row[:,None], col[None,:]).
            for index in selectors:
                if isinstance(index, Node) and isinstance(
                    index.meta.get("val"), torch.Tensor
                ):
                    shape = _shape(index)
                    if shape:
                        offset = len(coordinates) - len(shape)
                        if offset < 0:
                            raise _UnsupportedChain("advanced index rank")
                        coords = tuple(
                            "0" if size == 1 else coordinates[i + offset]
                            for i, size in enumerate(shape)
                        )
                        result.append(self.value(index, coords))
                        continue
                if (
                    index == slice(None)
                    or isinstance(index, Node)
                    and index.target
                    in (_tracing_ops._get_symnode, torch.ops.aten.sym_size.int)
                ):
                    raise _UnsupportedChain(
                        "mixed basic and broadcast advanced indexing"
                    )
                result.append(self.scalar(index))
            return result
        for index in selectors:
            vector = index == slice(None) or (
                isinstance(index, Node)
                and (
                    index.target
                    in (_tracing_ops._get_symnode, torch.ops.aten.sym_size.int)
                    or (
                        isinstance(index.meta.get("val"), torch.Tensor)
                        and bool(_shape(index))
                    )
                )
            )
            result.append(self._index(index, coordinates[dim] if vector else None))
            dim += int(vector)
        if dim != len(coordinates):
            raise _UnsupportedChain(f"advanced index shape at {node.name}")
        return result

    def _load(self, node: Node, coordinates: tuple[str, ...]) -> str:
        from ...language.memory_ops import _cute_scalar_load_expr

        source = cast("Node", node.args[0])
        selectors = cast("Sequence[object]", node.args[1])
        if source.target is not _tracing_ops._host_tensor and all(
            index is None or index == slice(None) for index in selectors
        ):
            value = self.value(
                source,
                tuple(
                    c
                    for c, s in zip(coordinates, selectors, strict=True)
                    if s is not None
                ),
            )
            return self._mask_internal_load(node, coordinates, value)
        indices = self.indices(node, coordinates)
        if source.target is not _tracing_ops._host_tensor:
            return self._mask_internal_load(
                node, coordinates, self.value(source, tuple(indices))
            )
        tensor = source.meta["val"]
        sizes = _host_shape(tensor)
        self.accesses.append((indices, tuple(tensor.stride())))
        self.global_accesses.append((source, indices))
        name = self.tensor_name(source)
        bounds = [
            f"0 <= ({index}) < {size}"
            for index, size in zip(indices, sizes, strict=True)
        ]
        mask = self._load_mask(node, coordinates)
        if mask is not None:
            bounds.append(mask)
        value = _cute_scalar_load_expr(name, indices, tensor.dtype)
        dtype = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
        if tensor.dtype == torch.bool:
            # The launcher exposes Boolean tensors as byte-addressed Uint8.
            # Restore the predicate before joining with the masked Boolean zero.
            value = f"cutlass.Boolean({value})"
        # Helion masked loads produce zero. The fourth positional argument is
        # an eviction-policy hint, not an alternate masked value.
        fallback = f"({value} if {' and '.join(bounds)} else {dtype}(0))"
        cached = (
            _scan_input_value(
                self,
                source,
                indices,
                _staged_input_value(self, source, indices, fallback),
            )
            if all(value is None for value in node.args[2:4])
            else fallback
        )
        result = self.bind_read(node, cached)
        self.loaded_inputs.append((node, coordinates, indices, result))
        return result

    def _load_mask(self, node: Node, coordinates: tuple[str, ...]) -> str | None:
        if len(node.args) <= 2 or node.args[2] is None:
            return None
        mask = node.args[2]
        if not isinstance(mask, Node):
            return self.scalar(mask)
        mask_shape = _shape(mask)
        offset = len(coordinates) - len(mask_shape)
        mask_coords = tuple(
            "0" if size == 1 else coordinates[i + offset]
            for i, size in enumerate(mask_shape)
        )
        return self.value(mask, mask_coords)

    def _mask_internal_load(
        self, node: Node, coordinates: tuple[str, ...], value: str
    ) -> str:
        mask = self._load_mask(node, coordinates)
        if mask is None:
            return value
        dtype = CompileEnvironment.current().backend.dtype_str(node.meta["val"].dtype)
        return self.bind(f"({value} if {mask} else {dtype}(0))")

    def _view(self, node: Node, coordinates: tuple[str, ...]) -> str:
        source = cast("Node", node.args[0])
        old_shape, new_shape = _shape(source), _shape(node)
        if node.target is _tracing_ops._new_var:
            old_coords = coordinates
        elif node.target is torch.ops.aten.permute.default:
            permutation = [
                axis % len(old_shape) for axis in cast("Sequence[int]", node.args[1])
            ]
            old_coords = tuple(
                coordinates[permutation.index(i)] for i in range(len(old_shape))
            )
        elif node.target in (torch.ops.aten.t.default, torch.ops.aten.transpose.int):
            perm = list(range(len(old_shape)))
            a, b = (
                (0, 1)
                if node.target is torch.ops.aten.t.default
                else cast("tuple[int, int]", node.args[1:3])
            )
            perm[a], perm[b] = perm[b], perm[a]
            old_coords = tuple(
                coordinates[perm.index(i)] for i in range(len(old_shape))
            )
        elif node.target is torch.ops.aten.unsqueeze.default:
            axis = cast("int", node.args[1]) % len(new_shape)
            old_coords = coordinates[:axis] + coordinates[axis + 1 :]
        elif node.target is torch.ops.aten.squeeze.dim:
            axis = cast("int", node.args[1]) % len(old_shape)
            old_coords = (
                (*coordinates[:axis], "0", *coordinates[axis:])
                if old_shape[axis] == 1
                else coordinates
            )
        elif node.target is torch.ops.aten.expand.default:
            offset = len(new_shape) - len(old_shape)
            old_coords = tuple(
                "0" if s == 1 else coordinates[i + offset]
                for i, s in enumerate(old_shape)
            )
        elif node.target is view_ops.subscript:
            selectors = cast("Sequence[object]", node.args[1])
            result: list[str] = []
            dim = 0
            for index in selectors:
                if isinstance(index, int):
                    result.append(str(index))
                    continue
                coordinate = coordinates[dim]
                dim += 1
                if index is None:
                    continue
                if isinstance(index, slice) and index != slice(None):
                    result.append(f"({index.start} + {coordinate})")
                else:
                    result.append(self._index(index, coordinate))
            if dim != len(coordinates):
                raise _UnsupportedChain("subscript coordinate rank")
            old_coords = tuple(result)
        else:
            flat = (
                " + ".join(
                    f"({c}) * {math.prod(new_shape[i + 1 :])}"
                    for i, c in enumerate(coordinates)
                )
                or "0"
            )
            old_coords = tuple(
                f"(({flat}) // {math.prod(old_shape[i + 1 :])}) % {s}"
                for i, s in enumerate(old_shape)
            )
        return self.value(source, old_coords)

    def bind_fragment_image(
        self, node: Node, coordinates: tuple[str, ...], value: str
    ) -> None:
        """Bind one exact image; never guess coordinates or replace an image."""
        if node in self.fragments or coordinates in self.fragment_images.get(node, {}):
            raise _UnsupportedChain("duplicate or mixed register fragment image")
        self.fragment_images.setdefault(node, {})[coordinates] = value

    def value(self, node: Node, coordinates: tuple[str, ...]) -> str:

        from ..aten_lowering import LoweringContext

        key = (node, coordinates)
        if node in self.fragment_images:
            if node in self.fragments or coordinates not in self.fragment_images[node]:
                raise _UnsupportedChain("missing or mixed register fragment image")
        if key in self.memo:
            return self.memo[key]
        shape = _shape(node)
        if len(coordinates) != len(shape):
            raise _UnsupportedChain(f"coordinate rank at {node.name}")
        if node in self.fragment_images:
            value = self.fragment_images[node][coordinates]
        elif node in self.fragments:
            expected, value = self.fragments[node]
            if coordinates != expected:
                raise _UnsupportedChain("cross-coordinate register fragment use")
        elif node in self.boundaries:
            self.accesses.append(
                (
                    list(coordinates),
                    tuple(math.prod(shape[i + 1 :]) for i in range(len(shape))),
                )
            )
            dtype = CompileEnvironment.current().backend.dtype_str(
                node.meta["val"].dtype
            )
            value = _materialized_value(
                self.boundaries[node], shape, coordinates, dtype
            )
            if self.bind_scan_reads and node in self.plan.scans:
                value = self.bind_read(node, value)
        elif self.prepared_widenings is not None and any(
            binding.transfer.widening is node
            for binding in self.prepared_widenings.bindings
        ):
            value = self.prepared_widenings.expression(self, node, coordinates)
        elif node.target is memory_ops.load:
            value = self._load(node, coordinates)
        elif node.target is _tracing_ops._mask_to:
            source = cast("Node", node.args[0])
            original = self.value(source, coordinates)
            bounds = _operand_domain(self.cg, source, coordinates, self.plan)
            dtype = CompileEnvironment.current().backend.dtype_str(
                node.meta["val"].dtype
            )
            value = (
                original
                if not bounds
                else f"({original} if {' and '.join(bounds)} else {dtype}({self.scalar(node.args[1])}))"
            )
        elif node.target in _VIEWS:
            value = self._view(node, coordinates)
        elif node.target is torch.ops.prims.iota.default:
            value = f"({node.kwargs.get('start', 0)} + ({coordinates[0]}) * {node.kwargs.get('step', 1)})"
        elif node.target is tile_index:
            value = f"({self.origins[self.block_id(cast('Node', node.args[0]))]} + {coordinates[0]})"
        elif node.target is torch.ops.aten.scalar_tensor.default:
            dtype = CompileEnvironment.current().backend.dtype_str(
                node.meta["val"].dtype
            )
            value = f"{dtype}({self.scalar(node.args[0])})"
        elif node.target is creation_ops.full:
            dtype = CompileEnvironment.current().backend.dtype_str(
                node.meta["val"].dtype
            )
            value = f"{dtype}({self.scalar(node.args[1])})"
        elif node.target is torch.ops.aten.where.self:
            choices: list[str] = []
            for arg in cast("tuple[Node, ...]", node.args):
                arg_shape = _shape(arg)
                offset = len(shape) - len(arg_shape)
                coords = tuple(
                    "0" if size == 1 else coordinates[i + offset]
                    for i, size in enumerate(arg_shape)
                )
                choices.append(self.value(arg, coords))
            value = self.bind(f"({choices[1]} if {choices[0]} else {choices[2]})")
        elif (
            node.target in _REMAINDER_TARGETS or _is_floor_divide(node)
        ) and node.meta["val"].dtype in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            dtype = CompileEnvironment.current().backend.dtype_str(
                node.meta["val"].dtype
            )
            operands: list[str] = []
            for arg in node.args[:2]:
                if isinstance(arg, Node) and isinstance(
                    arg.meta.get("val"), torch.Tensor
                ):
                    arg_shape = _shape(arg)
                    offset = len(shape) - len(arg_shape)
                    coords = tuple(
                        "0" if size == 1 else coordinates[i + offset]
                        for i, size in enumerate(arg_shape)
                    )
                    operand = self.value(arg, coords)
                else:
                    operand = self.scalar(arg)
                operands.append(self.bind(f"{dtype}({operand})"))
            left, right = operands
            shift = _power_of_two_divisor_shift(node.args[1], node.meta["val"].dtype)
            if shift is not None:
                # Arithmetic right shift is floor division, including negative
                # values and the signed minimum. The low-bit mask is the exact
                # nonnegative remainder; neither requires an intermediate
                # subtraction or CuTe's opposite-sign floordivsi correction.
                reduced = (
                    f"({left} >> {shift})"
                    if _is_floor_divide(node)
                    else f"({left} & {(1 << shift) - 1})"
                )
                # CuTe promotes Int8/Int16 bit operations to Int32. Restore the
                # source dtype before any following pointwise operation.
                value = self.bind(f"{dtype}({reduced})")
            elif _is_floor_divide(node):
                value = self.signed_floor(left, right)
            else:
                remainder = self.bind(f"({left} % {right})")
                value = self.bind(_signed_remainder_adjustment(remainder, right))
        elif (inputs := _pointwise_inputs(node)) is not None:
            values: list[ast.AST] = []
            for input_node in inputs:
                if not isinstance(input_node.meta.get("val"), torch.Tensor):
                    values.append(expr_from_string(self.scalar(input_node)))
                    continue
                input_shape = _shape(input_node)
                offset = len(shape) - len(input_shape)
                coords = tuple(
                    "0" if size == 1 else coordinates[i + offset]
                    for i, size in enumerate(input_shape)
                )
                values.append(expr_from_string(self.value(input_node, coords)))
            ctx = LoweringContext.__new__(LoweringContext)
            ctx.cg = self.cg
            ctx.env = {}
            statements: list[ast.AST] = []
            lowering = cast("PointwiseLowering", node.meta["lowering"])
            with self.cg.set_statements(statements), V.set_current_node(node):
                result = lowering.codegen_from_input_asts(ctx, node, values)
            assert isinstance(result, ast.AST)
            for statement in statements:
                self.record_statement(statement)
            value = self.bind(ast.unparse(result))
        else:
            raise _UnsupportedChain(f"expression {node.name}: {node.target}")
        self.memo[key] = value
        return value


class _DomainExpression(_Expression):
    """Trace logical padding provenance without evaluating memory or matmuls.

    Static iotas retain their original length in args[0], while their containing
    contraction operands may already have padded fake shapes. Follow that
    provenance through pointwise/views and surviving contraction output axes.
    """

    def __init__(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        node: Node,
        coordinates: tuple[str, ...],
    ) -> None:
        super().__init__(cg, plan, {})
        # Domain analysis follows the original conversion graph, not an emitted
        # recurrence-only storage binding or its instruction-selection helper.
        self.prepared_widenings = None
        self.bounds: list[str] = []
        self.unavailable_values: set[str] = set()
        self.coordinate_extents = dict(zip(coordinates, _shape(node), strict=True))

    def _unavailable_value(self, node: Node) -> str:
        name = f"chain_domain_value_{node.name}"
        self.unavailable_values.add(name)
        return name

    def _load(self, node: Node, coordinates: tuple[str, ...]) -> str:
        source = cast("Node", node.args[0])
        if source.target is _tracing_ops._host_tensor:
            self.indices(node, coordinates)
            return self._unavailable_value(node)
        return super()._load(node, coordinates)

    def value(self, node: Node, coordinates: tuple[str, ...]) -> str:
        from .chained_collectives import classify_collective

        self.bounds.extend(_shape_domain(node, coordinates, self.plan))
        if node.target is torch.ops.prims.iota.default:
            length_arg = node.args[0]
            if isinstance(length_arg, Node):
                length_arg = length_arg.meta.get("val")
            if not isinstance(length_arg, (int, torch.SymInt)):
                raise _UnsupportedChain("symbolic iota domain")
            length = _resolved_extent(length_arg)
            extent = self.coordinate_extents.get(coordinates[0])
            if coordinates[0] != "0" and (extent is None or length < extent):
                self.bounds.append(f"0 <= ({coordinates[0]}) < {length}")
        if node.target is dot:
            self.value(cast("Node", node.args[0]), (coordinates[0], "0"))
            self.value(cast("Node", node.args[1]), ("0", coordinates[1]))
            return self._unavailable_value(node)
        if node.target is scan_ops._associative_scan:
            self.value(cast("Node", node.args[1]), coordinates)
            return self._unavailable_value(node)
        if node.target is _tracing_ops._mask_to:
            return self.value(cast("Node", node.args[0]), coordinates)
        operation = classify_collective(node)
        if operation is not None and operation.kind == "sum":
            source_coords = list(coordinates)
            if node.meta["val"].ndim == operation.source.meta["val"].ndim:
                source_coords[operation.axis] = "0"
            else:
                source_coords.insert(operation.axis, "0")
            self.value(operation.source, tuple(source_coords))
            return self._unavailable_value(node)
        return super().value(node, coordinates)


def _domain_bounds(expression: _DomainExpression) -> list[str]:
    class InlineDomain(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id in expression.definitions:
                return self.visit(expr_from_string(expression.definitions[node.id]))
            return node

    # Domain tracing has its own temporary bindings, which are deliberately
    # not emitted as value computations. Gather coordinates can depend on those
    # bindings: return self-contained predicates rather than dangling names.
    bounds = []
    for bound in expression.bounds:
        tree = expr_from_string(bound)
        if _names(tree).intersection(expression.definitions):
            tree = InlineDomain().visit(tree)
        if _names(tree).intersection(expression.unavailable_values):
            # This proof-only traversal cannot replace a runtime index read
            # with a constant. Leave data-dependent domains to ordinary lowering.
            raise _UnsupportedChain("data-dependent contraction operand domain")
        bounds.append(
            ast.unparse(tree)
            if _names(expr_from_string(bound)).intersection(expression.definitions)
            else bound
        )
    return list(dict.fromkeys(bounds))


def _operand_domain(
    cg: GenerateAST,
    node: Node,
    coordinates: tuple[str, ...],
    plan: ChainedMatmulPlan,
) -> list[str]:
    expression = _DomainExpression(cg, plan, node, coordinates)
    expression.value(node, coordinates)
    return _domain_bounds(expression)


class _StoreDomainExpression(_DomainExpression):
    """Retain index-tile validity independently of destination addresses."""

    def _index(self, index: object, coordinate: str | None) -> str:
        if isinstance(index, Node) and index.target in (
            _tracing_ops._get_symnode,
            torch.ops.aten.sym_size.int,
        ):
            assert coordinate is not None
            axis = self.block_id(index)
            if self.plan.loop is not None and axis == self.plan.loop.block_id:
                self.bounds.append(
                    f"chain_loop_index + ({coordinate}) < chain_loop_end"
                )
            else:
                for block_id, extent, block in self.plan.axes:
                    if block_id == axis and extent % block:
                        self.bounds.append(
                            f"chain_origin_{axis} + ({coordinate}) < {extent}"
                        )
        return super()._index(index, coordinate)


def _store_domain(
    cg: GenerateAST,
    store: Node,
    coordinates: tuple[str, ...],
    plan: ChainedMatmulPlan,
) -> list[str]:
    # Traverse the selectors, not the stored value: tensor index shapes own
    # their automatic tile masks, even when arithmetic wraps padded indices
    # back inside the destination. indices() also preserves singleton broadcast
    # coordinates and maps raw tile selectors to the correct output dimension.
    expression = _StoreDomainExpression(
        cg, plan, cast("Node", store.args[2]), coordinates
    )
    expression.indices(store, coordinates)
    return _domain_bounds(expression)


def _indent(lines: Sequence[str], spaces: int = 4) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line.replace("\n", "\n" + prefix) for line in lines)


_COPY_INTEGER_CASTS = {
    name: sympy.Function(f"chain_copy_{name}", integer=True)
    for name in ("Int32", "Int64")
}


def _copy_index(
    text: str, definitions: dict[str, str], symbols: dict[str, sympy.Symbol]
) -> sympy.Expr:
    """Parse only integer index arithmetic; never treat memory reads as uniform."""

    def visit(node: ast.AST) -> sympy.Expr:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return sympy.Integer(node.value)
        if isinstance(node, ast.Name):
            if node.id in symbols:
                return symbols[node.id]
            if node.id in definitions:
                return _copy_index(definitions[node.id], definitions, symbols)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = visit(node.operand)
            if value.has(*_COPY_INTEGER_CASTS.values()):
                raise _UnsupportedChain("arithmetic after fixed-width index cast")
            return -value
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cutlass"
            and node.func.attr in _COPY_INTEGER_CASTS
            and len(node.args) == 1
            and not node.keywords
        ):
            # Preserve the cast as an opaque operation. A uniform cast can be
            # a complete source index; a varying cast is not proven affine.
            # Arithmetic after either cast needs a separate no-overflow proof.
            return cast(
                "sympy.Expr", _COPY_INTEGER_CASTS[node.func.attr](visit(node.args[0]))
            )
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if left.has(*_COPY_INTEGER_CASTS.values()) or right.has(
                *_COPY_INTEGER_CASTS.values()
            ):
                # SymPy uses unbounded integers. It must not cancel or widen
                # fixed-width operations such as (Int32(i) * 2**30) // 2**30.
                raise _UnsupportedChain("arithmetic after fixed-width index cast")
            if isinstance(node.op, ast.Add):
                return sympy.Add(left, right)
            if isinstance(node.op, ast.Sub):
                return sympy.Add(left, sympy.Mul(-1, right))
            if isinstance(node.op, ast.Mult):
                return sympy.Mul(left, right)
            if isinstance(node.op, ast.FloorDiv):
                return sympy.floor(sympy.Mul(left, sympy.Pow(right, -1)))
            if isinstance(node.op, ast.Mod):
                return cast("sympy.Expr", sympy.Mod(left, right))
        raise _UnsupportedChain("non-affine cooperative copy index")

    return visit(ast.parse(text, mode="eval").body)


def _copy_code(value: sympy.Basic) -> str:
    """Render integer arithmetic without floating division or math.floor."""
    if isinstance(value, sympy.Integer):
        return str(value)
    if isinstance(value, sympy.Symbol):
        env = CompileEnvironment.current()
        index_dtype = env.backend.dtype_str(env.index_dtype)
        return f"{index_dtype}({value})"
    for name, function in _COPY_INTEGER_CASTS.items():
        if value.func == function:
            result = f"cutlass.{name}({_copy_code(value.args[0])})"
            # Preserve explicit narrowing, then widen before physical stride
            # arithmetic when the tensor requires 64-bit addressing.
            if (
                name == "Int32"
                and CompileEnvironment.current().index_dtype == torch.int64
            ):
                result = f"cutlass.Int64({result})"
            return result
    if isinstance(value, (sympy.Add, sympy.Mul)):
        op = " + " if isinstance(value, sympy.Add) else " * "
        return "(" + op.join(_copy_code(arg) for arg in value.args) + ")"
    if value.func is sympy.floor:
        numerator, denominator = sympy.fraction(sympy.together(value.args[0]))
        return f"({_copy_code(numerator)} // {_copy_code(denominator)})"
    if isinstance(value, sympy.Mod):
        return f"({_copy_code(value.args[0])} % {_copy_code(value.args[1])})"
    raise _UnsupportedChain("noninteger cooperative copy index")


def _stage_input(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    operand: Node,
    shared: str,
    role: str,
    shape: tuple[int, int],
) -> _StagedInput | None:
    """Record a pure staged leaf with invertible unit-stride logical indices."""
    if not _direct_operand(operand):
        return None
    names = ("chain_cached_row", "chain_cached_col")
    expression = _Expression(cg, plan, {})
    try:
        expression.value(operand, names if role == "a" else names[::-1])
        if len(expression.global_accesses) != 1:
            return None
        source, indices = expression.global_accesses[0]
        symbols = {
            name: sympy.Symbol(name, integer=True)
            for name in (*names, *expression.origins.values())
        }
        coordinates = (symbols[names[0]], symbols[names[1]])
        values = tuple(
            sympy.expand(_copy_index(index, expression.definitions, symbols))
            for index in indices
        )
        inverse: list[tuple[int, int]] = []
        for axis, coordinate in enumerate(coordinates):
            for position, value in enumerate(values):
                slope = sympy.diff(value, coordinate)
                other_slope = sympy.diff(value, coordinates[1 - axis])
                if slope in (-1, 1) and other_slope == 0:
                    inverse.append((position, int(slope)))
                    break
            else:
                return None
    except _UnsupportedChain:
        return None
    return _StagedInput(
        source,
        operand,
        shared,
        role,
        shape,
        values,
        coordinates,
        (inverse[0], inverse[1]),
    )


def _staged_input_value(
    expression: _Expression, source: Node, indices: list[str], fallback: str
) -> str:
    """Reuse the last contraction's leaf only after proving its exact address map."""
    for staged in expression.staged_inputs:
        if source is not staged.source:
            continue
        symbols = {
            name: sympy.Symbol(name, integer=True)
            for name in (*expression.origins.values(), "chain_store")
        }
        try:
            query = tuple(
                sympy.expand(_copy_index(index, expression.definitions, symbols))
                for index in indices
            )
            zero = dict.fromkeys(staged.coordinates, 0)
            coordinates = tuple(
                sympy.expand(
                    sympy.Mul(
                        sign,
                        sympy.Add(
                            query[position],
                            sympy.Mul(-1, staged.indices[position].subs(zero)),
                        ),
                    )
                )
                for position, sign in staged.inverse_axes
            )
            substitution = dict(zip(staged.coordinates, coordinates, strict=True))
            if any(
                sympy.simplify(actual - cached.subs(substitution)) != 0
                for actual, cached in zip(query, staged.indices, strict=True)
            ):
                continue
            emitted = (_copy_code(coordinates[0]), _copy_code(coordinates[1]))
            bounds = [
                f"0 <= ({coordinate}) < {extent}"
                for coordinate, extent in zip(emitted, staged.shape, strict=True)
            ]
            bounds.extend(
                _operand_domain(
                    expression.cg,
                    staged.operand,
                    emitted if staged.role == "a" else emitted[::-1],
                    expression.plan,
                )
            )
            predicate = " & ".join(f"({bound})" for bound in bounds)
            return f"({staged.shared}[{emitted[0]}, {emitted[1]}] if {predicate} else {fallback})"
        except _UnsupportedChain:
            continue
    return fallback


def _scan_input(
    expression: _Expression,
    operand: Node,
    coordinates: tuple[str, ...],
    indices: list[str],
    shared: str,
    extent: int,
    scan_index: str,
) -> _ScanInput | None:
    """Prove a scan-prelude leaf's invertible one-dimensional address map."""
    if (
        operand.target is not memory_ops.load
        or not _direct_operand(operand)
        or coordinates != (scan_index,)
    ):
        return None
    source = cast("Node", operand.args[0])
    symbols = {
        name: sympy.Symbol(name, integer=True)
        for name in (scan_index, *expression.origins.values())
    }
    coordinate = symbols[scan_index]
    try:
        values = tuple(
            sympy.expand(_copy_index(index, expression.definitions, symbols))
            for index in indices
        )
        for position, value in enumerate(values):
            slope = sympy.diff(value, coordinate)
            if slope in (-1, 1):
                return _ScanInput(
                    source,
                    operand,
                    shared,
                    extent,
                    values,
                    coordinate,
                    (position, int(slope)),
                )
    except _UnsupportedChain:
        pass
    return None


def _scan_input_value(
    expression: _Expression, source: Node, indices: list[str], fallback: str
) -> str:
    """Read a cached leaf only for the same source, full index map and domain."""
    for cached in expression.scan_inputs:
        if source is not cached.source:
            continue
        symbols = {
            name: sympy.Symbol(name, integer=True)
            for name in (*expression.origins.values(), *expression.coordinate_names)
        }
        try:
            query = tuple(
                sympy.expand(_copy_index(index, expression.definitions, symbols))
                for index in indices
            )
            position, sign = cached.inverse_axis
            coordinate = sympy.expand(
                sign
                * (
                    query[position]
                    - cached.indices[position].subs(cached.coordinate, 0)
                )
            )
            if any(
                sympy.simplify(actual - original.subs(cached.coordinate, coordinate))
                != 0
                for actual, original in zip(query, cached.indices, strict=True)
            ):
                continue
            emitted = _copy_code(coordinate)
            bounds = [f"0 <= ({emitted}) < {cached.extent}"]
            bounds.extend(
                _operand_domain(
                    expression.cg, cached.operand, (emitted,), expression.plan
                )
            )
            predicate = " & ".join(f"({bound})" for bound in bounds)
            dtype = CompileEnvironment.current().backend.dtype_str(
                source.meta["val"].dtype
            )
            return (
                f"({dtype}({cached.shared}[{emitted}]) if {predicate} else {fallback})"
            )
        except _UnsupportedChain:
            continue
    return fallback


def _async_copy(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    operand: Node,
    prefix: str,
    role: str,
    shape: tuple[int, int],
    stride: tuple[int, int],
    inner: int,
    dtype: str,
    fallback: list[str],
    target: str | None = None,
    *,
    expression: _Expression | None = None,
    domain: Sequence[str] | None = None,
) -> list[str] | None:
    """Stage affine leaves using 128-bit copies, with uniform bounds/alignment guards.

    Arbitrary argument views need a runtime pointer-alignment check: allocator
    alignment or fake storage identity alone says nothing about a bound view's
    offset. The scalar path retains masking on boundary CTAs and misaligned views.
    """
    if not _direct_operand(operand):
        return None
    width, height = shape[inner], shape[1 - inner]
    thread_columns = min(plan.threads, width // 8)
    if (
        width % 8
        or thread_columns == 0
        or thread_columns & (thread_columns - 1)
        or height % (plan.threads // thread_columns)
    ):
        return None
    coordinates = ("chain_copy_row", "chain_copy_col")
    expression = expression or _Expression(cg, plan, {})
    coords = coordinates if role == "a" else coordinates[::-1]
    try:
        expression.value(operand, coords)
        if len(expression.global_accesses) != 1:
            return None
        source, indices = expression.global_accesses[0]
        fake = source.meta["val"]
        symbols = {
            name: sympy.Symbol(name, integer=True)
            for name in (*coordinates, *expression.origins.values())
        }
        row, col = (symbols[name] for name in coordinates)
        bases: list[sympy.Expr] = []
        coefficients: list[tuple[int, int]] = []
        bounds: list[str] = []
        for index, extent in zip(indices, fake.shape, strict=True):
            value = sympy.expand(_copy_index(index, expression.definitions, symbols))
            base = value.subs({row: 0, col: 0})
            delta = (sympy.diff(value, row), sympy.diff(value, col))
            if any(not isinstance(d, sympy.Integer) for d in delta):
                return None
            pair = (int(delta[0]), int(delta[1]))
            if (
                sympy.expand(
                    sympy.Add(
                        value, -base, sympy.Mul(-pair[0], row), sympy.Mul(-pair[1], col)
                    )
                )
                != 0
            ):
                return None
            bases.append(base)
            coefficients.append(pair)
            lower = sum(min(0, d * (s - 1)) for d, s in zip(pair, shape, strict=True))
            upper = sum(max(0, d * (s - 1)) for d, s in zip(pair, shape, strict=True))
            bounds.extend(
                [
                    f"0 <= ({_copy_code(base + lower)})",
                    f"({_copy_code(base + upper)}) < {extent}",
                ]
            )
        strides = tuple(fake.stride())
        physical = tuple(
            sum(c[axis] * s for c, s in zip(coefficients, strides, strict=True))
            for axis in (0, 1)
        )
        if (
            physical[inner] != 1
            or physical[1 - inner] <= 0
            or physical[1 - inner] % 8
            or stride[inner] != 1
            or stride[1 - inner] % 8
        ):
            return None
        base_offset = sympy.Add(
            *itertools.starmap(sympy.Mul, zip(bases, strides, strict=True))
        )
        name = expression.tensor_name(source)
        offset = _copy_code(base_offset)
        bounds.extend(
            f"{name}.layout.stride[{axis}] == {source_stride}"
            for axis, source_stride in enumerate(strides)
        )
        last = (str(shape[0] - 1), str(shape[1] - 1))
        bounds.extend(
            _operand_domain(cg, operand, last if role == "a" else last[::-1], plan)
            if domain is None
            else domain
        )
    except _UnsupportedChain:
        return None
    tag = f"{prefix}_{role}_async"
    fast = [
        f"{tag}_source = cute.make_tensor({tag}_pointer.align(16), cute.make_layout(({height}, {width}), stride=({physical[1 - inner]}, 1)))",
        f"{tag}_target = {target}"
        if target is not None
        else f"{tag}_target = cute.make_tensor({prefix}_{role}_ptr, cute.make_layout(({height}, {width}), stride=({stride[1 - inner]}, 1)))",
        f"{tag}_copy = cute.make_tiled_copy_tv(cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), {dtype}, num_bits_per_copy=128), cute.make_layout(({plan.threads // thread_columns}, {thread_columns}), stride=({thread_columns}, 1)), cute.make_layout((1, 8)))",
        f"{tag}_thread = {tag}_copy.get_slice(chain_thread)",
        f"cute.copy({tag}_copy, {tag}_thread.partition_S({tag}_source), {tag}_thread.partition_D({tag}_target))",
    ]
    return [
        f"{tag}_pointer = {name}.iterator + ({offset})",
        # CuTe's short-circuit AST rewrite can duplicate a long conjunction
        # exponentially. These comparisons are uniform and safe to evaluate
        # eagerly; none dereferences the pointer or depends on another guard.
        f"if {' & '.join(f'({bound})' for bound in [*bounds, f'{tag}_pointer.toint() % 16 == 0'])}:",
        _indent(fast),
        "else:",
        _indent(fallback),
    ]


def _operand_inner_axis(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    operand: Node,
    *,
    expression: _Expression | None = None,
) -> int:
    """Choose cooperative order from the operand's global-memory accesses."""
    expression = expression or _Expression(cg, plan, boundaries)
    coordinates = ("chain_coordinate_m", "chain_coordinate_n")
    expression.value(operand, coordinates)
    costs = [0, 0]
    for indices, strides in expression.accesses:
        for index, stride in zip(indices, strides, strict=True):
            dependencies = expression.dependencies(index)
            for axis, coordinate in enumerate(coordinates):
                if coordinate in dependencies:
                    costs[axis] += abs(stride)
    # A broadcast access costs nothing in its invariant direction.  If both
    # axes are shared-memory-only, keep the conventional row-major traversal.
    return 0 if costs[0] < costs[1] else 1


@dataclasses.dataclass(frozen=True)
class _RegisterBridge:
    role: str
    lines: tuple[str, ...]
    stage: int
    source: Node
    operand: Node
    revision: tuple[object, ...]
    epilogue: WarpOperandEpilogue | None = None
    keep_shared: bool = False

    def epilogue_facts(self) -> object:
        return (
            None if self.epilogue is None else self.epilogue.facts(),
            self.keep_shared,
        )

    def render(self) -> tuple[str, ...]:
        from .chained_fragment_epilogue import WarpOperandEpilogue

        if not isinstance(self.epilogue, WarpOperandEpilogue):
            raise _UnsupportedChain("register bridge lacks original epilogue")
        if (
            self.epilogue.value.source is not self.source
            or self.epilogue.value.target is not self.operand
            or self.epilogue.destination != f"chain_{self.stage}_{self.role}"
        ):
            raise _UnsupportedChain("register bridge epilogue relation changed")
        lines = self.epilogue.render()
        if lines != self.lines:
            raise _UnsupportedChain("register bridge epilogue publication changed")
        return lines


def _register_bridge_revision(plan: ChainedMatmulPlan) -> tuple[object, ...]:
    """Preserve the exact graph behind a late coordinate/exclusive-use proof."""
    return (
        plan.dots,
        plan.shapes,
        plan.threads,
        plan.scans,
        tuple(
            (
                node,
                node.op,
                node.target,
                node.args,
                tuple(node.kwargs.items()),
                (
                    node.meta["val"].dtype,
                    tuple(map(str, node.meta["val"].shape)),
                    tuple(map(str, node.meta["val"].stride())),
                )
                if isinstance(node.meta.get("val"), torch.Tensor)
                else str(node.meta.get("val")),
                tuple(node.users),
            )
            for node in plan.dots[0].graph.nodes
        ),
    )


def _register_bridges(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    dtype: str,
    scan_inputs: list[_ScanInput],
) -> dict[int, _RegisterBridge]:
    """Fuse a single-use, coordinate-preserving pointwise region into C registers.

    The producer must be the immediately preceding contraction. Requiring all
    producer warps active keeps register lifetimes out of conditional regions;
    unsupported permutations, broadcasts of C, and other consumers use shared C.
    """
    from .chained_fragment_epilogue import WarpOperandEpilogue
    from .chained_fragment_epilogue import bind_fragment_expression
    from .physical_use_frontier import PhysicalUseFrontier

    result: dict[int, _RegisterBridge] = {}
    use_frontier = PhysicalUseFrontier(tuple(plan.dots[0].graph.nodes))
    all_boundaries = {
        **boundaries,
        **{node: f"chain_{i}_c" for i, node in enumerate(plan.dots)},
    }
    revision: tuple[object, ...] | None = None
    for stage in range(1, len(plan.dots)):
        producer, consumer = plan.dots[stage - 1 : stage + 1]
        if plan.pointwise_cache is not None and any(
            producer in entry.dependencies for entry in plan.pointwise_cache.entries
        ):
            continue
        if plan.region is not None and any(
            producer in _ancestors(collective)
            for collective in (*plan.region.scans, *plan.region.reductions)
        ):
            continue
        accumulator = consumer.args[2]
        if isinstance(accumulator, Node) and producer in _ancestors(accumulator):
            # The FP32 producer remains live independently of its narrowed MMA
            # operand. A register-only bridge must not erase that boundary.
            continue
        dtype = CompileEnvironment.current().backend.dtype_str(
            plan.operand_dtype(stage)
        )
        producer_shape = plan.shapes[stage - 1][:2]
        if producer_shape[1] < plan.threads // 4:
            continue
        for role, operand in zip(("a", "b"), consumer.args[:2], strict=True):
            assert isinstance(operand, Node)
            if plan.strategy != "tcgen05_tmem" and plan.pointwise_cache is not None:
                # Warp bridge statements are built before cache publication.
                # Keep this operand in ordinary staging so it reads the cache.
                ancestors = _ancestors(operand)
                if any(
                    entry.node in ancestors for entry in plan.pointwise_cache.entries
                ):
                    continue
            other = cast("Node", consumer.args[1 if role == "a" else 0])
            if producer in _ancestors(other):
                continue
            logical_shape = _shape(operand)
            stored_shape = logical_shape if role == "a" else logical_shape[::-1]
            if stored_shape != producer_shape:
                continue
            keep_shared = not use_frontier.exclusive_use(producer, operand, consumer)
            if keep_shared and not (
                cg.device_function.config.config.get("cute_chained_fragment_epilogues")
                is True
                and plan.strategy != "tcgen05_tmem"
                and len(plan.dots) == 2
                and plan.loop is None
            ):
                continue
            prefix, previous = f"chain_{stage}_{role}_bridge", f"chain_{stage - 1}"
            coords = (f"{prefix}_row", f"{prefix}_col")
            try:
                expression = bind_fragment_expression(
                    cg,
                    plan,
                    all_boundaries,
                    scan_inputs,
                    producer,
                    operand,
                    coords,
                    coords if role == "a" else (coords[1], coords[0]),
                    f"{previous}_acc[{prefix}_index]",
                    dtype,
                )
            except _UnsupportedChain:
                continue
            if revision is None:
                revision = _register_bridge_revision(plan)
            epilogue = WarpOperandEpilogue(
                expression, prefix, previous, producer_shape, f"chain_{stage}_{role}"
            )
            result[stage] = _RegisterBridge(
                role,
                epilogue.render(),
                stage,
                producer,
                operand,
                revision,
                epilogue,
                keep_shared,
            )
            break
    use_frontier.check()
    return result


def _codegen_scans(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scan_inputs: list[_ScanInput],
    cache_inputs: bool,
) -> list[str]:
    lines: list[str] = []
    for stage, node in enumerate(plan.scans):
        extent = _shape(node)[0]
        prefix = f"chain_scan_{stage}"
        expression = _Expression(cg, plan, boundaries)
        value = expression.value(cast("Node", node.args[1]), (f"{prefix}_index",))
        cached_values: list[tuple[_ScanInput, str]] = []
        if cache_inputs:
            candidates = _scan_cache_candidates(node, plan.scans, plan.store)
            for operand, coords, indices, loaded in expression.loaded_inputs:
                if operand not in candidates:
                    continue
                cached = _scan_input(
                    expression,
                    operand,
                    coords,
                    indices,
                    f"{prefix}_input_{len(cached_values)}",
                    extent,
                    f"{prefix}_index",
                )
                if cached is not None:
                    cached_values.append((cached, loaded))
        cache_stores: list[str] = []
        for cached, loaded in cached_values:
            domain = _operand_domain(cg, cached.operand, (f"{prefix}_index",), plan)
            dtype = CompileEnvironment.current().backend.dtype_str(
                cached.operand.meta["val"].dtype
            )
            cache_stores.extend(
                [
                    f"if {prefix}_index < {extent}:",
                    f"    {cached.shared}[{prefix}_index] = {_masked_operand(loaded, dtype, domain)}",
                ]
            )
        body = [
            f"{prefix}_index = {prefix}_step * {plan.threads} + chain_thread",
            *expression.lines,
            *cache_stores,
            f"{prefix}_acc = cutlass.Float32({value}) if {prefix}_index < {extent} else cutlass.Float32(0)",
        ]
        for shift in (1, 2, 4, 8, 16):
            body.extend(
                [
                    f"{prefix}_other = cute.arch.shuffle_sync_up({prefix}_acc, offset={shift})",
                    f"{prefix}_acc += ({prefix}_other if chain_thread % 32 >= {shift} else cutlass.Float32(0))",
                ]
            )
        body.extend(
            [
                "if chain_thread % 32 == 31:",
                f"    {prefix}_totals[chain_thread // 32] = {prefix}_acc",
                "cute.arch.sync_threads()",
                f"{prefix}_carry = cutlass.Float32(0)",
                f"for {prefix}_warp in cutlass.range_constexpr({plan.threads // 32}):",
                f"    if {prefix}_warp < chain_thread // 32:",
                f"        {prefix}_carry += {prefix}_totals[{prefix}_warp]",
                f"if {prefix}_step > 0:",
                f"    {prefix}_carry += {prefix}_values[{prefix}_step * {plan.threads} - 1]",
                f"if {prefix}_index < {extent}:",
                f"    {prefix}_values[{prefix}_index] = {prefix}_acc + {prefix}_carry",
                "cute.arch.sync_threads()",
            ]
        )
        lines.extend(
            [
                f"{prefix}_pointer = cute.arch.alloc_smem(cutlass.Float32, {extent + plan.threads // 32}, alignment=128)",
                f"{prefix}_values = cute.make_tensor({prefix}_pointer, cute.make_layout({extent}))",
                f"{prefix}_totals = cute.make_tensor({prefix}_pointer + {extent}, cute.make_layout({plan.threads // 32}))",
                *(
                    # Keep the already-loaded leaf in its original scalar dtype.
                    # Scan arithmetic is FP32, but its raw inputs need no roundtrip.
                    f"{cached.shared} = cute.make_tensor(cute.arch.alloc_smem({CompileEnvironment.current().backend.dtype_str(cached.operand.meta['val'].dtype)}, {extent}, alignment=128), cute.make_layout({extent}))"
                    for cached, _ in cached_values
                ),
                f"for {prefix}_step in cutlass.range_constexpr({(extent + plan.threads - 1) // plan.threads}):",
                _indent(body),
            ]
        )
        scan_inputs.extend(cached for cached, _ in cached_values)
        boundaries[node] = f"{prefix}_values"
    return lines


def _emit_store(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    store: Node,
    boundaries: dict[Node, str],
    staged_inputs: list[_StagedInput],
    scan_inputs: list[_ScanInput],
    prefix: str = "chain_store",
    *,
    execution: ChainedExecution | None = None,
) -> list[str]:
    """Store a region live-out through its original coordinates and predicate."""
    from .chained_store_expression import lower_store_point

    shape = _shape(cast("Node", store.args[2]))
    if len(shape) != 2:
        raise _UnsupportedChain("contraction live-out rank")
    m, n = shape
    execution = execution or ChainedExecution(plan.threads)
    coords = (f"{prefix} // {n}", f"{prefix} % {n}")
    point = lower_store_point(
        cg,
        plan,
        store,
        boundaries,
        staged_inputs,
        scan_inputs,
        coordinates=coords,
        coordinate_names=(prefix,),
    )
    return [
        f"for {prefix}_step in cutlass.range_constexpr({(m * n + execution.threads - 1) // execution.threads}):",
        f"    {prefix} = {execution.thread} + {prefix}_step * {execution.threads}",
        f"    if {prefix} < {m * n}:",
        _indent(point.lines, 8),
        f"        if {' and '.join(point.bounds)}:",
        f"            {point.target}[{', '.join(point.indices)}] = {point.value}",
    ]


def codegen_chained_matmul(cg: GenerateAST) -> bool:
    from .native_matmul_metadata import body_attempt

    with body_attempt(cg):
        return _codegen_chained_matmul(cg)


def _codegen_chained_matmul(cg: GenerateAST) -> bool:
    """Emit a contraction DAG, preserving its pointwise regions and store."""
    from . import chained_collectives
    from . import chained_pointwise_residency
    from . import chained_tcgen_stage
    from .chained_cache_layout import PointwiseCacheLayouts
    from .chained_pointwise_unroll import BoundedProducerUnroll
    from .chained_scratch_layout import ScratchLayouts
    from .chained_seed_tiles import SeedTiling
    from .chained_vector_stage import VectorStaging
    from .chained_warp_mma import emit_warp_mma

    df = cg.device_function
    plan = df.cute_state.chained_matmul_plan
    root = cg.current_root_graph_info
    island_consumers = df.config.config.get("cute_chained_island_consumers", False)
    if type(island_consumers) is not bool or (
        island_consumers
        and (
            plan is None
            or plan.strategy != "tcgen05_tmem"
            or plan.loop is None
            or plan.preparation_pipeline is None
            or df.config.config.get("cute_chained_compact_preparation") is not True
            or df.config.config.get("cute_chained_register_islands") is not True
        )
    ):
        raise exc.BackendUnsupported(
            "cute",
            "island consumers require admitted compact register-island preparation",
        )
    if plan is None or root is None or root.graph_id != plan.root_graph_id:
        return False
    if plan.strategy == "tcgen05_tmem" and plan.loop is None:
        from .chained_plain_root import codegen_plain_root
        from .chained_tcgen05 import codegen_chained_tcgen05

        if codegen_plain_root(cg, plan):
            return True
        return codegen_chained_tcgen05(cg, plan)
    dtype = "cutlass.BFloat16" if plan.dtype is torch.bfloat16 else "cutlass.Float16"
    schedule = cast(
        "str", df.config.config.get("cute_chained_mma_schedule", "coalesced")
    )
    scratch = ScratchLayouts.for_plan(
        cast("str", df.config.config.get("cute_chained_scratch_layout", "row_major")),
        plan,
    )
    padding = 0 if schedule == "k_major" else 8
    output_padding = padding // 2
    vector_staging = VectorStaging(
        cast("bool", df.config.config.get("cute_chained_pointwise_vectorize", False)),
        group_enabled=cast(
            "bool", df.config.config.get("cute_chained_vector_group", False)
        ),
        async_enabled=cast(
            "bool", df.config.config.get("cute_chained_async_vector_store", False)
        ),
    )
    producer_unroll = (
        BoundedProducerUnroll(
            cast("int", df.config.config.get("cute_chained_pointwise_unroll", 1))
        )
        if plan.loop is not None and plan.strategy == "tcgen05_tmem"
        else None
    )
    seed_tiling = SeedTiling(
        cast("int", df.config.config.get("cute_chained_seed_tile_columns", 0))
    )
    cache_layouts = PointwiseCacheLayouts(
        cast("str", df.config.config.get("cute_chained_pointwise_cache_layout", "auto"))
    )
    lines = [
        "chain_thread = cutlass.Int32(cute.arch.thread_idx()[0])",
        "chain_pid = cutlass.Int32(cute.arch.block_idx()[0])",
    ]
    suffix = 1
    env = CompileEnvironment.current()
    index_dtype = env.backend.dtype_str(env.index_dtype)
    for axis_id, size, block in reversed(plan.axes):
        count = (size + block - 1) // block
        lines.append(
            f"chain_origin_{axis_id} = {index_dtype}(chain_pid // {suffix} % {count}) * {block}"
        )
        suffix *= count
    if plan.preparation_pipeline is not None:
        from .chained_preparation_pipeline import emit_preparation_pipeline

        assert producer_unroll is not None
        lines = emit_preparation_pipeline(
            cg,
            plan,
            plan.preparation_pipeline,
            lines,
            scratch,
            vector_staging,
            producer_unroll,
            seed_tiling,
            cache_layouts,
            **(
                {"compact_preparation": True}
                if df.config.config.get("cute_chained_compact_preparation") is True
                else {}
            ),
            **(
                {"leaf_issue_batching": True}
                if df.config.config.get("cute_chained_leaf_issue_batching") is True
                else {}
            ),
            **(
                {"broadcast_retention": True}
                if df.config.config.get("cute_chained_broadcast_retention") is True
                else {}
            ),
            **(
                {"completed_member_store": True}
                if df.config.config.get("cute_chained_completed_member_store") is True
                else {}
            ),
            **({"island_consumers": True} if island_consumers else {}),
            **(
                {"fragment_epilogues": True}
                if df.config.config.get("cute_chained_fragment_epilogues") is True
                else {}
            ),
        )
        scratch.validate()
        vector_staging.validate()
        producer_unroll.validate()
        seed_tiling.validate()
        cache_layouts.validate()
        return _install_chained_body(cg, plan, lines)
    m, n = _shape(cast("Node", plan.store.args[2]))
    boundaries: dict[Node, str] = {}
    prepared_warp_accepted = False
    try:
        loop_setup: list[str] = []
        if plan.loop is not None:
            from .chained_loop import initialize_loop

            loop_setup = list(lines)
            if plan.loop_workspace is not None:
                loop_setup.append(
                    f"chain_c_workspace = cute.arch.alloc_smem(cutlass.Float32, {plan.loop_workspace.allocated_bytes // 4}, alignment=128)"
                )
            loop_setup.extend(initialize_loop(cg, plan, scratch, plan.loop_workspace))
            lines = []
        scan_inputs: list[_ScanInput] = []
        general_collectives = chained_collectives.uses_general_collectives(plan) or (
            plan.loop is not None and bool(plan.scans)
        )
        if general_collectives:
            boundaries.update(chained_collectives.collective_bindings(plan))
            (loop_setup if plan.loop is not None else lines).extend(
                chained_collectives.allocate_collectives(plan, scratch)
            )
        else:
            lines.extend(
                _codegen_scans(
                    cg,
                    plan,
                    boundaries,
                    scan_inputs,
                    schedule == "cp_async_register_reuse_scan",
                )
            )
        if plan.pointwise_cache is not None:
            (loop_setup if plan.loop is not None else lines).extend(
                chained_pointwise_residency.allocate_pointwise_cache(
                    plan.pointwise_cache, scratch, cache_layouts
                )
            )
        bridges = (
            _register_bridges(cg, plan, boundaries, dtype, scan_inputs)
            if plan.loop is None
            and schedule
            in (
                "cp_async_register",
                "cp_async_register_reuse",
                "cp_async_register_reuse_scan",
            )
            else {}
        )
        if (
            df.config.config.get("cute_chained_fragment_epilogues") is True
            and not bridges
        ):
            raise _UnsupportedChain(
                "fragment epilogues lack an admitted original result image"
            )
        from .chained_warp_bridge import root_warp_bridge_sequence

        warp_sequence = root_warp_bridge_sequence(
            cg,
            plan,
            boundaries,
            scan_inputs,
            bridges,
            schedule,
            padding,
            dtype,
            general_collectives=general_collectives,
        )
        if (
            any(bridge.keep_shared for bridge in bridges.values())
            and warp_sequence is None
        ):
            raise _UnsupportedChain(
                "shared-plus-fragment requires the original completed root body"
            )
        a_workspace = max(m * k + 8 * max(m, k) for m, _, k in plan.shapes)
        b_workspace = max(n * k + 8 * max(n, k) for _, n, k in plan.shapes)
        tcgen_geometries = ()
        if plan.strategy == "tcgen05_tmem":
            tcgen_geometries = tuple(
                chained_tcgen_stage.stage_geometry(shape) for shape in plan.shapes
            )
            assert all(geometry is not None for geometry in tcgen_geometries)
            loop_setup.extend(
                chained_tcgen_stage.allocate_stages(
                    cast(
                        "tuple[chained_tcgen_stage.StageGeometry, ...]",
                        tcgen_geometries,
                    ),
                    plan.contraction_groups,
                    plan.loop_workspace or plan_contraction_workspace(plan),
                    scratch,
                    plan.threads,
                    workspace_allocated=plan.loop_workspace is not None,
                )
            )
        else:
            (loop_setup if plan.loop is not None else lines).extend(
                [
                    f"chain_a_workspace = cute.arch.alloc_smem({dtype}, {a_workspace}, alignment=128)",
                    f"chain_b_workspace = cute.arch.alloc_smem({dtype}, {b_workspace}, alignment=128)",
                ]
            )
        staged_inputs: list[_StagedInput] = []
        for stage, (node, (rows, columns, reduction)) in enumerate(
            zip(plan.dots, plan.shapes, strict=True)
        ):
            if general_collectives:
                lines.extend(
                    chained_collectives.emit_collectives_before(
                        cg, plan, boundaries, stage
                    )
                )
            if plan.pointwise_cache is not None:
                lines.extend(
                    chained_pointwise_residency.emit_pointwise_cache_before(
                        cg, plan, boundaries, plan.pointwise_cache, stage
                    )
                )
            if plan.strategy == "tcgen05_tmem":
                assert plan.loop is not None
                geometry = tcgen_geometries[stage]
                assert geometry is not None
                group = None
                if plan.contraction_groups is not None:
                    group = next(
                        (
                            item
                            for item in plan.contraction_groups
                            if stage in item.stages
                        ),
                        None,
                    )
                    assert group is not None
                    if stage != group.stages[0]:
                        continue
                    geometry = group.geometries[0]
                step = df.resolved_block_size(plan.loop.block_id)
                phase = f"((chain_loop_index - chain_loop_begin) // {step}) & 1"
                lines.extend(
                    chained_tcgen_stage.emit_stage(
                        cg,
                        plan,
                        boundaries,
                        stage,
                        geometry,
                        phase,
                        group,
                        vector_staging,
                        producer_unroll,
                        seed_tiling=seed_tiling,
                    )
                )
                continue
            prefix = f"chain_{stage}"
            dtype = env.backend.dtype_str(plan.operand_dtype(stage))
            # Keep cooperative staging at the configured CTA width. A narrow
            # contraction may need fewer MMA warps, without serializing every
            # other contraction and all global-memory copies in this root.
            stage_threads = 32 * min(
                plan.threads // 32, 2 ** (columns.bit_length() - 4)
            )
            from .chained_root_warp_stage import emit_original_warp_operands
            from .chained_root_warp_stage import root_warp_stage_action

            warp_action = warp_sequence or root_warp_stage_action(
                cg,
                plan,
                boundaries,
                schedule,
                padding,
                dtype,
                general_collectives=general_collectives,
            )
            if warp_action is not None:
                from .chained_body_program import bind_root_warp_body
                from .chained_body_program import emit_body_program

                prepared_warp_accepted = True
                warp_body = bind_root_warp_body(
                    cg, plan, warp_action, boundaries, staged_inputs, scratch, lines
                )
                lines = emit_body_program(
                    cg,
                    plan,
                    None,
                    warp_body.execution,
                    None,
                    None,
                    scratch,
                    root_warp=warp_body,
                )
                warp_body.validate_return(lines)
                break

            operands = emit_original_warp_operands(
                cg,
                plan,
                boundaries,
                scan_inputs,
                bridges,
                staged_inputs,
                stage,
                schedule,
                padding,
                dtype,
            )
            lines.extend(operands.lines)
            inner_axes = dict(operands.axes)
            async_stage = operands.asynchronous
            if async_stage:
                lines.extend(
                    [
                        "cute.arch.cp_async_commit_group()",
                        "cute.arch.cp_async_wait_group(0)",
                    ]
                )
            accumulator = node.args[2]
            seed = [f"{prefix}_acc.fill(0.0)"]
            if isinstance(accumulator, Node):
                coords = (f"{prefix}_seed_row", f"{prefix}_seed_col")
                seed_expression = _Expression(cg, plan, boundaries)
                seed_expression.scan_inputs = scan_inputs
                seed_expression.coordinate_names.update(coords)
                seed_value = seed_expression.value(accumulator, coords)
                seed = [
                    f"{prefix}_seed_coords = {prefix}_thr.partition_C(cute.make_identity_tensor(({rows}, {columns})))",
                    f"for {prefix}_seed_index in cutlass.range_constexpr(cute.size({prefix}_acc)):",
                    f"    {coords[0]}, {coords[1]} = {prefix}_seed_coords[{prefix}_seed_index]",
                    _indent(seed_expression.lines),
                    f"    {prefix}_acc[{prefix}_seed_index] = cutlass.Float32({seed_value})",
                ]
            compute = emit_warp_mma(
                prefix,
                dtype,
                (rows, columns, reduction),
                stage_threads,
                inner_axes,
                seed,
            )
            lines.append("cute.arch.sync_threads()")
            if stage + 1 not in bridges or bridges[stage + 1].keep_shared:
                (loop_setup if plan.loop is not None else lines).extend(
                    [
                        f"{prefix}_c_ptr = cute.arch.alloc_smem(cutlass.Float32, {rows * (columns + output_padding)}, alignment=128)",
                        f"{prefix}_c = cute.make_tensor({prefix}_c_ptr, {scratch.layout(f'{prefix}_c', (rows, columns), row_stride=columns + output_padding)})",
                    ]
                )
                compute.extend(
                    [
                        f"{prefix}_sc = {prefix}_thr.partition_C({prefix}_c)",
                        f"cute.autovec_copy({prefix}_acc, {prefix}_sc)",
                    ]
                )
            if stage_threads < plan.threads:
                # The predicate is warp-uniform. Cooperative staging and both
                # CTA barriers remain outside; active warps cover the whole C tile.
                lines.extend([f"if chain_thread < {stage_threads}:", _indent(compute)])
            else:
                lines.extend(compute)
            lines.append("cute.arch.sync_threads()")
            boundaries[node] = f"{prefix}_c"
        if warp_sequence is not None:
            warp_sequence.validate(cg, plan, boundaries, staged_inputs, lines)
        if general_collectives:
            lines.extend(
                chained_collectives.emit_collectives_before(
                    cg, plan, boundaries, len(plan.dots)
                )
            )
        if producer_unroll is not None:
            producer_unroll.validate()
        seed_tiling.validate()
        stores = plan.loop.region.stores if plan.loop is not None else (plan.store,)
        for index, store in enumerate(stores):
            lines.extend(
                _emit_store(
                    cg,
                    plan,
                    store,
                    boundaries,
                    staged_inputs,
                    scan_inputs,
                    "chain_store" if index == 0 else f"chain_store_{index}",
                )
            )
        if plan.loop is not None:
            from .chained_loop import advance_carries

            lines.extend(advance_carries(cg, plan, boundaries, scratch))
            step = df.resolved_block_size(plan.loop.block_id)
            lines = [
                *loop_setup,
                f"for chain_loop_index in cutlass.range(chain_loop_begin, chain_loop_end, {step}, unroll=1):",
                _indent(lines),
            ]
            for index, store in enumerate(plan.loop.final_stores):
                lines.extend(
                    _emit_store(
                        cg, plan, store, {}, [], [], f"chain_final_store_{index}"
                    )
                )
            if plan.strategy == "tcgen05_tmem":
                lines.extend(chained_tcgen_stage.free_stages())
        scratch.validate()
        vector_staging.validate()
        cache_layouts.validate()
    except _UnsupportedChain as error:
        if prepared_warp_accepted:
            raise exc.BackendUnsupported(
                "cute", f"accepted prepared warp stage changed: {error}"
            ) from error
        if cache_layouts.mode != "auto":
            raise exc.BackendUnsupported(
                "cute", f"unsupported chain with explicit cache layout: {error}"
            ) from error
        return False
    return _install_chained_body(cg, plan, lines)


def _install_chained_body(
    cg: GenerateAST, plan: ChainedMatmulPlan, lines: Sequence[str]
) -> bool:
    df = cg.device_function
    # The root is independent and uses the same flattened task count.  All
    # thread coordinates and per-element boundaries are owned by this body.
    df.preamble = []
    template = _GeneratedCodeTemplate("chain", tuple(plan.tensor_aliases), df.new_var)
    body = ast.parse(template.render("\n".join(lines))).body
    if df.config.get("cute_grid_work_order") is not None:
        from .work_order import UnsupportedWorkOrder
        from .work_order import discover_work_order
        from .work_order import emit_work_permutation
        from .work_order import requested_work_axis

        try:
            axis = requested_work_axis(cg)
            assert axis is not None
            ordering = discover_work_order(cg, axis)
            origins = {
                item: ast.Name(
                    id=template.render(f"chain_origin_{item}"), ctx=ast.Load()
                )
                for item in ordering.leaf_axes
            }
            permutation = emit_work_permutation(cg, ordering, origins)
            # These original origin assignments are the whole-root adapter's
            # genuine pre-effect boundary, before captures, carries and roles.
            cut = 2 + len(plan.axes)
            expected = {
                template.render(f"chain_origin_{item}")
                for item, _size, _block in plan.axes
            }
            actual = {
                target.id
                for statement in body[2:cut]
                if isinstance(statement, ast.Assign)
                for target in statement.targets
                if isinstance(target, ast.Name)
            }
            if actual != expected or df.cute_state.work_order_plan is not None:
                raise UnsupportedWorkOrder(
                    "original common-root coordinate boundary changed"
                )
            body[cut:cut] = [
                *permutation.statements,
                ast.fix_missing_locations(
                    ast.Assign(
                        targets=[ast.Name(id=origins[axis].id, ctx=ast.Store())],
                        value=permutation.selected,
                    )
                ),
            ]
            df.cute_state.work_order_plan = ordering
        except UnsupportedWorkOrder as error:
            raise exc.BackendUnsupported("cute", str(error)) from error
    aliases = ast.parse(
        "\n".join(f"{alias} = {name}" for name, alias in plan.tensor_aliases.items())
    ).body
    df.body = [*aliases, *body]
    from .native_matmul_metadata import commit_body

    commit_body(cg, plan, lines)
    return True
