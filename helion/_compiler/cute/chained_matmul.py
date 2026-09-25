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

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import scan_ops
from ...language import tile_index
from ...language import tile_ops
from ...language import view_ops
from ...language.matmul_ops import dot
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from ..inductor_lowering import PointwiseLowering
from .cute_reshape import _get_tile_shape
from .fragment_epilogue import _has_fresh_output_allocation
from .fx_matcher import _GeneratedCodeTemplate
from .mma_support import get_cute_mma_support
from .tcgen05_config import CuteTcgen05Config

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST


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
        and isinstance(node.args[0], Node)
        and node.args[0].target is _tracing_ops._host_tensor
    }
    return {
        node
        for node in _ancestors(cast("Node", scan.args[1]))
        if node.target is memory_ops.load
        and node.args[0] in sources
        and len(_shape(node)) == 1
        and node.meta["val"].dtype in (torch.float16, torch.bfloat16, torch.float32)
    }


def _supported(node: Node) -> bool:
    if node.op == "output":
        return True
    if node.op != "call_function":
        return False
    if node.target in {
        _tracing_ops._host_tensor,
        _tracing_ops._get_symnode,
        tile_ops.tile_begin,
        tile_index,
        torch.ops.prims.iota.default,
        torch.ops.aten.scalar_tensor.default,
        torch.ops.aten.where.self,
        dot,
        scan_ops._associative_scan,
        *_VIEWS,
        *_SCALAR_BINARY,
    }:
        return True
    if node.target in (memory_ops.load, memory_ops.store):
        mask_index = 2 if node.target is memory_ops.load else 3
        return len(node.args) <= mask_index or node.args[mask_index] is None
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

    if len(node.args) != 5 or node.args[2:] != (0, False, False):
        return False
    source = node.args[1]
    if not isinstance(source, Node) or source.meta["val"].ndim != 1:
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
        and not any(n.target is dot for n in _ancestors(source))
    )


@dataclasses.dataclass(frozen=True)
class _ChainedGraph:
    root: GraphInfo
    nodes: tuple[Node, ...]
    dots: tuple[Node, ...]
    scans: tuple[Node, ...]
    store: Node
    exports: None


def _classify_chained_graph(graphs: Sequence[GraphInfo]) -> _ChainedGraph | None:
    """Shared structural admission; physical layout/config proofs come later.

    Recompute for the graph supplied by each caller. Discovery and lowering
    can observe different graph revisions, so these facts are not cached.
    """
    root = _root_graph(graphs)
    if root is None:
        return None
    nodes = tuple(root.graph.nodes)
    dots = tuple(node for node in nodes if node.target is dot)
    scans = tuple(node for node in nodes if node.target is scan_ops._associative_scan)
    stores = tuple(node for node in nodes if node.target is memory_ops.store)

    allowed = frozenset()
    if (
        not dots
        or (len(stores) != 1)
        or not all(_supported(node) or node in allowed for node in nodes)
    ):
        return None
    if not all(map(_additive_scan, scans)):
        return None
    store = stores[0]
    if not isinstance(store.args[0], Node) or not _has_fresh_output_allocation(
        store.args[0]
    ):
        return None
    if not isinstance(store.args[2], Node) or store.args[2].meta["val"].ndim != 2:
        return None
    ancestors = _ancestors(store.args[2])
    for node in dots:
        if len(node.args) != 4 or node.args[2:] != (None, None):
            return None
        if any(not isinstance(arg, Node) for arg in node.args[:2]):
            return None
        lhs, rhs = (arg.meta["val"] for arg in node.args[:2])
        if lhs.ndim != 2 or rhs.ndim != 2 or lhs.dtype != rhs.dtype:
            return None
        if lhs.dtype not in (torch.bfloat16, torch.float16):
            return None
        if node not in ancestors:
            return None
    if (
        len(dots) == 1
        and all(_direct_operand(cast("Node", arg)) for arg in dots[0].args[:2])
        and _ordinary_mma_supported(dots[0])
    ):
        return None
    return _ChainedGraph(root, nodes, dots, scans, store, None)


def detect_chained_matmul_search(graphs: Sequence[GraphInfo]) -> bool:
    """Config-independent admission for the computed-contraction search family."""
    return _classify_chained_graph(graphs) is not None


def plan_chained_matmul(graphs: Sequence[GraphInfo]) -> ChainedMatmulPlan | None:
    from ..device_function import DeviceFunction
    from ..host_function import HostFunction

    env = CompileEnvironment.current()
    df = DeviceFunction.current()
    if env.backend.name != "cute" or df.config.pid_type != "flat":
        return None
    support = get_cute_mma_support()
    if not (support.warp_f16bf16):
        return None
    graph = _classify_chained_graph(graphs)
    if graph is None:
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
            or lhs.meta["val"].dtype != dtype
            or rhs.meta["val"].dtype != dtype
            or node.meta["val"].dtype != torch.float32
        ):
            return None
        m, k, n = *left, right[1]
        if m % 16 or n % 8 or k % 16 or min(m, n, k) <= 0:
            return None
        shapes.append((m, n, k))
    # Bound the resident workspace; larger contractions use other schedules.
    cache_inputs = (
        df.config.config.get("cute_chained_mma_schedule")
        == "cp_async_register_reuse_scan"
    )
    shared_allocations = [
        2 * max(m * k + 8 * max(m, k) for m, _, k in shapes),
        2 * max(n * k + 8 * max(n, k) for _, n, k in shapes),
        *(4 * m * (n + 4) for m, n, _ in shapes),
        *(4 * (_shape(scan)[0] + min(df.config.num_warps, 8)) for scan in scans),
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
    if smem_bytes > CuteTcgen05Config.per_cta_smem_capacity_bytes(first_input.device):
        return None
    output = store.args[0]
    if not isinstance(output, Node) or output.target is not _tracing_ops._host_tensor:
        return None
    output_value = output.meta["val"]
    if output_value.dtype not in (dtype, torch.float32):
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
    return ChainedMatmulPlan(
        graph.root.graph_id,
        dots,
        store,
        tuple(axes),
        tuple(shapes),
        dtype,
        32 * min(df.config.num_warps, 8),
        scans,
    )


def _names(node: ast.AST) -> frozenset[str]:
    return frozenset(item.id for item in ast.walk(node) if isinstance(item, ast.Name))


@dataclasses.dataclass
class _Statement:
    code: str
    target: str | None
    inputs: frozenset[str]


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
        self.boundaries = boundaries
        self.statements: list[_Statement] = []
        self.memo: dict[tuple[Node, tuple[str, ...]], str] = {}
        self.definitions: dict[str, str] = {}
        self.definition_inputs: dict[str, frozenset[str]] = {}
        self.accesses: list[tuple[list[str], tuple[int, ...]]] = []
        self.global_accesses: list[tuple[Node, list[str]]] = []
        self.loaded_inputs: list[tuple[Node, tuple[str, ...], list[str], str]] = []
        self.fragments: dict[Node, tuple[tuple[str, ...], str]] = {}
        self.staged_inputs: list[_StagedInput] = []
        self.scan_inputs: list[_ScanInput] = []
        self.coordinate_names: set[str] = set()
        self.origins = {
            axis_id: f"chain_origin_{axis_id}" for axis_id, _, _ in plan.axes
        }

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
            _, extent, block = next(a for a in self.plan.axes if a[0] == axis)
            bounds = (0, max(0, (extent - 1) // block * block))
        elif isinstance(arg, Node) and arg.target is _tracing_ops._get_symnode:
            axis = self.block_id(arg)
            block = next(block for i, _, block in self.plan.axes if i == axis)
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
        if arg.target is tile_ops.tile_begin:
            return self.origins[self.block_id(cast("Node", arg.args[0]))]
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
        if arg.target is _tracing_ops._get_symnode:
            axis = self.block_id(arg)
            return str({i: b for i, _, b in self.plan.axes}[axis])
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
            if index.target is _tracing_ops._get_symnode:
                assert coordinate is not None
                return f"({self.origins[self.block_id(index)]} + {coordinate})"
            if isinstance(index.meta.get("val"), torch.Tensor) and _shape(index):
                assert coordinate is not None
                return self.value(index, (coordinate,))
        return self.scalar(index)

    def indices(self, node: Node, coordinates: tuple[str, ...]) -> list[str]:
        result: list[str] = []
        dim = 0
        for index in cast("Sequence[object]", node.args[1]):
            vector = index == slice(None) or (
                isinstance(index, Node)
                and (
                    index.target is _tracing_ops._get_symnode
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
            return self.value(
                source,
                tuple(
                    c
                    for c, s in zip(coordinates, selectors, strict=True)
                    if s is not None
                ),
            )
        indices = self.indices(node, coordinates)
        if source.target is not _tracing_ops._host_tensor:
            return self.value(source, tuple(indices))
        tensor = source.meta["val"]
        if any(not isinstance(size, int) for size in tensor.shape):
            raise _UnsupportedChain("dynamic host shapes")
        self.accesses.append((indices, tuple(tensor.stride())))
        self.global_accesses.append((source, indices))
        name = self.tensor_name(source)
        bounds = [
            f"0 <= ({index}) < {size}"
            for index, size in zip(indices, tensor.shape, strict=True)
        ]
        value = _cute_scalar_load_expr(name, indices, tensor.dtype)
        dtype = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
        fallback = f"({value} if {' and '.join(bounds)} else {dtype}(0))"
        result = self.bind(
            _scan_input_value(
                self,
                source,
                indices,
                _staged_input_value(self, source, indices, fallback),
            )
        )
        self.loaded_inputs.append((node, coordinates, indices, result))
        return result

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

    def value(self, node: Node, coordinates: tuple[str, ...]) -> str:
        from ..aten_lowering import LoweringContext

        key = (node, coordinates)
        if key in self.memo:
            return self.memo[key]
        shape = _shape(node)
        if len(coordinates) != len(shape):
            raise _UnsupportedChain(f"coordinate rank at {node.name}")
        if node in self.fragments:
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
            bounds = " and ".join(
                f"0 <= ({coordinate}) < {size}"
                for coordinate, size in zip(coordinates, shape, strict=True)
            )
            dtype = CompileEnvironment.current().backend.dtype_str(
                node.meta["val"].dtype
            )
            value = f"({self.boundaries[node]}[{', '.join(coordinates)}] if {bounds} else {dtype}(0))"
        elif node.target is memory_ops.load:
            value = self._load(node, coordinates)
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
            if _is_floor_divide(node):
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
        self.bounds: list[str] = []
        self.coordinate_extents = dict(zip(coordinates, _shape(node), strict=True))

    def _load(self, node: Node, coordinates: tuple[str, ...]) -> str:
        source = cast("Node", node.args[0])
        if source.target is _tracing_ops._host_tensor:
            self.indices(node, coordinates)
            return "0"
        return super()._load(node, coordinates)

    def value(self, node: Node, coordinates: tuple[str, ...]) -> str:
        self.bounds.extend(_shape_domain(node, coordinates, self.plan))
        if node.target is torch.ops.prims.iota.default:
            length = node.args[0]
            if not isinstance(length, int):
                raise _UnsupportedChain("symbolic iota domain")
            extent = self.coordinate_extents.get(coordinates[0])
            if coordinates[0] != "0" and (extent is None or length < extent):
                self.bounds.append(f"0 <= ({coordinates[0]}) < {length}")
        if node.target is dot:
            self.value(cast("Node", node.args[0]), (coordinates[0], "0"))
            self.value(cast("Node", node.args[1]), ("0", coordinates[1]))
            return "0"
        if node.target is scan_ops._associative_scan:
            return self.value(cast("Node", node.args[1]), coordinates)
        return super().value(node, coordinates)


def _operand_domain(
    cg: GenerateAST,
    node: Node,
    coordinates: tuple[str, ...],
    plan: ChainedMatmulPlan,
) -> list[str]:
    expression = _DomainExpression(cg, plan, node, coordinates)
    expression.value(node, coordinates)
    return list(dict.fromkeys(expression.bounds))


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
    if coordinates != (scan_index,):
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
    expression = _Expression(cg, plan, {})
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
) -> int:
    """Choose cooperative order from the operand's global-memory accesses."""
    expression = _Expression(cg, plan, boundaries)
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
    result: dict[int, _RegisterBridge] = {}
    all_boundaries = {
        **boundaries,
        **{node: f"chain_{i}_c" for i, node in enumerate(plan.dots)},
    }
    for stage in range(1, len(plan.dots)):
        producer, consumer = plan.dots[stage - 1 : stage + 1]
        producer_shape = plan.shapes[stage - 1][:2]
        if producer_shape[1] < plan.threads // 4:
            continue
        for role, operand in zip(("a", "b"), consumer.args[:2], strict=True):
            assert isinstance(operand, Node)
            other = cast("Node", consumer.args[1 if role == "a" else 0])
            if producer in _ancestors(other):
                continue
            logical_shape = _shape(operand)
            stored_shape = logical_shape if role == "a" else logical_shape[::-1]
            if stored_shape != producer_shape:
                continue
            allowed = _ancestors(operand)
            if producer not in allowed:
                continue
            pending, visited, exclusive = [producer], set(), True
            while pending:
                current = pending.pop()
                if current in visited:
                    continue
                visited.add(current)
                permitted = {consumer} if current is operand else allowed
                if any(user not in permitted for user in current.users):
                    exclusive = False
                    break
                if current is not operand:
                    pending.extend(current.users)
            if not exclusive:
                continue
            prefix, previous = f"chain_{stage}_{role}_bridge", f"chain_{stage - 1}"
            coords = (f"{prefix}_row", f"{prefix}_col")
            expression = _Expression(cg, plan, all_boundaries)
            expression.scan_inputs = scan_inputs
            expression.coordinate_names.update(coords)
            expression.fragments[producer] = (coords, f"{previous}_acc[{prefix}_index]")
            try:
                value = expression.value(
                    operand, coords if role == "a" else coords[::-1]
                )
                domain = _operand_domain(
                    cg, operand, coords if role == "a" else coords[::-1], plan
                )
            except _UnsupportedChain:
                continue
            result[stage] = _RegisterBridge(
                role,
                (
                    f"{prefix}_coords = {previous}_thr.partition_C(cute.make_identity_tensor({producer_shape!r}))",
                    f"{prefix}_values = cute.make_rmem_tensor({previous}_acc.shape, {dtype})",
                    f"for {prefix}_index in cutlass.range_constexpr(cute.size({prefix}_values)):",
                    f"    {coords[0]}, {coords[1]} = {prefix}_coords[{prefix}_index]",
                    _indent(expression.lines),
                    f"    {prefix}_values[{prefix}_index] = {_masked_operand(value, dtype, domain)}",
                    f"cute.autovec_copy({prefix}_values, {previous}_thr.partition_C(chain_{stage}_{role}))",
                ),
            )
            break
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


def codegen_chained_matmul(cg: GenerateAST) -> bool:
    """Emit a contraction DAG, preserving its pointwise regions and store."""
    df = cg.device_function
    plan = df.cute_state.chained_matmul_plan
    root = cg.current_root_graph_info
    if plan is None or root is None or root.graph_id != plan.root_graph_id:
        return False
    dtype = "cutlass.BFloat16" if plan.dtype is torch.bfloat16 else "cutlass.Float16"
    schedule = df.config.config.get("cute_chained_mma_schedule", "coalesced")
    padding = 0 if schedule == "k_major" else 8
    output_padding = padding // 2
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
    m, n = _shape(cast("Node", plan.store.args[2]))
    boundaries: dict[Node, str] = {}
    try:
        scan_inputs: list[_ScanInput] = []
        lines.extend(
            _codegen_scans(
                cg,
                plan,
                boundaries,
                scan_inputs,
                schedule == "cp_async_register_reuse_scan",
            )
        )
        bridges = (
            _register_bridges(cg, plan, boundaries, dtype, scan_inputs)
            if schedule
            in (
                "cp_async_register",
                "cp_async_register_reuse",
                "cp_async_register_reuse_scan",
            )
            else {}
        )
        a_workspace = max(m * k + 8 * max(m, k) for m, _, k in plan.shapes)
        b_workspace = max(n * k + 8 * max(n, k) for _, n, k in plan.shapes)
        lines.extend(
            [
                f"chain_a_workspace = cute.arch.alloc_smem({dtype}, {a_workspace}, alignment=128)",
                f"chain_b_workspace = cute.arch.alloc_smem({dtype}, {b_workspace}, alignment=128)",
            ]
        )
        staged_inputs: list[_StagedInput] = []
        for stage, (node, (rows, columns, reduction)) in enumerate(
            zip(plan.dots, plan.shapes, strict=True)
        ):
            prefix = f"chain_{stage}"
            # Keep cooperative staging at the configured CTA width. A narrow
            # contraction may need fewer MMA warps, without serializing every
            # other contraction and all global-memory copies in this root.
            stage_threads = 32 * min(
                plan.threads // 32, 2 ** (columns.bit_length() - 4)
            )
            inner_axes: dict[str, int] = {}
            async_stage = False
            for role, shape in (("a", (rows, reduction)), ("b", (columns, reduction))):
                operand = cast("Node", node.args[0 if role == "a" else 1])
                inner = _operand_inner_axis(cg, plan, boundaries, operand)
                if role == "b":
                    inner = 1 - inner
                if schedule in ("k_major", "k_major_padded"):
                    inner = 1
                inner_axes[role] = inner
                if inner == 1:
                    x, y = f"{prefix}_load // {shape[1]}", f"{prefix}_load % {shape[1]}"
                    stride = (shape[1] + padding, 1)
                else:
                    x, y = f"{prefix}_load % {shape[0]}", f"{prefix}_load // {shape[0]}"
                    stride = (1, shape[0] + padding)
                coords = (x, y) if role == "a" else (y, x)
                expression = _Expression(cg, plan, boundaries)
                expression.scan_inputs = scan_inputs
                expression.coordinate_names.add(f"{prefix}_load")
                value = expression.value(operand, coords)
                domain = _operand_domain(cg, operand, coords, plan)
                vector = 1 if schedule == "k_major" else min(8, shape[inner])
                steps = (math.prod(shape) + plan.threads * vector - 1) // (
                    plan.threads * vector
                )
                load_range = (
                    f"cutlass.range({steps}, unroll=1)"
                    if schedule
                    in (
                        "coalesced",
                        "cp_async",
                        "cp_async_register",
                        "cp_async_register_reuse",
                        "cp_async_register_reuse_scan",
                    )
                    else f"cutlass.range_constexpr({steps})"
                )
                lines.extend(
                    [
                        f"{prefix}_{role}_ptr = chain_{role}_workspace",
                        f"{prefix}_{role} = cute.make_tensor({prefix}_{role}_ptr, cute.make_layout({shape!r}, stride={stride!r}))",
                    ]
                )
                if (
                    schedule
                    in ("cp_async_register_reuse", "cp_async_register_reuse_scan")
                    and stage == len(plan.dots) - 1
                ):
                    staged = _stage_input(
                        cg, plan, operand, f"{prefix}_{role}", role, shape
                    )
                    if staged is not None:
                        staged_inputs.append(staged)
                bridge = bridges.get(stage)
                if bridge is not None and bridge.role == role:
                    lines.extend(bridge.lines)
                    continue
                fallback = [
                    f"for {prefix}_load_step in {load_range}:",
                    f"    for {prefix}_load_vec in cutlass.range_constexpr({vector}):",
                    f"        {prefix}_load = chain_thread * {vector} + {prefix}_load_step * {plan.threads * vector} + {prefix}_load_vec",
                    f"        if {prefix}_load < {math.prod(shape)}:",
                    _indent(expression.lines, 12),
                    f"            {prefix}_{role}[{x}, {y}] = {_masked_operand(value, dtype, domain)}",
                ]
                async_lines = (
                    _async_copy(
                        cg,
                        plan,
                        operand,
                        prefix,
                        role,
                        shape,
                        stride,
                        inner,
                        dtype,
                        fallback,
                    )
                    if schedule
                    in (
                        "cp_async",
                        "cp_async_register",
                        "cp_async_register_reuse",
                        "cp_async_register_reuse_scan",
                    )
                    else None
                )
                async_stage |= async_lines is not None
                lines.extend(fallback if async_lines is None else async_lines)
            if async_stage:
                lines.extend(
                    [
                        "cute.arch.cp_async_commit_group()",
                        "cute.arch.cp_async_wait_group(0)",
                    ]
                )
            compute = [
                f"{prefix}_mma = cute.make_tiled_mma(cute.make_mma_atom(cute.nvgpu.warp.MmaF16BF16Op({dtype}, cutlass.Float32, (16, 8, 16))), atom_layout_mnk=(1, {stage_threads // 32}, 1))",
                f"{prefix}_thr = {prefix}_mma.get_slice(chain_thread)",
                f"{prefix}_sa = {prefix}_thr.partition_A({prefix}_a)",
                f"{prefix}_sb = {prefix}_thr.partition_B({prefix}_b)",
                f"{prefix}_ra = {prefix}_mma.make_fragment_A((cute.shape({prefix}_sa)[0], cute.shape({prefix}_sa)[1], 1))",
                f"{prefix}_rb = {prefix}_mma.make_fragment_B((cute.shape({prefix}_sb)[0], cute.shape({prefix}_sb)[1], 1))",
                f"{prefix}_copy_a = cute.make_tiled_copy_A(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose={inner_axes['a'] == 0!r}, num_matrices=4), {dtype}), {prefix}_mma)",
                f"{prefix}_copy_b = cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose={inner_axes['b'] == 0!r}, num_matrices=2), {dtype}), {prefix}_mma)",
                f"{prefix}_copy_thr_a = {prefix}_copy_a.get_slice(chain_thread)",
                f"{prefix}_copy_thr_b = {prefix}_copy_b.get_slice(chain_thread)",
                f"{prefix}_copy_sa = {prefix}_copy_thr_a.partition_S({prefix}_a)",
                f"{prefix}_copy_sb = {prefix}_copy_thr_b.partition_S({prefix}_b)",
                f"{prefix}_copy_ra = {prefix}_copy_thr_a.retile({prefix}_ra)",
                f"{prefix}_copy_rb = {prefix}_copy_thr_b.retile({prefix}_rb)",
                f"{prefix}_acc = cute.make_rmem_tensor({prefix}_mma.partition_shape_C(({rows}, {columns})), cutlass.Float32)",
                f"{prefix}_acc.fill(0.0)",
                f"for {prefix}_kk in cutlass.range_constexpr(cute.size({prefix}_sa, mode=[2])):",
                f"    cute.copy({prefix}_copy_a, {prefix}_copy_sa[None, None, {prefix}_kk], {prefix}_copy_ra[None, None, 0])",
                f"    cute.copy({prefix}_copy_b, {prefix}_copy_sb[None, None, {prefix}_kk], {prefix}_copy_rb[None, None, 0])",
                f"    cute.gemm({prefix}_mma, {prefix}_acc, {prefix}_ra[None, None, 0], {prefix}_rb[None, None, 0], {prefix}_acc)",
            ]
            lines.append("cute.arch.sync_threads()")
            if stage + 1 not in bridges:
                lines.extend(
                    [
                        f"{prefix}_c_ptr = cute.arch.alloc_smem(cutlass.Float32, {rows * (columns + output_padding)}, alignment=128)",
                        f"{prefix}_c = cute.make_tensor({prefix}_c_ptr, cute.make_layout(({rows}, {columns}), stride=({columns + output_padding}, 1)))",
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
        expression = _Expression(cg, plan, boundaries)
        expression.staged_inputs = staged_inputs
        expression.scan_inputs = scan_inputs
        expression.coordinate_names.add("chain_store")
        coords = (f"chain_store // {n}", f"chain_store % {n}")
        value = expression.value(cast("Node", plan.store.args[2]), coords)
        indices = expression.indices(plan.store, coords)
        output = cast("Node", plan.store.args[0])
        fake = output.meta["val"]
        name = expression.tensor_name(output)
        bounds = " and ".join(
            f"0 <= ({i}) < {s}" for i, s in zip(indices, fake.shape, strict=True)
        )
        lines.extend(
            [
                f"for chain_store_step in cutlass.range_constexpr({(m * n + plan.threads - 1) // plan.threads}):",
                f"    chain_store = chain_thread + chain_store_step * {plan.threads}",
                f"    if chain_store < {m * n}:",
                _indent(expression.lines, 8),
                f"        if {bounds}:",
                f"            {name}[{', '.join(indices)}] = {value}",
            ]
        )
    except _UnsupportedChain:
        return False
    # The root is independent and uses the same flattened task count.  All
    # thread coordinates and per-element boundaries are owned by this body.
    df.preamble = []
    template = _GeneratedCodeTemplate("chain", tuple(plan.tensor_aliases), df.new_var)
    body = ast.parse(template.render("\n".join(lines))).body
    aliases = ast.parse(
        "\n".join(f"{alias} = {name}" for name, alias in plan.tensor_aliases.items())
    ).body
    df.body = [*aliases, *body]
    cg.cute_uses_matmul = True
    return True
