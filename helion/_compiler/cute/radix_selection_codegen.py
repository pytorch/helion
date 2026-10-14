"""Generate bounded radix selection and the proved scalar consumer DAG."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import NoReturn
from typing import cast

import torch

from ... import exc
from ...language import _tracing_ops
from ...language import memory_ops
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..program_id import XYZProgramIDs
from ..tile_strategy import DeviceGridState
from .memory_ops import cute_tensor_base_is_aligned
from .radix_selection import RadixSelectionGraph
from .radix_selection import RadixSelectionPlan
from .radix_selection import radix_packet_base_is_aligned
from .radix_selection import strip_alias
from .radix_selection import tensor_value
from .row_fragment import RowFragmentEmitter
from .row_fragment import RowFragmentLayout
from .row_fragment import row_fragment_arange
from .row_fragment import row_fragment_tensor_inputs

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from torch.fx import Node

    from ..generate_ast import GenerateAST


@dataclass(frozen=True)
class _Format:
    width: int
    cute_name: str
    infinity: int


_FORMATS = {
    torch.float16: _Format(16, "Float16", 0x7C00),
    torch.bfloat16: _Format(16, "BFloat16", 0x7F80),
    torch.float32: _Format(32, "Float32", 0x7F800000),
}


def _selected_ordinal(graph: RadixSelectionGraph, node: object) -> bool:
    node = strip_alias(node)
    if node in graph.index_getters:
        return True
    if not isinstance(node, torch.fx.Node):
        return False
    if node.target in (
        torch.ops.aten.clone.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.constant_pad_nd.default,
    ):
        return _selected_ordinal(graph, node.args[0])
    if node.target is torch.ops.prims.convert_element_type.default:
        original, value = tensor_value(node.args[0]), tensor_value(node)
        return (
            original is not None
            and value is not None
            and original.dtype in (torch.int32, torch.int64)
            and value.dtype in (torch.int32, torch.int64)
            and _selected_ordinal(graph, node.args[0])
        )
    return False


def has_exact_cached_payload_consumer(graph: RadixSelectionGraph) -> bool:
    if graph.value_getters:
        return True
    for load in graph.loads:
        value = tensor_value(load)
        indices = load.args[1]
        if (
            load is not graph.source_load
            and load.args[0] is graph.x_node
            and value is not None
            and value.dtype == graph.x.dtype
            and isinstance(indices, (tuple, list))
            and len(indices) == 2
            and _selected_ordinal(graph, indices[1])
        ):
            return True
    return False


@dataclass(frozen=True)
class _RadixCodegenSpec:
    plan: RadixSelectionPlan
    input_alignment: int

    @property
    def recipe(self) -> str:
        return "float_rank"

    @property
    def dtype(self) -> torch.dtype:
        return self.plan.graph.x.dtype

    @property
    def threads(self) -> int:
        return self.plan.threads

    @property
    def cluster(self) -> int:
        return self.plan.cluster

    @property
    def packet(self) -> int:
        return self.plan.packet

    @property
    def cta_cols(self) -> int:
        return self.plan.cta_cols

    @property
    def cols(self) -> int:
        return self.plan.graph.cols

    @property
    def logical_cols(self) -> int:
        return self.plan.graph.logical_cols

    @property
    def k(self) -> int:
        return self.plan.graph.k

    @property
    def largest(self) -> bool:
        return self.plan.graph.largest

    @property
    def candidate_capacity(self) -> int:
        return self.plan.candidate_capacity

    @property
    def primary_width(self) -> int:
        return self.plan.primary_width

    @property
    def cache_keys_enabled(self) -> bool:
        return self.plan.cache_keys_enabled

    @property
    def cached_key_scalar(self) -> str:
        return "cutlass.Uint16" if self.primary_width == 16 else "cutlass.Uint32"

    @property
    def scalar_candidates(self) -> bool:
        return self.plan.scalar_candidates

    @property
    def valid_domain(self) -> bool:
        return self.plan.valid_domain

    @property
    def ticket_stores(self) -> bool:
        return self.plan.ticket_stores

    @property
    def packet_bits(self) -> int:
        return self.plan.packet_bits

    @property
    def scan_unroll(self) -> int:
        return self.plan.scan_unroll

    @property
    def cached_payload_inverse(self) -> bool:
        return self.plan.cached_payload_inverse

    @property
    def column_scalar(self) -> str:
        return (
            "cutlass.Int32"
            if self.plan.column_dtype == torch.int32
            else "cutlass.Int64"
        )


class _ScalarConsumer:
    """One lexical evaluation; loads and dtype boundaries remain explicit."""

    def __init__(
        self,
        cg: GenerateAST,
        plan: RadixSelectionPlan,
        *,
        row: str,
        length: str,
        column: str,
        slot: str,
        raw_value: str | None = None,
        known_valid: bool = False,
    ) -> None:
        self.cg = cg
        self.fn = cg.device_function
        self.backend = CompileEnvironment.current().backend
        self.graph = plan.graph
        self.row = row
        self.length = length
        self.column = column
        self.slot = slot
        self.raw_value = raw_value
        self.known_valid = known_valid
        self.ticket_in_range = plan.ticket_stores and known_valid
        self.memo: dict[tuple[Node, str], ast.AST] = {}
        self.pointwise = RowFragmentEmitter(
            cg,
            lambda node: self._unexpected_fragment(node),
            layout=RowFragmentLayout(32, 1, "0"),
            valid_row="True",
            row_expr=row,
        )

    def _unexpected_fragment(self, node: Node) -> NoReturn:
        raise exc.InvalidConfig(f"radix scalar consumer requested a fragment: {node}")

    def cast(self, value: ast.AST, dtype: torch.dtype) -> ast.AST:
        return self.backend.cast_ast(value, dtype)

    def emit(self, source: str) -> None:
        self.cg.add_statement(statement_from_string(source))

    def selected_ordinal(self, node: object) -> bool:
        """Identity at a selected slot, whose original ordinal fits Int32."""
        return _selected_ordinal(self.graph, node)

    def prefix_predicate(self, node: object) -> bool:
        """The retained consumer mask implies the packet's raw load guard."""
        node = strip_alias(node)
        if not isinstance(node, torch.fx.Node):
            return False
        if node.target is torch.ops.aten.bitwise_and.Tensor:
            return any(self.prefix_predicate(argument) for argument in node.args)
        return (
            node.target is torch.ops.aten.lt.Tensor
            and len(node.args) == 2
            and self.selected_ordinal(node.args[0])
            and strip_alias(node.args[1]) is self.graph.length
        )

    def proved_true(self, node: Node) -> bool:
        if not self.known_valid or len(node.args) != 2:
            return False
        left, right = node.args
        original_column = strip_alias(
            left
        ) is self.graph.columns or self.selected_ordinal(left)
        if node.target is torch.ops.aten.lt.Tensor:
            return original_column and strip_alias(right) is self.graph.length
        if node.target is torch.ops.aten.lt.Scalar and type(right) is int:
            if original_column:
                return right >= self.graph.cols
            if isinstance(left, torch.fx.Node):
                progression = row_fragment_arange(left)
                return (
                    left not in self.graph.source_nodes
                    and progression is not None
                    and progression[:2] == (0, 1)
                    and right >= self.graph.k
                )
        return False

    def read(self, node: Node, coordinate: str) -> ast.AST:
        value = tensor_value(node)
        assert value is not None
        domain = (
            (value.size(0) if value.ndim else 0)
            if node in self.graph.source_nodes
            else self.graph.slot_domains[node]
        )
        coord = "0" if domain in (0, 1) else coordinate
        key = (node, coord)
        if key in self.memo:
            return self.memo[key]
        if self.proved_true(node):
            result = expr_from_string("True")
        elif node is self.graph.length:
            result = expr_from_string(self.length)
        elif node in self.graph.index_getters:
            result = expr_from_string(f"cutlass.Int64({self.column})")
        elif node in self.graph.value_getters:
            result = self.read(self.graph.source, self.column)
        elif node.target is memory_ops.load:
            result = self.load(node, coord)
        elif row_fragment_arange(node) is not None:
            result = expr_from_string(f"cutlass.Int64({coord})")
        elif node.target is torch.ops.aten.constant_pad_nd.default:
            source = cast("Node", node.args[0])
            source_value = cast("torch.Tensor", tensor_value(source))
            name = self.fn.new_var("radix_padded_value")
            other = self.cast(expr_from_string(repr(node.args[2])), value.dtype)
            self.cg.add_statement(
                statement_from_string(f"{name} = {{value}}", value=other)
            )
            branch = statement_from_string(
                f"if ({coord}) < {source_value.size(0)}:\n    pass"
            )
            assert isinstance(branch, ast.If)
            branch.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                self.cg.add_statement(
                    statement_from_string(
                        f"{name} = {{value}}", value=self.read(source, coord)
                    )
                )
            self.cg.add_statement(branch)
            result = expr_from_string(name)
        elif node.target is torch.ops.aten.view.dtype:
            operand = self.read(cast("Node", node.args[0]), coord)
            result = expr_from_string(
                f"({{value}}).bitcast({self.backend.dtype_str(value.dtype)})",
                value=operand,
            )
        elif node.target in (
            _tracing_ops._new_var,
            torch.ops.aten.alias.default,
            torch.ops.aten.clone.default,
            torch.ops.aten.detach.default,
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten._unsafe_view.default,
        ):
            result = self.read(cast("Node", node.args[0]), coord)
        else:
            inputs = [
                self.read(source, coord) for source in row_fragment_tensor_inputs(node)
            ]
            result = self.pointwise.emit_pointwise_scalar(
                node, inputs, logical_column=coord
            )
        result = self.cast(result, value.dtype)
        # The named cast retains every traced dtype boundary before consumers.
        named = self.cg.lift(result, prefix="radix_scalar")
        self.memo[key] = named
        return named

    def load(self, node: Node, coordinate: str) -> ast.AST:
        if (
            self.raw_value is not None
            and node is self.graph.source_load
            and (coordinate == self.column or self.graph.logical_cols == 1)
        ):
            # This exact root load was evaluated at the same original column
            # by the packet, including its typed zero for invalid positions.
            return expr_from_string(self.raw_value)
        target_node = cast("Node", node.args[0])
        target = cast("torch.Tensor", tensor_value(target_node))
        value = cast("torch.Tensor", tensor_value(node))
        name = self.fn.tensor_arg(target).name
        indices = cast("tuple[object, ...]", node.args[1])
        expressions = [self.row]
        if target.ndim == 2:
            expressions.append(
                ast.unparse(self.read(cast("Node", indices[1]), coordinate))
            )
        exact_selected_read = (
            target_node is self.graph.x_node
            and target.ndim == 2
            and value.dtype == self.graph.x.dtype
            and (
                self.selected_ordinal(indices[1])
                or node is self.graph.source_load
                and (coordinate == self.column or self.graph.logical_cols == 1)
            )
        )
        masks = [
            f"(({index}) >= 0) & (({index}) < {size})"
            for index, size in zip(expressions, target.shape, strict=True)
        ]
        if self.known_valid and exact_selected_read:
            # The long canonical prefix proves this exact original read is
            # within both row/column bounds. Different tensors, offset indices
            # and the short-row branch retain their automatic bounds.
            masks = []
        if node.args[2] is not None:
            masks.append(ast.unparse(self.read(cast("Node", node.args[2]), coordinate)))
        predicate = " & ".join(f"({part})" for part in masks) or "True"
        result = self.fn.new_var("radix_payload")
        zero = self.cast(expr_from_string("0"), value.dtype)
        self.cg.add_statement(
            statement_from_string(f"{result} = {{value}}", value=zero)
        )
        offset = " + ".join(
            f"cutlass.Int64({index}) * cutlass.Int64({stride})"
            for index, stride in zip(expressions, target.stride(), strict=True)
        )
        forward = (
            self.raw_value is not None
            and target_node is self.graph.x_node
            and target.ndim == 2
            and exact_selected_read
            and (self.known_valid or self.prefix_predicate(node.args[2]))
        )
        loaded = self.cast(
            expr_from_string(
                cast("str", self.raw_value)
                if forward
                else f"({name}.iterator + ({offset})).load()"
            ),
            value.dtype,
        )
        self.cg.add_statement(
            statement_from_string(
                f"if {predicate}:\n    {result} = {{value}}", value=loaded
            )
        )
        return expr_from_string(result)

    def store(self, node: Node) -> None:
        target = cast("torch.Tensor", tensor_value(node.args[0]))
        indices = cast("tuple[Node, Node]", node.args[1])
        domain = self.graph.slot_domains[indices[1]]
        branch = statement_from_string(
            f"if ({self.slot}) < {min(domain, target.size(1))}:\n    pass"
        )
        assert isinstance(branch, ast.If)
        branch.body.clear()
        with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
            self.memo = {}
            stored = self.cast(
                self.read(cast("Node", node.args[2]), self.slot), target.dtype
            )
            mask = (
                self.read(cast("Node", node.args[3]), self.slot)
                if node.args[3] is not None
                else expr_from_string("True")
            )
            name = self.fn.tensor_arg(target).name
            offset = (
                f"cutlass.Int64({self.row}) * cutlass.Int64({target.stride(0)})"
                f" + cutlass.Int64({self.slot}) * cutlass.Int64({target.stride(1)})"
            )
            self.cg.add_statement(
                statement_from_string(
                    f"if ({{mask}}):\n    ({name}.iterator + ({offset})).store({{value}})",
                    mask=mask,
                    value=stored,
                )
            )
        if self.ticket_in_range and min(domain, target.size(1)) >= self.graph.k:
            # Above tickets cover [0,K-R); admitted equal tickets cover
            # [K-R,K). Preserve smaller output domains and every traced mask.
            for statement in branch.body:
                self.cg.add_statement(statement)
        else:
            self.cg.add_statement(branch)


class _RenameCore(ast.NodeTransformer):
    def __init__(self, names: dict[str, str]) -> None:
        self.names = names

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.names:
            return ast.copy_location(
                ast.Name(id=self.names[node.id], ctx=node.ctx), node
            )
        return node


class _BindConsumers(ast.NodeTransformer):
    def __init__(
        self, cg: GenerateAST, plan: RadixSelectionPlan, names: dict[str, str]
    ) -> None:
        self.cg = cg
        self.plan = plan
        self.names = names

    def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.AST]:
        if not (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_helion_radix_consumer"
        ):
            return node
        assert len(node.value.args) in (2, 3)
        column, slot = map(ast.unparse, node.value.args[:2])
        raw_value = (
            ast.unparse(node.value.args[2]) if len(node.value.args) == 3 else None
        )
        known_valid = any(
            keyword.arg == "known_valid"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.value.keywords
        )
        consumer = _ScalarConsumer(
            self.cg,
            self.plan,
            row=self.names["row"],
            length=self.names["selection_length"],
            column=column,
            slot=slot,
            raw_value=raw_value,
            known_valid=known_valid,
        )
        statements: list[ast.AST] = []
        with self.cg.set_statements(statements):
            for store in self.plan.graph.stores:
                consumer.store(store)
        return statements


def codegen_radix_selection_root(cg: GenerateAST, plan: RadixSelectionPlan) -> bool:
    graph = plan.graph
    fn = cg.device_function
    root = cg.current_root_graph_info
    grid = cg.current_grid_state
    if (
        root is None
        or root.graph is not graph.root_graph
        or not isinstance(grid, DeviceGridState)
        or fn.pid is None
        or len(fn.pid.pid_info) != 1
        or fn.pid.pid_info[0].block_id != graph.row_block_id
    ):
        return False
    info = grid.block_id_to_info[graph.row_block_id]
    if info.begin_expr != 0 or info.end_expr != graph.rows:
        return False
    # Reserve every argument before allocating generated locals. User parameter
    # names may coincide with any of the core's implementation-local names.
    for node in graph.root_graph.nodes:
        if node.target is _tracing_ops._host_tensor:
            fn.tensor_arg(
                cast("torch.Tensor", tensor_value(node)),
                prefer_name=cast("str", node.args[0]),
            )
    # The complete body uses this divided row for every source and consumer.
    # The logical host grid remains rows; its physical cluster multiplier is C.
    fn.pid = XYZProgramIDs(pid_info=[fn.pid.pid_info[0]._replace(block_size_var="1")])
    env = CompileEnvironment.current()
    input_alignment = (
        32
        if plan.packet_bits == 256 and radix_packet_base_is_aligned(env, graph.x)
        else (16 if cute_tensor_base_is_aligned(env, graph.x, 16) else 1)
    )
    spec = _RadixCodegenSpec(plan, input_alignment)
    body = ast.parse(_core_source(spec)).body
    core_names = sorted(
        {
            node.id
            for statement in body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
    )
    names = {name: fn.new_var(f"radix_{name}") for name in core_names}
    transform = _RenameCore(names)
    body = [transform.visit(statement) for statement in body]
    argument_names = {
        "x": fn.tensor_arg(graph.x).name,
        "lengths": fn.tensor_arg(
            cast("torch.Tensor", tensor_value(graph.length.args[0]))
        ).name,
    }
    arguments = _RenameCore(argument_names)
    body = [arguments.visit(statement) for statement in body]
    bind = _BindConsumers(cg, plan, names)
    statements = []
    for statement in body:
        result = bind.visit(statement)
        statements.extend(result if isinstance(result, list) else [result])
    fn.preamble = []
    fn.body = statements
    fn.cute_state.cluster_shape = (plan.cluster, 1, 1) if plan.cluster > 1 else None
    fn.cute_state.launch_grid_multiplier = plan.cluster
    fn.cute_state.owned_root_block_dims = (plan.threads, 1, 1)
    return True


def _primary_key(
    w: _Writer, spec: _RadixCodegenSpec, value: str, column: str, name: str
) -> None:
    """An order/equality surrogate; observed values use the traced source DAG."""
    width = spec.primary_width
    mask = (1 << width) - 1
    _rank(w, spec, value, name + "_rank")
    w.line(
        f"{name} = (cutlass.Uint32({name}_rank) ^ cutlass.Uint32({1 << (width - 1)})) & cutlass.Uint32({mask})"
    )
    if not spec.valid_domain:
        with w.block(
            f"if not ({_boolean(f'{column} < {spec.cols}', f'{column} < selection_length')})"
        ):
            w.line(f"{name} = cutlass.Uint32(0)")
    if not spec.largest:
        w.line(f"{name} = {name} ^ cutlass.Uint32({mask})")


def _histogram_update(w: _Writer, spec: _RadixCodegenSpec) -> None:
    w.line("members = cute.arch.vote_ballot_sync(matched)")
    with w.block("if matched"):
        if spec.plan.histogram_mode == "direct":
            w.line(
                "cute.arch.atomic_add((histogram.iterator + digit).llvm_ptr, cutlass.Int32(1), sem='relaxed', scope='cta')"
            )
        else:
            w.line("peers = cute.arch.match_sync(members, digit)")
            w.line("peer_count = cutlass.Int32(cute.arch.popc(peers))")
            with w.block("if (peers & lane_mask) == cutlass.Uint32(0)"):
                w.line(
                    "cute.arch.atomic_add((histogram.iterator + digit).llvm_ptr, peer_count, sem='relaxed', scope='cta')"
                )


def _compact(w: _Writer, spec: _RadixCodegenSpec) -> None:
    w.line("primary_boundary = primary_prefix[0]")
    w.line("boundary_remaining = remaining[0]")
    w.line(f"global_above = cutlass.Int32({spec.k}) - boundary_remaining")
    w.line("cute.arch.sync_threads()")
    _scan_length(w, spec)
    w.line("use_candidates = cutlass.Boolean(False)")
    if spec.candidate_capacity:
        with w.block(f"if candidate_counter[0] <= {spec.candidate_capacity}"):
            w.line(f"scan_length = {spec.column_scalar}(candidate_counter[0])")
            w.line("use_candidates = cutlass.Boolean(True)")
    _scan_round_count(w, spec, candidates=bool(spec.candidate_capacity))
    _packet_scan(
        w, spec, candidates=bool(spec.candidate_capacity), work=_compact_packet_work
    )
    w.line("cute.arch.sync_threads()")


def _emit_consumer(
    w: _Writer,
    spec: _RadixCodegenSpec,
    *,
    column: str = "column",
    raw_value: str | None = None,
    known_valid: bool = False,
) -> None:
    # This marker is substituted with the real typed FX DAG before publication.
    raw_argument = f", {raw_value}" if raw_value is not None else ""
    valid_argument = ", known_valid=True" if known_valid else ""
    w.line(
        f"_helion_radix_consumer({column}, output_column{raw_argument}{valid_argument})"
    )


def _core_source(spec: _RadixCodegenSpec) -> str:
    w = _Writer()
    w.line("thread = cutlass.Int32(cute.arch.thread_idx()[0])")
    w.line("lane = thread % 32")
    w.line("warp = thread // 32")
    w.line(
        "lane_mask = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)"
    )
    w.line(f"row = cutlass.Int32(cute.arch.block_idx()[0]) // {spec.cluster}")
    w.line(
        "cta_rank = cutlass.Int32(cute.arch.block_idx_in_cluster())"
        if spec.cluster > 1
        else "cta_rank = cutlass.Int32(0)"
    )
    w.line(
        f"cta_start = {spec.column_scalar}(cta_rank) * {spec.column_scalar}({spec.cta_cols})"
    )
    w.line(
        f"input_row_base = cutlass.Int64(row) * cutlass.Int64({spec.plan.graph.x.stride(0)})"
    )
    w.line("allocator = cutlass.utils.SmemAllocator()")
    unsigned = {"primary_prefix", "primary_mask", "coarse_prefix", "coarse_mask"}
    for name, dtype, count in spec.plan.allocations:
        scalar = (
            "cutlass.Uint32"
            if name in unsigned
            else {
                torch.int32: "cutlass.Int32",
                torch.int64: "cutlass.Int64",
                torch.uint16: "cutlass.Uint16",
                torch.uint32: "cutlass.Uint32",
            }[dtype]
        )
        w.line(
            f"{name} = allocator.allocate_tensor({scalar}, cute.make_layout(({count},)), byte_alignment=16)"
        )
    dtype = "cutlass." + _FORMATS[spec.dtype].cute_name
    w.line(
        f"input_packet = cute.make_rmem_tensor(cute.make_layout(({spec.packet},)), {dtype})"
    )
    w.line(
        f"input_columns = cute.make_rmem_tensor(cute.make_layout(({spec.packet},)), {spec.column_scalar})"
    )
    w.line(
        f"input_members = cute.make_rmem_tensor(cute.make_layout(({spec.packet},)), cutlass.Boolean)"
    )
    if spec.cache_keys_enabled:
        w.line(
            f"input_keys = cute.make_rmem_tensor(cute.make_layout(({spec.packet},)), cutlass.Uint32)"
        )
    with w.block("if thread == 0"):
        w.line("primary_prefix[0] = cutlass.Uint32(0)")
        w.line("primary_mask[0] = cutlass.Uint32(0)")
        w.line(f"remaining[0] = cutlass.Int32({spec.k})")
        w.line("candidate_counter[0] = cutlass.Int32(0)")
        w.line("counters[0] = cutlass.Int32(0)")
        w.line("counters[1] = cutlass.Int32(0)")
        lengths = cast("torch.Tensor", tensor_value(spec.plan.graph.length.args[0]))
        scalar = "cutlass.Int32" if lengths.dtype == torch.int32 else "cutlass.Int64"
        w.line(
            f"length_slot[0] = {scalar}((lengths.iterator + cutlass.Int64(row) * cutlass.Int64({lengths.stride(0)})).load())"
        )
    w.line("cute.arch.sync_threads()")
    w.line("selection_length = length_slot[0]")
    # Every admitted primary key has at least two byte passes. The first
    # histogram release/wait publishes initialized counters before any DSM
    # ticket use, which starts in pass one; shortfill never reads DSM tickets.
    # The unconditional final lifetime barrier is retained for both branches.
    if spec.valid_domain:
        # Keep the raw typed length for every traced consumer. Clamp in Int64
        # before narrowing so huge positive or negative Int64 lengths cannot
        # wrap into the valid physical prefix.
        w.line("clamped_length = cutlass.Int64(selection_length)")
        with w.block(f"if clamped_length > {spec.plan.valid_cols}"):
            w.line(f"clamped_length = cutlass.Int64({spec.plan.valid_cols})")
        with w.block("if clamped_length < 0"):
            w.line("clamped_length = cutlass.Int64(0)")
        w.line(f"valid_count = {spec.column_scalar}(clamped_length)")
        with (
            w.block(f"if valid_count < {spec.k}"),
            w.block(
                f"for short_slot in range(thread + cta_rank * {spec.threads}, {spec.k}, {spec.threads * spec.cluster})"
            ),
        ):
            # All valid ordinals plus distinct MIN32 padding ordinals are an
            # exact unordered top-k. The original typed suffix handles them.
            w.line("output_source_column = cutlass.Int64(short_slot)")
            w.line("output_column = cutlass.Int64(short_slot)")
            _emit_consumer(w, spec, column="output_source_column")
        w.line("else:")
        w.indent += 1
    _primary_passes(w, spec)
    _compact(w, spec)
    # All peer reads drain before the scalar suffix. It reads private scratch
    # and disjoint global tensors only, so no CTA can retire a live DSM source.
    _collect(w, spec)
    if spec.valid_domain:
        w.indent -= 1
        # Both uniform row branches participate; empty CTAs cannot retire while
        # a peer still holds a histogram or rank-zero ticket pointer.
        w.line("cute.arch.sync_threads()")
        _synchronize_cluster(w, spec, release=True)
    return "\n".join(w.lines) + "\n"


class _Writer:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.indent = 0

    def line(self, text: str = "") -> None:
        self.lines.append("    " * self.indent + text)

    @contextmanager
    def block(self, header: str) -> Iterator[None]:
        self.line(header + ":")
        self.indent += 1
        try:
            yield
        finally:
            self.indent -= 1


def _boolean(*terms: str) -> str:
    return " & ".join(f"({term})" for term in terms)


def _rank(
    w: _Writer, spec: _RadixCodegenSpec, value: str, name: str, *, wide: bool = False
) -> None:
    kind = _FORMATS[spec.dtype]
    assert kind.infinity is not None
    width = 32 if wide else kind.width
    mask = (1 << width - 1) - 1
    infinity = 2139095040 if wide else kind.infinity
    if wide and kind.width != 32:
        w.line(f"{name}_value = cutlass.Float32({value})")
        value = f"{name}_value"
    bits = "cutlass.Int16" if width == 16 else "cutlass.Int32"
    w.line(f"{name}_bits = cutlass.Int32({value}.bitcast({bits}))")
    w.line(f"{name}_magnitude = {name}_bits & cutlass.Int32({mask})")
    w.line(f"{name}_sign = {name}_bits >> cutlass.Int32(31)")
    w.line(f"{name} = ({name}_magnitude ^ {name}_sign) - {name}_sign")
    with w.block(f"if {name}_magnitude > cutlass.Int32({infinity})"):
        w.line(f"{name} = cutlass.Int32({mask})")


def _packet(w: _Writer, spec: _RadixCodegenSpec, *, candidates: bool) -> None:
    dtype = "cutlass." + _FORMATS[spec.dtype].cute_name
    p = spec.packet
    column_scalar = spec.column_scalar
    w.line(
        f"scan_base = {column_scalar}(packet_round) * {column_scalar}({spec.threads * p}) + {column_scalar}(thread) * {column_scalar}({p})"
    )
    with w.block(f"for prepare in cutlass.range_constexpr({p})"):
        w.line(f"scan_slot = scan_base + {column_scalar}(prepare)")
        w.line("column = cta_start + scan_slot")
        w.line("member = scan_slot < scan_length")
        if candidates:
            with w.block("if use_candidates"):
                w.line(f"column = {column_scalar}(0)")
                with w.block("if member"):
                    w.line(
                        f"column = {column_scalar}(candidate_indices[cutlass.Int32(scan_slot)])"
                    )
        w.line("input_columns[prepare] = column")
        if spec.valid_domain:
            w.line("input_members[prepare] = cutlass.Boolean(member)")
        else:
            w.line(
                f"input_members[prepare] = cutlass.Boolean({_boolean('member', f'column < {spec.logical_cols}')})"
            )
        w.line(f"input_packet[prepare] = {dtype}(0)")
    read_terms = [
        _boolean(
            f"input_members[{i}]",
            f"input_columns[{i}] >= 0",
            f"input_columns[{i}] < {spec.cols}",
            *(
                (f"input_columns[{i}] < selection_length",)
                if spec.recipe in ("packed_float_index", "float_rank")
                else ()
            ),
        )
        for i in (range(p) if candidates else (0, p - 1))
    ]
    if spec.valid_domain:
        read_terms = [
            f"input_members[{i}]" for i in (range(p) if candidates else (0, p - 1))
        ]
    guard = _boolean(
        *read_terms,
        f"(input_row_base + input_columns[0]) % {p} == 0",
        *(
            (f"input_columns[{i}] == input_columns[0] + {i}" for i in range(1, p))
            if candidates
            else ()
        ),
    )
    alignment = spec.packet_bits // 8
    if spec.input_alignment >= alignment:
        with w.block(f"if {guard}"):
            w.line(
                f"packet_source = cute.make_tensor((x.iterator + input_row_base + input_columns[0]).align({alignment}), cute.make_layout(({p},)))"
            )
            w.line(
                f"cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), {dtype}, num_bits_per_copy={spec.packet_bits}), packet_source, input_packet)"
            )
        w.line("else:")
        w.indent += 1
    with w.block(f"for prepare in cutlass.range_constexpr({p})"):
        load_guard = _boolean(
            "input_members[prepare]",
            "input_columns[prepare] >= 0",
            f"input_columns[prepare] < {spec.cols}",
            *(
                ("input_columns[prepare] < selection_length",)
                if spec.recipe in ("packed_float_index", "float_rank")
                else ()
            ),
        )
        if spec.valid_domain:
            load_guard = "input_members[prepare]"
        with w.block(f"if {load_guard}"):
            w.line(
                f"input_packet[prepare] = {dtype}((x.iterator + input_row_base + input_columns[prepare]).load())"
            )
    if spec.input_alignment >= alignment:
        w.indent -= 1


def _cached_packet(w: _Writer, spec: _RadixCodegenSpec) -> None:
    # The complete-band count predicate dominates every paired read. There is
    # no access to a partially filled cache after overflow.
    scalar = spec.column_scalar
    w.line(
        f"scan_base = {scalar}(packet_round) * {scalar}({spec.threads * spec.packet}) + {scalar}(thread) * {scalar}({spec.packet})"
    )
    with w.block(f"for prepare in cutlass.range_constexpr({spec.packet})"):
        w.line(f"scan_slot = scan_base + {scalar}(prepare)")
        w.line("member = scan_slot < scan_length")
        w.line(f"input_columns[prepare] = {scalar}(0)")
        w.line("input_members[prepare] = cutlass.Boolean(member)")
        w.line("input_keys[prepare] = cutlass.Uint32(0)")
        with w.block("if member"):
            w.line(
                f"input_columns[prepare] = {scalar}(candidate_indices[cutlass.Int32(scan_slot)])"
            )
            w.line(
                "input_keys[prepare] = cutlass.Uint32(candidate_keys[cutlass.Int32(scan_slot)])"
            )


def _selection_packet(w: _Writer, spec: _RadixCodegenSpec, *, candidates: bool) -> None:
    if not spec.cache_keys_enabled or not candidates:
        _packet(w, spec, candidates=candidates)
        return
    with w.block("if use_candidates"):
        _cached_packet(w, spec)
    w.line("else:")
    w.indent += 1
    _packet(w, spec, candidates=False)
    w.indent -= 1


def _selection_key(w: _Writer, spec: _RadixCodegenSpec, *, candidates: bool) -> None:
    if not spec.cache_keys_enabled or not candidates:
        _primary_key(w, spec, "input_packet[vec]", "input_columns[vec]", "key")
        return
    w.line("key = cutlass.Uint32(0)")
    with w.block("if use_candidates"):
        w.line("key = cutlass.Uint32(input_keys[vec])")
    w.line("else:")
    w.indent += 1
    _primary_key(w, spec, "input_packet[vec]", "input_columns[vec]", "key")
    w.indent -= 1


def _scalar_candidate_packet(w: _Writer, spec: _RadixCodegenSpec) -> None:
    scalar = spec.column_scalar
    dtype = "cutlass." + _FORMATS[spec.dtype].cute_name
    w.line(
        f"scan_slot = {scalar}(packet_round) * {scalar}({spec.threads}) + {scalar}(thread)"
    )
    w.line("member = scan_slot < scan_length")
    w.line(f"input_columns[0] = {scalar}(0)")
    w.line("input_members[0] = cutlass.Boolean(member)")
    if spec.cache_keys_enabled:
        w.line("input_keys[0] = cutlass.Uint32(0)")
    else:
        w.line(f"input_packet[0] = {dtype}(0)")
    with w.block("if member"):
        w.line(
            f"input_columns[0] = {scalar}(candidate_indices[cutlass.Int32(scan_slot)])"
        )
        if spec.cache_keys_enabled:
            w.line(
                "input_keys[0] = cutlass.Uint32(candidate_keys[cutlass.Int32(scan_slot)])"
            )
        else:
            guard = _boolean(
                "input_columns[0] >= 0",
                f"input_columns[0] < {spec.cols}",
                "input_columns[0] < selection_length",
            )
            with w.block("if member" if spec.valid_domain else f"if {guard}"):
                w.line(
                    f"input_packet[0] = {dtype}((x.iterator + input_row_base + input_columns[0]).load())"
                )


def _scan_round_count(w: _Writer, spec: _RadixCodegenSpec, *, candidates: bool) -> None:
    w.line(
        f"scan_rounds = (scan_length + {spec.threads * spec.packet - 1}) // {spec.threads * spec.packet}"
    )
    if spec.scalar_candidates and candidates:
        with w.block("if use_candidates"):
            w.line(
                f"scan_rounds = (scan_length + {spec.threads - 1}) // {spec.threads}"
            )


def _packet_scan(
    w: _Writer,
    spec: _RadixCodegenSpec,
    *,
    candidates: bool,
    work: Callable[[_Writer, _RadixCodegenSpec, bool], None],
) -> None:
    if spec.scan_unroll > 1:
        if candidates:
            # Preserve uniform candidate scheduling. Only the source scan gets
            # an explicit unroll factor; all tail lanes retain their predicates.
            with (
                w.block("if use_candidates"),
                w.block("for packet_round in range(scan_rounds)"),
            ):
                if spec.scalar_candidates:
                    _scalar_candidate_packet(w, spec)
                    with w.block("for vec in cutlass.range_constexpr(1)"):
                        work(w, spec, True)
                else:
                    _selection_packet(w, spec, candidates=True)
                    with w.block(f"for vec in cutlass.range_constexpr({spec.packet})"):
                        work(w, spec, True)
            w.line("else:")
            w.indent += 1
        with w.block(
            f"for packet_round in cutlass.range(scan_rounds, unroll={spec.scan_unroll})"
        ):
            _packet(w, spec, candidates=False)
            with w.block(f"for vec in cutlass.range_constexpr({spec.packet})"):
                work(w, spec, False)
        if candidates:
            w.indent -= 1
        return
    with w.block("for packet_round in range(scan_rounds)"):
        if spec.scalar_candidates and candidates:
            # The count and mode are uniform within each CTA. Every lane enters
            # the same number of votes/shuffles, including partial final rounds.
            with w.block("if use_candidates"):
                _scalar_candidate_packet(w, spec)
                with w.block("for vec in cutlass.range_constexpr(1)"):
                    work(w, spec, True)
            w.line("else:")
            w.indent += 1
            _packet(w, spec, candidates=False)
            with w.block(f"for vec in cutlass.range_constexpr({spec.packet})"):
                work(w, spec, False)
            w.indent -= 1
        else:
            _selection_packet(w, spec, candidates=candidates)
            with w.block(f"for vec in cutlass.range_constexpr({spec.packet})"):
                work(w, spec, candidates)


def _primary_packet_work(w: _Writer, spec: _RadixCodegenSpec, candidates: bool) -> None:
    word = "cutlass.Uint32"
    _selection_key(w, spec, candidates=candidates)
    w.line(f"digit = cutlass.Int32((key >> {word}(primary_shift)) & {word}(255))")
    w.line(
        f"matched = cutlass.Boolean({_boolean('input_members[vec]', '(key & pass_mask) == pass_prefix')})"
    )
    _histogram_update(w, spec)
    if spec.candidate_capacity:
        _cache_update(w, spec)
    with w.block("if primary_round == 1"):
        w.line(
            f"coarse_above = cutlass.Boolean({_boolean('input_members[vec]', '(key & pass_mask) > pass_prefix')})"
        )
        _reserve(w, spec, "above", "coarse_above", candidates=candidates)


def _compact_packet_work(w: _Writer, spec: _RadixCodegenSpec, candidates: bool) -> None:
    _selection_key(w, spec, candidates=candidates)
    w.line("better = cutlass.Boolean(key > primary_boundary)")
    w.line("same = cutlass.Boolean(key == primary_boundary)")
    w.line(
        "coarse_member = cutlass.Boolean((key & coarse_mask[0]) == coarse_prefix[0])"
    )
    w.line(
        f"take_above = cutlass.Boolean({_boolean('input_members[vec]', 'coarse_member', 'better')})"
    )
    w.line(
        f"take_equal = cutlass.Boolean({_boolean('input_members[vec]', 'coarse_member', 'same')})"
    )
    _reserve(w, spec, "above", "take_above", candidates=candidates)
    _reserve(w, spec, "equal", "take_equal", candidates=candidates)


def _scan_length(w: _Writer, spec: _RadixCodegenSpec) -> None:
    w.line(f"scan_length = {spec.column_scalar}({spec.cta_cols})")
    if spec.valid_domain:
        with w.block("if scan_length > valid_count - cta_start"):
            w.line(f"scan_length = {spec.column_scalar}(valid_count - cta_start)")
        with w.block("if scan_length < 0"):
            w.line(f"scan_length = {spec.column_scalar}(0)")


def _synchronize_cluster(w: _Writer, spec: _RadixCodegenSpec, *, release: bool) -> None:
    if spec.cluster > 1:
        w.line(
            "cute.arch.cluster_arrive()"
            if release
            else "cute.arch.cluster_arrive_relaxed()"
        )
        w.line("cute.arch.cluster_wait()")


def _cache_update(w: _Writer, spec: _RadixCodegenSpec) -> None:
    with w.block("if primary_round == 1"):
        w.line("candidate_count = cutlass.Int32(cute.arch.popc(members))")
        w.line("candidate_base = cutlass.Int32(0)")
        with w.block(f"if {_boolean('lane == 0', 'candidate_count > 0')}"):
            w.line(
                "candidate_base = cute.arch.atomic_add(candidate_counter.iterator.llvm_ptr, candidate_count, sem='relaxed', scope='cta')"
            )
        w.line("candidate_base = cute.arch.shuffle_sync(candidate_base, 0)")
        with w.block("if matched"):
            w.line(
                "candidate_slot = candidate_base + cutlass.Int32(cute.arch.popc(members & lane_mask))"
            )
            with w.block(f"if candidate_slot < {spec.candidate_capacity}"):
                w.line(
                    "candidate_indices[candidate_slot] = cutlass.Int32(input_columns[vec])"
                )
                if spec.cache_keys_enabled:
                    w.line(
                        f"candidate_keys[candidate_slot] = {spec.cached_key_scalar}(key)"
                    )


def _histogram_choose(
    w: _Writer,
    spec: _RadixCodegenSpec,
    *,
    prefix: str,
    mask: str,
    word: str,
    shift: str,
) -> None:
    w.line("cute.arch.sync_threads()")
    _synchronize_cluster(w, spec, release=True)
    w.line("bin_count = cutlass.Int32(0)")
    with w.block("if thread < 256"):
        if spec.cluster > 1:
            with w.block(f"for peer in cutlass.range_constexpr({spec.cluster})"):
                w.line(
                    "remote_bin = cute.arch.map_dsmem_ptr(histogram.iterator + thread, cutlass.Int32(peer))"
                )
                w.line("bin_count = bin_count + cutlass.Int32(remote_bin.load())")
        else:
            w.line("bin_count = histogram[thread]")
    w.line("suffix_count = bin_count")
    for delta in (1, 2, 4, 8, 16):
        w.line(f"scan_peer = cute.arch.shuffle_sync_down(suffix_count, {delta})")
        with w.block(f"if lane + {delta} < 32"):
            w.line("suffix_count = suffix_count + scan_peer")
    with w.block(f"if {_boolean('thread < 256', 'lane == 0')}"):
        w.line("warp_totals[warp] = suffix_count")
    w.line("cute.arch.sync_threads()")
    with w.block("if thread < 256"):
        w.line("greater_count = suffix_count - bin_count")
        with (
            w.block("for other_warp in cutlass.range_constexpr(8)"),
            w.block("if other_warp > warp"),
        ):
            w.line("greater_count = greater_count + warp_totals[other_warp]")
        chosen = _boolean(
            "bin_count > 0",
            "greater_count < pass_remaining",
            "pass_remaining <= greater_count + bin_count",
        )
        with w.block(f"if {chosen}"):
            w.line(f"{prefix}[0] = pass_prefix | ({word}(thread) << {word}({shift}))")
            w.line(f"{mask}[0] = pass_mask | ({word}(255) << {word}({shift}))")
            w.line("remaining[0] = pass_remaining - greater_count")
    w.line("cute.arch.sync_threads()")
    _synchronize_cluster(w, spec, release=False)


def _primary_passes(w: _Writer, spec: _RadixCodegenSpec) -> None:
    word = "cutlass.Uint64" if spec.primary_width == 64 else "cutlass.Uint32"
    count = spec.primary_width // 8
    with w.block(f"for primary_round in cutlass.range_constexpr({count})"):
        w.line(f"primary_shift = cutlass.Int32(({count - 1} - primary_round) * 8)")
        with w.block("if cutlass.const_expr(primary_round == 0)"):
            w.line(f"pass_prefix = {word}(0)")
            w.line(f"pass_mask = {word}(0)")
            w.line(f"pass_remaining = cutlass.Int32({spec.k})")
        w.line("else:")
        w.indent += 1
        w.line("pass_prefix = primary_prefix[0]")
        w.line("pass_mask = primary_mask[0]")
        w.line("pass_remaining = remaining[0]")
        w.indent -= 1
        with w.block(f"if {_boolean('primary_round == 1', 'thread == 0')}"):
            w.line("coarse_prefix[0] = pass_prefix")
            w.line("coarse_mask[0] = pass_mask")
        if spec.candidate_capacity:
            with w.block(f"if {_boolean('primary_round == 1', 'thread == 0')}"):
                w.line("candidate_counter[0] = cutlass.Int32(0)")
        with w.block(f"for initialize in range(thread, 256, {spec.threads})"):
            w.line("histogram[initialize] = cutlass.Int32(0)")
        w.line("cute.arch.sync_threads()")
        _scan_length(w, spec)
        w.line("use_candidates = cutlass.Boolean(False)")
        if spec.candidate_capacity:
            condition = _boolean(
                "primary_round > 1",
                f"candidate_counter[0] <= {spec.candidate_capacity}",
            )
            with w.block(f"if {condition}"):
                w.line(f"scan_length = {spec.column_scalar}(candidate_counter[0])")
                w.line("use_candidates = cutlass.Boolean(True)")
        candidates = bool(spec.candidate_capacity) and count > 2
        _scan_round_count(w, spec, candidates=candidates)
        _packet_scan(w, spec, candidates=candidates, work=_primary_packet_work)
        _histogram_choose(
            w,
            spec,
            prefix="primary_prefix",
            mask="primary_mask",
            word=word,
            shift="primary_shift",
        )


def _reserve(
    w: _Writer,
    spec: _RadixCodegenSpec,
    group: str,
    predicate: str,
    *,
    candidates: bool,
) -> None:
    index = 0 if group == "above" else 1
    w.line(f"{group}_members = cute.arch.vote_ballot_sync({predicate})")
    w.line(f"{group}_count = cutlass.Int32(cute.arch.popc({group}_members))")
    w.line(f"{group}_base = cutlass.Int32(0)")
    with w.block(f"if {_boolean('lane == 0', f'{group}_count > 0')}"):
        if spec.ticket_stores and spec.cluster > 1:
            w.line(
                f"{group}_counter = cute.arch.map_dsmem_ptr(counters.iterator + {index}, cutlass.Int32(0))"
            )
            w.line(
                f"{group}_base = cute.arch.atomic_add({group}_counter.llvm_ptr, {group}_count, sem='relaxed', scope='cluster')"
            )
        else:
            w.line(
                f"{group}_base = cute.arch.atomic_add((counters.iterator + {index}).llvm_ptr, {group}_count, sem='relaxed', scope='cta')"
            )
    w.line(f"{group}_base = cute.arch.shuffle_sync({group}_base, 0)")
    with w.block(f"if {predicate}"):
        w.line(
            f"{group}_ticket = {group}_base + cutlass.Int32(cute.arch.popc({group}_members & lane_mask))"
        )
        if spec.ticket_stores:
            if group == "equal":
                with w.block("if equal_ticket < boundary_remaining"):
                    _direct_output(w, spec, group, candidates=candidates)
            else:
                _direct_output(w, spec, group, candidates=candidates)
        elif group == "above":
            w.line("selected_indices[above_ticket] = cutlass.Int32(input_columns[vec])")
        else:
            with w.block("if equal_ticket < boundary_remaining"):
                w.line(
                    "selected_indices[global_above + equal_ticket] = cutlass.Int32(input_columns[vec])"
                )


def _direct_output(
    w: _Writer, spec: _RadixCodegenSpec, group: str, *, candidates: bool
) -> None:
    # Keep the source ordinal separate from the packet's Int32 scan variable.
    # Every callback observes the same typed index/value pair at its global
    # output ticket, regardless of which CTA reserved it.
    w.line("output_source_column = cutlass.Int64(input_columns[vec])")
    w.line(
        "output_column = cutlass.Int64(above_ticket)"
        if group == "above"
        else "output_column = cutlass.Int64(global_above + equal_ticket)"
    )
    if candidates and spec.cache_keys_enabled:
        with w.block("if use_candidates"):
            if spec.cached_payload_inverse:
                _inverse_cached_payload(w, spec)
            _emit_consumer(
                w,
                spec,
                column="output_source_column",
                raw_value="cached_payload_raw" if spec.cached_payload_inverse else None,
                known_valid=spec.valid_domain,
            )
        w.line("else:")
        w.indent += 1
        _emit_consumer(
            w,
            spec,
            column="output_source_column",
            raw_value="input_packet[vec]",
            known_valid=spec.valid_domain,
        )
        w.indent -= 1
    else:
        _emit_consumer(
            w,
            spec,
            column="output_source_column",
            raw_value="input_packet[vec]",
            known_valid=spec.valid_domain,
        )


def _inverse_cached_payload(w: _Writer, spec: _RadixCodegenSpec) -> None:
    width = spec.primary_width
    sign = 1 << (width - 1)
    mask = (1 << width) - 1
    unsigned = f"cutlass.Uint{width}"
    signed = f"cutlass.Int{width}"
    dtype = "cutlass." + _FORMATS[spec.dtype].cute_name
    w.line(f"inverse_key = {unsigned}(input_keys[vec])")
    if not spec.largest:
        w.line(f"inverse_key = inverse_key ^ {unsigned}({mask})")
    w.line(f"cached_payload_raw = {dtype}(0)")
    with w.block(
        f"if (inverse_key == {unsigned}({sign})) | (inverse_key == {unsigned}({mask}))"
    ):
        # Zero signs and canonical NaN payload/sign are not represented by the
        # rank key. The original typed readonly load is required in those cases.
        w.line(
            f"cached_payload_raw = {dtype}((x.iterator + input_row_base + output_source_column).load())"
        )
    w.line("else:")
    w.indent += 1
    w.line(
        f"inverse_rank = cutlass.Int32((inverse_key ^ {unsigned}({sign})).bitcast({signed}))"
    )
    w.line(f"inverse_raw_bits = {unsigned}(inverse_rank)")
    with w.block("if inverse_rank < 0"):
        w.line(f"inverse_raw_bits = {unsigned}({sign}) | {unsigned}(-inverse_rank)")
    w.line(f"cached_payload_raw = inverse_raw_bits.bitcast({dtype})")
    w.indent -= 1


def _collect(w: _Writer, spec: _RadixCodegenSpec) -> None:
    if spec.ticket_stores:
        if not spec.valid_domain:
            w.line("cute.arch.sync_threads()")
            _synchronize_cluster(w, spec, release=True)
        return
    with w.block("if thread == 0"):
        w.line("metadata[0] = counters[0]")
        w.line("local_equal = counters[1]")
        with w.block("if local_equal > boundary_remaining"):
            w.line("local_equal = boundary_remaining")
        w.line("metadata[1] = cutlass.Int32(local_equal)")
    w.line("cute.arch.sync_threads()")
    _synchronize_cluster(w, spec, release=True)
    with w.block("if thread == 0"):
        w.line("above_offset = cutlass.Int32(0)")
        w.line("equal_offset = cutlass.Int32(0)")
        if spec.cluster > 1:
            with (
                w.block(f"for peer in cutlass.range_constexpr({spec.cluster})"),
                w.block("if peer < cta_rank"),
            ):
                w.line(
                    "remote_above = cute.arch.map_dsmem_ptr(metadata.iterator, cutlass.Int32(peer))"
                )
                w.line(
                    "remote_equal = cute.arch.map_dsmem_ptr(metadata.iterator + 1, cutlass.Int32(peer))"
                )
                w.line(
                    "above_offset = above_offset + cutlass.Int32(remote_above.load())"
                )
                w.line(
                    "equal_offset = equal_offset + cutlass.Int32(remote_equal.load())"
                )
        w.line("metadata[2] = above_offset")
        w.line("metadata[3] = equal_offset")
    w.line("cute.arch.sync_threads()")
    _synchronize_cluster(w, spec, release=False)
    with w.block(f"for slot in range(thread, metadata[0], {spec.threads})"):
        w.line("column = cutlass.Int64(selected_indices[slot])")
        w.line("output_column = cutlass.Int64(metadata[2] + slot)")
        _emit_consumer(w, spec, known_valid=spec.valid_domain)
    with w.block(f"for slot in range(thread, metadata[1], {spec.threads})"):
        w.line("output_column = cutlass.Int64(global_above + metadata[3] + slot)")
        with w.block(f"if output_column < {spec.k}"):
            w.line("column = cutlass.Int64(selected_indices[global_above + slot])")
            _emit_consumer(w, spec, known_valid=spec.valid_domain)
