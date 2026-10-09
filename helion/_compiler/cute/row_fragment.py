"""Compose ordinary graph operations over subgroup-owned register fragments.

The caller supplies loads and operation boundaries; this module owns scalar
pointwise lowering, broadcasting, and last-axis reductions.  A fragment's
layout describes unique logical elements, rather than a particular producer
such as a sorting network or a matrix multiplication.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from dataclasses import replace
import hashlib
import itertools
import math
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.virtualized import V
from torch.fx.node import map_arg

from ... import exc
from ...language._tracing_ops import _mask_to
from ...language._tracing_ops import _new_var
from ...language.memory_ops import load
from ...language.tile_ops import tile_index
from ...language.view_ops import subscript
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..aten_lowering import LoweringContext
from ..compile_environment import CompileEnvironment
from ..inductor_lowering import PointwiseLowering

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.fx.node import Argument
    from torch.fx.node import Node

    from ..generate_ast import GenerateAST
    from .register_tensor import RegisterTensorMap
    from .scalar_lowering import ScalarLoweringContextCache


_VIEW_TARGETS = frozenset(
    {
        _new_var,
        subscript,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.expand.default,
    }
)
_REDUCTION_TARGETS: dict[object, str] = {
    torch.ops.aten.sum.dim_IntList: "sum",
    torch.ops.aten.amax.default: "max",
    torch.ops.aten.amin.default: "min",
    torch.ops.aten.any.dim: "max",
    torch.ops.aten.any.dims: "max",
    torch.ops.aten.all.dim: "min",
    torch.ops.aten.all.dims: "min",
}
_BOOLEAN_REDUCTION_TARGETS = frozenset(
    {
        torch.ops.aten.any.dim,
        torch.ops.aten.any.dims,
        torch.ops.aten.all.dim,
        torch.ops.aten.all.dims,
    }
)
_CAST_TARGETS = frozenset({torch.ops.prims.convert_element_type.default})


@dataclass(frozen=True)
class RowFragmentLayout:
    """Contiguous vectors distributed cyclically over a lane subgroup."""

    lanes: int
    vector_width: int
    lane_expr: str
    owner_lanes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        assert 0 < self.lanes <= 32 and self.lanes & (self.lanes - 1) == 0
        assert self.vector_width > 0
        if self.owner_lanes is not None:
            assert sorted(self.owner_lanes) == list(range(self.lanes))

    def column(self, register: str | int) -> str:
        vector = self.vector_width
        return (
            f"((({register}) // {vector} * {self.lanes} + ({self.lane_expr}))"
            f" * {vector} + ({register}) % {vector})"
        )

    def num_registers(self, extent: int) -> int:
        width = self.lanes * self.vector_width
        return (extent + width - 1) // width * self.vector_width

    def owner(self, column: int) -> tuple[int, int]:
        """Return the physical subgroup lane and register for a logical column."""
        vector = self.vector_width
        lane = column // vector % self.lanes
        if self.owner_lanes is not None:
            lane = self.owner_lanes[lane]
        return lane, column // (self.lanes * vector) * vector + column % vector

    def replicated_gather_map(
        self, extent: int, source_registers: int
    ) -> RegisterTensorMap:
        """Map a locally replicated vector to this layout, padding with typed zero."""
        assert 0 < extent <= source_registers
        rows = [
            [source_registers] * self.num_registers(extent) for _ in range(self.lanes)
        ]
        for column in range(extent):
            owner, register = self.owner(column)
            rows[owner][register] = column
        unique_rows: list[list[int]] = []
        row_ids: dict[tuple[int, ...], int] = {}
        owners = []
        for row in rows:
            key = tuple(row)
            if key not in row_ids:
                row_ids[key] = len(unique_rows)
                unique_rows.append(row)
            owners.append(row_ids[key])
        return {"rows": unique_rows, "owners": owners}


def emit_replicated_register_gather(
    cg: GenerateAST,
    source: str,
    *,
    extent: int,
    source_registers: int,
    layout: RowFragmentLayout,
    physical_lane: str,
) -> str:
    """Return a gather expression without changing source dtype or logical owners."""
    mapping = layout.replicated_gather_map(extent, source_registers)
    encoded = repr(mapping)
    name = "_cute_register_map_" + hashlib.sha256(encoded.encode()).hexdigest()
    module = "helion.runtime.cute.register_tensor"
    helper = "_cute_gather_registers"
    if not any(
        isinstance(statement, ast.ImportFrom)
        and statement.module == module
        and any(alias.name == helper for alias in statement.names)
        for statement in cg.module_statements
    ):
        cg.module_statements.append(
            ast.ImportFrom(module=module, names=[ast.alias(name=helper)], level=0)
        )
    if not any(
        isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in statement.targets
        )
        for statement in cg.module_statements
    ):
        cg.module_statements.append(statement_from_string(f"{name} = {encoded}"))
    return f"{helper}({source}, {name}, {physical_lane})"


@dataclass(frozen=True)
class RowFragmentOrder:
    """Monotone values with NaNs at the numerically greatest endpoint."""

    descending: bool

    def maximum(self, extent: int) -> int:
        return 0 if self.descending else extent - 1


@dataclass(frozen=True)
class RowFragment:
    """A named 1-D rmem tensor and its logical element ownership.

    Replicated fragments contain one scalar per row, identical in all lanes.
    Other fragments contain each logical column exactly once; padded register
    slots are not logical elements and never contribute to reductions.
    """

    name: str
    dtype: torch.dtype
    extent: int
    layout: RowFragmentLayout
    replicated: bool = False
    order: RowFragmentOrder | None = None

    def __post_init__(self) -> None:
        assert self.extent > 0
        assert not self.replicated or self.extent == 1

    @property
    def num_registers(self) -> int:
        return 1 if self.replicated else self.layout.num_registers(self.extent)

    def element(self, register: str | int) -> ast.AST:
        return expr_from_string(f"{self.name}[{0 if self.replicated else register}]")


def _tensor(node: Node) -> torch.Tensor | None:
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def row_fragment_tensor_inputs(node: Node) -> list[Node]:
    """Tensor operands in ordinary lowering order, excluding scheduler edges."""
    inputs: list[Node] = []

    def visit(argument: Node) -> Node:
        if _tensor(argument) is not None:
            inputs.append(argument)
        return argument

    map_arg((node.args, {**node.kwargs, "_extra_deps": None}), visit)
    return inputs


def row_fragment_logical_extent(size: int | torch.SymInt) -> int | None:
    """Resolve a tile symbol to its logical extent, excluding padding."""
    if isinstance(size, int):
        return size
    env = CompileEnvironment.current()
    block_id = env.get_block_id(size)
    if block_id is None:
        return None
    logical_size = env.block_sizes[block_id].size
    return logical_size if isinstance(logical_size, int) else None


def _same_extent(left: int | torch.SymInt, right: int | torch.SymInt) -> bool:
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    return CompileEnvironment.current().known_equal(left, right)


def _preserves_row_columns(node: Node) -> bool:
    """Allow inserting/removing unit axes without moving row/column axes."""
    inputs = row_fragment_tensor_inputs(node)
    output = _tensor(node)
    if len(inputs) != 1 or output is None:
        return False
    source = inputs[0].meta["val"]
    if node.target is subscript:
        if (
            len(node.args) != 2
            or node.kwargs
            or not isinstance(node.args[1], (tuple, list))
        ):
            return False
        indices = node.args[1]
        if any(index is not None and index != slice(None) for index in indices):
            return False
        if sum(index is not None for index in indices) != source.ndim:
            return False
    if node.target is torch.ops.aten.expand.default:
        # Expand can broadcast a row scalar, but cannot change a non-unit
        # logical dimension or introduce another non-unit row axis.
        if source.ndim > output.ndim:
            return False
        padded = [1] * (output.ndim - source.ndim) + list(source.shape)
        return all(
            _same_extent(before, 1) or _same_extent(before, after)
            for before, after in zip(padded, output.shape, strict=True)
        )
    source_shape = [size for size in source.shape if not _same_extent(size, 1)]
    output_shape = [size for size in output.shape if not _same_extent(size, 1)]
    return len(source_shape) == len(output_shape) and all(
        itertools.starmap(_same_extent, zip(source_shape, output_shape, strict=True))
    )


def _row_reduction(node: Node) -> str | None:
    reduction = _REDUCTION_TARGETS.get(node.target)
    inputs = row_fragment_tensor_inputs(node)
    if reduction is None or len(inputs) != 1:
        return None
    source = inputs[0].meta["val"]
    if node.target in _BOOLEAN_REDUCTION_TARGETS and source.dtype is not torch.bool:
        return None
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    if isinstance(dim, int):
        dim = [dim]
    if (
        source.ndim < 2
        or not isinstance(dim, (tuple, list))
        or len(dim) != 1
        or dim[0] not in (-1, source.ndim - 1)
    ):
        return None
    return reduction


def row_fragment_arange(node: Node) -> tuple[int, int, int] | None:
    """Prove a static integer progression independent of the tile schedule."""
    output = _tensor(node)
    if (
        node.target not in (torch.ops.aten.arange.default, torch.ops.prims.iota.default)
        or len(node.args) != 1
        or type(node.args[0]) is not int
        or output is None
        or output.ndim != 1
        or output.dtype not in (torch.int32, torch.int64)
        or not isinstance(output.size(0), int)
        or output.size(0) != node.args[0]
        or output.size(0) <= 0
    ):
        return None
    start, step = node.kwargs.get("start", 0), node.kwargs.get("step", 1)
    if type(start) is not int or type(step) is not int:
        return None
    return start, step, output.size(0)


def supports_row_fragment_node(node: Node) -> bool:
    """Whether this node has a producer-independent fragment implementation.

    Loads are boundaries delegated to the caller.  Their indexing and alias
    safety, and the availability of every graph dependency, are caller proofs.
    """
    if node.op != "call_function" or _tensor(node) is None:
        return False
    if node.target is tile_index or row_fragment_arange(node) is not None:
        return True
    if node.target is load or node.target is _mask_to or node.target in _CAST_TARGETS:
        return True
    if node.target in _VIEW_TARGETS:
        return _preserves_row_columns(node)
    if node.target in _REDUCTION_TARGETS:
        return _row_reduction(node) is not None
    if node.target in (
        torch.ops.aten.where.self,
        torch.ops.aten.scalar_tensor.default,
    ):
        return True
    if isinstance(node.meta.get("lowering"), PointwiseLowering):
        return True
    # Discovery can precede prepare_graph_lowerings.  Actual emission still
    # requires the ordinary lowering to produce a pointwise operation.
    return isinstance(node.target, torch._ops.OpOverload) and (
        torch.Tag.pointwise in node.target.tags
    )


class RowFragmentEmitter:
    """Emit a graph DAG while retaining register ownership across operations."""

    def __init__(
        self,
        cg: GenerateAST,
        load: Callable[[Node], RowFragment],
        *,
        layout: RowFragmentLayout,
        valid_row: str = "True",
        resolve: Callable[[Node], Node] | None = None,
        numeric_uses_only: Callable[[Node], bool] | None = None,
        row_expr: str | None = None,
        reuse_scalar_lowering: bool = False,
    ) -> None:
        self.cg = cg
        self.load = load
        self.layout = layout
        self.valid_row = valid_row
        self.resolve = resolve
        self.numeric_uses_only = numeric_uses_only
        self.row_expr = row_expr
        self.fragments: dict[Node, RowFragment] = {}
        self._scalar_lowering_context: ScalarLoweringContextCache | None = None
        if reuse_scalar_lowering:
            from .scalar_lowering import ScalarLoweringContextCache

            self._scalar_lowering_context = ScalarLoweringContextCache(cg)

    def bind(self, node: Node, fragment: RowFragment) -> None:
        self.fragments[node] = fragment
        if self.resolve is not None:
            self.fragments[self.resolve(node)] = fragment

    def _unsupported(self, node: Node, detail: str) -> exc.BackendUnsupported:
        return exc.BackendUnsupported("cute", f"row fragment {node.target}: {detail}")

    def _add(self, source: str) -> None:
        for statement in ast.parse(source).body:
            self.cg.add_statement(statement)

    def _new_fragment(
        self,
        node: Node,
        extent: int,
        layout: RowFragmentLayout,
        *,
        replicated: bool = False,
    ) -> RowFragment:
        output = _tensor(node)
        assert output is not None
        name = self.cg.device_function.new_var("row_fragment")
        fragment = RowFragment(name, output.dtype, extent, layout, replicated)
        dtype = CompileEnvironment.current().backend.dtype_str(fragment.dtype)
        self._add(
            f"{name} = cute.make_rmem_tensor({fragment.num_registers}, {dtype})\n"
            f"{name}.fill({dtype}(0))"
        )
        return fragment

    def emit_pointwise_scalar(
        self,
        node: Node,
        input_values: list[ast.AST],
        *,
        valid: str = "True",
        logical_column: str | None = None,
    ) -> ast.AST:
        """Apply ordinary scalar lowering, preserving each node's dtype boundary.

        The caller can also use this for recomputation at a selected logical
        index.  Statements needed by a lowering are emitted through ``cg``.
        """
        backend = CompileEnvironment.current().backend
        output = _tensor(node)
        assert output is not None
        if (progression := row_fragment_arange(node)) is not None:
            if logical_column is None:
                raise self._unsupported(node, "missing logical column")
            start, step, _ = progression
            result = expr_from_string(f"({logical_column}) * {step} + {start}")
        elif node.target is tile_index:
            if self.row_expr is None:
                raise self._unsupported(node, "missing logical row")
            result = expr_from_string(self.row_expr)
        elif node.target in _VIEW_TARGETS or node.target in _CAST_TARGETS:
            assert len(input_values) == 1
            result = input_values[0]
        elif node.target is _mask_to:
            assert len(input_values) == 1
            other = node.args[1]
            assert isinstance(other, (bool, int, float))
            result = expr_from_string(
                "({value} if {valid} else {other})",
                value=backend.cast_ast(input_values[0], output.dtype),
                valid=expr_from_string(valid),
                other=backend.cast_ast(
                    expr_from_string(constant_repr(other)), output.dtype
                ),
            )
        elif node.target is torch.ops.aten.scalar_tensor.default:
            scalar = node.args[0]
            if not isinstance(scalar, (bool, int, float)):
                raise self._unsupported(node, "nonliteral scalar")
            result = expr_from_string(constant_repr(scalar))
        else:
            inputs = row_fragment_tensor_inputs(node)
            if len(inputs) != len(input_values):
                raise self._unsupported(node, "unexpected scalar operand")
            context = LoweringContext.__new__(LoweringContext)
            context.cg = self.cg
            input_by_node = dict(zip(inputs, input_values, strict=True))
            context.env = cast("dict[Node, Argument]", input_by_node)
            lowering = node.meta.get("lowering")
            if lowering is None:
                from ..inductor_lowering import FakeGraphLowering
                from ..inductor_lowering import prepare_node_lowering

                graph_lowering = FakeGraphLowering()
                with V.set_graph_handler(graph_lowering):
                    prepare_node_lowering(graph_lowering, node)
                lowering = node.meta.get("lowering")
                # Preparation may remove duplicate/unused tensor operands.
                input_values = [
                    input_by_node[argument]
                    for argument in row_fragment_tensor_inputs(node)
                ]
            with V.set_current_node(node):
                if isinstance(lowering, PointwiseLowering):
                    if len(input_values) != len(lowering.input_names):
                        raise self._unsupported(node, "non-tensor pointwise operand")
                    kernel_handlers = None
                    if self._scalar_lowering_context is not None:
                        kernel_handlers = self._scalar_lowering_context.install(
                            node,
                            dict(zip(lowering.input_names, input_values, strict=True)),
                        )
                    result = lowering.codegen_from_input_asts(
                        context, node, input_values, kernel_handlers=kernel_handlers
                    )
                elif node.target in (
                    torch.ops.aten.where.self,
                    torch.ops.aten.fmin.default,
                    torch.ops.aten.fmax.default,
                ):
                    assert lowering is not None
                    result = lowering.codegen(context, node)
                else:
                    raise self._unsupported(node, "ordinary lowering is not pointwise")
            if not isinstance(result, ast.AST):
                raise self._unsupported(node, "pointwise result is not scalar")
        return backend.cast_ast(result, output.dtype)

    def _emit_pointwise(self, node: Node) -> RowFragment:
        inputs = row_fragment_tensor_inputs(node)
        fragments = [self.emit(source) for source in inputs]
        owned = [fragment for fragment in fragments if not fragment.replicated]
        if owned:
            layout = owned[0].layout
            extent = owned[0].extent
            if any(
                fragment.layout != layout or fragment.extent != extent
                for fragment in owned[1:]
            ):
                raise self._unsupported(node, "incompatible fragment ownership")
        else:
            layout, extent = self.layout, 1
        result = self._new_fragment(node, extent, layout, replicated=not owned)
        result = replace(result, order=self._pointwise_order(node, fragments))
        register = self.cg.device_function.new_var("row_register")
        valid = self.valid_row
        if not result.replicated:
            valid = f"({valid}) and ({layout.column(register)} < {extent})"
        statements: list[ast.AST] = []
        with self.cg.set_statements(statements):
            scalar = self.emit_pointwise_scalar(
                node,
                [fragment.element(register) for fragment in fragments],
                valid=valid,
            )
            self.cg.add_statement(
                statement_from_string(
                    f"{result.name}[{register}] = {{value}}", value=scalar
                )
            )
        body = ast.If(
            test=cast("ast.expr", expr_from_string(valid)),
            body=cast("list[ast.stmt]", statements),
            orelse=[],
        )
        self.cg.add_statement(
            ast.fix_missing_locations(
                ast.For(
                    target=ast.Name(id=register, ctx=ast.Store()),
                    iter=cast(
                        "ast.expr",
                        expr_from_string(
                            f"cutlass.range_constexpr({result.num_registers})"
                        ),
                    ),
                    body=[body],
                    orelse=[],
                )
            )
        )
        return result

    def _pointwise_order(
        self, node: Node, fragments: list[RowFragment]
    ) -> RowFragmentOrder | None:
        ordered = [fragment for fragment in fragments if fragment.order is not None]
        output = _tensor(node)
        if len(ordered) != 1 or output is None or not output.dtype.is_floating_point:
            return None
        order = ordered[0].order
        if node.target in _CAST_TARGETS or node.target is _mask_to:
            return order
        if node.target in (
            torch.ops.aten.exp.default,
            torch.ops.aten.exp2.default,
            torch.ops.aten.sigmoid.default,
            torch.ops.aten.tanh.default,
        ):
            return order
        # A finite scalar offset or positive finite scale preserves both the
        # numerical ordering and the location of an existing NaN endpoint.
        if len(node.args) < 2 or len(fragments) != 1:
            return None
        scalar = node.args[1]
        if not isinstance(scalar, (int, float)) or not math.isfinite(scalar):
            return None
        limits = torch.finfo(output.dtype)
        if abs(scalar) > limits.max:
            return None
        alpha = node.kwargs.get("alpha", 1)
        if not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
            return None
        if abs(scalar * alpha) > limits.max:
            return None
        if node.target in (
            torch.ops.aten.add.Tensor,
            torch.ops.aten.add.Scalar,
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.sub.Scalar,
        ) or (
            scalar >= limits.tiny
            and node.target
            in (
                torch.ops.aten.mul.Tensor,
                torch.ops.aten.mul.Scalar,
                torch.ops.aten.div.Tensor,
                torch.ops.aten.div.Scalar,
            )
        ):
            return order
        return None

    def _emit_view(self, node: Node) -> RowFragment:
        source_node = row_fragment_tensor_inputs(node)[0]
        source = self.emit(source_node)
        output = _tensor(node)
        assert output is not None
        source_tensor = source_node.meta["val"]
        if not source.replicated:
            if output.ndim == 0 or (
                row_fragment_logical_extent(output.shape[-1]) != source.extent
            ):
                raise self._unsupported(node, "view moves the distributed column axis")
            return source
        if output.ndim < 2 or _same_extent(output.shape[-1], 1):
            return source
        if node.target is not torch.ops.aten.expand.default:
            raise self._unsupported(node, "view moves the replicated row axis")
        # A replicated rank-one tensor represents rows, so PyTorch's
        # right-aligned expansion cannot reinterpret those rows as columns.
        if source_tensor.ndim == 1 and not _same_extent(source_tensor.shape[0], 1):
            raise self._unsupported(node, "expand broadcasts rows into columns")
        extent = row_fragment_logical_extent(output.shape[-1])
        if extent is None:
            raise self._unsupported(node, "dynamic broadcast column extent")
        result = self._new_fragment(node, extent, self.layout)
        register = self.cg.device_function.new_var("row_expand_register")
        self._add(
            f"for {register} in cutlass.range_constexpr({result.num_registers}):\n"
            f"    {result.name}[{register}] = {source.name}[0]"
        )
        return result

    def _emit_reduction(self, node: Node, reduction: str) -> RowFragment:
        source = self.emit(row_fragment_tensor_inputs(node)[0])
        output = _tensor(node)
        assert output is not None
        result = self._new_fragment(node, 1, source.layout, replicated=True)
        backend = CompileEnvironment.current().backend
        accumulation_dtype = output.dtype
        if accumulation_dtype in (torch.float16, torch.bfloat16):
            accumulation_dtype = torch.float32
        dtype = backend.dtype_str(accumulation_dtype)
        output_dtype = backend.dtype_str(output.dtype)
        if source.replicated:
            self._add(f"{result.name}[0] = {output_dtype}({source.name}[0])")
            return result
        fallback_guard = None
        if reduction == "max" and source.order is not None:
            lane, register = source.layout.owner(source.order.maximum(source.extent))
            maximum = self.cg.device_function.new_var("row_known_maximum")
            self._add(f"{maximum} = {dtype}({source.name}[{register}])")
            if source.layout.lanes > 1:
                self._add(
                    f"{maximum} = cute.arch.shuffle_sync({maximum}, offset={lane}, "
                    f"mask_and_clamp={((32 - source.layout.lanes) << 8) | 31})"
                )
            self._add(f"{result.name}[0] = {output_dtype}({maximum})")
            if self.numeric_uses_only is not None and self.numeric_uses_only(node):
                return result
            # Preserve the original choice between zero signs or NaN payloads.
            # The fallback contains shuffles: every thread named in their warp
            # mask must take it, even when only one row subgroup needs recovery.
            fallback_guard = (
                f"cute.arch.vote_ballot_sync(({self.valid_row}) and "
                f"(({maximum} == 0) or ({maximum} != {maximum}))) != 0"
            )
        full_reduction: list[ast.AST] = []
        with self.cg.set_statements(full_reduction):
            if reduction == "sum":
                identity = "0"
            elif accumulation_dtype.is_floating_point:
                identity = "-float('inf')" if reduction == "max" else "float('inf')"
            elif accumulation_dtype is torch.bool:
                identity = "False" if reduction == "max" else "True"
            else:
                limits = torch.iinfo(accumulation_dtype)
                identity = str(limits.min if reduction == "max" else limits.max)
            accumulator = self.cg.device_function.new_var("row_reduce")
            register = self.cg.device_function.new_var("row_reduce_register")
            value = self.cg.device_function.new_var("row_reduce_value")

            def combine(left: str, right: str) -> str:
                if reduction == "sum":
                    return f"({left} + {right})"
                comparison = ">" if reduction == "max" else "<"
                condition = f"({left} {comparison} {right})"
                if accumulation_dtype.is_floating_point:
                    # A NaN from either operand must survive local and subgroup
                    # reductions; hardware fmax/fmin may ignore a NaN operand.
                    condition += f" or ({left} != {left})"
                return f"({left} if {condition} else {right})"

            self._add(
                f"{accumulator} = {dtype}({identity})\n"
                f"for {register} in cutlass.range_constexpr({source.num_registers}):\n"
                f"    if ({self.valid_row}) and ({source.layout.column(register)} < {source.extent}):\n"
                f"        {value} = {dtype}({source.name}[{register}])\n"
                f"        {accumulator} = {combine(accumulator, value)}"
            )
            for stage in range(source.layout.lanes.bit_length() - 1):
                peer = self.cg.device_function.new_var("row_reduce_peer")
                self._add(
                    f"{peer} = cute.arch.shuffle_sync_bfly({accumulator}, offset={1 << stage})\n"
                    f"{accumulator} = {combine(accumulator, peer)}"
                )
            self._add(f"{result.name}[0] = {output_dtype}({accumulator})")
        if fallback_guard is None:
            for statement in full_reduction:
                self.cg.add_statement(statement)
        else:
            self.cg.add_statement(
                ast.If(
                    test=cast("ast.expr", expr_from_string(fallback_guard)),
                    body=cast("list[ast.stmt]", full_reduction),
                    orelse=[],
                )
            )
        return result

    def _emit_coordinate(self, node: Node) -> RowFragment:
        progression = row_fragment_arange(node)
        is_row = node.target is tile_index
        if not is_row:
            assert progression is not None
        extent = 1 if progression is None else progression[2]
        result = self._new_fragment(node, extent, self.layout, replicated=is_row)
        register = self.cg.device_function.new_var("row_coordinate")
        expression = self.emit_pointwise_scalar(
            node, [], logical_column=self.layout.column(register)
        )
        self._add(
            f"for {register} in cutlass.range_constexpr({result.num_registers}):\n"
            f"    {result.name}[{register}] = {ast.unparse(expression)}"
        )
        return result

    def emit(self, node: Node) -> RowFragment:
        if node in self.fragments:
            return self.fragments[node]
        if self.resolve is not None and (resolved := self.resolve(node)) is not node:
            fragment = self.emit(resolved)
            self.fragments[node] = fragment
            return fragment
        if not supports_row_fragment_node(node):
            raise self._unsupported(node, "unsupported operation or shape")
        if node.target is tile_index or row_fragment_arange(node) is not None:
            fragment = self._emit_coordinate(node)
        elif node.target is load:
            fragment = self.load(node)
        elif node.target in _VIEW_TARGETS:
            fragment = self._emit_view(node)
        elif (reduction := _row_reduction(node)) is not None:
            fragment = self._emit_reduction(node, reduction)
        else:
            fragment = self._emit_pointwise(node)
        self.bind(node, fragment)
        return fragment
