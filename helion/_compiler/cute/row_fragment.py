"""Compose ordinary graph operations over subgroup-owned register fragments.

The caller supplies loads and operation boundaries; this module owns scalar
pointwise lowering, broadcasting, and last-axis reductions.  A fragment's
layout describes unique logical elements, rather than a particular producer
such as a sorting network or a matrix multiplication.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import itertools
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.virtualized import V
from torch.fx.node import map_arg

from ... import exc
from ...language._tracing_ops import _mask_to
from ...language.memory_ops import load
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


_VIEW_TARGETS = frozenset(
    {
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.unsqueeze.default,
    }
)
_CAST_TARGETS = frozenset({torch.ops.prims.convert_element_type.default})


@dataclass(frozen=True)
class RowFragmentLayout:
    """Contiguous vectors distributed cyclically over a lane subgroup."""

    lanes: int
    vector_width: int
    lane_expr: str

    def __post_init__(self) -> None:
        assert 0 < self.lanes <= 32 and self.lanes & (self.lanes - 1) == 0
        assert self.vector_width > 0

    def column(self, register: str | int) -> str:
        vector = self.vector_width
        return (
            f"((({register}) // {vector} * {self.lanes} + ({self.lane_expr}))"
            f" * {vector} + ({register}) % {vector})"
        )

    def num_registers(self, extent: int) -> int:
        width = self.lanes * self.vector_width
        return (extent + width - 1) // width * self.vector_width


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
    source_shape = [size for size in source.shape if not _same_extent(size, 1)]
    output_shape = [size for size in output.shape if not _same_extent(size, 1)]
    return len(source_shape) == len(output_shape) and all(
        itertools.starmap(_same_extent, zip(source_shape, output_shape, strict=True))
    )


def supports_row_fragment_node(node: Node) -> bool:
    """Whether this node has a producer-independent fragment implementation.

    Loads are boundaries delegated to the caller.  Their indexing and alias
    safety, and the availability of every graph dependency, are caller proofs.
    """
    if node.op != "call_function" or _tensor(node) is None:
        return False
    if node.target is load or node.target is _mask_to or node.target in _CAST_TARGETS:
        return True
    if node.target in _VIEW_TARGETS:
        return _preserves_row_columns(node)
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
    ) -> None:
        self.cg = cg
        self.load = load
        self.layout = layout
        self.valid_row = valid_row
        self.resolve = resolve
        self.fragments: dict[Node, RowFragment] = {}

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
    ) -> ast.AST:
        """Apply ordinary scalar lowering, preserving each node's dtype boundary.

        The caller can also use this for recomputation at a selected logical
        index.  Statements needed by a lowering are emitted through ``cg``.
        """
        backend = CompileEnvironment.current().backend
        output = _tensor(node)
        assert output is not None
        if node.target in _VIEW_TARGETS or node.target in _CAST_TARGETS:
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
                    result = lowering.codegen_from_input_asts(
                        context, node, input_values
                    )
                elif node.target is torch.ops.aten.where.self:
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

    def _emit_view(self, node: Node) -> RowFragment:
        source = self.emit(row_fragment_tensor_inputs(node)[0])
        output = _tensor(node)
        assert output is not None
        if source.replicated:
            if output.ndim < 2 or _same_extent(output.shape[-1], 1):
                return source
            raise self._unsupported(node, "view moves the replicated row axis")
        if output.ndim == 0 or (
            row_fragment_logical_extent(output.shape[-1]) != source.extent
        ):
            raise self._unsupported(node, "view moves the distributed column axis")
        return source

    def emit(self, node: Node) -> RowFragment:
        if node in self.fragments:
            return self.fragments[node]
        if self.resolve is not None and (resolved := self.resolve(node)) is not node:
            fragment = self.emit(resolved)
            self.fragments[node] = fragment
            return fragment
        if not supports_row_fragment_node(node):
            raise self._unsupported(node, "unsupported operation or shape")
        if node.target is load:
            fragment = self.load(node)
        elif node.target in _VIEW_TARGETS:
            fragment = self._emit_view(node)
        else:
            fragment = self._emit_pointwise(node)
        self.bind(node, fragment)
        return fragment
