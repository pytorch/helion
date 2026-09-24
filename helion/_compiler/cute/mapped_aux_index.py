"""Conservative, geometry-independent addresses for mapped epilogue loads.

Only pointwise integer arithmetic and register full-slice/None broadcasting are
accepted. Canonical symbolic axes, not equal extents, establish correspondence
to output coordinates. Every original intermediate must fit signed int32 before
the existing logical-index interpreter may simplify it. Pointer/alias safety,
output validity predicates and transport selection remain the caller's job.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import tile_index
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction
from .fragment_epilogue import _Index
from .fragment_epilogue import _logical_index_value
from .fragment_epilogue import _UnsupportedFragment

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...runtime.config import Config


_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1


@dataclasses.dataclass(frozen=True)
class MappedIntegerProof:
    """An original FX intermediate's type/range before simplification."""

    node_name: str
    operation: str
    dtype: torch.dtype
    minimum: int
    maximum: int
    block_dependencies: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class MappedAuxIndex:
    """Immutable coordinate expressions; no mutable FX nodes are retained."""

    host_name: str
    input_shape: tuple[int, ...]
    input_strides: tuple[int, ...]
    input_dtype: torch.dtype
    output_block_ids: tuple[int, ...]
    output_shape: tuple[int, ...]
    indices: tuple[_Index, ...]
    integer_proofs: tuple[MappedIntegerProof, ...]
    block_dependencies: tuple[int, ...]


class _UnsupportedMappedIndex(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class _ValueProof:
    axes: tuple[int | None, ...]
    minimum: int
    maximum: int
    dependencies: frozenset[int]


def _static_positive(values: Sequence[object]) -> tuple[int, ...]:
    if not all(type(value) is int and value > 0 for value in values):
        raise _UnsupportedMappedIndex
    return cast("tuple[int, ...]", tuple(values))


def _broadcast_axes(
    left: tuple[int | None, ...], right: tuple[int | None, ...]
) -> tuple[int | None, ...]:
    rank = max(len(left), len(right))
    result: list[int | None] = []
    for a, b in zip(
        (None,) * (rank - len(left)) + left,
        (None,) * (rank - len(right)) + right,
        strict=True,
    ):
        if a is not None and b is not None and a != b:
            raise _UnsupportedMappedIndex
        result.append(a if a is not None else b)
    return tuple(result)


def _prove_rendered_range(index: _Index) -> None:
    """Simplification must not introduce overflowing intermediate arithmetic."""
    bounds = {str(symbol): (lower, upper) for symbol, lower, upper in index.bounds}
    expression = ast.parse(index.render({name: name for name in bounds}), mode="eval")

    def visit(node: ast.AST) -> tuple[int, int]:
        if isinstance(node, ast.Name) and node.id in bounds:
            lower, upper = bounds[node.id]
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            lower = upper = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            low, high = visit(node.operand)
            lower, upper = -high, -low
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cutlass"
            and node.func.attr == "Int32"
            and len(node.args) == 1
            and not node.keywords
        ):
            lower, upper = visit(node.args[0])
        elif isinstance(node, ast.BinOp):
            lo, hi = visit(node.left)
            rlo, rhi = visit(node.right)
            if isinstance(node.op, ast.Add):
                lower, upper = lo + rlo, hi + rhi
            elif isinstance(node.op, ast.Mult):
                products = (lo * rlo, lo * rhi, hi * rlo, hi * rhi)
                lower, upper = min(products), max(products)
            elif isinstance(node.op, (ast.FloorDiv, ast.Mod)):
                if lo < 0 or rlo != rhi or rlo <= 0:
                    raise _UnsupportedMappedIndex
                if isinstance(node.op, ast.FloorDiv):
                    lower, upper = lo // rlo, hi // rlo
                else:
                    lower, upper = (lo, hi) if hi < rlo else (0, rlo - 1)
            else:
                raise _UnsupportedMappedIndex
        else:
            raise _UnsupportedMappedIndex
        if lower < _INT32_MIN or upper > _INT32_MAX:
            raise _UnsupportedMappedIndex
        return lower, upper

    visit(expression.body)


class _Analyzer:
    def __init__(
        self, output_ids: tuple[int, ...], output_shape: tuple[int, ...]
    ) -> None:
        self.env = CompileEnvironment.current()
        self.extents = dict(zip(output_ids, output_shape, strict=True))
        self.proofs: dict[Node, _ValueProof] = {}
        self.records: list[MappedIntegerProof] = []

    def axes(self, node: Node) -> tuple[int | None, ...]:
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            raise _UnsupportedMappedIndex
        axes: list[int | None] = []
        for size in value.shape:
            if type(size) is int and size == 1:
                axes.append(None)
                continue
            block_id = self.env.get_block_id(size)
            if block_id is None:
                raise _UnsupportedMappedIndex
            block_id = self.env.canonical_block_id(block_id)
            if block_id not in self.extents:
                raise _UnsupportedMappedIndex
            axes.append(block_id)
        return tuple(axes)

    def argument(self, value: object) -> _ValueProof:
        if isinstance(value, Node):
            return self.visit(value)
        if type(value) is int and _INT32_MIN <= value <= _INT32_MAX:
            return _ValueProof((), value, value, frozenset())
        raise _UnsupportedMappedIndex

    def visit(self, node: Node) -> _ValueProof:
        if node in self.proofs:
            return self.proofs[node]
        value = node.meta.get("val")
        if (
            node.op != "call_function"
            or not isinstance(value, torch.Tensor)
            or value.dtype not in (torch.int32, torch.int64)
        ):
            raise _UnsupportedMappedIndex
        axes = self.axes(node)
        target = node.target
        operation: str
        if target is tile_index:
            if node.kwargs or len(node.args) != 1 or len(axes) != 1:
                raise _UnsupportedMappedIndex
            size_node = node.args[0]
            size = (
                size_node.meta.get("val") if isinstance(size_node, Node) else size_node
            )
            if not isinstance(size, (int, torch.SymInt, sympy.Basic)):
                raise _UnsupportedMappedIndex
            block_id = self.env.get_block_id(size)
            if block_id is None:
                raise _UnsupportedMappedIndex
            block_id = self.env.canonical_block_id(block_id)
            if axes != (block_id,) or block_id not in self.extents:
                raise _UnsupportedMappedIndex
            result = _ValueProof(
                axes, 0, self.extents[block_id] - 1, frozenset((block_id,))
            )
            operation = "tile_index"
        elif target is memory_ops.load:
            if (
                node.kwargs
                or not 2 <= len(node.args) <= 4
                or any(item is not None for item in node.args[2:])
            ):
                raise _UnsupportedMappedIndex
            source, indices = node.args[:2]
            if not isinstance(source, Node) or not isinstance(indices, (list, tuple)):
                raise _UnsupportedMappedIndex
            source_proof = self.visit(source)  # A host/data leaf is rejected here.
            expected: list[int | None] = []
            source_axis = 0
            for index in indices:
                if index is None:
                    expected.append(None)
                elif index == slice(None) and source_axis < len(source_proof.axes):
                    expected.append(source_proof.axes[source_axis])
                    source_axis += 1
                else:
                    raise _UnsupportedMappedIndex
            if source_axis != len(source_proof.axes) or tuple(expected) != axes:
                raise _UnsupportedMappedIndex
            result = dataclasses.replace(source_proof, axes=axes)
            operation = "register_fullslice_none"
        else:
            if len(node.args) != 2:
                raise _UnsupportedMappedIndex
            left = self.argument(node.args[0])
            right = self.argument(node.args[1])
            if _broadcast_axes(left.axes, right.axes) != axes:
                raise _UnsupportedMappedIndex
            lo, hi = left.minimum, left.maximum
            rlo, rhi = right.minimum, right.maximum
            if target in (torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar):
                if node.kwargs not in ({}, {"alpha": 1}):
                    raise _UnsupportedMappedIndex
                lower, upper = lo + rlo, hi + rhi
                operation = "add"
            elif target in (torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar):
                if node.kwargs not in ({}, {"alpha": 1}):
                    raise _UnsupportedMappedIndex
                lower, upper = lo - rhi, hi - rlo
                operation = "sub"
            elif target in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar):
                if node.kwargs:
                    raise _UnsupportedMappedIndex
                products = (lo * rlo, lo * rhi, hi * rlo, hi * rhi)
                lower, upper = min(products), max(products)
                operation = "mul"
            elif target in (
                torch.ops.aten.div.Tensor_mode,
                torch.ops.aten.floor_divide.default,
                torch.ops.aten.remainder.Scalar,
            ):
                divisor = node.args[1]
                if type(divisor) is not int or divisor <= 0 or lo < 0:
                    raise _UnsupportedMappedIndex
                expected_kwargs = (
                    {"rounding_mode": "floor"}
                    if target is torch.ops.aten.div.Tensor_mode
                    else {}
                )
                if node.kwargs != expected_kwargs:
                    raise _UnsupportedMappedIndex
                if target is torch.ops.aten.remainder.Scalar:
                    lower, upper = (lo, hi) if hi < divisor else (0, divisor - 1)
                    operation = "mod"
                else:
                    lower, upper = lo // divisor, hi // divisor
                    operation = "floor_div"
            else:
                raise _UnsupportedMappedIndex
            result = _ValueProof(
                axes, lower, upper, left.dependencies | right.dependencies
            )
        # Also restrict int64 expressions to a proven int32-safe subset, because
        # the shared interpreter renders signed Int32 constants/coordinates.
        if result.minimum < _INT32_MIN or result.maximum > _INT32_MAX:
            raise _UnsupportedMappedIndex
        self.proofs[node] = result
        self.records.append(
            MappedIntegerProof(
                node.name,
                operation,
                value.dtype,
                result.minimum,
                result.maximum,
                tuple(sorted(result.dependencies)),
            )
        )
        return result


def analyze_mapped_aux_load(
    load: Node,
    *,
    output_block_ids: Sequence[int],
    output_shape: Sequence[int],
    config: Config | None = None,
) -> MappedAuxIndex | None:
    """Prove a host load's addresses for full, static rank2/3 output domains.

    Requires active CompileEnvironment AND HostFunction. Pre-config callers may
    omit config: default_config only resolves shapes at local flat zero; it is
    not a selected tuning configuration. Axis correspondence is proved before
    replacing tile origins by global coordinates, so the result is independent
    of the supplied legal tile configuration. Unknown geometry fails closed.
    """
    env = CompileEnvironment.current()
    HostFunction.current()
    try:
        shape = _static_positive(output_shape)
        if len(shape) not in (2, 3) or len(shape) != len(output_block_ids):
            return None
        if any(
            type(block_id) is not int or not 0 <= block_id < len(env.block_sizes)
            for block_id in output_block_ids
        ):
            return None
        ids = tuple(env.canonical_block_id(block_id) for block_id in output_block_ids)
        if len(set(ids)) != len(ids) or any(size - 1 > _INT32_MAX for size in shape):
            return None
        if any(
            type(env.block_sizes[block_id].size) is not int
            or env.block_sizes[block_id].size != size
            for block_id, size in zip(ids, shape, strict=True)
        ):
            return None
        if (
            load.op != "call_function"
            or load.target is not memory_ops.load
            or load.kwargs
            or not 2 <= len(load.args) <= 4
            or any(item is not None for item in load.args[2:])
        ):
            return None
        host, indices = load.args[:2]
        if (
            not isinstance(host, Node)
            or host.op != "call_function"
            or host.target is not _tracing_ops._host_tensor
            or len(host.args) != 1
            or not isinstance(host.args[0], str)
            or host.kwargs
            or not isinstance(indices, (list, tuple))
        ):
            return None
        tensor = host.meta.get("val")
        if (
            not isinstance(tensor, torch.Tensor)
            or not indices
            or len(indices) != tensor.ndim
        ):
            return None
        input_shape = _static_positive(tensor.shape)
        strides = tuple(tensor.stride())
        if not all(type(stride) is int and stride >= 0 for stride in strides):
            return None
        analyzer = _Analyzer(ids, shape)
        integer_indices: list[Node] = []
        for index, extent in zip(indices, input_shape, strict=True):
            if not isinstance(index, Node):
                return None
            proof = analyzer.visit(index)
            if len(proof.axes) > len(ids):
                return None
            aligned = (None,) * (len(ids) - len(proof.axes)) + proof.axes
            if any(
                axis is not None and axis != expected
                for axis, expected in zip(aligned, ids, strict=True)
            ):
                return None
            if proof.minimum < 0 or proof.maximum >= extent:
                return None
            integer_indices.append(index)
        resolver_config = (
            config if config is not None else env.config_spec.default_config()
        )
        origins = {
            block_id: _Index.variable(f"output_{axis}", extent - 1)
            for axis, (block_id, extent) in enumerate(zip(ids, shape, strict=True))
        }
        # Geometry independence follows from the restricted operation and axis
        # proof above. Every local coordinate is zero; origins become globals.
        expressions = tuple(
            _logical_index_value(
                index,
                _Index.constant(0),
                config=resolver_config,
                tile_origins=origins,
                memo={},
            )
            for index in integer_indices
        )
        for expression in expressions:
            _prove_rendered_range(expression)
        dependencies = tuple(
            sorted(
                set().union(*(proof.dependencies for proof in analyzer.proofs.values()))
            )
        )
        return MappedAuxIndex(
            host.args[0],
            input_shape,
            cast("tuple[int, ...]", strides),
            tensor.dtype,
            ids,
            shape,
            expressions,
            tuple(analyzer.records),
            dependencies,
        )
    except (_UnsupportedMappedIndex, _UnsupportedFragment):
        return None


def render_mapped_aux_indices(
    mapped: MappedAuxIndex, global_coordinates: Sequence[ast.expr]
) -> tuple[ast.expr, ...]:
    """Render tensor element subscripts, not byte addresses or raw pointers.

    Coordinates must be authoritative GLOBAL output coordinates in the declared
    canonical-axis order and inside output_shape. Caller supplies predicates and
    applies input strides/layout; no lane or tile-order inference happens here.
    """
    if len(global_coordinates) != len(mapped.output_block_ids):
        raise ValueError("Mapped auxiliary coordinates must match output rank")
    names = {
        f"output_{axis}": f"cutlass.Int32({ast.unparse(value)})"
        for axis, value in enumerate(global_coordinates)
    }
    return tuple(
        ast.parse(index.render(names), mode="eval").body for index in mapped.indices
    )
