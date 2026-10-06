"""Retain packed-byte provenance from a proved scaled contraction.

This is a narrower consumer of ``scaled_contraction``'s arithmetic proof. The
native format has eight ordered packed bytes and one E4M3 scale per logical
K=16 group. Recipes contain integer coordinates and metadata constants only;
memory indirection, predicates and opaque calls remain on the BF16 path.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import tile_ops
from ...language import view_ops
from ...language.quantized_ops import float4_e2m1fn_x2_to_float32
from ..compile_environment import CompileEnvironment
from ..indexing_strategy import subscript_tile_info

if TYPE_CHECKING:
    from torch.fx import Node

    from .scaled_contraction import _ScaledPair


BLOCK_SCALED_PROVENANCE = "cute_block_scaled_provenance"


@dataclass(frozen=True)
class IntegerRecipe:
    operation: str
    operands: tuple[IntegerRecipe, ...] = ()
    value: int | sympy.Expr = 0
    bits: int = 64


@dataclass(frozen=True)
class PackedOperand:
    tensor: torch.Tensor
    scale: torch.Tensor
    byte_indices: tuple[tuple[IntegerRecipe, ...], ...]
    scale_indices: tuple[IntegerRecipe, ...]
    row_axis: int


@dataclass(frozen=True)
class BlockScaledProvenance:
    lhs: PackedOperand
    rhs: PackedOperand
    group_axis: int
    loop_graph_id: int
    fact_id: int
    operand_fingerprint: tuple[object, ...]


def operand_fingerprint(
    operands: tuple[Node, Node], *, budget: int = 4096
) -> tuple[object, ...] | None:
    """Seal the successful arithmetic proof across later FX graph copies.

    Number dependencies once in postorder rather than expanding shared DAGs.
    Any later arithmetic, index, mask or dtype edit invalidates provenance.
    """
    memo: dict[torch.fx.Node, int] = {}
    definitions: list[object] = []
    remaining = budget
    declined = False

    def visit(value: object, depth: int = 0) -> object:
        nonlocal remaining, declined
        remaining -= 1
        if remaining < 0 or depth > 128:
            declined = True
            return None
        if isinstance(value, torch.fx.Node):
            if value not in memo:
                arguments = visit(value.args, depth + 1)
                keywords = visit(dict(value.kwargs), depth + 1)
                metadata = value.meta.get("val")
                typed = (
                    (str(metadata.dtype), tuple(map(str, metadata.shape)))
                    if isinstance(metadata, torch.Tensor)
                    else None
                )
                memo[value] = len(definitions)
                definitions.append((value.op, value.target, arguments, keywords, typed))
            return ("node", memo[value])
        if isinstance(value, (list, tuple)):
            return (
                type(value).__name__,
                tuple(visit(item, depth + 1) for item in value),
            )
        if isinstance(value, dict):
            return (
                "dict",
                tuple(
                    (visit(key, depth + 1), visit(item, depth + 1))
                    for key, item in value.items()
                ),
            )
        if isinstance(value, slice):
            return (
                "slice",
                visit(value.start, depth + 1),
                visit(value.stop, depth + 1),
                visit(value.step, depth + 1),
            )
        if value is None or isinstance(
            value,
            (int, float, str, torch.dtype, torch.device, torch.SymInt, sympy.Basic),
        ):
            return (type(value).__name__, str(value))
        declined = True
        return None

    result = visit(operands)
    return None if declined else (result, tuple(definitions))


_INTEGER_BINARY: dict[object, str] = {
    operator.add: "add",
    operator.sub: "sub",
    operator.mul: "mul",
    operator.floordiv: "floordiv",
    operator.mod: "mod",
    torch.ops.aten.add.Tensor: "add",
    torch.ops.aten.sub.Tensor: "sub",
    torch.ops.aten.mul.Tensor: "mul",
    torch.ops.aten.remainder.Scalar: "mod",
    torch.ops.aten.remainder.Tensor: "mod",
}


def _integer_bits(node: Node) -> int | None:
    value = node.meta.get("val")
    if isinstance(value, torch.Tensor):
        return {torch.int32: 32, torch.int64: 64}.get(value.dtype)
    if type(value) is int or isinstance(value, torch.SymInt):
        return 64
    return None


def _size_axis(value: object, env: CompileEnvironment) -> int | None:
    if isinstance(value, torch.fx.Node):
        value = value.meta.get("val")
    if isinstance(value, torch.SymInt):
        expression = value._sympy_()
        return next(
            (
                env.canonical_block_id(index)
                for index, block in enumerate(env.block_sizes)
                if expression == sympy.sympify(block.var)
            ),
            None,
        )
    return None


def integer_recipe(
    node: object,
    env: CompileEnvironment,
    *,
    subscript: bool = False,
    budget: int = 512,
) -> IntegerRecipe | None:
    """Translate a bounded, typed integer DAG without evaluating input data."""
    memo: dict[torch.fx.Node, IntegerRecipe | None] = {}
    remaining = budget
    depth = 0

    def visit(value: object, as_subscript: bool = False) -> IntegerRecipe | None:
        nonlocal remaining, depth
        remaining -= 1
        if remaining < 0 or depth > 128:
            return None
        if type(value) is int:
            return IntegerRecipe("constant", value=value)
        if as_subscript:
            info = subscript_tile_info(env, value)
            if info is not None:
                if not env.known_equal(info.offset, 0) or info.block_size is not None:
                    return None
                return IntegerRecipe(
                    "coordinate",
                    value=env.canonical_block_id(info.block_id),
                    bits=32 if env.index_dtype == torch.int32 else 64,
                )
        if not isinstance(value, torch.fx.Node) or value.op != "call_function":
            return None
        if value in memo:
            return memo[value]
        memo[value] = None
        depth += 1
        result = translate(value)
        depth -= 1
        memo[value] = result
        return result

    def translate(node: Node) -> IntegerRecipe | None:
        bits = _integer_bits(node)
        if (
            node.target is tile_ops.tile_index
            and len(node.args) == 1
            and not node.kwargs
        ):
            axis = _size_axis(node.args[0], env)
            if axis is not None and bits is not None:
                return IntegerRecipe("coordinate", value=axis, bits=bits)
            return None
        if node.target in (memory_ops.load, view_ops.subscript):
            if len(node.args) not in (2, 4) or node.kwargs:
                return None
            if len(node.args) == 4 and node.args[2:] != (None, None):
                return None
            indices = node.args[1]
            if not isinstance(indices, (tuple, list)) or any(
                item is not None and item != slice(None) for item in indices
            ):
                return None
            # This only unwraps a tile-index tensor. A host tensor cannot be
            # translated below, so a tensor load never becomes an integer fact.
            return visit(node.args[0])
        if bits is None:
            return None
        if (
            node.target is torch.ops.prims.convert_element_type.default
            and len(node.args) == 2
            and node.args[1] in (torch.int32, torch.int64)
            and not node.kwargs
        ):
            operand = visit(node.args[0])
            return IntegerRecipe("cast", (operand,), bits=bits) if operand else None
        operation = _INTEGER_BINARY.get(node.target)
        if node.target is torch.ops.aten.div.Tensor_mode and node.kwargs == {
            "rounding_mode": "floor"
        }:
            operation = "floordiv"
        elif node.kwargs:
            return None
        if operation is not None and len(node.args) == 2:
            left, right = (visit(argument) for argument in node.args)
            if left is not None and right is not None:
                return IntegerRecipe(operation, (left, right), bits=bits)
            return None
        if node.target not in (_tracing_ops._get_symnode, torch.ops.aten.sym_size.int):
            return None
        value = node.meta.get("val")
        if type(value) is int:
            return IntegerRecipe("constant", value=value, bits=bits)
        if isinstance(value, torch.SymInt):
            # Metadata expressions may be resolved later, only after the owning
            # input-metadata specialization fact has been registered.
            expression = value._sympy_()
            if not expression.free_symbols.intersection(
                sympy.sympify(block.var) for block in env.block_sizes
            ):
                return IntegerRecipe(
                    "constant", value=cast("sympy.Expr", expression), bits=bits
                )
        return None

    return visit(node, subscript)


def _plain_load(node: object, dtype: torch.dtype) -> Node | None:
    if (
        not isinstance(node, torch.fx.Node)
        or node.target is not memory_ops.load
        or len(node.args) != 4
        or node.args[2:] != (None, None)
        or node.kwargs
        or not isinstance(node.args[0], torch.fx.Node)
    ):
        return None
    tensor = node.args[0].meta.get("val")
    return node if isinstance(tensor, torch.Tensor) and tensor.dtype == dtype else None


def _packed_pair(pair: tuple[Node, Node]) -> Node | None:
    low, high = pair
    if (
        low.target is not operator.getitem
        or high.target is not operator.getitem
        or len(low.args) != 2
        or len(high.args) != 2
        or low.args[1] != 0
        or high.args[1] != 1
        or low.args[0] is not high.args[0]
    ):
        return None
    convert = low.args[0]
    if (
        not isinstance(convert, torch.fx.Node)
        or convert.target is not float4_e2m1fn_x2_to_float32
        or len(convert.args) != 1
        or convert.kwargs
    ):
        return None
    return _plain_load(convert.args[0], torch.float4_e2m1fn_x2)


def _indices(load: Node, env: CompileEnvironment) -> tuple[IntegerRecipe, ...] | None:
    tensor = cast("Node", load.args[0]).meta["val"]
    indices = load.args[1]
    if not isinstance(indices, (list, tuple)) or len(indices) != tensor.ndim:
        return None
    recipes = tuple(integer_recipe(value, env, subscript=True) for value in indices)
    if any(recipe is None for recipe in recipes):
        return None
    return tuple(recipe for recipe in recipes if recipe is not None)


def capture_block_scaled_provenance(
    pairs: list[_ScaledPair],
    *,
    lowered_operands: tuple[Node, Node],
    group_axis: int,
    row_axes: tuple[int, int],
    loop_graph_id: int,
    fact_id: int,
) -> BlockScaledProvenance | None:
    """Consume the successful arithmetic proof before decoded nodes are erased."""
    if len(pairs) != 8:
        return None
    fingerprint = operand_fingerprint(lowered_operands)
    if fingerprint is None:
        return None
    env = CompileEnvironment.current()
    operands: list[PackedOperand] = []
    for side, row_axis in zip(("lhs", "rhs"), row_axes, strict=True):
        terms = [pair.lhs if side == "lhs" else pair.rhs for pair in pairs]
        scales = [pair.lhs_scale if side == "lhs" else pair.rhs_scale for pair in pairs]
        if any(scale is not scales[0] for scale in scales[1:]):
            return None
        scale = _plain_load(scales[0].args[0], torch.float8_e4m3fn)
        loads = [_packed_pair(pair) for pair in terms]
        if scale is None or any(load is None for load in loads):
            return None
        checked_loads = [load for load in loads if load is not None]
        tensor = cast("Node", checked_loads[0].args[0]).meta["val"]
        if any(
            cast("Node", load.args[0]).meta["val"] is not tensor
            for load in checked_loads
        ):
            return None
        indices = tuple(_indices(load, env) for load in checked_loads)
        scale_indices = _indices(scale, env)
        if scale_indices is None or any(item is None for item in indices):
            return None
        operands.append(
            PackedOperand(
                tensor=tensor,
                scale=cast("Node", scale.args[0]).meta["val"],
                byte_indices=tuple(item for item in indices if item is not None),
                scale_indices=scale_indices,
                row_axis=row_axis,
            )
        )
    return BlockScaledProvenance(
        *operands,
        group_axis=group_axis,
        loop_graph_id=loop_graph_id,
        fact_id=fact_id,
        operand_fingerprint=fingerprint,
    )
