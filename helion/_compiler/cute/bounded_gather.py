"""Static logical bounds for opt-in complete-fragment gathers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._inductor.ir import Reduction
from torch.fx import Node

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo


@dataclass(frozen=True)
class GatherBounds:
    source_shape: tuple[sympy.Expr, ...]
    index_shape: tuple[sympy.Expr, ...]
    lower: int
    upper: int


def all_axis_reduction_shape(
    node: Node,
    infer: Callable[[Node], tuple[sympy.Expr, ...] | None],
    dimension: Callable[[object], sympy.Expr],
) -> tuple[sympy.Expr, ...] | None:
    """Prove only an existing value reduction's canonical all-axis domain.

    Bind the current call against its exact schema. The emitter still owns
    masks, accumulation and dtype conversions; padding is not a logical axis.
    """
    from ..inductor_lowering import ReductionLowering

    kinds = {
        torch.ops.aten.amin.default: "min",
        torch.ops.aten.amax.default: "max",
        torch.ops.aten.min.default: "min",
        torch.ops.aten.max.default: "max",
        torch.ops.aten.sum.default: "sum",
        torch.ops.aten.sum.dim_IntList: "sum",
        torch.ops.aten.prod.default: "prod",
    }
    lowering = node.meta.get("lowering")
    if (
        node.target not in kinds
        or not isinstance(lowering, ReductionLowering)
        or lowering.reduction_type != kinds[node.target]
    ):
        return None
    target = cast("torch._ops.OpOverload", node.target)
    schema = target._schema.arguments
    if set(node.kwargs) - {arg.name for arg in schema}:
        return None
    bound: dict[str, object] = {}
    position = 0
    for arg in schema:
        if not arg.kwarg_only and position < len(node.args):
            if arg.name in node.kwargs:
                return None
            bound[arg.name] = node.args[position]
            position += 1
        elif arg.name in node.kwargs:
            bound[arg.name] = node.kwargs[arg.name]
        elif arg.has_default_value():
            bound[arg.name] = arg.default_value
        else:
            return None
    if position != len(node.args):
        return None
    if "dim" in bound:
        dims = bound["dim"]
        if dims is None:
            if target is not torch.ops.aten.sum.dim_IntList:
                return None
        elif not isinstance(dims, (list, tuple)) or dims:
            # All explicit nonempty dimension lists retain their old path.
            return None
    keepdim = bound.get("keepdim", False)
    if type(keepdim) is not bool:
        return None
    source = bound.get("self")
    output = node.meta.get("val")
    if not isinstance(source, Node) or not isinstance(output, torch.Tensor):
        return None
    fake = source.meta.get("val")
    logical = infer(source)
    if (
        not isinstance(fake, torch.Tensor)
        or logical is None
        or len(logical) != fake.ndim
    ):
        return None
    physical = tuple(dimension(size) for size in fake.shape)
    if any(
        sympy.expand(size) != sympy.expand(capacity)
        or sympy.StrictGreaterThan(size, 0) is not sympy.true
        for size, capacity in zip(logical, physical, strict=True)
    ):
        return None
    # Rank zero contains one scalar; a zero-volume tensor does not gain support.
    volume = sympy.prod(logical)
    data = lowering.buffer.data
    if not isinstance(data, Reduction):
        return None
    if (
        sympy.expand(volume)
        != sympy.expand(sympy.prod(dimension(size) for size in data.reduction_ranges))
        or output.dtype != lowering.buffer.get_dtype()
        or (bound.get("dtype") is not None and bound["dtype"] != output.dtype)
    ):
        return None
    result = (sympy.Integer(1),) * len(logical) if keepdim else ()
    if output.ndim != len(result) or any(
        dimension(size) != expected
        for size, expected in zip(output.shape, result, strict=True)
    ):
        return None
    # The existing emitter may flatten singleton output axes. Their volume and
    # every retained range must still be exactly one for an all-axis result.
    if any(dimension(size) != 1 for size in data.ranges):
        return None
    return result


def inline_asm_shape(
    node: Node,
    infer: Callable[[Node], tuple[sympy.Expr, ...] | None],
    dimension: Callable[[object], sympy.Expr],
) -> tuple[sympy.Expr, ...] | None:
    """Prove the existing pure scalar-assembly broadcast's logical domain.

    Physical shapes check the emitter's coordinate mapping only. No extent or
    integer value bound is inferred from the opaque assembly or padded output.
    Tuple outputs and empty/context-dependent operand domains remain unsupported.
    """
    if (
        len(node.args) != 6
        or node.args[4] is not True
        or type(node.args[5]) is not int
        or node.args[5] != 1
        or not isinstance(node.args[3], torch.dtype)
    ):
        return None
    output = node.meta.get("val")
    operands = node.args[2]
    if (
        not isinstance(output, torch.Tensor)
        or output.dtype != node.args[3]
        or not isinstance(operands, (list, tuple))
        or not operands
    ):
        return None
    logical = []
    physical = []
    for operand in operands:
        if not isinstance(operand, Node):
            return None
        fake = operand.meta.get("val")
        shape = infer(operand)
        if (
            not isinstance(fake, torch.Tensor)
            or shape is None
            or len(shape) != fake.ndim
            or any(
                (size == 1) != (dimension(capacity) == 1)
                for size, capacity in zip(shape, fake.shape, strict=True)
            )
        ):
            return None
        logical.append(shape)
        physical.append(tuple(dimension(size) for size in fake.shape))
    rank = max(map(len, logical))
    if output.ndim != rank:
        return None
    result: list[sympy.Expr] = [sympy.Integer(1)] * rank
    for shape in logical:
        for axis, size in enumerate((sympy.Integer(1),) * (rank - len(shape)) + shape):
            if size == 1:
                continue
            if result[axis] != 1 and sympy.expand(result[axis]) != sympy.expand(size):
                return None
            result[axis] = size
    # Fragment.broadcast right-aligns operands and uses coordinate zero only
    # for physical singleton axes. Check exactly that mapping against output.
    for axis, capacity in enumerate(output.shape):
        sizes = [
            ((sympy.Integer(1),) * (rank - len(shape)) + shape)[axis]
            for shape in physical
        ]
        non_singleton = [size for size in sizes if size != 1]
        expected = non_singleton[0] if non_singleton else sympy.Integer(1)
        if any(
            sympy.expand(size) != sympy.expand(expected) for size in non_singleton
        ) or sympy.expand(dimension(capacity)) != sympy.expand(expected):
            return None
    return tuple(result)


def indexed_load_shapes(
    env: CompileEnvironment,
    indices: list[object],
    infer: Callable[[Node], tuple[sympy.Expr, ...] | None],
    dimension: Callable[[object], sympy.Expr],
) -> dict[int, tuple[sympy.Expr, ...]] | None:
    """Prove tensor-index axes using the existing physical indexing mode.

    Index values, address masks and padding do not determine output domains.
    The physical helpers select only the axis mapping; every extent comes from
    the index producer's logical shape. Unknown or incompatible axes decline.
    """
    tensors = {
        i: index.meta["val"]
        for i, index in enumerate(indices)
        if isinstance(index, Node) and isinstance(index.meta.get("val"), torch.Tensor)
    }
    logical: dict[int, tuple[sympy.Expr, ...]] = {}
    for i, tensor in tensors.items():
        shape = infer(cast("Node", indices[i]))
        if shape is None or len(shape) != tensor.ndim:
            return None
        logical[i] = shape
    if not tensors:
        return {}
    fake_indices = [
        index.meta.get("val") if isinstance(index, Node) else index for index in indices
    ]
    if env.should_broadcast_tensor_indexers(fake_indices):
        if all(tensor.ndim == 1 for tensor in tensors.values()):
            # Multiple 1D indexers use the existing Cartesian convention.
            result = tuple(shape[0] for shape in logical.values())
        else:
            rank = max(map(len, logical.values()))
            merged: list[sympy.Expr] = [sympy.Integer(1)] * rank
            for shape in logical.values():
                for axis, size in enumerate(
                    (sympy.Integer(1),) * (rank - len(shape)) + shape
                ):
                    if size == 1:
                        continue
                    if merged[axis] != 1 and sympy.expand(merged[axis]) != sympy.expand(
                        size
                    ):
                        return None
                    merged[axis] = size
            result = tuple(merged)
        if len(result) != len(
            env.tensor_indexer_broadcast_shape(list(tensors.values()))
        ):
            return None
        # Singleton axes must mean the same thing to the existing emitter.
        if any(
            (size == 1) != (dimension(physical) == 1)
            for i, tensor in tensors.items()
            for size, physical in zip(logical[i], tensor.shape, strict=True)
        ):
            return None
        first = next(iter(tensors))
        return {i: result if i == first else () for i in tensors}
    result_by_index = {}
    for i, tensor in tensors.items():
        shape = logical[i]
        if any(
            (size == 1) != (dimension(physical) == 1)
            for size, physical in zip(shape, tensor.shape, strict=True)
        ):
            return None
        non_singleton = tuple(size for size in shape if size != 1)
        width = len(env.tensor_indexer_dims(tensor))
        if non_singleton:
            if len(non_singleton) != width:
                return None
            result_by_index[i] = non_singleton
        elif width == 0 and not shape:
            result_by_index[i] = ()
        elif width == 1 and shape:
            result_by_index[i] = (sympy.Integer(1),)
        else:
            return None
    return result_by_index


def integer_bounds(
    node: object, *, captured_bounds: dict[Node, tuple[int, int]] | None = None
) -> tuple[int, int] | None:
    """Bound integer expressions without observing input tensor contents.

    Each intermediate must fit its actual dtype. Unknown values can become
    bounded through positive remainder or an explicit min/max clamp. This
    proves logical indices; physical padding is never an enlarged index domain.
    """
    visiting: set[Node] = set()
    memo: dict[Node, tuple[int, int] | None] = {}

    def bounds(value: object) -> tuple[int, int] | None:
        if type(value) is int:
            return value, value
        if not isinstance(value, Node) or value in visiting:
            return None
        if value in memo:
            return memo[value]
        fake = value.meta.get("val")
        if not isinstance(fake, torch.Tensor) or fake.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            return None
        visiting.add(value)
        result = calculate(value)
        visiting.remove(value)
        limits = torch.iinfo(fake.dtype)
        if result is None or not limits.min <= result[0] <= result[1] <= limits.max:
            result = None
        memo[value] = result
        return result

    def calculate(value: Node) -> tuple[int, int] | None:
        if captured_bounds is not None and value in captured_bounds:
            return captured_bounds[value]
        target = value.target
        args = value.args
        if target is torch.ops.prims.iota.default:
            length = args[0]
            start = value.kwargs.get("start", 0)
            step = value.kwargs.get("step", 1)
            if (
                type(length) is int
                and type(start) is int
                and type(step) is int
                and length > 0
            ):
                end = start + (length - 1) * step
                return min(start, end), max(start, end)
            return None
        if target is torch.ops.aten.scalar_tensor.default:
            return bounds(args[0])
        if target is torch.ops.aten.full.default:
            return bounds(args[1])
        from ...language import _tracing_ops
        from ...language import creation_ops

        if captured_bounds is not None and target is _tracing_ops._new_var:
            return bounds(args[0])

        if target is creation_ops.full:
            return bounds(args[1])
        if target in (
            torch.ops.aten._to_copy.default,
            torch.ops.prims.convert_element_type.default,
            torch.ops.aten.clone.default,
            torch.ops.aten.alias.default,
            torch.ops.aten.expand.default,
            torch.ops.aten.unsqueeze.default,
            torch.ops.aten.squeeze.dim,
            torch.ops.aten.permute.default,
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
        ):
            return bounds(args[0])
        if target is torch.ops.aten.where.self:
            left, right = bounds(args[1]), bounds(args[2])
            if left is not None and right is not None:
                return min(left[0], right[0]), max(left[1], right[1])
            return None
        if len(args) < 2:
            return None
        left, right = bounds(args[0]), bounds(args[1])

        # Unknown tensor values retain their full signed integer range. Bounds
        # narrowed by explicit clamps/remainders are still independently proved.
        def full_range(
            arg: object, known: tuple[int, int] | None
        ) -> tuple[int, int] | None:
            if known is not None:
                return known
            if isinstance(arg, Node):
                fake = arg.meta.get("val")
                if isinstance(fake, torch.Tensor) and fake.dtype in (
                    torch.int32,
                    torch.int64,
                ):
                    limits = torch.iinfo(fake.dtype)
                    return limits.min, limits.max
            return None

        if target in (torch.ops.aten.minimum.default, torch.ops.aten.maximum.default):
            left, right = full_range(args[0], left), full_range(args[1], right)
        if target in (torch.ops.aten.remainder.Tensor, torch.ops.aten.remainder.Scalar):
            if right is not None and right[0] == right[1] and right[0] > 0:
                return 0, right[0] - 1
            return None
        if left is None or right is None:
            return None
        a, b = left
        c, d = right
        if target in (torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar):
            if value.kwargs.get("alpha", 1) != 1:
                return None
            return a + c, b + d
        if target in (torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar):
            if value.kwargs.get("alpha", 1) != 1:
                return None
            return a - d, b - c
        if target in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar):
            products = a * c, a * d, b * c, b * d
            return min(products), max(products)
        if target is torch.ops.aten.minimum.default:
            return min(a, c), min(b, d)
        if target is torch.ops.aten.maximum.default:
            return max(a, c), max(b, d)
        if target in (
            torch.ops.aten.bitwise_and.Tensor,
            torch.ops.aten.bitwise_and.Scalar,
        ):
            if a >= 0 and c == d:
                return 0, min(b, c) if c >= 0 else b
            if c >= 0 and a == b:
                return 0, min(d, a) if a >= 0 else d
            return None
        if target in (
            torch.ops.aten.bitwise_or.Tensor,
            torch.ops.aten.bitwise_or.Scalar,
            torch.ops.aten.bitwise_xor.Tensor,
            torch.ops.aten.bitwise_xor.Scalar,
        ):
            if a >= 0 and c >= 0:
                return 0, (1 << max(b, d).bit_length()) - 1
            return None
        if target in (
            torch.ops.aten.div.Tensor_mode,
            torch.ops.aten.div.Scalar_mode,
        ) and value.kwargs.get("rounding_mode") in ("floor", "trunc"):
            if a >= 0 and c == d and c > 0:
                return a // c, b // c
        return None

    return bounds(node)


def prove_gather(
    env: CompileEnvironment, node: Node, *, graphs: Sequence[GraphInfo] | None = None
) -> GatherBounds | None:
    from .computed_fragment import _fragment_logical_shape
    from .gather_domains import loop_domain_facts

    if node.target is not torch.ops.aten.gather.default:
        return None
    source, dim, index = node.args[:3]
    if not isinstance(source, Node) or not isinstance(index, Node):
        return None
    sf, ix = source.meta.get("val"), index.meta.get("val")
    if (
        not isinstance(sf, torch.Tensor)
        or not isinstance(ix, torch.Tensor)
        or sf.ndim < 1
        or sf.ndim != ix.ndim
        or dim not in (-1, sf.ndim - 1)
        or ix.dtype != torch.int64
        or sf.dtype
        not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
        )
    ):
        return None
    if graphs is None:
        from ..host_function import HostFunction

        graphs = HostFunction.current().device_ir.graphs
    facts = loop_domain_facts(env, graphs)
    source_shape = _fragment_logical_shape(
        env,
        source,
        scalar_indexed_loads=True,
        tensor_indexed_loads=True,
        pure_inline_asm=True,
        all_axis_reductions=True,
        proven_domains=facts.shapes,
    )
    index_shape = _fragment_logical_shape(
        env,
        index,
        scalar_indexed_loads=True,
        tensor_indexed_loads=True,
        pure_inline_asm=True,
        all_axis_reductions=True,
        proven_domains=facts.shapes,
    )
    if (
        source_shape is None
        or index_shape is None
        or len(source_shape) != sf.ndim
        or len(index_shape) != ix.ndim
        or len(source_shape) != len(index_shape)
        or any(
            sympy.expand(a) != sympy.expand(b)
            for a, b in zip(source_shape[:-1], index_shape[:-1], strict=True)
        )
        or not source_shape[-1].is_Integer
        or int(source_shape[-1]) <= 0
    ):
        return None
    interval = integer_bounds(index, captured_bounds=facts.readonly_ranges)
    if interval is None or not 0 <= interval[0] <= interval[1] < int(source_shape[-1]):
        return None
    return GatherBounds(source_shape, index_shape, *interval)
