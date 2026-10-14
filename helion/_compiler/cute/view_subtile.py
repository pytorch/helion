"""Attach local coordinates to views that split one tiled dimension.

A user-written ``val.view(..., D // F, F)`` turns a tile dimension ``D``
distributed over one thread axis into two logical dimensions. Those new
dimensions carry no intrinsic block id, so without metadata ``hl.split``'s
CuTe lowering cannot tell which pair element a thread holds. A matching
``hl.join`` also needs the minor coordinate to select the reconstructed
element.

This pass records ``{block_id, divisor, modulus}`` mappings that
``cute_reshape._subtile_coord_expr`` expands into block-local coordinates,
and carries them through the permute/unsqueeze/expand/cast views that may
sit between the split view and ``hl.split`` (and after ``hl.join``). It is a
no-op unless one tiled dimension is split exactly.
"""

from __future__ import annotations

from itertools import starmap
import operator
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast

import sympy
import torch
from torch.utils._sympy.functions import FloorDiv

from ... import exc
from ...language import memory_ops
from ...language._tracing_ops import _mask_to
from ...language.view_ops import join as hl_join
from ...language.view_ops import split as hl_split
from ...language.view_ops import subscript as hl_subscript
from ..compile_environment import CompileEnvironment
from .cute_reshape import CUTE_DIM_LOCAL_COORD_META
from .cute_reshape import _rebound_consumer_dims
from .cute_reshape import _rebound_operands
from .cute_reshape import _resolve_dim_block_id
from .cute_reshape import _resolve_tile_extent
from .cute_reshape import operand_positions
from .cute_reshape import tensor_dim_block_ids

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...runtime.config import Config
    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from ..helper_function import CodegenInterface

_VIEW_TARGETS = (
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten._unsafe_view.default,
)

# Views and casts that keep every thread's element in place; split-view
# coordinates propagate through them (reordered or padded with ``None``).
_PASS_THROUGH_TARGETS = (
    torch.ops.aten.permute.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.clone.default,
    torch.ops.aten._to_copy.default,
    torch.ops.prims.convert_element_type.default,
)


def annotate_view_subtiles(graphs: list[GraphInfo], config: Config) -> None:
    """Annotate split views and matching joins with local coordinates."""
    env = CompileEnvironment.current()
    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or CUTE_DIM_LOCAL_COORD_META in node.meta:
                continue
            meta: list[object | None] | None = None
            if is_noop_view(node):
                meta = _source_coord_meta(node)
            elif node.target in _VIEW_TARGETS and _feeds_split(node):
                meta = _split_subtile_coord_meta(node, env, config)
            elif node.target is torch.ops.aten.stack.default:
                meta = _stack_coord_meta(node)
            elif node.target in _PASS_THROUGH_TARGETS:
                meta = _propagated_coord_meta(node)
            elif node.target is operator.getitem:
                meta = _split_output_coord_meta(node)
            elif node.target is hl_join:
                meta = _join_subtile_coord_meta(node)
            elif node.target is hl_subscript:
                meta = _subscript_coord_meta(node)
            else:
                meta = _pointwise_coord_meta(node)
            if meta is not None:
                node.meta[CUTE_DIM_LOCAL_COORD_META] = meta


def _feeds_split(node: torch.fx.Node) -> bool:
    for user in node.users:
        if user.op != "call_function":
            continue
        if user.target is hl_split:
            return True
        if (
            user.target in _PASS_THROUGH_TARGETS or is_noop_view(user)
        ) and _feeds_split(user):
            return True
    return False


def is_noop_view(node: torch.fx.Node) -> bool:
    """A view to the shape its input already has.

    ``torch.unbind`` reshapes its input to put a literal 2 at the unbound dim
    (``view_ops._torch_unbind``); every thread keeps its element.
    """
    if node.op != "call_function" or node.target not in _VIEW_TARGETS:
        return False
    source = node.args[0] if node.args else None
    value = node.meta.get("val")
    source_value = source.meta.get("val") if isinstance(source, torch.fx.Node) else None
    if not isinstance(value, torch.Tensor) or not isinstance(
        source_value, torch.Tensor
    ):
        return False
    env = CompileEnvironment.current()
    return value.ndim == source_value.ndim and all(
        starmap(env.known_equal, zip(value.shape, source_value.shape, strict=True))
    )


def unbound_stack(split_input: torch.fx.Node) -> torch.fx.Node | None:
    """The ``torch.stack`` an ``hl.split`` unbinds along its stacked dim.

    ``torch.unbind(torch.stack((a, b), d), d)`` lowers to a permute moving
    ``d`` last (and a no-op view) feeding ``hl.split``, which then is just
    ``(a, b)``.  Returns the stack when the views between it and the split
    move exactly its stacked dim to the end.
    """
    value = split_input.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return None
    perm = list(range(value.ndim))
    node = split_input
    while node.op == "call_function":
        if node.target is torch.ops.aten.permute.default:
            dims = node.args[1]
            assert isinstance(dims, (list, tuple))
            perm = [cast("int", dims[position]) for position in perm]
        elif not is_noop_view(node):
            break
        source = node.args[0]
        assert isinstance(source, torch.fx.Node)
        node = source
    if node.target is not torch.ops.aten.stack.default:
        return None
    tensors = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
    if not isinstance(tensors, (list, tuple)) or len(tensors) != 2:
        return None
    assert isinstance(dim, int)
    dim %= value.ndim
    if perm != [*(d for d in range(value.ndim) if d != dim), dim]:
        return None
    return node


def _source_coord_meta(node: torch.fx.Node) -> list[object | None] | None:
    source = node.args[0] if node.args else None
    if not isinstance(source, torch.fx.Node):
        return None
    meta = source.meta.get(CUTE_DIM_LOCAL_COORD_META)
    return [*meta] if isinstance(meta, (list, tuple)) else None


def _propagated_coord_meta(node: torch.fx.Node) -> list[object | None] | None:
    """Carry split-view coordinates through a pass-through view or cast."""
    meta = _source_coord_meta(node)
    if meta is None:
        return None
    if node.target is torch.ops.aten.permute.default:
        dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dims")
        if not isinstance(dims, (list, tuple)):
            return None
        perm = [dim for dim in dims if isinstance(dim, int)]
        if len(perm) != len(dims) or len(perm) != len(meta):
            return None
        return [meta[dim] for dim in perm]
    if node.target is torch.ops.aten.unsqueeze.default:
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        if not isinstance(dim, int):
            return None
        dim %= len(meta) + 1
        return [*meta[:dim], None, *meta[dim:]]
    if node.target is torch.ops.aten.expand.default:
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor) or value.ndim < len(meta):
            return None
        return [*([None] * (value.ndim - len(meta))), *meta]
    return meta


def _stack_coord_meta(node: torch.fx.Node) -> list[object | None] | None:
    """A stack of operands with one set of coordinates keeps them; the new
    dim has none."""
    tensors = node.args[0] if node.args else None
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
    value = node.meta.get("val")
    if (
        not isinstance(tensors, (list, tuple))
        or not isinstance(dim, int)
        or not isinstance(value, torch.Tensor)
    ):
        return None
    metas = [
        tensor.meta.get(CUTE_DIM_LOCAL_COORD_META)
        if isinstance(tensor, torch.fx.Node)
        else None
        for tensor in tensors
    ]
    meta = metas[0]
    if not isinstance(meta, (list, tuple)) or any(other != meta for other in metas):
        return None
    dim %= value.ndim
    return [*meta[:dim], None, *meta[dim:]]


def _subscript_coord_meta(node: torch.fx.Node) -> list[object | None] | None:
    """``hl.subscript`` with ``None`` and ``:`` entries is a series of unsqueezes."""
    meta = _source_coord_meta(node)
    index = node.args[1] if len(node.args) > 1 else None
    if meta is None or not isinstance(index, (list, tuple)):
        return None
    source = iter(meta)
    result: list[object | None] = []
    for entry in index:
        if entry is None:
            result.append(None)
        elif isinstance(entry, slice) and entry == slice(None):
            result.append(next(source, None))
        else:
            return None
    return [*result, *source]


def _split_output_coord_meta(node: torch.fx.Node) -> list[object | None] | None:
    """``hl.split`` outputs keep the input's coordinates minus the pair dim."""
    split_node = node.args[0] if node.args else None
    if not isinstance(split_node, torch.fx.Node) or split_node.target is not hl_split:
        return None
    meta = _source_coord_meta(split_node)
    if meta is None or not meta:
        return None
    return meta[:-1]


def _pointwise_coord_meta(node: torch.fx.Node) -> list[object | None] | None:
    """A per-thread element-wise op keeps the coordinates of its operands.

    Each dim takes the coordinates of the same-rank operands that carry them
    there (``check_view_coord_consumers`` refuses operands that would meet a
    split-view dim with another coordinate), so stores address a split
    output, or arithmetic on it, at the coordinate the thread actually holds.
    """
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return None
    from ..inductor_lowering import PointwiseLowering

    if node.target is not torch.ops.aten.where.self and not isinstance(
        node.meta.get("lowering"), PointwiseLowering
    ):
        return None
    result: list[object | None] = [None] * value.ndim
    for input_node in node.all_input_nodes:
        input_value = input_node.meta.get("val")
        meta = input_node.meta.get(CUTE_DIM_LOCAL_COORD_META)
        if (
            not isinstance(input_value, torch.Tensor)
            or input_value.ndim != value.ndim
            or not isinstance(meta, (list, tuple))
        ):
            continue
        for dim, info in enumerate(meta):
            if not isinstance(info, dict):
                continue
            if result[dim] is None:
                result[dim] = info
            elif result[dim] != info:
                return None
    if not any(isinstance(info, dict) for info in result):
        return None
    return result


def _dim_block_id(env: CompileEnvironment, size: int | torch.SymInt) -> int | None:
    """Block id owning a tile dim.

    A static full-slice extent (``x[tile, :]``) is not a block symbol; like
    load addressing, map it to the unique reduction dim of that size.
    """
    block_id = env.get_block_id(size)
    if block_id is not None:
        return block_id
    candidates = [
        info.block_id
        for info in env.block_sizes
        if info.reduction
        and isinstance(info.size, (int, torch.SymInt))
        and env.known_equal(info.size, size)
    ]
    return candidates[0] if len(candidates) == 1 else None


def _split_subtile_coord_meta(
    node: torch.fx.Node,
    env: CompileEnvironment,
    config: Config,
) -> list[object | None] | None:
    """Return coordinate metadata when one tiled dimension becomes two."""
    from .cute_reshape import _get_tile_shape

    output_val = node.meta.get("val")
    source = node.args[0] if node.args else None
    if not isinstance(source, torch.fx.Node):
        return None
    input_val = source.meta.get("val")
    if not isinstance(output_val, torch.Tensor) or not isinstance(
        input_val, torch.Tensor
    ):
        return None
    if output_val.ndim != input_val.ndim + 1:
        return None

    input_shape = _get_tile_shape(input_val, env, config)
    output_shape = _get_tile_shape(output_val, env, config)
    source_meta = source.meta.get(CUTE_DIM_LOCAL_COORD_META)
    input_meta = (
        [*source_meta]
        if isinstance(source_meta, (list, tuple)) and len(source_meta) == input_val.ndim
        else [None] * input_val.ndim
    )
    for dim, input_extent in enumerate(input_shape):
        if (
            input_shape[:dim] != output_shape[:dim]
            or input_shape[dim + 1 :] != output_shape[dim + 2 :]
        ):
            continue
        outer_extent, inner_extent = output_shape[dim : dim + 2]
        if (
            outer_extent < 1
            or inner_extent < 2
            or outer_extent * inner_extent != input_extent
        ):
            continue

        old_coord = input_meta[dim]
        if isinstance(old_coord, dict) and isinstance(old_coord.get("block_id"), int):
            block_id = old_coord["block_id"]
            divisor = old_coord.get("divisor", 1)
            if not isinstance(divisor, int):
                continue
        else:
            block_id = _dim_block_id(env, input_val.shape[dim])
            divisor = 1
        if block_id is None or env.is_jagged_tile(block_id):
            continue
        block_size = env.block_sizes[block_id].from_config(config)
        if not isinstance(block_size, int) or block_size % input_extent:
            continue

        return [
            *input_meta[:dim],
            {
                "block_id": block_id,
                "divisor": divisor * inner_extent,
                "modulus": outer_extent,
            },
            {
                "block_id": block_id,
                "divisor": divisor,
                "modulus": inner_extent,
            },
            *input_meta[dim + 1 :],
        ]
    return None


def _join_subtile_coord_meta(node: torch.fx.Node) -> list[object | None] | None:
    """Recover the split input's coordinates when ``join`` rebuilds its shape."""
    output_val = node.meta.get("val")
    sources = node.args[:2]
    if len(sources) != 2 or not all(
        isinstance(source, torch.fx.Node) for source in sources
    ):
        return None
    left, right = sources
    assert isinstance(left, torch.fx.Node) and isinstance(right, torch.fx.Node)
    input_val = left.meta.get("val")
    right_val = right.meta.get("val")
    if (
        not isinstance(output_val, torch.Tensor)
        or not isinstance(input_val, torch.Tensor)
        or not isinstance(right_val, torch.Tensor)
        or right_val.shape != input_val.shape
        or output_val.ndim != input_val.ndim + 1
        or output_val.shape[-1] != 2
        or input_val.ndim == 0
    ):
        return None

    left_meta = _split_source_coord_meta(left)
    right_meta = _split_source_coord_meta(right)
    if (
        left_meta is None
        or left_meta != right_meta
        or len(left_meta) != output_val.ndim
    ):
        return None
    return left_meta


def _split_source_coord_meta(node: torch.fx.Node) -> list[object | None] | None:
    """Trace a pointwise join operand back to the split input it reconstructs.

    Returns that input's full coordinate list (its last entry is the minor
    pair coordinate the join selects with).
    """
    if node.op != "call_function":
        return None
    if node.target is operator.getitem and node.args:
        split_node = node.args[0]
        if (
            isinstance(split_node, torch.fx.Node)
            and split_node.target is hl_split
            and split_node.args
            and isinstance(split_node.args[0], torch.fx.Node)
        ):
            split_input_meta = split_node.args[0].meta.get(CUTE_DIM_LOCAL_COORD_META)
            if (
                isinstance(split_input_meta, (list, tuple))
                and split_input_meta
                and isinstance(split_input_meta[-1], dict)
            ):
                return [*split_input_meta]
        return None

    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return None
    from ..inductor_lowering import PointwiseLowering

    if not isinstance(node.meta.get("lowering"), PointwiseLowering):
        return None
    discovered: list[list[object | None]] = []
    for input_node in node.all_input_nodes:
        input_value = input_node.meta.get("val")
        if (
            not isinstance(input_value, torch.Tensor)
            or input_value.shape != value.shape
        ):
            continue
        meta = _split_source_coord_meta(input_node)
        if meta is not None and meta not in discovered:
            discovered.append(meta)
    return discovered[0] if len(discovered) == 1 else None


def _split_minor_coord_meta(node: torch.fx.Node) -> dict[object, object] | None:
    """Trace a pointwise join operand back to the split's minor coordinate."""
    meta = _split_source_coord_meta(node)
    if meta is None:
        return None
    minor = meta[-1]
    return dict(minor) if isinstance(minor, dict) else None


def view_coord_dims(node: torch.fx.Node) -> list[int]:
    """The dims of ``node`` that carry split-view coordinates."""
    meta = node.meta.get(CUTE_DIM_LOCAL_COORD_META)
    if not isinstance(meta, (list, tuple)):
        return []
    return [
        dim
        for dim, info in enumerate(meta)
        if isinstance(info, dict) and isinstance(info.get("block_id"), int)
    ]


# Ops that take a split-view operand without combining it with anything and
# carry its coordinates to their result (``annotate_view_subtiles``).
_VIEW_COORD_CARRIERS = frozenset({*_PASS_THROUGH_TARGETS, hl_subscript})

# A dim's lane coordinate ``(block, divisor, modulus)``, standing for
# ``(coord(block) // divisor) % modulus``; ``None`` when no lane owns the dim.
_Descriptor = tuple[int, int, int] | None


class _Slot(NamedTuple):
    """A dim a load or store subscript addresses.

    ``kind`` is ``"unit"`` (``None``), ``"tile"``, ``"full"`` (``:``),
    ``"partial"`` (another slice) or ``"index"``, whose ``indexers`` are the
    tensor indexer dims meeting there.  A tile addresses the dim by its
    block's coordinate below its extent (a ``tile_with_offset`` sub-tile of
    the epilogue subtiling masks the rest), ``lane``.
    """

    kind: str
    indexers: tuple[tuple[torch.fx.Node, int], ...] = ()
    lane: _Descriptor = None


def check_view_coord_consumers(cg: CodegenInterface, node: torch.fx.Node) -> None:
    """Refuse an op that would read a split-view dim at another coordinate.

    A dim of a split view (``x[tile, :].view(tile, 2, half)`` feeding
    ``hl.split``), of the split's results and of what
    ``annotate_view_subtiles`` carries them to has no block of its own: every
    thread of the source block holds the element at the sub-coordinate
    ``(coord // divisor) % modulus`` of its lane (several threads the same
    one).  An op reads it right only when it addresses that dim by the same
    sub-coordinate: the views and casts carrying the metadata, ``hl.split``,
    an element-wise op, ``hl.join`` or ``torch.where`` whose operands all
    have that coordinate at the position (or broadcast there), a reshape
    that merges such dims back into a block's own coordinate, a store that
    re-addresses a full slice by it (``_apply_cute_value_coord_meta``), and a
    gather whose index dim has it.  Anything else, such as a reduction, scan
    or matmul over it, a device loop that drops the metadata, or an operand
    distributed by another block's lane (``a * bias[tile, :]`` after a split
    of ``x[tile, :]``), would combine elements of different positions.
    """
    from ..inductor_lowering import PointwiseLowering

    operands = [operand for operand in node.all_input_nodes if view_coord_dims(operand)]
    if not operands or node.target in (hl_split, torch.ops.aten.sym_size.int):
        return
    if node.target in _VIEW_COORD_CARRIERS:
        conflict = (
            None
            if view_coord_dims(node)
            else f"{operands[0].name} into a result without its coordinates"
        )
    elif node.target in (memory_ops.load, memory_ops.store):
        conflict = _memory_op_conflict(cg, node, operands)
    elif node.target in _VIEW_TARGETS:
        conflict = (
            None
            if CUTE_DIM_LOCAL_COORD_META in node.meta or _merges_view_coords(cg, node)
            else f"{operands[0].name} into dims whose lanes are not its coordinates"
        )
    elif node.target is torch.ops.aten.stack.default:
        # The stacked dim has no lane; only ``torch.unbind`` of it, which
        # ``hl.split`` resolves to the operands, reads the stack.
        conflict = _combination_conflict(cg, node)
        if conflict is None and not _only_unbound(node):
            conflict = f"{operands[0].name} into a stacked dim no lane owns"
    elif node.target in (hl_join, torch.ops.aten.where.self) or isinstance(
        node.meta.get("lowering"), PointwiseLowering
    ):
        conflict = _combination_conflict(cg, node)
    else:
        conflict = f"{operands[0].name} by block coordinates"
    if conflict is not None:
        raise exc.BackendUnsupported(
            "cute",
            f"{_target_name(node)} reads {conflict}, but dims "
            f"{view_coord_dims(operands[0])} of {operands[0].name} are hl.split "
            "view coordinates (a sub-coordinate of their source block's lane)",
        )


def _only_unbound(stack: torch.fx.Node) -> bool:
    """Whether every use of ``stack`` is an ``hl.split`` that unbinds it."""

    def unbinds(node: torch.fx.Node) -> bool:
        for user in node.users:
            if user.target is hl_split:
                source = user.args[0]
                assert isinstance(source, torch.fx.Node)
                if unbound_stack(source) is not stack:
                    return False
            elif not (
                user.target is torch.ops.aten.permute.default or is_noop_view(user)
            ) or not unbinds(user):
                return False
        return True

    return unbinds(stack)


def _target_name(node: torch.fx.Node) -> str:
    target = node.target
    if target is _mask_to:
        return "the operand masking of a reduction or matmul"
    if callable(target) and not isinstance(target, torch._ops.OpOverload):
        return target.__name__
    return str(target)


def _dim_descriptor(
    cg: CodegenInterface, node: torch.fx.Node, value: torch.Tensor, dim: int
) -> _Descriptor:
    """The lane coordinate distributing ``value``'s dim ``dim``.

    ``(block, divisor, modulus)`` as ``_subtile_coord_expr`` and
    ``_get_dim_local_coord`` address it: by the dim's split-view metadata, as
    ``coord // k`` for a ``block_size // k`` extent, or as the coordinate
    ``(block, 1, extent)`` of a block's own dim.  The modulus is clipped to
    what the block's extent leaves above the divisor, so equal coordinates
    compare equal.
    """
    env = CompileEnvironment.current()
    config = cg.device_function.config
    size = value.shape[dim]
    meta = node.meta.get(CUTE_DIM_LOCAL_COORD_META)
    info = meta[dim] if isinstance(meta, (list, tuple)) and dim < len(meta) else None
    divisor: object = 1
    modulus: object = None
    if isinstance(info, dict) and isinstance(info.get("block_id"), int):
        block_id = info["block_id"]
        # A tile split by ``tile.block_size // k`` records symbolic terms.
        divisor, modulus = (
            _resolve_tile_extent(term, env, config)
            if isinstance(term, torch.SymInt)
            else term
            for term in (info.get("divisor", 1), info.get("modulus"))
        )
    elif (block_id := env.get_block_id(size)) is None:
        expr = size.node._expr if isinstance(size, torch.SymInt) else None
        if isinstance(expr, FloorDiv) and isinstance(expr.args[1], sympy.Integer):
            block_id = env.get_block_id(expr.args[0])
            divisor = int(expr.args[1])
        else:
            block_id = _resolve_dim_block_id(cast("GenerateAST", cg), value, dim)
    if block_id is None:
        return None
    block_id = env.canonical_block_id(block_id)
    extent = env.block_sizes[block_id].from_config(config)
    if modulus is None and isinstance(extent, int) and isinstance(divisor, int):
        modulus = extent // divisor if extent % divisor == 0 else None
    if not (
        isinstance(extent, int)
        and isinstance(divisor, int)
        and isinstance(modulus, int)
    ):
        return None
    return block_id, divisor, min(modulus, -(-extent // divisor))


def _first_view_conflict(
    cg: CodegenInterface,
    entries: list[list[tuple[torch.fx.Node, torch.Tensor, int]]],
) -> str | None:
    """Describe the first position whose split-view dims meet another coordinate.

    ``entries`` lists, per position, the operand dims that meet there.
    """
    for position, dims in enumerate(entries):
        descriptors: dict[_Descriptor, str] = {}
        has_view = False
        for operand, value, dim in dims:
            if (
                value.stride(dim) == 0
                or CompileEnvironment.current().size_hint(value.shape[dim]) == 1
            ):
                continue
            has_view = has_view or dim in view_coord_dims(operand)
            descriptors.setdefault(
                _dim_descriptor(cg, operand, value, dim), f"{operand.name} dim {dim}"
            )
        if has_view and len(descriptors) > 1:
            return f"{' and '.join(descriptors.values())} at position {position} with different lane coordinates"
    return None


def _combination_conflict(cg: CodegenInterface, node: torch.fx.Node) -> str | None:
    """Operands of an element-wise combination whose split-view dims misalign.

    Operands meet as ``rebound_block_dims`` aligns them (a lower-rank
    operand's split-view dim is a non-tile dim, matched by extent); the
    result must keep the coordinates of every position a split-view dim
    reaches (``_pointwise_coord_meta`` takes them from same-rank operands).
    """
    consumer = _rebound_consumer_dims(node)
    if consumer is None:
        return "split-view operands it cannot align"
    sizes, lower_rank_by_block_id = consumer
    env = CompileEnvironment.current()
    block_ids = tensor_dim_block_ids(env, sizes)
    entries: list[list[tuple[torch.fx.Node, torch.Tensor, int]]] = [[] for _ in sizes]
    view_positions: set[int] = set()
    for operand in _rebound_operands(node):
        value = operand.meta.get("val")
        if not isinstance(value, torch.Tensor):
            continue
        positions = operand_positions(
            env, value, sizes, block_ids, lower_rank_by_block_id=lower_rank_by_block_id
        )
        for dim, position in enumerate(positions):
            if position >= 0:
                entries[position].append((operand, value, dim))
        view_positions.update(positions[dim] for dim in view_coord_dims(operand))
    if (conflict := _first_view_conflict(cg, entries)) is not None:
        return conflict
    if node.target is torch.ops.aten.stack.default:
        # The operands meet the result without its stacked dim.
        stacked = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        assert isinstance(stacked, int)
        stacked %= len(sizes) + 1
        view_positions = {
            position + (position >= stacked) for position in view_positions
        }
    if not view_positions <= set(view_coord_dims(node)):
        return "split-view operands into a result without their coordinates"
    return None


def _subscript_slots(
    cg: CodegenInterface, subscript: Sequence[object]
) -> list[_Slot] | None:
    """The dims a load or store subscript addresses, one ``_Slot`` per dim.

    Several tensor indexers meet in one dim when they broadcast together,
    placed at the first one as ``_subscript_slot_dims`` places them.  ``None``
    for an entry the walk does not know.
    """
    env = CompileEnvironment.current()

    def fake(entry: object) -> object:
        if isinstance(entry, torch.fx.Node):
            return entry.meta.get("val")
        return entry

    indexers = [
        entry
        for entry in subscript
        if isinstance(entry, torch.fx.Node)
        and isinstance(entry.meta.get("val"), torch.Tensor)
    ]
    broadcast = env.should_broadcast_tensor_indexers(
        [fake(entry) for entry in subscript]
    )
    shape = (
        torch.broadcast_shapes(*(indexer.meta["val"].shape for indexer in indexers))
        if broadcast
        else ()
    )
    slots: list[_Slot] = []
    for entry in subscript:
        value = fake(entry)
        if entry is None:
            slots.append(_Slot("unit"))
        elif isinstance(value, torch.Tensor):
            assert isinstance(entry, torch.fx.Node)
            if not broadcast:
                slots.extend(
                    _Slot("index", ((entry, dim),)) for dim in range(value.ndim)
                )
            elif entry is indexers[0]:
                for position in range(len(shape)):
                    slots.append(
                        _Slot(
                            "index",
                            tuple(
                                (indexer, position - offset)
                                for indexer in indexers
                                if position
                                >= (offset := len(shape) - indexer.meta["val"].ndim)
                            ),
                        )
                    )
        elif isinstance(value, torch.SymInt):
            offset_info = (
                entry.meta.get("tile_with_offset")
                if isinstance(entry, torch.fx.Node)
                else None
            )
            block_id = (
                env.get_block_id(value)
                if offset_info is None
                else offset_info["block_id"]
            )
            if block_id is not None:
                slots.append(_Slot("tile", lane=_tile_lane(cg, block_id, offset_info)))
        elif isinstance(entry, slice):
            slots.append(_Slot("full" if entry == slice(None) else "partial"))
        elif not isinstance(entry, int):
            return None
    return slots


def _tile_lane(
    cg: CodegenInterface, block_id: int, offset_info: dict[str, object] | None
) -> _Descriptor:
    """The lane coordinate a tile subscript entry addresses its dim by."""
    env = CompileEnvironment.current()
    config = cg.device_function.config
    block_id = env.canonical_block_id(block_id)
    extent = env.block_sizes[block_id].from_config(config)
    size = None if offset_info is None else offset_info.get("block_size")
    limit = (
        extent
        if size is None
        else _resolve_tile_extent(cast("int | torch.SymInt", size), env, config)
    )
    if not isinstance(extent, int) or not isinstance(limit, int):
        return None
    return block_id, 1, min(limit, extent)


def _memory_op_conflict(
    cg: CodegenInterface, node: torch.fx.Node, operands: list[torch.fx.Node]
) -> str | None:
    """A load or store addressing a split-view operand at another coordinate.

    A tensor indexer and an ``extra_mask`` are read per thread like
    element-wise operands (the mask right-aligned to the accessed dims, as
    ``tl.load`` / ``tl.store`` AND it), so their dims must have the
    coordinate of the element the thread accesses there: the loaded value's
    own (a ``block_size // k`` extent), or the stored value's.  A stored
    value's split-view dim must otherwise land on a full slice, which
    ``_apply_cute_value_coord_meta`` re-addresses by the value's coordinate,
    or on a tile of the same coordinate.
    """
    subscript = node.args[1]
    if node.target is memory_ops.store:
        value_node = node.args[2]
        mask = node.args[3] if len(node.args) > 3 else None
    else:
        value_node = node
        mask = node.args[2] if len(node.args) > 2 else None
    slots = (
        _subscript_slots(cg, subscript)
        if isinstance(subscript, (list, tuple))
        else None
    )
    indexers = {indexer for slot in slots or () for indexer, _ in slot.indexers}
    for operand in operands:
        if (
            operand not in indexers
            and operand is not value_node
            and operand is not mask
        ):
            return f"{operand.name} outside its subscript, value and mask"
    value = (
        value_node.meta.get("val") if isinstance(value_node, torch.fx.Node) else None
    )
    if slots is None or not isinstance(value, torch.Tensor):
        return f"{operands[0].name} through a subscript the check cannot walk"
    assert isinstance(value_node, torch.fx.Node)
    entries: list[list[tuple[torch.fx.Node, torch.Tensor, int]]] = [[] for _ in slots]
    aligned: list[tuple[torch.fx.Node, torch.Tensor]] = [(value_node, value)]
    if isinstance(mask, torch.fx.Node) and isinstance(
        mask_value := mask.meta.get("val"), torch.Tensor
    ):
        aligned.append((mask, mask_value))
    for operand, operand_value in aligned:
        offset = len(slots) - operand_value.ndim
        if offset < 0:
            return f"{operand.name} at a higher rank than its subscript"
        for dim in range(operand_value.ndim):
            entries[offset + dim].append((operand, operand_value, dim))
    offset = len(slots) - value.ndim
    for dim in view_coord_dims(value_node):
        slot = slots[offset + dim]
        # A full slice is re-addressed by the value's coordinate; a tile only
        # addresses the coordinate it has.
        if slot.kind not in ("index", "full") and (
            slot.lane is None
            or slot.lane != _dim_descriptor(cg, value_node, value, dim)
        ):
            kind = "slice" if slot.kind == "partial" else slot.kind
            return f"{value_node.name} dim {dim} through a {kind} other than ':'"
    for position, slot in enumerate(slots):
        for indexer, dim in slot.indexers:
            entries[position].append((indexer, indexer.meta["val"], dim))
    return _first_view_conflict(cg, entries)


def _merges_view_coords(cg: CodegenInterface, node: torch.fx.Node) -> bool:
    """Whether a reshape of a split view lands each element on its own lane.

    Groups the non-unit dims of the input and the output into runs of equal
    element counts and requires each input run, composed digit by digit
    (``(b, d * m, m')`` above ``(b, d, m)`` is ``(b, d, m * m')``), to be
    the coordinate the output run's dims own: the hl.join, permute, reshape
    that rebuilds the tile a split view split.
    """
    source = node.args[0]
    value = node.meta.get("val")
    if not isinstance(source, torch.fx.Node) or not isinstance(value, torch.Tensor):
        return False
    source_value = source.meta.get("val")
    if not isinstance(source_value, torch.Tensor):
        return False
    env = CompileEnvironment.current()
    config = cg.device_function.config

    def digits(
        owner: torch.fx.Node, tensor: torch.Tensor
    ) -> list[tuple[int, _Descriptor]] | None:
        result = []
        for dim, size in enumerate(tensor.shape):
            extent = _resolve_tile_extent(size, env, config)
            if extent is None:
                return None
            if extent != 1:
                result.append((extent, _dim_descriptor(cg, owner, tensor, dim)))
        return result

    def compose(run: list[tuple[int, _Descriptor]]) -> _Descriptor:
        composed = run[-1][1]
        for _, descriptor in reversed(run[:-1]):
            if composed is None or descriptor is None:
                return None
            block_id, divisor, modulus = descriptor
            if block_id != composed[0] or divisor != composed[1] * composed[2]:
                return None
            composed = (block_id, composed[1], modulus * composed[2])
        return composed

    inputs = digits(source, source_value)
    outputs = digits(node, value)
    if inputs is None or outputs is None:
        return False
    while inputs or outputs:
        if not inputs or not outputs:
            return False
        input_run = [inputs.pop(0)]
        output_run = [outputs.pop(0)]
        input_count = input_run[0][0]
        output_count = output_run[0][0]
        while input_count != output_count:
            if input_count < output_count and inputs:
                input_run.append(inputs.pop(0))
                input_count *= input_run[-1][0]
            elif output_count < input_count and outputs:
                output_run.append(outputs.pop(0))
                output_count *= output_run[-1][0]
            else:
                return False
        composed = compose(input_run)
        if composed is None or composed != compose(output_run):
            return False
    return True
