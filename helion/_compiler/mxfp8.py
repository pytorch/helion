"""Backend-neutral recognition of packed MXFP8 scale expressions.

DeepGEMM-style MXFP8 kernels store four E8M0 scale bytes per int32 word and
extract one byte per K group in the kernel::

    words = hl.load(packed, [rows, group // 4])
    scale = ((words >> ((group % 4) * 8)) & 255).to(torch.uint8)

``match_packed_scale`` proves that a ``dot_scaled`` scale operand is exactly
this recipe so a backend can load the words once and unpack them natively.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from .._compat import is_hip
from ..language import _tracing_ops
from ..language import memory_ops
from ..language import tile_ops
from ..language import view_ops
from ..language.matmul_ops import dot_scaled
from .compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .device_ir import GraphInfo

_GROUP = sympy.Symbol("group", integer=True, nonnegative=True)
_WORD = cast("sympy.Expr", sympy.floor(sympy.Mul(_GROUP, sympy.Rational(1, 4))))
_SHIFT = sympy.Mul(sympy.Mod(_GROUP, 4), 8)
# The coordinate source of a full ``arange(groups)`` (a tile index records
# its block id instead).
_FULL_RANGE = "full_range"


@dataclass(frozen=True)
class PackedScaleMatch:
    """A proved ``uint8((load(words, [row, group // 4]) >> shift) & 255)``."""

    tensor: torch.Tensor
    row: object
    word_index: torch.fx.Node
    mask: object | None
    # The K-tile block whose ``tile.index`` is the group coordinate, or None
    # when the coordinate is a full ``arange(groups)`` starting at zero.
    group_block_id: int | None


def strip_new_var(node: object) -> object:
    while (
        isinstance(node, torch.fx.Node)
        and node.target is _tracing_ops._new_var
        and len(node.args) == 1
        and not node.kwargs
    ):
        node = node.args[0]
    return node


def host_tensor(node: object) -> torch.Tensor | None:
    if (
        not isinstance(node, torch.fx.Node)
        or node.target is not _tracing_ops._host_tensor
    ):
        return None
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) and value.ndim == 2 else None


def tile_load(
    node: object, *, allow_mask: bool = False
) -> tuple[torch.Tensor, list[object]] | None:
    """A 2-D ``hl.load`` of a host tensor, as ``(tensor, indices)``."""
    if (
        not isinstance(node, torch.fx.Node)
        or node.target is not memory_ops.load
        or len(node.args) != 4
        or node.args[3] is not None
        or (not allow_mask and node.args[2] is not None)
        or node.kwargs
        or not isinstance(node.args[1], (list, tuple))
        or len(node.args[1]) != 2
    ):
        return None
    tensor = host_tensor(node.args[0])
    return (tensor, list(node.args[1])) if tensor is not None else None


def packed_group_expression(
    node: object,
    groups: int | None,
    *,
    sources: set[object],
    allow_tile: bool = False,
    budget: int = 128,
) -> sympy.Expr | None:
    """Prove a broadcasted group coordinate and its bounded integer arithmetic.

    The coordinate is a full ``arange(groups)`` or, with ``allow_tile``, the
    ``tile.index`` of a K tile.  Each coordinate's source (``_FULL_RANGE`` or
    the tile's block id) is added to ``sources``; a caller combining several
    expressions must check that they share one source.
    """
    remaining = budget
    if groups is not None and (groups <= 0 or groups >= 1 << 26):
        return None
    remainder = sympy.Mod(_GROUP, 4)
    if not isinstance(remainder, sympy.Expr):
        return None

    def full_range() -> sympy.Expr:
        sources.add(_FULL_RANGE)
        return _GROUP

    def visit(value: object) -> sympy.Expr | None:
        nonlocal remaining
        remaining -= 1
        if remaining < 0:
            return None
        if type(value) is int:
            return sympy.Integer(value)
        if not isinstance(value, torch.fx.Node) or value.op != "call_function":
            return None
        if (
            value.target is _tracing_ops._new_var
            and len(value.args) == 1
            and not value.kwargs
        ):
            return visit(value.args[0])
        if value.target is tile_ops.tile_index:
            if not allow_tile or len(value.args) != 1 or value.kwargs:
                return None
            tile = value.args[0]
            tile_value = (
                tile.meta.get("val") if isinstance(tile, torch.fx.Node) else tile
            )
            if not isinstance(tile_value, torch.SymInt):
                return None
            symbol = tile_value._sympy_()
            block_id = next(
                (
                    info.block_id
                    for info in CompileEnvironment.current().block_sizes
                    if info.symbol() == symbol
                ),
                None,
            )
            if block_id is None:
                return None
            sources.add(block_id)
            return _GROUP
        if value.target in (memory_ops.load, view_ops.subscript):
            if len(value.args) not in (2, 4) or value.kwargs:
                return None
            if len(value.args) == 4 and value.args[2:] != (None, None):
                return None
            if not isinstance(value.args[1], (list, tuple)) or any(
                item is not None and item != slice(None) for item in value.args[1]
            ):
                return None
            return visit(value.args[0])
        if value.target is torch.ops.aten.unsqueeze.default and len(value.args) == 2:
            return visit(value.args[0])
        if value.target is torch.ops.prims.convert_element_type.default:
            if len(value.args) != 2 or value.args[1] not in (torch.int32, torch.int64):
                return None
            return visit(value.args[0])
        if value.target in (
            torch.ops.aten.arange.default,
            torch.ops.aten.arange.start,
            torch.ops.aten.arange.start_step,
        ):
            if groups is None or value.kwargs.get("dtype", torch.int64) not in (
                torch.int32,
                torch.int64,
            ):
                return None
            if tuple(value.args) not in ((groups,), (0, groups), (0, groups, 1)):
                return None
            return full_range()
        if value.target is torch.ops.prims.iota.default:
            if (
                groups is None
                or value.args != (groups,)
                or value.kwargs.get("start", 0) != 0
                or value.kwargs.get("step", 1) != 1
                or value.kwargs.get("dtype", torch.int64)
                not in (torch.int32, torch.int64)
            ):
                return None
            return full_range()
        if len(value.args) != 2:
            return None
        left, right = (visit(item) for item in value.args)
        if left is None or right is None:
            return None
        # Match the bounded byte recipe without algebraic cancellation. A
        # wider integer expression can overflow before simplifying to this
        # formula, so arbitrary arithmetic must not become native provenance.
        if value.target in (
            operator.floordiv,
            torch.ops.aten.div.Tensor_mode,
        ):
            if value.target is torch.ops.aten.div.Tensor_mode:
                if value.kwargs != {"rounding_mode": "floor"}:
                    return None
            elif value.kwargs:
                return None
            return _WORD if left == _GROUP and right == 4 else None
        if value.kwargs:
            return None
        if value.target in (
            operator.mod,
            torch.ops.aten.remainder.Scalar,
            torch.ops.aten.remainder.Tensor,
        ):
            return remainder if left == _GROUP and right == 4 else None
        if value.target in (operator.mul, torch.ops.aten.mul.Tensor):
            if (left == remainder and right == 8) or (right == remainder and left == 8):
                return _SHIFT
        return None

    return visit(node)


def match_packed_scale(
    node: object,
    row: object | None,
    groups: int | None,
    *,
    allow_mask: bool = False,
    allow_tile: bool = False,
) -> PackedScaleMatch | None:
    """Match ``uint8((load(int32) >> ((group % 4) * 8)) & 255)``.

    ``row`` (when given) must be the word load's row index.  ``groups`` is the
    extent of a full ``arange`` group coordinate; ``allow_tile`` also accepts
    a K tile's ``tile.index`` (of any extent), and ``allow_mask`` a masked
    word load.  Both coordinates must come from the same source.
    """
    node = strip_new_var(node)
    if (
        not isinstance(node, torch.fx.Node)
        or node.target is not torch.ops.prims.convert_element_type.default
        or node.args[1:] != (torch.uint8,)
        or node.kwargs
    ):
        return None
    masked = node.args[0]
    if (
        not isinstance(masked, torch.fx.Node)
        or masked.target not in (operator.and_, torch.ops.aten.bitwise_and.Scalar)
        or len(masked.args) != 2
        or masked.args[1] != 255
        or masked.kwargs
    ):
        return None
    shifted = masked.args[0]
    if (
        not isinstance(shifted, torch.fx.Node)
        or shifted.target
        not in (
            operator.rshift,
            torch.ops.aten.bitwise_right_shift.Tensor,
            torch.ops.aten.__rshift__.Tensor,
        )
        or len(shifted.args) != 2
        or shifted.kwargs
    ):
        return None
    load_node = shifted.args[0]
    words = tile_load(load_node, allow_mask=allow_mask)
    if (
        words is None
        or words[0].dtype != torch.int32
        or (row is not None and words[1][0] != row)
        or not isinstance(words[1][1], torch.fx.Node)
    ):
        return None
    index_value = words[1][1].meta.get("val")
    shift_value = (
        shifted.args[1].meta.get("val")
        if isinstance(shifted.args[1], torch.fx.Node)
        else None
    )
    if (
        not isinstance(index_value, torch.Tensor)
        or index_value.ndim != 1
        or not isinstance(shift_value, torch.Tensor)
        or not (
            shift_value.ndim == 1
            or (
                shift_value.ndim == 2
                and type(shift_value.shape[0]) is int
                and shift_value.shape[0] == 1
            )
        )
    ):
        return None
    # Full iota provenance below proves the exact group extent. A broadcasted
    # view's metadata may use a symbolic full-slice axis before config binding;
    # requiring a literal shape here would lose the same source's native seeds.
    sources: set[object] = set()
    word_expr = packed_group_expression(
        words[1][1], groups, sources=sources, allow_tile=allow_tile
    )
    shift_expr = packed_group_expression(
        shifted.args[1], groups, sources=sources, allow_tile=allow_tile
    )
    if word_expr != _WORD or shift_expr != _SHIFT or len(sources) != 1:
        return None
    (source,) = sources
    assert isinstance(load_node, torch.fx.Node)
    return PackedScaleMatch(
        words[0],
        words[1][0],
        words[1][1],
        load_node.args[2],
        source if type(source) is int else None,
    )


def native_packed_mxfp8_target(env: CompileEnvironment) -> bool:
    """Whether the Triton backend lowers packed MXFP8 dots natively here
    (``triton/packed_mxfp8.py``): CUDA sm100/sm103."""
    capability = env.config_spec.target_device_capability
    return (
        env.backend_name == "triton"
        and env.device.type == "cuda"
        and not is_hip()
        and capability is not None
        and capability[0] == 10
    )


def find_packed_mxfp8_dot(graphs: Iterable[GraphInfo]) -> torch.fx.Node | None:
    """The kernel's only ``dot_scaled``, when it contracts E4M3 operands with
    packed E8M0 scale words."""
    dots = [
        node
        for graph in graphs
        for node in graph.graph.find_nodes(op="call_function", target=dot_scaled)
    ]
    if len(dots) != 1:
        return None
    (dot,) = dots
    if len(dot.args) < 6 or dot.args[2] != "e4m3" or dot.args[5] != "e4m3":
        return None
    for scale in (dot.args[1], dot.args[4]):
        value = scale.meta.get("val") if isinstance(scale, torch.fx.Node) else None
        groups = (
            value.shape[1]
            if isinstance(value, torch.Tensor)
            and value.ndim == 2
            and type(value.shape[1]) is int
            else None
        )
        if (
            match_packed_scale(scale, None, groups, allow_mask=True, allow_tile=True)
            is None
        ):
            return None
    return dot
