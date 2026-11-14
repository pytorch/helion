"""Analysis: the FX graph-walk primitives shared by the attention matchers.

Role: analysis.  Leaf helper module of ``cute/analysis/attention_match`` (dense, causal
and score-modified attention), ``cute/flex_match`` / ``cute/flex_bwd_match``
(FlexAttention forward / backward), ``cute/cute_flash_gated`` and
``cute/cute_flash_bwd`` (the dense backward), and of the FX-level admission
analyses (``binary_pair``, ``node_tensor``).  Every helper reads one FX node
(or one block-size entry) and returns a plain value; none mutates a graph or
reads the compile environment.  The host-scalar readers (``symnode_expr``,
``symnode_sympy``, ``scale_operand``, ``split_scale``) return the traced
SymInt / SymFloat's sympy expression so symbol identity survives to codegen
(``DeviceFunction.sympy_expr`` lifts it as the kernel argument it came from).

Literal acceptance differs per family and is explicit here: the dense
matchers accept Python numbers and ``aten.scalar_tensor(c)`` only
(``scalar_value(node)``); the flex matchers also accept a 0-d
``aten.full([], c)`` / ``hl.full([], c)`` (``allow_full=True``).  The dense
family keeps its narrower acceptance: widening it would admit kernels to the
tcgen05 flash lowering that it was never validated on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple

import sympy
import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Container

    from ...compile_environment import CompileEnvironment

_aten = torch.ops.aten


def is_call(node: object, *targets: object) -> bool:
    """``node`` is a ``call_function`` of one of ``targets`` (by identity)."""
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target in targets
    )


def binary_pair(
    node: object, targets: Container[object]
) -> tuple[object, object] | None:
    """The two operands of an exactly-binary ``call_function`` (no keywords)
    whose target is in ``targets``; None for anything else."""
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target not in targets
        or len(node.args) != 2
        or node.kwargs
    ):
        return None
    return node.args[0], node.args[1]


def node_tensor(node: object) -> torch.Tensor | None:
    """The fake tensor (``meta["val"]``) of an FX node; None for a non-node
    or a non-tensor value."""
    if isinstance(node, torch.fx.Node):
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            return value
    return None


def new_var_source(node: torch.fx.Node) -> torch.fx.Node | None:
    """The node one ``_new_var`` wrapper renames; None for any other node."""
    from ....language._tracing_ops import _new_var

    if node.op != "call_function" or node.target is not _new_var:
        return None
    if not node.args or not isinstance(node.args[0], torch.fx.Node):
        return None
    return node.args[0]


def strip_new_var(node: object) -> object:
    """``node`` with every ``_new_var`` wrapper removed."""
    from ....language._tracing_ops import _new_var

    while is_call(node, _new_var):
        assert isinstance(node, torch.fx.Node)
        node = node.args[0]
    return node


def host_tensor_name(node: object) -> str | None:
    """The name of a ``_host_tensor(name)`` node; None for anything else."""
    from ....language._tracing_ops import _host_tensor

    if is_call(node, _host_tensor):
        assert isinstance(node, torch.fx.Node)
        name = node.args[0] if node.args else None
        return name if isinstance(name, str) else None
    return None


def scalar_value(node: object, *, allow_full: bool = False) -> float | None:
    """A Python number (``bool`` included) or ``aten.scalar_tensor(c)`` as a
    float; with ``allow_full`` also a 0-d ``aten.full([], c)`` /
    ``hl.full([], c)``. None for anything else."""
    if isinstance(node, (int, float)):
        return float(node)
    if is_call(node, _aten.scalar_tensor.default):
        assert isinstance(node, torch.fx.Node)
        value = node.args[0] if node.args else None
        return float(value) if isinstance(value, (int, float)) else None
    if not allow_full:
        return None
    from ....language import creation_ops

    if is_call(node, _aten.full.default, creation_ops.full):
        assert isinstance(node, torch.fx.Node)
        if len(node.args) < 2 or node.args[0] not in ([], ()):
            return None
        value = node.args[1]
        return float(value) if isinstance(value, (int, float)) else None
    return None


def is_neg_inf(node: object, *, allow_full: bool = False) -> bool:
    """``node`` is the literal ``-inf`` (see :func:`scalar_value`)."""
    value = scalar_value(node, allow_full=allow_full)
    return value is not None and value == float("-inf")


def unit_fixed_block(env: CompileEnvironment, block_id: int) -> bool:
    """``block_id`` is pinned to one element: ``hl.tile(..., block_size=1)``
    or an ``hl.grid`` axis of unit step. An autotuned block's ``tile.begin``
    is its first row only, so a scalar index of it does not address the
    tile."""
    from ...compile_environment import FixedBlockSizeSource

    source = env.block_sizes[block_id].block_size_source
    return (
        isinstance(source, FixedBlockSizeSource)
        and isinstance(source.value, int)
        and source.value == 1
    )


def host_tensor_load(
    node: object, *, index_len: int
) -> tuple[str, torch.fx.Node] | None:
    """Match ``load(_host_tensor(name), [block, (:,)?], None, None)``: a host
    tensor read at one block index (``index_len`` 1) or at a block index and
    the full last dim (``index_len`` 2).

    Returns (host tensor name, block-index node) or None.
    """
    from ....language import memory_ops

    if not is_call(node, memory_ops.load):
        return None
    assert isinstance(node, torch.fx.Node)
    name = host_tensor_name(node.args[0])
    index = node.args[1]
    if name is None or not isinstance(index, (list, tuple)) or len(index) != index_len:
        return None
    if index_len == 2 and index[1] != slice(None):
        return None
    block = index[0]
    if not isinstance(block, torch.fx.Node):
        return None
    return name, block


def symnode_expr(node: object) -> str | None:
    """The expression text of a ``_get_symnode(text)`` node; None otherwise."""
    from ....language._tracing_ops import _get_symnode

    if is_call(node, _get_symnode):
        assert isinstance(node, torch.fx.Node)
        expr = node.args[0]
        if isinstance(expr, str):
            return expr
    return None


def symnode_sympy(node: object) -> sympy.Expr | None:
    """The traced SymInt / SymFloat's sympy expression behind an FX node
    (its ``meta["val"]``), or None."""
    if not isinstance(node, torch.fx.Node):
        return None
    value = node.meta.get("val")
    if not isinstance(value, (torch.SymInt, torch.SymFloat)):
        return None
    expr = value._sympy_()
    return expr if isinstance(expr, sympy.Expr) else None


def scale_operand(node: object) -> tuple[sympy.Expr | float, str] | None:
    """A runtime scale operand: a traced host SymFloat (its sympy expression
    and host expression text) or a Python number baked into the graph."""
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return float(node), repr(float(node))
    text = symnode_expr(node)
    sym = symnode_sympy(node)
    if text is None or sym is None:
        return None
    return sym, text


def split_scale(
    node: object,
) -> tuple[object, tuple[sympy.Expr | float, str]] | None:
    """``mul(x, scale)`` -> (x, scale operand); any other node is ``x * 1``."""
    if is_call(node, _aten.mul.Tensor):
        assert isinstance(node, torch.fx.Node)
        scale = scale_operand(node.args[1])
        if scale is None:
            return None
        return node.args[0], scale
    return node, (1.0, "1.0")


class FlattenedRowOffset(NamedTuple):
    """``tile.begin // rows_per_group * group_stride`` of block ``block_id``:
    the row offset of a tile's group in a tensor whose groups lie
    ``group_stride`` rows apart.  A kernel that tiles ``B*H*M`` query rows on
    one axis reads its batch head's ``N`` key rows of a ``[B*H*N, D]`` tensor
    at ``tile_n.index + tile_m.begin // M * N``."""

    block_id: int
    rows_per_group: int
    group_stride: int


def flattened_row_offset(
    node: object, tile_begin_block: Callable[[sympy.Symbol], int | None]
) -> FlattenedRowOffset | None:
    """The folded ``tile.begin // M * N`` behind a ``_get_symnode`` node: the
    host traces it as ``N*FloorDiv(u, M)`` with ``u`` a tile-begin symbol,
    which ``tile_begin_block`` maps to its block id (None for any other
    symbol).  None for any other expression."""
    from torch.utils._sympy.functions import FloorDiv

    expr = symnode_sympy(node)
    if expr is None:
        return None
    stride, quotient = expr.as_coeff_Mul()
    if not (
        isinstance(stride, sympy.Integer)
        and isinstance(quotient, FloorDiv)
        and len(quotient.args) == 2
    ):
        return None
    begin, rows = quotient.args
    if not (isinstance(begin, sympy.Symbol) and isinstance(rows, sympy.Integer)):
        return None
    block_id = tile_begin_block(begin)
    if block_id is None:
        return None
    return FlattenedRowOffset(block_id, int(rows), int(stride))


# The packed two-operand fp32 multiply ``hl.inline_asm_elementwise(...,
# pack=2)`` spells as PTX (``examples/blackwell_attention.py``): both fp32
# pairs are moved into 64-bit registers, multiplied by one ``mul.f32x2``
# (round-to-nearest, like ``aten.mul``) and unpacked.  Whitespace-normalized.
_PACKED_F32X2_MULTIPLY_ASM = (
    "{ .reg .b64 ra, rb, rc; mov.b64 ra, { $2, $3 }; mov.b64 rb, { $4, $5 }; "
    "mul.f32x2 rc, ra, rb; mov.b64 { $0, $1 }, rc; }"
)
_PACKED_F32X2_BINARY_CONSTRAINTS = "=r,=r,r,r,r,r"


def packed_f32x2_multiply(node: object) -> tuple[torch.fx.Node, torch.fx.Node] | None:
    """The two tensor operands of a pure, packed-by-two fp32
    ``inline_asm_elementwise`` whose asm is the ``mul.f32x2`` idiom: an
    elementwise ``aten.mul.Tensor`` under another spelling.  None for any
    other call."""
    from ....language.inline_asm_ops import inline_asm_elementwise

    if not is_call(node, inline_asm_elementwise):
        return None
    assert isinstance(node, torch.fx.Node)
    if len(node.args) < 6:
        return None
    asm, constraints, operands, dtype, is_pure, pack = node.args[:6]
    if not (
        isinstance(asm, str)
        and constraints == _PACKED_F32X2_BINARY_CONSTRAINTS
        and dtype is torch.float32
        and is_pure is True
        and pack == 2
        and " ".join(asm.split()) == _PACKED_F32X2_MULTIPLY_ASM
    ):
        return None
    if not isinstance(operands, (list, tuple)) or len(operands) != 2:
        return None
    lhs, rhs = operands
    if not (isinstance(lhs, torch.fx.Node) and isinstance(rhs, torch.fx.Node)):
        return None
    return lhs, rhs
