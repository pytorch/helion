"""Triton lowering of packed MXFP8 ``hl.dot_scaled`` on sm100/sm103.

A packed MXFP8 GEMM loads E4M3 operands and int32 words holding four E8M0
scale bytes, and extracts each K group's byte in the kernel
(``mxfp8.match_packed_scale``).  The generic lowering loads the scale words
once per byte and leaves every operand on pointer loads.  Here the operands
and scale words are loaded through host tensor descriptors instead, the words
are unpacked once into the order tcgen05's UTCCP scale copy expects, and a
16- or 32-row tile is issued as the transposed MMA (``swap``).

Each rewrite is used only when it reads exactly the values of the load it
replaces:

* every descriptor offset is the start of an ``iota + scalar`` index,
* every load mask only clips lanes outside the tensor (a descriptor zero
  fills those) -- a bound at or past the extent, or the end of a K loop whose
  tiles never cross it before the end of the tensor, and
* the K tiles of an unpacked scale start on a word (a multiple of 4 groups).

The last two are proved for the config being compiled: a split-K kernel's
partition tunable decides both.

Inside a warp-specialized loop the dot is lowered natively only as a whole:
a pointer-loaded operand beside descriptor loads and the TMA output store
gives wrong, nondeterministic results there (Triton 3.9), so such a config
is rejected (``InvalidConfig``).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import functools
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ... import exc
from ..._compat import tensor_descriptor_supports_dtype
from ...language import _tracing_ops
from ...language import view_ops
from ..ast_extension import expr_from_string
from ..compile_environment import CUDA_TENSOR_DESCRIPTOR_MAX_BLOCK_SIZE
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction
from ..indexing_strategy import _contiguous_integer_tensor_index
from ..indexing_strategy import _in_warp_specialized_loop
from ..indexing_strategy import _tensor_dimension_fits_int32
from ..matmul_utils import _resolve_dim_size
from ..mxfp8 import PackedScaleMatch
from ..mxfp8 import find_packed_mxfp8_dot
from ..mxfp8 import match_packed_scale
from ..mxfp8 import native_packed_mxfp8_target
from ..mxfp8 import strip_new_var
from ..mxfp8 import tile_load
from ..variable_origin import BlockSizeOrigin
from ..variable_origin import TileBeginOrigin

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState

# All-127 bytes: E8M0 scale 1.0 for padded words outside the scale tensor.
_UNIT_SCALE_WORDS = 0x7F7F7F7F
# A TMA store stages its whole tile in shared memory beside the main loop's
# operand ring: 64 KiB (a 128x256 bf16 tile) still fits a 3-stage 128x256
# FP8 ring on sm100.
_MAX_STORE_TILE_BYTES = 64 * 1024


@dataclass(frozen=True)
class PackedMxfp8Dot:
    """Replacements for a packed MXFP8 ``dot_scaled``'s argument ASTs.

    A ``None`` operand keeps its generic load.  ``swap`` issues the dot as
    ``(rhs^T @ lhs^T)^T`` so a 16- or 32-row tile fills tcgen05's 128-row M.
    """

    lhs: ast.AST | None
    lhs_scale: ast.AST | None
    rhs: ast.AST | None
    rhs_scale: ast.AST | None
    swap: bool


@dataclass(frozen=True)
class _Operand:
    load: torch.fx.Node
    tensor: torch.Tensor
    scale: PackedScaleMatch
    # The scale tile: operand rows by K groups.
    tile: tuple[int, int]


def packed_mxfp8_dot(state: CodegenState) -> PackedMxfp8Dot | None:
    """Plan the native lowering of the ``dot_scaled`` being generated."""
    if not native_packed_mxfp8_target(CompileEnvironment.current()):
        return None
    operands = _match_operands(state)
    if operands is None:
        return None
    lhs, rhs = operands
    swap = lhs.tile[0] in (16, 32) and rhs.tile[0] >= 128
    if not all(_scales_word_aligned(state, operand) for operand in operands):
        return PackedMxfp8Dot(None, None, None, None, swap)
    lhs_scale = _packed_scale_ast(state, lhs, prefix="mxfp8_lhs")
    rhs_scale = _packed_scale_ast(state, rhs, prefix="mxfp8_rhs")
    if lhs_scale is None or rhs_scale is None:
        return PackedMxfp8Dot(None, None, None, None, swap)
    lhs_operand = _operand_ast(state, lhs)
    rhs_operand = _operand_ast(state, rhs)
    if (lhs_operand is None or rhs_operand is None) and _in_warp_specialized_loop(
        state
    ):
        raise exc.InvalidConfig(
            "a warp-specialized loop needs every packed MXFP8 operand loaded "
            "through a tensor descriptor; this config leaves one on pointer loads"
        )
    if rhs_operand is not None:
        rhs_operand = expr_from_string("tl.permute({value}, [1, 0])", value=rhs_operand)
    return PackedMxfp8Dot(lhs_operand, lhs_scale, rhs_operand, rhs_scale, swap)


def _unmask_zero_operand(node: object) -> object:
    while (
        isinstance(node, torch.fx.Node)
        and node.target is _tracing_ops._mask_to
        and len(node.args) == 2
        and not node.kwargs
        and node.args[1] == 0
    ):
        node = node.args[0]
    return node


def _match_operands(state: CodegenState) -> tuple[_Operand, _Operand] | None:
    """``lhs [M, K]`` and K-major ``rhs [N, K]`` (used as ``rhs.T``) loads of
    E4M3 data with packed scales indexed by the same rows."""
    node = state.fx_node
    if node is None or state.proxy_args[2] != "e4m3" or state.proxy_args[5] != "e4m3":
        return None
    rhs_node = _unmask_zero_operand(node.args[3])
    if (
        not isinstance(rhs_node, torch.fx.Node)
        or rhs_node.target is not torch.ops.aten.permute.default
        or rhs_node.args[1:] != ([1, 0],)
        or rhs_node.kwargs
    ):
        return None
    operands = []
    for load, scale, scale_proxy in (
        (_unmask_zero_operand(node.args[0]), node.args[1], state.proxy_args[1]),
        (rhs_node.args[0], node.args[4], state.proxy_args[4]),
    ):
        loaded = tile_load(load, allow_mask=True)
        if (
            loaded is None
            or not isinstance(load, torch.fx.Node)
            or loaded[0].dtype != torch.float8_e4m3fn
            or not isinstance(scale_proxy, torch.Tensor)
            or scale_proxy.ndim != 2
        ):
            return None
        tile = tuple(_resolve_dim_size(size) for size in scale_proxy.shape)
        if not all(type(size) is int and size > 0 for size in tile):
            return None
        tile = cast("tuple[int, int]", tile)
        match = match_packed_scale(
            scale, loaded[1][0], tile[1], allow_mask=True, allow_tile=True
        )
        if match is None:
            return None
        operands.append(_Operand(load, loaded[0], match, tile))
    lhs, rhs = operands
    if lhs.tile[1] != rhs.tile[1]:
        return None
    return lhs, rhs


# -- proofs ----------------------------------------------------------------


def _expr(value: object) -> sympy.Expr | None:
    if isinstance(value, torch.fx.Node):
        value = value.meta.get("val")
    if isinstance(value, torch.SymInt):
        value = value._sympy_()
    if type(value) is int:
        return sympy.Integer(value)
    if isinstance(value, sympy.Expr):
        return CompileEnvironment.current().shape_env.replace(value)
    return None


def _static(value: object) -> int | None:
    """A tensor size or stride every input of this bound kernel shares."""
    expr = _expr(value)
    if expr is None:
        return None
    expr = CompileEnvironment.current().specialize_expr(expr)
    return int(expr) if isinstance(expr, sympy.Integer) else None


def _tensor_extent(value: object) -> sympy.Expr | None:
    """A tensor dimension, specialized when the bound kernel keys on it."""
    expr = _expr(value)
    if expr is None:
        return None
    return CompileEnvironment.current().specialize_expr(expr)


def _configured(state: CodegenState, value: object) -> sympy.Expr | None:
    """``value`` with the block sizes and tunables of the config compiled."""
    expr = _expr(value)
    if expr is None:
        return None
    env = CompileEnvironment.current()
    origins = HostFunction.current().expr_to_origin
    expr = env.specialize_expr(expr)
    substitutions: dict[sympy.Basic, sympy.Basic] = {}
    for symbol in expr.free_symbols:
        origin = origins.get(symbol)
        if symbol in env.config_value_expressions:
            substitutions[symbol] = sympy.Integer(
                env.config_value_expressions[symbol].evaluate(state.config)
            )
        elif origin is not None and isinstance(origin.origin, BlockSizeOrigin):
            size = _block_size(state, origin.origin.block_id)
            if size is not None:
                substitutions[symbol] = sympy.Integer(size)
    # Grid and tile-begin symbols stay free: a proof holds for every tile.
    return cast("sympy.Expr", expr.xreplace(substitutions))


def _known_nonnegative(expr: sympy.Expr) -> bool:
    return sympy.simplify(expr).is_nonnegative is True


def _known_multiple(expr: sympy.Expr, divisor: int) -> bool:
    return sympy.Mod(sympy.expand(expr), divisor) == 0


def _parent(
    state: CodegenState, graph: torch.fx.Graph
) -> tuple[torch.fx.Node, int] | None:
    """The one control-flow node capturing ``graph``'s inputs, and its slot."""
    # device_ir imports this backend's codegen modules; import it on use.
    from ..device_ir import control_flow_child_graph_ids
    from ..device_ir import control_flow_parent_entries

    graphs = state.codegen.codegen_graphs
    info = next((info for info in graphs if info.graph is graph), None)
    if info is None:
        return None
    references = sum(
        control_flow_child_graph_ids(node).count(info.graph_id)
        for other in graphs
        for node in other.graph.nodes
    )
    if references != 1:
        return None
    return control_flow_parent_entries(graphs).get(info.graph_id)


def _definition(state: CodegenState, node: object) -> object:
    """Follow ``_new_var`` copies and loop-invariant captures to a value's
    definition in an enclosing graph."""
    for _ in range(8):
        node = strip_new_var(node)
        if not isinstance(node, torch.fx.Node) or node.op != "placeholder":
            return node
        entry = _parent(state, node.graph)
        if entry is None:
            return node
        parent, slot = entry
        captured = parent.args[slot]
        position = list(node.graph.find_nodes(op="placeholder")).index(node)
        if not isinstance(captured, (list, tuple)) or position >= len(captured):
            return node
        source = captured[position]
        # A loop carry rebinds the placeholder after the first iteration.
        if isinstance(source, torch.fx.Node) and any(
            user.target is _tracing_ops._phi
            and len(user.args) == 2
            and user.args[0] is source
            for user in source.users
        ):
            return node
        node = source
    return node


def _contiguous(
    state: CodegenState, index: object
) -> tuple[sympy.Expr, sympy.Expr] | None:
    """``(start, extent)`` of an ``iota(extent) + start`` index vector."""
    definition = _definition(state, index)
    value = (
        definition.meta.get("val") if isinstance(definition, torch.fx.Node) else None
    )
    if not isinstance(value, torch.Tensor):
        return None
    info = _contiguous_integer_tensor_index(value, definition)
    if info is None:
        return None
    start: sympy.Expr = sympy.Integer(0)
    for sign, term in info.base_terms:
        term_expr = _expr(term)
        if term_expr is None:
            # A device-produced (tensor) base: contiguous, but not affine in
            # the loop symbols this module reasons about.
            return None
        start = sympy.Add(start, sympy.Mul(sign, term_expr))
    extent = _expr(info.extent)
    return (start, extent) if extent is not None else None


def _loop_bounds(
    state: CodegenState, block_id: int
) -> tuple[sympy.Expr, sympy.Expr] | None:
    """The configured begin and end of the device loop over ``block_id``."""
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import device_loop_bounds

    graphs = state.codegen.codegen_graphs
    loops = [
        info
        for info in graphs
        if isinstance(info, ForLoopGraphInfo) and block_id in info.block_ids
    ]
    if len(loops) != 1:
        return None
    entry = _parent(state, loops[0].graph)
    if entry is None:
        return None
    bounds = device_loop_bounds(loops[0], {loops[0].graph_id: entry[0]}, block_id)
    if bounds is None:
        return None
    begin, end = (_configured(state, value) for value in bounds)
    return (begin, end) if begin is not None and end is not None else None


def _block_size(state: CodegenState, block_id: int) -> int | None:
    size = CompileEnvironment.current().block_sizes[block_id].from_config(state.config)
    return size if type(size) is int else None


def _tile_begins(state: CodegenState, start: sympy.Expr) -> tuple[int, int] | None:
    """``(block_id, c)`` when ``start`` is ``c * tile.begin`` of a K tile."""
    origins = HostFunction.current().expr_to_origin
    for symbol in start.free_symbols:
        origin = origins.get(symbol)
        if origin is None or not isinstance(origin.origin, TileBeginOrigin):
            continue
        coefficient = sympy.simplify(sympy.Mul(start, sympy.Pow(symbol, -1)))
        if coefficient.is_Integer and int(coefficient) > 0:
            return origin.origin.block_id, int(coefficient)
    return None


def _bound_clips_only_outside(
    state: CodegenState, index: object, bound: object, extent: int | sympy.Expr
) -> bool:
    """Whether ``index < bound`` is false only where ``index >= extent``,
    on every tile the config executes.  A symbolic ``extent`` (a dimension
    the bound kernel does not specialize) only passes the tile proof."""
    configured = _configured(state, bound)
    if configured is None:
        return False
    if _known_nonnegative(sympy.Add(configured, -extent)):
        return True
    # A split-K partition ends inside the tensor: ``k < c * loop_end`` for an
    # index ``c * tile.begin + iota(c * block)`` of that loop is outside the
    # loop's own tiles unless a tile crosses the partition end.
    contiguous = _contiguous(state, index)
    if contiguous is None:
        return False
    start, width = (_configured(state, value) for value in contiguous)
    tile = None if start is None else _tile_begins(state, start)
    if tile is None:
        return False
    block_id, scale = tile
    size = _block_size(state, block_id)
    bounds = _loop_bounds(state, block_id)
    if size is None or bounds is None or width != scale * size:
        return False
    begin, end = bounds
    if sympy.simplify(sympy.Add(configured, sympy.Mul(-scale, end))) != 0:
        return False
    # ``end`` is a minimum of arms; each must either end a whole number of
    # tiles after ``begin`` or lie at or past the tensor's end.
    arms = end.args if isinstance(end, sympy.Min) else (end,)
    return all(
        _known_multiple(sympy.Add(arm, sympy.Mul(-1, begin)), size)
        or _known_nonnegative(sympy.Add(sympy.Mul(scale, arm), -extent))
        for arm in arms
    )


def _mask_conjuncts(mask: object) -> list[object] | None:
    mask = strip_new_var(mask)
    if mask is None:
        return []
    if (
        isinstance(mask, torch.fx.Node)
        and mask.target in (operator.and_, torch.ops.aten.bitwise_and.Tensor)
        and len(mask.args) == 2
        and not mask.kwargs
    ):
        left, right = (_mask_conjuncts(arg) for arg in mask.args)
        return None if left is None or right is None else [*left, *right]
    return [mask]


def _broadcast_source(node: object, dim: int) -> object:
    """``x`` for ``x[:, None]`` (dim 0) or ``x[None, :]`` (dim 1)."""
    node = strip_new_var(node)
    pattern = [slice(None), None] if dim == 0 else [None, slice(None)]
    if (
        isinstance(node, torch.fx.Node)
        and node.target is view_ops.subscript
        and len(node.args) == 2
        and not node.kwargs
        and isinstance(node.args[1], (list, tuple))
        and list(node.args[1]) == pattern
    ):
        return strip_new_var(node.args[0])
    return None


def mask_clips_only_outside(
    state: CodegenState,
    mask: object,
    indices: tuple[object, object],
    shape: tuple[int | sympy.Expr, int | sympy.Expr],
) -> bool:
    """Whether a 2-D access's mask equals its out-of-bounds lanes.

    ``mask`` must be a conjunction of ``index[:, None] < bound`` and
    ``index[None, :] < bound`` tests (``_bound_clips_only_outside``).
    """
    conjuncts = _mask_conjuncts(mask)
    if conjuncts is None:
        return False
    definitions = [_definition(state, index) for index in indices]
    for conjunct in conjuncts:
        if (
            not isinstance(conjunct, torch.fx.Node)
            or conjunct.target
            not in (operator.lt, torch.ops.aten.lt.Scalar, torch.ops.aten.lt.Tensor)
            or len(conjunct.args) != 2
            or conjunct.kwargs
        ):
            return False
        lhs, bound = conjunct.args
        if not any(
            (source := _broadcast_source(lhs, dim)) is not None
            and _definition(state, source) is definitions[dim]
            and _bound_clips_only_outside(state, indices[dim], bound, shape[dim])
            for dim in (0, 1)
        ):
            return False
    return True


def _scales_word_aligned(state: CodegenState, operand: _Operand) -> bool:
    """Whether every K tile of the scale starts on a whole packed word."""
    block_id = operand.scale.group_block_id
    if block_id is None:
        return True  # a full ``arange(groups)`` starts at group 0
    size = _block_size(state, block_id)
    bounds = _loop_bounds(state, block_id)
    return (
        size is not None
        and size % 4 == 0
        and bounds is not None
        and _known_multiple(bounds[0], 4)
    )


# -- codegen ---------------------------------------------------------------


def _node_ast(state: CodegenState, node: object) -> ast.AST | None:
    if not isinstance(node, torch.fx.Node):
        return None
    value = state.env.get(node)
    return value if isinstance(value, ast.AST) else None


def _descriptor_eligible(
    tensor: torch.Tensor, *, stride_one_dim: int | None = None
) -> bool:
    """The config-independent checks for a 2-D host descriptor over ``tensor``
    (its layout signature and base alignment are part of the bound kernel's
    key)."""
    env = CompileEnvironment.current()
    layout = env.tensor_descriptor_layout_signature(tensor)
    return (
        tensor.ndim == 2
        # Offsets are cast to int32 (as ``TensorDescriptorIndexingStrategy``).
        and (
            env.index_dtype != torch.int64
            or all(_tensor_dimension_fits_int32(size) for size in tensor.shape)
        )
        and tensor_descriptor_supports_dtype(tensor.dtype)
        and layout is not None
        and layout[0] is not None
        and stride_one_dim in (None, layout[0])
        and all(aligned or dim == layout[0] for dim, aligned in enumerate(layout[1]))
        and env.tensor_descriptor_base_is_aligned(tensor)
    )


def descriptor_operands(dot: torch.fx.Node) -> bool:
    """Whether both operands of a packed MXFP8 ``dot`` and its MN-major scale
    words admit this lowering's host descriptors.

    A seed that warp-specializes the dot's loop checks this: the rest of the
    proof (contiguous indices, masks, word-aligned K tiles) then holds for
    the seed's own tiles, so its config lowers every operand natively.
    """
    rhs = _unmask_zero_operand(dot.args[3])
    if (
        not isinstance(rhs, torch.fx.Node)
        or rhs.target is not torch.ops.aten.permute.default
    ):
        return False
    tensors: list[tuple[torch.Tensor, int | None]] = []
    for load in (_unmask_zero_operand(dot.args[0]), rhs.args[0]):
        loaded = tile_load(load, allow_mask=True)
        if loaded is None:
            return False
        tensors.append((loaded[0], None))
    for scale in (dot.args[1], dot.args[4]):
        value = scale.meta.get("val") if isinstance(scale, torch.fx.Node) else None
        groups = (
            value.shape[1]
            if isinstance(value, torch.Tensor)
            and value.ndim == 2
            and type(value.shape[1]) is int
            else None
        )
        match = match_packed_scale(
            scale, None, groups, allow_mask=True, allow_tile=True
        )
        if match is None:
            return False
        # The words load as a [words, rows] block of MN-major storage.
        tensors.append((match.tensor, 0))
    return all(
        _descriptor_eligible(tensor, stride_one_dim=dim) for tensor, dim in tensors
    )


def descriptor_tile(
    state: CodegenState,
    tensor: torch.Tensor,
    indices: tuple[object, object],
    block_shape: tuple[int, int],
    *,
    physical_layout: bool = False,
    load: bool = True,
) -> tuple[str, list[ast.AST], list[int] | None] | None:
    """A host descriptor over ``tensor`` for a 2-D block at ``indices``.

    Returns the descriptor name, its offsets in descriptor-dimension order,
    and the inverse permutation that restores the tensor-dimension order
    (``None`` when the descriptor is not permuted or ``physical_layout`` asks
    for the descriptor order).  ``indices`` must be proven contiguous.
    """
    env = CompileEnvironment.current()
    index_asts = [
        index for index in map(functools.partial(_node_ast, state), indices) if index
    ]
    layout = env.tensor_descriptor_layout_signature(tensor)
    if (
        len(index_asts) != 2
        # Fences around cross-thread syncs only cover TMA the function knows of.
        or not state.device_function.uses_async_proxy_global
        or not native_packed_mxfp8_target(env)
        or not _descriptor_eligible(tensor)
        or layout is None
        or layout[0] is None
        or not all(
            0 < size <= CUDA_TENSOR_DESCRIPTOR_MAX_BLOCK_SIZE and size & (size - 1) == 0
            for size in block_shape
        )
        or block_shape[layout[0]] * tensor.element_size() < 16
    ):
        return None
    descriptor = state.device_function.tensor_descriptor_arg(
        tensor, list(block_shape), force_host=True
    )
    if load is False:
        state.device_function.note_tma_store(
            warp_specialized=_in_warp_specialized_loop(state)
        )
    order = descriptor.permutation or [0, 1]
    offsets = [
        expr_from_string("tl.cast(tl.min({index}), tl.int32)", index=index_asts[dim])
        for dim in order
    ]
    inverse = (
        None
        if descriptor.permutation is None or physical_layout
        else descriptor.inverse_permutation
    )
    return descriptor.name, offsets, inverse


def _descriptor_load(
    state: CodegenState,
    tensor: torch.Tensor,
    indices: tuple[object, object],
    block_shape: tuple[int, int],
    *,
    physical_layout: bool = False,
) -> ast.AST | None:
    descriptor = descriptor_tile(
        state, tensor, indices, block_shape, physical_layout=physical_layout
    )
    if descriptor is None:
        return None
    name, offsets, inverse = descriptor
    placeholders = {f"offset_{i}": offset for i, offset in enumerate(offsets)}
    value = expr_from_string(
        f"{name}.load([{', '.join(f'{{{key}}}' for key in placeholders)}])",
        **placeholders,
    )
    if inverse is not None:
        value = expr_from_string(f"tl.permute({{value}}, {inverse!r})", value=value)
    return value


def _operand_ast(state: CodegenState, operand: _Operand) -> ast.AST | None:
    """The operand tile through a descriptor, when that reads the same values."""
    indices = cast(
        "tuple[object, object]", tuple(cast("list[object]", operand.load.args[1]))
    )
    rows, groups = operand.tile
    # A dimension the bound kernel does not specialize (for example the K of
    # ``b`` when only ``a``'s K is passed to ``hl.specialize``) stays symbolic:
    # the mask proof then has to hold for every value of it.
    shape = tuple(_tensor_extent(size) for size in operand.tensor.shape)
    value = operand.load.meta.get("val")
    if (
        not isinstance(value, torch.Tensor)
        or tuple(_configured(state, size) for size in value.shape)
        != (rows, groups * 32)
        or any(size is None for size in shape)
        or any(_contiguous(state, index) is None for index in indices)
        or not mask_clips_only_outside(
            state,
            operand.load.args[2],
            indices,
            cast("tuple[sympy.Expr, sympy.Expr]", shape),
        )
    ):
        return None
    return _descriptor_load(state, operand.tensor, indices, (rows, groups * 32))


def _packed_scale_ast(
    state: CodegenState, operand: _Operand, *, prefix: str
) -> ast.AST | None:
    """Load packed words once and expose scales in tcgen05's UTCCP order."""
    match = operand.scale
    block_rows, block_groups = operand.tile
    if block_rows % 128 or block_groups % 4:
        return None
    row_ast = _node_ast(state, match.row)
    word_index_ast = _node_ast(state, match.word_index)
    if row_ast is None or word_index_ast is None:
        return None
    tensor = match.tensor
    words = block_groups // 4
    loaded = None
    layout = CompileEnvironment.current().tensor_descriptor_layout_signature(tensor)
    # MN-major words (rows contiguous) arrive in descriptor order [words, rows].
    # The layout signature is part of the bound kernel's key.
    if (
        layout is not None
        and layout[0] == 0
        and _contiguous(state, match.row) is not None
    ):
        loaded = _descriptor_load(
            state,
            tensor,
            (match.row, match.word_index),
            (block_rows, words),
            physical_layout=True,
        )
    if loaded is None:
        # The pointer form spells out sizes and strides, so they must be the
        # same for every input of this bound kernel, and indexes in int32.
        rows, words_per_row = (_static(size) for size in tensor.shape)
        row_stride, word_stride = (_static(stride) for stride in tensor.stride())
        if (
            rows is None
            or words_per_row is None
            or row_stride != 1
            or word_stride is None
            or word_stride < rows
            or words_per_row * word_stride >= 2**31
        ):
            return None
        pointer = state.device_function.tensor_arg(tensor).name
        loaded = expr_from_string(
            (
                "tl.load({pointer} + "
                f"({{word_start}} + tl.arange(0, {words}))[:, None] * {word_stride} + "
                f"{{rows}}[None, :], "
                f"mask=(({{word_start}} + tl.arange(0, {words}))[:, None] < {words_per_row}) "
                f"& ({{rows}}[None, :] < {rows}), other={_UNIT_SCALE_WORDS})"
            ),
            pointer=expr_from_string(pointer),
            word_start=expr_from_string(
                "tl.min({word_index})", word_index=word_index_ast
            ),
            rows=row_ast,
        )
    loaded = state.codegen.lift(loaded, dce=True, prefix=f"{prefix}_words")
    unpacked = expr_from_string(
        (
            "(({words}[:, :, None] >> "
            "(tl.arange(0, 4)[None, None, :] * 8)) & 255).to(tl.uint8)"
        ),
        words=loaded,
    )
    unpacked = state.codegen.lift(unpacked, dce=True, prefix=f"{prefix}_unpacked")
    # The words arrive as [K/128, rows, 4 bytes].  The reshape and transposes
    # are the identity on the logical [rows, groups] scale tile; they match
    # Triton's block-scaled tutorial so the scales reach TMEM through UTCCP.
    scale = expr_from_string(
        (
            f"{{scale}}.reshape({words}, {block_rows // 128}, 4, 32, 4)"
            ".trans(1, 0, 3, 2, 4)"
            ".trans(0, 3, 2, 1, 4)"
            f".reshape({block_rows}, {block_groups})"
        ),
        scale=unpacked,
    )
    if match.mask is not None:
        mask_ast = _node_ast(state, match.mask)
        if mask_ast is None:
            return None
        scale = expr_from_string(
            "tl.where({mask}, {scale}, 0)", mask=mask_ast, scale=scale
        )
    return scale


def _flattened_indices(
    index: object, row_stride: int | torch.SymInt
) -> tuple[object, object] | None:
    """``(rows, cols)`` of ``rows[:, None] * row_stride + cols[None, :]``."""
    index = strip_new_var(index)
    if (
        not isinstance(index, torch.fx.Node)
        or index.target not in (operator.add, torch.ops.aten.add.Tensor)
        or len(index.args) != 2
        or index.kwargs
    ):
        return None
    env = CompileEnvironment.current()
    for row_term, col_term in (index.args, index.args[::-1]):
        row_term = strip_new_var(row_term)
        cols = _broadcast_source(col_term, 1)
        if (
            cols is None
            or not isinstance(row_term, torch.fx.Node)
            or row_term.target not in (operator.mul, torch.ops.aten.mul.Tensor)
            or len(row_term.args) != 2
            or row_term.kwargs
        ):
            continue
        for rows_term, stride in (row_term.args, row_term.args[::-1]):
            rows = _broadcast_source(rows_term, 0)
            if (
                rows is not None
                and isinstance(stride, (int, torch.SymInt))
                and env.known_equal(stride, row_stride)
            ):
                return rows, cols
    return None


def flat_output_descriptor_store(
    state: CodegenState, tensor: torch.Tensor, value: ast.AST
) -> ast.AST | None:
    """Store a tile of a flattened 2-D output through a 2-D host descriptor.

    ``out.view(m * n)[rows[:, None] * n + cols[None, :]] = tile`` under the
    mask ``(rows < m) & (cols < n)`` writes the ``[rows, cols]`` block of
    ``out``.  In a warp-specialized loop, a TMA store of that block lets the
    epilogue overlap the next tile's main loop.
    """
    node = state.fx_node
    base = tensor._base
    env = CompileEnvironment.current()
    if (
        node is None
        or not native_packed_mxfp8_target(env)
        or find_packed_mxfp8_dot(HostFunction.current().device_ir.graphs) is None
        or base is None
        or tensor.ndim != 1
        or base.ndim != 2
        or not base.is_contiguous()
        or base not in HostFunction.current().tensor_to_origin
        or tensor.stride() != (1,)
        or not env.known_equal(tensor.storage_offset(), base.storage_offset())
        or not env.known_equal(tensor.numel(), base.numel())
        or not _in_warp_specialized_loop(state)
    ):
        return None
    subscript = node.args[1]
    value_proxy = state.proxy_arg(2)
    if (
        not isinstance(subscript, (list, tuple))
        or len(subscript) != 1
        or not isinstance(value_proxy, torch.Tensor)
        or value_proxy.ndim != 2
        or value_proxy.dtype != base.dtype
    ):
        return None
    indices = _flattened_indices(subscript[0], base.shape[1])
    block = tuple(_resolve_dim_size(size) for size in value_proxy.shape)
    shape = tuple(_resolve_dim_size(size) for size in base.shape)
    if indices is None or not all(type(size) is int for size in (*block, *shape)):
        return None
    ranges = [_contiguous(state, index) for index in indices]
    if (
        # The value fills the [rows, cols] block rather than broadcasting.
        any(
            index_range is None or _configured(state, index_range[1]) != size
            for index_range, size in zip(ranges, block, strict=True)
        )
        or math.prod(cast("tuple[int, int]", block)) * base.element_size()
        > _MAX_STORE_TILE_BYTES
        or not mask_clips_only_outside(
            state, node.args[3], indices, cast("tuple[int, int]", shape)
        )
    ):
        return None
    descriptor = descriptor_tile(
        state, base, indices, cast("tuple[int, int]", block), load=False
    )
    if descriptor is None or descriptor[2] is not None:
        return None
    name, (row, col), _ = descriptor
    return expr_from_string(
        f"{name}.store([{{row}}, {{col}}], {{value}})", row=row, col=col, value=value
    )
