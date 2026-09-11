"""Whole-root lowering for the resident BT16 KDA chain.

This is deliberately a narrow schedule specialization.  It recognizes a
semantic four-contraction recurrence over the five-factor workspace written by
``chunk_prepare`` and replaces the complete root with a capability-selected
CuTe schedule: a 512-thread SM100 TMEM DV2 path (device ABI 2) or a 192-thread
SM100 warp-MMA DV4 path (device ABI 3). Both consume workspace layout v2;
mismatches leave no partial lowering behind.
"""

from __future__ import annotations

import ast
from collections import Counter
import dataclasses
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ...autotuner.config_spec import CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY
from ...language.matmul_ops import dot
from ..compile_environment import CompileEnvironment
from .chunk_prepare import _packed_workspace_is_proven
from .chunk_prepare import _register_packed_workspace_specialization
from .fx_matcher import _canonical_root_axis_ids
from .fx_matcher import _is_call
from .fx_matcher import _is_matrix_transpose_of
from .fx_matcher import _linear_offsets_fit_i32
from .fx_matcher import _load_ref
from .fx_matcher import _memory_indices
from .fx_matcher import _same_ref
from .fx_matcher import _store_ref
from .fx_matcher import _strip_value_casts
from .fx_matcher import _TensorRef
from .fx_matcher import _xyz_grid_fits

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from ..tile_dispatch import TileStrategyDispatch


_BT = 16
_DK = 128
_DV = 128
_SM100_TMEM_THREADS = 512
_SM100_TMEM_SMEM_BYTES = 138_240
_SM100_TMEM_DEVICE_ABI = 2
_SM100_WARP_DV4_THREADS = 192
_SM100_WARP_DV4_SMEM_BYTES = 97_536
_SM100_WARP_DV4_DEVICE_ABI = 3
_SM100_WARP_DV4_INPUT_STAGES = 6
_SM100_WARP_DV4_MAX_CHUNKS = 32


def _select_sm100_dv_partitions(
    *, total_chunks: int, sequences: int, heads: int, num_sm: int
) -> int:
    """Select DV4 only inside its measured short, underfilled envelope.

    DV2 launches two CTAs per ``(sequence, head)``.  Splitting into four DV
    partitions is useful when doubling that grid fills at most one machine
    wave; once DV2 already fills half a wave, the duplicated factor traffic of
    warp HMMA loses to the resident tcgen05 schedule.  Because only aggregate
    chunk count is statically known, multi-sequence inputs conservatively stay
    on DV2 rather than assuming their individual lengths are balanced.
    """

    if min(total_chunks, sequences, heads, num_sm) <= 0:
        raise ValueError("recurrence geometry must be positive")
    dv2_ctas = 2 * sequences * heads
    short_single_sequence = (
        sequences == 1 and total_chunks <= _SM100_WARP_DV4_MAX_CHUNKS
    )
    underfilled = 2 * dv2_ctas <= num_sm
    return 4 if short_single_sequence and underfilled else 2


@dataclasses.dataclass(frozen=True)
class CuteChunkRecurrencePlan:
    """Compile-time contract for the BT16 resident recurrence chain root."""

    root_graph_id: int
    heads: int
    sequences: int
    total_tokens: int
    total_chunks: int
    kd: _TensorRef
    qd: _TensorRef
    ak: _TensorRef
    aq: _TensorRef
    g_total: _TensorRef
    values: _TensorRef
    output: _TensorRef
    state: _TensorRef
    cu_seqlens: _TensorRef
    cu_chunks: _TensorRef
    output_scale: sympy.Expr
    workspace_layout_version: int
    schedule: str
    threads: int
    smem_bytes: int
    device_abi: int
    input_stages: int
    tma_stages: int
    factor_tma_value_splits: int
    output_acc_stages: int
    output_smem_stages: int
    output_store_wait_groups: int
    tmem_cols: int
    dv_partitions: int


@dataclasses.dataclass(frozen=True)
class CuteChunkRecurrenceSearchGeometry:
    """Static geometry exposed to the recurrence autotuner heuristic."""

    total_chunks: int
    sequences: int
    heads: int


@dataclasses.dataclass(frozen=True)
class _CuteChunkRecurrenceMatch:
    """Exact semantic/storage match shared by planning and autotune gating."""

    root_graph_id: int
    root_phase_index: int
    heads: int
    sequences: int
    total_tokens: int
    total_chunks: int
    kd: _TensorRef
    qd: _TensorRef
    ak: _TensorRef
    aq: _TensorRef
    g_total: _TensorRef
    values: _TensorRef
    output: _TensorRef
    state: _TensorRef
    cu_seqlens: _TensorRef
    cu_chunks: _TensorRef
    sequence_coord: torch.fx.Node
    head_coord: torch.fx.Node
    value_coord: torch.fx.Node
    output_scale: sympy.Expr | None


def detect_chunk_recurrence_search_geometry(
    graphs: Sequence[GraphInfo],
) -> CuteChunkRecurrenceSearchGeometry | None:
    """Recognize the exact BT16 recurrence before config codegen."""

    match = _match_chunk_recurrence_graphs(graphs)
    if (
        match is None
        or match.output_scale is None
        or min(match.total_tokens, match.total_chunks, match.sequences, match.heads)
        <= 0
    ):
        return None
    env = CompileEnvironment.current()
    if not _register_packed_workspace_specialization(
        env,
        tuple(
            ref.fake for ref in (match.kd, match.qd, match.ak, match.aq, match.g_total)
        ),
    ):
        return None
    return CuteChunkRecurrenceSearchGeometry(
        match.total_chunks,
        match.sequences,
        match.heads,
    )


def _tensor_refs(
    nodes: Sequence[torch.fx.Node],
) -> list[tuple[torch.fx.Node, _TensorRef]]:
    result: list[tuple[torch.fx.Node, _TensorRef]] = []
    for node in nodes:
        ref = _load_ref(node)
        if ref is not None:
            result.append((node, ref))
    return result


def _single_ref_load(node: object) -> tuple[torch.fx.Node, _TensorRef] | None:
    current = _strip_value_casts(node)
    if current is None:
        return None
    ref = _load_ref(current)
    return (current, ref) if ref is not None else None


def _ancestors(node: object) -> set[torch.fx.Node]:
    if not isinstance(node, torch.fx.Node):
        return set()
    pending = [node]
    seen: set[torch.fx.Node] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(current.all_input_nodes)
    return seen


def _contains_target(node: object, targets: set[object]) -> bool:
    return any(
        current.op == "call_function" and current.target in targets
        for current in _ancestors(node)
    )


def _output_store_contract(
    nodes: Sequence[torch.fx.Node], output_value: torch.fx.Node
) -> tuple[torch.fx.Node, sympy.Expr | None] | None:
    """Match either ABI-v1 direct output or ABI-v2 post-accumulator scaling."""

    from ...language._tracing_ops import _get_symnode

    matches: list[tuple[torch.fx.Node, sympy.Expr | None]] = []
    multiply_targets = {operator.mul, torch.ops.aten.mul.Tensor}
    for node in nodes:
        if _store_ref(node) is None or len(node.args) < 3:
            continue
        stored = _strip_value_casts(node.args[2])
        if stored is output_value:
            matches.append((node, None))
            continue
        if (
            not isinstance(stored, torch.fx.Node)
            or stored.op != "call_function"
            or stored.target not in multiply_targets
        ):
            continue
        operands = tuple(_strip_value_casts(value) for value in stored.args[:2])
        if operands[0] is output_value:
            scale_node = operands[1]
        elif operands[1] is output_value:
            scale_node = operands[0]
        else:
            continue
        if not _is_call(scale_node, _get_symnode):
            continue
        assert isinstance(scale_node, torch.fx.Node)
        value = scale_node.meta.get("val")
        if not isinstance(value, torch.SymFloat):
            continue
        scale = value._sympy_()
        if isinstance(scale, sympy.Expr):
            matches.append((node, scale))
    return matches[0] if len(matches) == 1 else None


def _is_bf16_roundtrip(node: object) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    convert = torch.ops.prims.convert_element_type.default
    if (
        not _is_call(node, convert)
        or len(node.args) < 2
        or node.args[1] is not torch.float32
    ):
        return False
    inner = node.args[0]
    return (
        isinstance(inner, torch.fx.Node)
        and _is_call(inner, convert)
        and len(inner.args) >= 2
        and inner.args[1] is torch.bfloat16
    )


def _binary_args(
    node: object, targets: tuple[object, ...]
) -> tuple[object, object] | None:
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target not in targets
        or len(node.args) != 2
    ):
        return None
    return node.args[0], node.args[1]


def _is_convert(node: object, source: object, dtype: torch.dtype) -> bool:
    return bool(
        isinstance(node, torch.fx.Node)
        and _is_call(node, torch.ops.prims.convert_element_type.default)
        and len(node.args) == 2
        and node.args == (source, dtype)
    )


def _is_full_slice(value: object) -> bool:
    return isinstance(value, slice) and value == slice(None)


def _subscript_source(
    node: object, indices: tuple[object, ...]
) -> torch.fx.Node | None:
    from ...language import view_ops

    if (
        not isinstance(node, torch.fx.Node)
        or not _is_call(node, view_ops.subscript)
        or len(node.args) != 2
        or not isinstance(node.args[0], torch.fx.Node)
        or not isinstance(node.args[1], (list, tuple))
        or len(node.args[1]) != len(indices)
    ):
        return None
    for actual, expected in zip(node.args[1], indices, strict=True):
        if expected == "full":
            if not _is_full_slice(actual):
                return None
        elif actual is not expected:
            return None
    return node.args[0]


def _static_int(value: object) -> int | None:
    if type(value) is int:
        return value
    if isinstance(value, torch.SymInt):
        try:
            return int(value)
        except TypeError:
            return None
    if isinstance(value, torch.fx.Node):
        return _static_int(value.meta.get("val"))
    return None


def _same_scalar_index(lhs: object, rhs: object) -> bool:
    if lhs is rhs:
        return True
    if not isinstance(lhs, torch.fx.Node) or not isinstance(rhs, torch.fx.Node):
        return False
    lhs_value = lhs.meta.get("val")
    rhs_value = rhs.meta.get("val")
    if isinstance(lhs_value, torch.SymInt) and isinstance(rhs_value, torch.SymInt):
        return lhs_value._sympy_() == rhs_value._sympy_()
    return bool(
        lhs.op == rhs.op == "call_function"
        and lhs.target is rhs.target
        and len(lhs.args) == len(rhs.args)
        and all(
            left is right or left == right
            for left, right in zip(lhs.args, rhs.args, strict=True)
        )
        and lhs.kwargs == rhs.kwargs
    )


def _same_tile_index(lhs: object, rhs: object) -> bool:
    from ...language.tile_ops import tile_index

    return bool(
        isinstance(lhs, torch.fx.Node)
        and isinstance(rhs, torch.fx.Node)
        and _is_call(lhs, tile_index)
        and _is_call(rhs, tile_index)
        and len(lhs.args) == len(rhs.args) == 1
        and lhs.args[0] is rhs.args[0]
    )


def _recurrence_target_counts() -> tuple[Counter[object], Counter[object]]:
    from ...language import memory_ops
    from ...language import view_ops
    from ...language._tracing_ops import _for_loop
    from ...language._tracing_ops import _get_symnode
    from ...language._tracing_ops import _host_tensor
    from ...language._tracing_ops import _new_var
    from ...language._tracing_ops import _phi
    from ...language.tile_ops import tile_id
    from ...language.tile_ops import tile_index

    inner: Counter[object] = Counter(
        {
            _new_var: 4,
            _get_symnode: 4,
            tile_id: 2,
            torch.ops.aten.add.Tensor: 10,
            tile_index: 4,
            torch.ops.aten.lt.Tensor: 1,
            torch.ops.aten.mul.Tensor: 6,
            operator.mul: 1,
            torch.ops.prims.iota.default: 2,
            torch.ops.aten.bitwise_xor.Scalar: 3,
            view_ops.subscript: 11,
            _host_tensor: 7,
            memory_ops.load: 6,
            torch.ops.aten.permute.default: 3,
            torch.ops.prims.convert_element_type.default: 10,
            dot: 4,
            torch.ops.aten.sym_size.int: 1,
            torch.ops.aten.sub.Tensor: 1,
            torch.ops.aten.__rshift__.Scalar: 1,
            torch.ops.aten.bitwise_and.Scalar: 1,
            torch.ops.aten.__lshift__.Scalar: 1,
            torch.ops.aten.bitwise_xor.Tensor: 1,
            torch.ops.aten.div.Tensor_mode: 1,
            memory_ops.store: 1,
        }
    )
    root: Counter[object] = Counter(
        {
            _get_symnode: 3,
            tile_id: 2,
            _host_tensor: 3,
            memory_ops.load: 4,
            torch.ops.prims.convert_element_type.default: 5,
            operator.add: 1,
            torch.ops.aten.sub.Tensor: 1,
            _for_loop: 1,
            operator.getitem: 1,
            _phi: 1,
            memory_ops.store: 1,
        }
    )
    return inner, root


def _validate_recurrence_semantics(
    *,
    root: object,
    loop: object,
    projection: torch.fx.Node,
    output_state: torch.fx.Node,
    output_residual: torch.fx.Node,
    update: torch.fx.Node,
    state_phi: torch.fx.Node,
    kd_load: torch.fx.Node,
    qd_load: torch.fx.Node,
    ak_load: torch.fx.Node,
    aq_load: torch.fx.Node,
    value_load: torch.fx.Node,
    decay_load: torch.fx.Node,
    output_store: torch.fx.Node,
    state: _TensorRef,
    cu_seqlens: _TensorRef,
    cu_chunks: _TensorRef,
    heads: int,
    total_chunks: int,
) -> bool:
    """Validate the complete carrier dataflow, not merely its op multiset."""

    from ...language._tracing_ops import _for_loop
    from ...language._tracing_ops import _get_symnode
    from ...language._tracing_ops import _new_var
    from ...language._tracing_ops import _phi
    from ...language.tile_ops import tile_id
    from ...language.tile_ops import tile_index

    inner_nodes = list(loop.graph.nodes)  # type: ignore[attr-defined]
    root_nodes = list(root.graph.nodes)  # type: ignore[attr-defined]
    expected_inner, expected_root = _recurrence_target_counts()
    if (
        Counter(node.target for node in inner_nodes if node.op == "call_function")
        != expected_inner
        or Counter(node.target for node in root_nodes if node.op == "call_function")
        != expected_root
    ):
        return False

    # The four contractions and every precision boundary are positional parts
    # of the ABI implemented by the external schedule.
    if any(
        len(node.args) != 4 or node.args[3] is not torch.float32
        for node in (
            projection,
            output_state,
            output_residual,
            update,
        )
    ):
        return False
    if projection.args[2] is not None or output_state.args[2] is not None:
        return False
    if output_residual.args[2] is not output_state or update.args[2] is not None:
        return False

    def bf16_transpose_of(node: object, source: torch.fx.Node) -> bool:
        if not isinstance(node, torch.fx.Node) or not _is_convert(
            node, node.args[0] if node.args else None, torch.bfloat16
        ):
            return False
        transpose = node.args[0]
        return bool(
            isinstance(transpose, torch.fx.Node)
            and _is_call(transpose, torch.ops.aten.permute.default)
            and transpose.args == (source, [1, 0])
        )

    if not bf16_transpose_of(projection.args[1], state_phi) or not bf16_transpose_of(
        output_state.args[1], state_phi
    ):
        return False

    residual_to_bf16 = output_residual.args[1]
    if not isinstance(residual_to_bf16, torch.fx.Node) or not _is_convert(
        residual_to_bf16,
        residual_to_bf16.args[0] if residual_to_bf16.args else None,
        torch.bfloat16,
    ):
        return False
    residual = residual_to_bf16.args[0]
    if not isinstance(residual, torch.fx.Node) or not _is_convert(
        residual, residual.args[0] if residual.args else None, torch.float32
    ):
        return False
    rounded_residual = residual.args[0]
    if not isinstance(rounded_residual, torch.fx.Node) or not _is_convert(
        rounded_residual,
        rounded_residual.args[0] if rounded_residual.args else None,
        torch.bfloat16,
    ):
        return False
    residual_sub = rounded_residual.args[0]
    residual_args = _binary_args(residual_sub, (torch.ops.aten.sub.Tensor,))
    if residual_args is None or residual_args[1] is not projection:
        return False
    value_f32 = residual_args[0]
    if not _is_convert(value_f32, value_load, torch.float32):
        return False
    if not bf16_transpose_of(update.args[0], residual) or update.args[1] is not ak_load:
        return False
    if projection.args[0] is not kd_load or output_state.args[0] is not qd_load:
        return False
    if output_residual.args[0] is not aq_load:
        return False

    carried = next(iter(loop.graph.find_nodes(op="output"))).args[0]  # type: ignore[attr-defined]
    if not isinstance(carried, (list, tuple)) or len(carried) != 1:
        return False
    next_state = carried[0]
    if not isinstance(next_state, torch.fx.Node) or not _is_convert(
        next_state, next_state.args[0] if next_state.args else None, torch.float32
    ):
        return False
    next_state_bf16 = next_state.args[0]
    if not isinstance(next_state_bf16, torch.fx.Node) or not _is_convert(
        next_state_bf16,
        next_state_bf16.args[0] if next_state_bf16.args else None,
        torch.bfloat16,
    ):
        return False
    state_add = next_state_bf16.args[0]
    state_add_args = _binary_args(state_add, (torch.ops.aten.add.Tensor,))
    if state_add_args is None or state_add_args[1] is not update:
        return False
    state_decay = state_add_args[0]
    state_decay_args = _binary_args(state_decay, (torch.ops.aten.mul.Tensor,))
    if state_decay_args is None or state_decay_args[0] is not state_phi:
        return False
    decay_broadcast = state_decay_args[1]
    if _subscript_source(decay_broadcast, (None, "full")) is not decay_load:
        return False

    # Recover the root loop values, then pin the loop extent and carry order.
    root_loads = _tensor_refs(root_nodes)
    state_loads = [node for node, ref in root_loads if _same_ref(ref, state)]
    seq_loads = [node for node, ref in root_loads if _same_ref(ref, cu_seqlens)]
    chunk_loads = [node for node, ref in root_loads if _same_ref(ref, cu_chunks)]
    root_stores = [node for node in root_nodes if _store_ref(node) is not None]
    loop_calls = [node for node in root_nodes if _is_call(node, _for_loop)]
    if not (
        len(state_loads) == 1
        and len(seq_loads) == 2
        and len(chunk_loads) == 1
        and len(root_stores) == 1
        and len(loop_calls) == 1
    ):
        return False
    state_load = state_loads[0]
    chunk_load = chunk_loads[0]
    loop_call = loop_calls[0]
    state_indices = _memory_indices(state_load)
    state_store_indices = _memory_indices(root_stores[0])
    if (
        state_indices is None
        or state_store_indices is None
        or len(state_indices) != 4
        or len(state_store_indices) != 4
        or any(
            (not _is_full_slice(lhs) if _is_full_slice(rhs) else lhs is not rhs)
            for lhs, rhs in zip(state_indices, state_store_indices, strict=True)
        )
        or not _is_full_slice(state_indices[3])
        or any(
            len(node.args) != 4 or node.args[2:] != (None, None)
            for node in (
                state_load,
                *seq_loads,
                chunk_load,
            )
        )
    ):
        return False
    sequence_index = state_indices[0]
    head_index = state_indices[1]
    value_index = state_indices[2]
    if (
        not _is_call(sequence_index, tile_id)
        or not _is_call(head_index, tile_id)
        or not _is_call(value_index, _get_symnode)
        or _same_scalar_index(sequence_index, head_index)
    ):
        return False

    begin_load = next(
        (node for node in seq_loads if _memory_indices(node) == (sequence_index,)),
        None,
    )
    end_load = next(
        (
            node
            for node in seq_loads
            if (
                (indices := _memory_indices(node)) is not None
                and len(indices) == 1
                and _binary_args(indices[0], (operator.add,)) == (sequence_index, 1)
            )
        ),
        None,
    )
    if (
        begin_load is None
        or end_load is None
        or _memory_indices(chunk_load) != (sequence_index,)
    ):
        return False
    begin = next(
        (
            node
            for node in begin_load.users
            if _is_convert(node, begin_load, torch.int64)
        ),
        None,
    )
    end = next(
        (node for node in end_load.users if _is_convert(node, end_load, torch.int64)),
        None,
    )
    chunk_begin = next(
        (
            node
            for node in chunk_load.users
            if _is_convert(node, chunk_load, torch.int64)
        ),
        None,
    )
    initial_state = next(
        (
            node
            for node in state_load.users
            if _is_convert(node, state_load, torch.float32)
        ),
        None,
    )
    if any(value is None for value in (begin, end, chunk_begin, initial_state)):
        return False
    extent = loop_call.args[2] if len(loop_call.args) > 2 else None
    initial = loop_call.args[3] if len(loop_call.args) > 3 else None
    if (
        loop_call.args[0] != loop.graph_id  # type: ignore[attr-defined]
        or loop_call.args[1] != [0]
        or not isinstance(extent, (list, tuple))
        or len(extent) != 1
        or _binary_args(extent[0], (torch.ops.aten.sub.Tensor,)) != (end, begin)
        or initial != [end, begin, chunk_begin, initial_state]
    ):
        return False
    placeholders = [node for node in inner_nodes if node.op == "placeholder"]
    new_vars = [node for node in inner_nodes if _is_call(node, _new_var)]
    if (
        len(placeholders) != 4
        or len(new_vars) != 4
        or any(
            node.args != (placeholder,)
            for node, placeholder in zip(new_vars, placeholders, strict=True)
        )
        or state_phi is not new_vars[3]
    ):
        return False
    end_phi, begin_phi, chunk_phi, _ = new_vars

    value_size_nodes = [
        node for node in inner_nodes if _is_call(node, torch.ops.aten.sym_size.int)
    ]
    if (
        len(value_size_nodes) != 1
        or value_size_nodes[0].args != (placeholders[3], 0)
        or not _same_scalar_index(value_index, value_size_nodes[0])
    ):
        return False
    value_size = value_size_nodes[0]

    tile_indices = [node for node in inner_nodes if _is_call(node, tile_index)]
    tile_ids = [node for node in inner_nodes if _is_call(node, tile_id)]
    iotas = [
        node for node in inner_nodes if _is_call(node, torch.ops.prims.iota.default)
    ]
    if len(tile_indices) != 4 or len(tile_ids) != 2 or len(iotas) != 2:
        return False
    token_tile_id = tile_ids[0]
    inner_head = tile_ids[1]
    token_lane = tile_indices[0]
    if (
        not _same_scalar_index(inner_head, head_index)
        or len(token_tile_id.args) != 1
        or len(token_lane.args) != 1
        or token_tile_id.args[0] is not token_lane.args[0]
    ):
        return False
    feature = next((node for node in iotas if _static_int(node.args[0]) == _DK), None)
    logical_col_iota = next(
        (node for node in iotas if _static_int(node.args[0]) == _BT), None
    )
    if feature is None or logical_col_iota is None:
        return False
    global_chunk = next(
        (
            node
            for node in inner_nodes
            if _binary_args(node, (torch.ops.aten.add.Tensor,))
            == (chunk_phi, token_tile_id)
        ),
        None,
    )
    token = next(
        (
            node
            for node in inner_nodes
            if _binary_args(node, (torch.ops.aten.add.Tensor,))
            == (begin_phi, token_lane)
        ),
        None,
    )
    if global_chunk is None or token is None:
        return False
    valid = next(
        (
            node
            for node in inner_nodes
            if _binary_args(node, (torch.ops.aten.lt.Tensor,)) == (token, end_phi)
        ),
        None,
    )
    row_mul = next(
        (
            node
            for node in inner_nodes
            if _binary_args(node, (torch.ops.aten.mul.Tensor,)) == (token, heads)
        ),
        None,
    )
    row = next(
        (
            node
            for node in inner_nodes
            if row_mul is not None
            and _binary_args(node, (torch.ops.aten.add.Tensor,))
            == (row_mul, inner_head)
        ),
        None,
    )
    head_chunks = next(
        (
            node
            for node in inner_nodes
            if _binary_args(node, (operator.mul,)) == (inner_head, total_chunks)
        ),
        None,
    )
    chunk_head = next(
        (
            node
            for node in inner_nodes
            if head_chunks is not None
            and _binary_args(node, (torch.ops.aten.add.Tensor,))
            == (global_chunk, head_chunks)
        ),
        None,
    )
    factor_base = next(
        (
            node
            for node in inner_nodes
            if chunk_head is not None
            and (args := _binary_args(node, (torch.ops.aten.mul.Tensor,))) is not None
            and args[0] is chunk_head
            and _static_int(args[1]) == _BT
        ),
        None,
    )
    factor_row = next(
        (
            node
            for node in inner_nodes
            if factor_base is not None
            and (args := _binary_args(node, (torch.ops.aten.add.Tensor,))) is not None
            and args[0] is factor_base
            and _same_tile_index(args[1], token_lane)
        ),
        None,
    )
    if any(
        value is None for value in (valid, row, chunk_head, factor_base, factor_row)
    ):
        return False

    kd_indices = _memory_indices(kd_load)
    qd_indices = _memory_indices(qd_load)
    ak_indices = _memory_indices(ak_load)
    aq_indices = _memory_indices(aq_load)
    decay_indices = _memory_indices(decay_load)
    value_indices = _memory_indices(value_load)
    if any(
        indices is None
        for indices in (
            kd_indices,
            qd_indices,
            ak_indices,
            aq_indices,
            decay_indices,
            value_indices,
        )
    ):
        return False
    assert kd_indices is not None
    assert qd_indices is not None
    assert ak_indices is not None
    assert aq_indices is not None
    assert decay_indices is not None
    assert value_indices is not None
    if (
        not all(
            len(indices) == 2
            for indices in (
                kd_indices,
                qd_indices,
                ak_indices,
                aq_indices,
                value_indices,
            )
        )
        or len(decay_indices) != 2
    ):
        return False
    kd_row = _subscript_source(kd_indices[0], ("full", None))
    qd_row = _subscript_source(qd_indices[0], ("full", None))
    kd_feature = _subscript_source(kd_indices[1], (None, "full"))
    qd_feature = _subscript_source(qd_indices[1], (None, "full"))
    physical_feature_args = _binary_args(
        kd_feature, (torch.ops.aten.bitwise_xor.Scalar,)
    )
    if not (
        kd_row is factor_row
        and qd_row is factor_row
        and kd_feature is qd_feature
        and physical_feature_args == (feature, 8)
    ):
        return False
    if not all(
        len(node.args) == 4 and node.args[2:] == (None, None)
        for node in (kd_load, qd_load, aq_load, ak_load, decay_load)
    ):
        return False
    value_mask = _subscript_source(value_load.args[2], ("full", None))
    if (
        value_indices[0] is not row
        or value_indices[1] is not value_size
        or value_mask is not valid
        or len(value_load.args) != 4
        or value_load.args[3] is not None
    ):
        return False

    ak_row = _subscript_source(ak_indices[0], ("full", None))
    ak_feature = _subscript_source(ak_indices[1], (None, "full"))
    ak_row_args = _binary_args(ak_row, (torch.ops.aten.add.Tensor,))
    if (
        ak_row_args is None
        or ak_row_args[0] is not factor_base
        or ak_feature is not feature
    ):
        return False
    ak_lane_args = _binary_args(ak_row_args[1], (torch.ops.aten.bitwise_xor.Scalar,))
    if (
        ak_lane_args is None
        or not _same_tile_index(ak_lane_args[0], token_lane)
        or ak_lane_args[1] != 8
    ):
        return False
    if (
        not _same_scalar_index(aq_indices[0], chunk_head)
        or not _same_scalar_index(decay_indices[0], chunk_head)
        or not _is_full_slice(decay_indices[1])
    ):
        return False

    pair_index = aq_indices[1]
    pair_args = _binary_args(pair_index, (torch.ops.aten.div.Tensor_mode,))
    if (
        pair_args is None
        or pair_args[1] != 2
        or not isinstance(pair_index, torch.fx.Node)
        or pair_index.kwargs != {"rounding_mode": "floor"}
    ):
        return False
    pair_xor_args = _binary_args(pair_args[0], (torch.ops.aten.bitwise_xor.Tensor,))
    if pair_xor_args is None:
        return False
    byte_offset, shifted = pair_xor_args
    byte_args = _binary_args(byte_offset, (torch.ops.aten.mul.Tensor,))
    shifted_args = _binary_args(shifted, (torch.ops.aten.__lshift__.Scalar,))
    if (
        byte_args is None
        or byte_args[1] != 2
        or shifted_args is None
        or shifted_args[1] != 4
    ):
        return False
    offset_args = _binary_args(byte_args[0], (torch.ops.aten.add.Tensor,))
    masked_args = _binary_args(shifted_args[0], (torch.ops.aten.bitwise_and.Scalar,))
    if offset_args is None or masked_args is None or masked_args[1] != 1:
        return False
    row_offset_args = _binary_args(offset_args[0], (torch.ops.aten.mul.Tensor,))
    storage_col_args = _binary_args(
        offset_args[1], (torch.ops.aten.bitwise_xor.Scalar,)
    )
    shifted_byte_args = _binary_args(
        masked_args[0], (torch.ops.aten.__rshift__.Scalar,)
    )
    logical_row = (
        _subscript_source(row_offset_args[0], ("full", None))
        if row_offset_args is not None
        else None
    )
    logical_col = (
        _subscript_source(storage_col_args[0], (None, "full"))
        if storage_col_args is not None
        else None
    )
    if not (
        row_offset_args is not None
        and _static_int(row_offset_args[1]) == _BT
        and storage_col_args is not None
        and storage_col_args[1] == 8
        and shifted_byte_args == (byte_offset, 7)
        and logical_row is not None
        and _same_tile_index(logical_row, token_lane)
        and logical_col is logical_col_iota
    ):
        return False

    stored = output_store.args[2]
    if (
        len(output_store.args) != 4
        or not isinstance(stored, torch.fx.Node)
        or not _is_convert(
            stored, stored.args[0] if stored.args else None, torch.bfloat16
        )
    ):
        return False
    scaled_output = stored.args[0]
    scale_args = _binary_args(scaled_output, (torch.ops.aten.mul.Tensor,))
    output_indices = _memory_indices(output_store)
    output_mask = _subscript_source(output_store.args[3], ("full", None))
    if (
        scale_args is None
        or scale_args[0] is not output_residual
        or not _is_call(scale_args[1], _get_symnode)
        or output_indices != value_indices
        or output_mask is not valid
    ):
        return False

    # Pin root writeback to the loop-carried value and preserve operand order.
    getitems = [node for node in root_nodes if _is_call(node, operator.getitem)]
    phis = [node for node in root_nodes if _is_call(node, _phi)]
    if len(getitems) != 1 or len(phis) != 1:
        return False
    if getitems[0].args != (loop_call, 0) or phis[0].args != (
        initial_state,
        getitems[0],
    ):
        return False
    stored_state = root_stores[0].args[2]
    return bool(
        _is_convert(stored_state, phis[0], torch.bfloat16)
        and len(root_stores[0].args) == 4
        and root_stores[0].args[3] is None
    )


def _match_chunk_recurrence_graphs(
    graphs: Sequence[GraphInfo],
) -> _CuteChunkRecurrenceMatch | None:
    """Recognize only the supported semantic and storage contract."""

    from ...language._tracing_ops import _for_loop
    from ...language._tracing_ops import _new_var
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import RootGraphInfo

    roots = [graph for graph in graphs if isinstance(graph, RootGraphInfo)]
    loops = [graph for graph in graphs if isinstance(graph, ForLoopGraphInfo)]
    if len(graphs) != 2 or len(roots) != 1 or len(loops) != 1:
        return None
    root, loop = roots[0], loops[0]
    if len(loop.block_ids) != 1:
        return None

    inner = list(loop.graph.nodes)
    dots = [node for node in inner if _is_call(node, dot)]
    if len(dots) != 4:
        return None
    projection, output_state, output_residual, update = dots
    if any(
        not isinstance(node.meta.get("val"), torch.Tensor)
        or cast("torch.Tensor", node.meta["val"]).dtype is not torch.float32
        for node in dots
    ):
        return None

    projection_lhs = _single_ref_load(projection.args[0])
    output_state_lhs = _single_ref_load(output_state.args[0])
    output_residual_lhs = _single_ref_load(output_residual.args[0])
    update_rhs = _single_ref_load(update.args[1])
    if any(
        item is None
        for item in (
            projection_lhs,
            output_state_lhs,
            output_residual_lhs,
            update_rhs,
        )
    ):
        return None
    assert projection_lhs is not None
    assert output_state_lhs is not None
    assert output_residual_lhs is not None
    assert update_rhs is not None
    kd_load, kd = projection_lhs
    qd_load, qd = output_state_lhs
    aq_load, aq = output_residual_lhs
    ak_load, ak = update_rhs

    projection_state = _strip_value_casts(projection.args[1])
    state_phi = (
        _strip_value_casts(projection_state.args[0])
        if isinstance(projection_state, torch.fx.Node) and projection_state.args
        else None
    )
    if (
        not _is_call(state_phi, _new_var)
        or not _is_matrix_transpose_of(
            projection.args[1], cast("torch.fx.Node", state_phi)
        )
        or not _is_matrix_transpose_of(
            output_state.args[1], cast("torch.fx.Node", state_phi)
        )
    ):
        return None
    assert isinstance(state_phi, torch.fx.Node)

    # Both residual consumers must see the same explicit BF16-rounded value.
    residual_operand = output_residual.args[1]
    residual = (
        residual_operand.args[0]
        if isinstance(residual_operand, torch.fx.Node)
        and _is_call(residual_operand, torch.ops.prims.convert_element_type.default)
        and len(residual_operand.args) >= 2
        and residual_operand.args[1] is torch.bfloat16
        else None
    )
    if (
        len(output_residual.args) < 3
        or output_residual.args[2] is not output_state
        or not isinstance(residual, torch.fx.Node)
        or not _is_bf16_roundtrip(residual)
        or not _is_matrix_transpose_of(update.args[0], residual)
    ):
        return None
    residual_source = cast("torch.fx.Node", residual.args[0]).args[0]
    if not isinstance(residual_source, torch.fx.Node):
        return None
    # The source contains exactly V plus GTotal after removing the four factors;
    # choose V by ancestry of the rounded residual rather than by source name.
    ancestors = _ancestors(residual_source)
    if projection not in ancestors or not _contains_target(
        residual_source, {operator.sub, torch.ops.aten.sub.Tensor}
    ):
        return None
    candidate_value_loads = [
        node
        for node in ancestors
        if _load_ref(node) is not None and node is not kd_load
    ]
    if len(candidate_value_loads) != 1:
        return None
    value_load = candidate_value_loads[0]
    values = _load_ref(value_load)
    if values is None:
        return None

    output_contract = _output_store_contract(inner, output_residual)
    if output_contract is None:
        return None
    output_store, output_scale = output_contract
    output = _store_ref(output_store)
    if output_store is None or output is None:
        return None

    # The only remaining inner load is GTotal.  Its value must feed the state
    # update together with the fourth dot, and that result must round via BF16.
    all_loads = _tensor_refs(inner)
    factor_and_value_nodes = {kd_load, qd_load, aq_load, ak_load, value_load}
    remaining = [
        (node, ref) for node, ref in all_loads if node not in factor_and_value_nodes
    ]
    if len(remaining) != 1:
        return None
    decay_load, g_total = remaining[0]
    loop_outputs = list(loop.graph.find_nodes(op="output"))
    if len(loop_outputs) != 1 or len(loop_outputs[0].args) != 1:
        return None
    carried = loop_outputs[0].args[0]
    if (
        not isinstance(carried, (list, tuple))
        or len(carried) != 1
        or not _is_bf16_roundtrip(carried[0])
        or not _contains_target(carried[0], {operator.add, torch.ops.aten.add.Tensor})
    ):
        return None
    carried_ancestors = _ancestors(carried[0])
    if not {update, decay_load, state_phi}.issubset(carried_ancestors):
        return None

    inner_load_nodes = {node for node, _ref in all_loads}
    inner_stores = {node for node in inner if _store_ref(node) is not None}
    if inner_load_nodes != {
        kd_load,
        qd_load,
        aq_load,
        ak_load,
        value_load,
        decay_load,
    } or inner_stores != {output_store}:
        return None
    if any(
        len(node.args) >= 3 and node.args[2] is not None
        for node in (kd_load, qd_load, aq_load, ak_load, decay_load)
    ):
        return None
    ak_indices = _memory_indices(ak_load)
    aq_indices = _memory_indices(aq_load)
    if not ak_indices or not _contains_target(
        ak_indices[0], {torch.ops.aten.bitwise_xor.Scalar}
    ):
        return None
    if (
        not aq_indices
        or len(aq_indices) < 2
        or not _contains_target(
            aq_indices[1],
            {
                torch.ops.aten.bitwise_xor.Scalar,
                torch.ops.aten.bitwise_xor.Tensor,
            },
        )
    ):
        return None

    root_nodes = list(root.graph.nodes)
    loop_calls = [node for node in root_nodes if _is_call(node, _for_loop)]
    root_stores = [node for node in root_nodes if _store_ref(node) is not None]
    root_loads = _tensor_refs(root_nodes)
    if len(loop_calls) != 1 or len(root_stores) != 1 or len(root_loads) != 4:
        return None
    state = _store_ref(root_stores[0])
    state_loads = [(node, ref) for node, ref in root_loads if _same_ref(ref, state)]
    if state is None or len(state_loads) != 1:
        return None
    metadata = [(node, ref) for node, ref in root_loads if not _same_ref(ref, state)]
    by_ref: dict[tuple[str, int], list[tuple[torch.fx.Node, _TensorRef]]] = {}
    for node, ref in metadata:
        by_ref.setdefault((ref.name, id(ref.fake)), []).append((node, ref))
    groups = sorted(by_ref.values(), key=len, reverse=True)
    if len(groups) != 2 or tuple(map(len, groups)) != (2, 1):
        return None
    cu_seqlens = groups[0][0][1]
    cu_chunks = groups[1][0][1]

    refs = (kd, qd, ak, aq, g_total, values, output, state, cu_seqlens, cu_chunks)
    if any(ref.fake.device.type != "cuda" for ref in refs):
        return None
    if any(not ref.fake.is_contiguous() for ref in refs):
        return None
    if any(
        ref.fake.dtype is not torch.bfloat16
        for ref in (kd, qd, ak, aq, values, output, state)
    ):
        return None
    if g_total.fake.dtype is not torch.float32:
        return None
    if (
        cu_seqlens.fake.dtype is not torch.int32
        or cu_chunks.fake.dtype is not torch.int32
    ):
        return None

    if state.fake.ndim != 4:
        return None
    sequences, heads, value_size, key_size = map(int, state.fake.shape)
    if key_size != _DK or value_size != _DV or sequences <= 0 or heads <= 0:
        return None
    if values.fake.ndim != 2 or output.fake.ndim != 2:
        return None
    if tuple(values.fake.shape) != tuple(output.fake.shape):
        return None
    total_tokens_times_heads, activation_width = map(int, values.fake.shape)
    if activation_width != _DV or total_tokens_times_heads % heads:
        return None
    total_tokens = total_tokens_times_heads // heads
    if aq.fake.ndim != 2 or aq.fake.shape[1] != _BT * _BT:
        return None
    total_chunk_heads = int(aq.fake.shape[0])
    if total_chunk_heads % heads:
        return None
    total_chunks = total_chunk_heads // heads
    rows = heads * total_chunks * _BT
    if any(tuple(ref.fake.shape) != (rows, _DK) for ref in (kd, qd, ak)):
        return None
    if tuple(g_total.fake.shape) != (heads * total_chunks, _DK):
        return None
    if tuple(cu_seqlens.fake.shape) != (sequences + 1,) or tuple(
        cu_chunks.fake.shape
    ) != (sequences + 1,):
        return None
    if not _validate_recurrence_semantics(
        root=root,
        loop=loop,
        projection=projection,
        output_state=output_state,
        output_residual=output_residual,
        update=update,
        state_phi=state_phi,
        kd_load=kd_load,
        qd_load=qd_load,
        ak_load=ak_load,
        aq_load=aq_load,
        value_load=value_load,
        decay_load=decay_load,
        output_store=output_store,
        state=state,
        cu_seqlens=cu_seqlens,
        cu_chunks=cu_chunks,
        heads=heads,
        total_chunks=total_chunks,
    ):
        return None
    state_indices = _memory_indices(state_loads[0][0])
    if (
        state_indices is None
        or len(state_indices) != 4
        or not all(isinstance(index, torch.fx.Node) for index in state_indices[:3])
    ):
        return None
    return _CuteChunkRecurrenceMatch(
        root_graph_id=root.graph_id,
        root_phase_index=root.phase_index,
        heads=heads,
        sequences=sequences,
        total_tokens=total_tokens,
        total_chunks=total_chunks,
        kd=kd,
        qd=qd,
        ak=ak,
        aq=aq,
        g_total=g_total,
        values=values,
        output=output,
        state=state,
        cu_seqlens=cu_seqlens,
        cu_chunks=cu_chunks,
        sequence_coord=cast("torch.fx.Node", state_indices[0]),
        head_coord=cast("torch.fx.Node", state_indices[1]),
        value_coord=cast("torch.fx.Node", state_indices[2]),
        output_scale=output_scale,
    )


def _plan_chunk_recurrence(
    graphs: Sequence[GraphInfo],
    _tile_strategy: TileStrategyDispatch,
) -> CuteChunkRecurrencePlan | None:
    from ..device_function import DeviceFunction
    from ..host_function import HostFunction

    if DeviceFunction.current().config.pid_type != "flat":
        return None
    match = _match_chunk_recurrence_graphs(graphs)
    if (
        match is None
        or match.output_scale is None
        or min(match.total_tokens, match.total_chunks, match.sequences, match.heads)
        <= 0
    ):
        return None
    env = CompileEnvironment.current()
    if not _packed_workspace_is_proven(env):
        return None
    target = env.config_spec.target_device_capability
    if target is None or target[0] != 10:
        return None

    # The exact body owns a DV64 CTA.  Confirm the carrier has exactly the three
    # semantic outer axes and that only the value axis is split.
    device_ir = HostFunction.current().device_ir
    grid_ids = _canonical_root_axis_ids(
        device_ir,
        root_phase_index=match.root_phase_index,
        coordinates=(
            match.sequence_coord,
            match.head_coord,
            match.value_coord,
        ),
        expected_extents=(match.sequences, match.heads, _DV),
    )
    if grid_ids is None or [
        DeviceFunction.current().resolved_block_size(block_id) for block_id in grid_ids
    ] != [1, 1, 64]:
        return None

    preferred_dv_partitions = _select_sm100_dv_partitions(
        total_chunks=match.total_chunks,
        sequences=match.sequences,
        heads=match.heads,
        num_sm=env.config_spec.num_sm,
    )
    configured_dv_partitions = DeviceFunction.current().config.get(
        CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY,
        preferred_dv_partitions,
    )
    if type(configured_dv_partitions) is not int or configured_dv_partitions not in (
        2,
        4,
    ):
        return None
    dv_partitions = configured_dv_partitions
    use_sm100_warp_dv4 = dv_partitions == 4
    schedule = "sm100_warp_dv4" if use_sm100_warp_dv4 else "sm100_tmem"
    refs = (
        match.kd,
        match.qd,
        match.ak,
        match.aq,
        match.g_total,
        match.values,
        match.output,
        match.state,
        match.cu_seqlens,
        match.cu_chunks,
    )
    grid = (
        (match.sequences * 4, match.heads, 1)
        if use_sm100_warp_dv4
        else (match.sequences, match.heads * 2, 1)
    )
    if not _linear_offsets_fit_i32(
        match.sequences * match.heads * dv_partitions,
        tuple(ref.fake.numel() for ref in refs),
    ) or not _xyz_grid_fits(grid):
        return None
    return CuteChunkRecurrencePlan(
        root_graph_id=match.root_graph_id,
        heads=match.heads,
        sequences=match.sequences,
        total_tokens=match.total_tokens,
        total_chunks=match.total_chunks,
        kd=match.kd,
        qd=match.qd,
        ak=match.ak,
        aq=match.aq,
        g_total=match.g_total,
        values=match.values,
        output=match.output,
        state=match.state,
        cu_seqlens=match.cu_seqlens,
        cu_chunks=match.cu_chunks,
        output_scale=match.output_scale,
        workspace_layout_version=2,
        schedule=schedule,
        threads=(
            _SM100_WARP_DV4_THREADS if use_sm100_warp_dv4 else _SM100_TMEM_THREADS
        ),
        smem_bytes=(
            _SM100_WARP_DV4_SMEM_BYTES if use_sm100_warp_dv4 else _SM100_TMEM_SMEM_BYTES
        ),
        device_abi=(
            _SM100_WARP_DV4_DEVICE_ABI if use_sm100_warp_dv4 else _SM100_TMEM_DEVICE_ABI
        ),
        input_stages=(_SM100_WARP_DV4_INPUT_STAGES if use_sm100_warp_dv4 else 8),
        tma_stages=6,
        factor_tma_value_splits=(1 if use_sm100_warp_dv4 else 2),
        output_acc_stages=2,
        output_smem_stages=(3 if use_sm100_warp_dv4 else 7),
        output_store_wait_groups=(0 if use_sm100_warp_dv4 else 6),
        tmem_cols=(0 if use_sm100_warp_dv4 else 512),
        dv_partitions=dv_partitions,
    )


def plan_chunk_recurrence(
    graphs: Sequence[GraphInfo], tile_strategy: TileStrategyDispatch
) -> None:
    from ..device_function import DeviceFunction

    DeviceFunction.current().cute_state.chunk_recurrence_plan = _plan_chunk_recurrence(
        graphs, tile_strategy
    )


def _tensor_arg(cg: GenerateAST, ref: _TensorRef) -> str:
    return cg.device_function.tensor_arg(ref.fake, prefer_name=ref.name).name


def _emit_module_imports(cg: GenerateAST, schedule: str) -> None:
    if getattr(cg, "_helion_chunk_recurrence_module_emitted", False):
        return
    cg._helion_chunk_recurrence_module_emitted = True  # type: ignore[attr-defined]
    if schedule == "sm100_tmem":
        helper_name = cg.device_function.new_var("_helion_sm100_chain_host")
        source = (
            "from helion._compiler.cute.chunk_recurrence_sm100 "
            f"import host_chain_dv2 as {helper_name}"
        )
    elif schedule == "sm100_warp_dv4":
        helper_name = cg.device_function.new_var("_helion_sm100_warp_dv4_host")
        source = (
            "from helion._compiler.cute.chunk_recurrence_dv4_sm100 "
            f"import _recurrence_entry as {helper_name}"
        )
    else:
        raise ValueError(f"unsupported chunk-recurrence schedule {schedule!r}")
    cg.module_statements.extend(ast.parse(source).body)


def codegen_chunk_recurrence(cg: GenerateAST) -> bool:
    """Replace the matched root with the vendored exact device body."""

    df = cg.device_function
    plan = df.cute_state.chunk_recurrence_plan
    root = cg.current_root_graph_info
    if plan is None or root is None or root.graph_id != plan.root_graph_id:
        return False

    refs = (
        plan.kd,
        plan.qd,
        plan.ak,
        plan.aq,
        plan.g_total,
        plan.values,
        plan.output,
        plan.state,
        plan.cu_seqlens,
        plan.cu_chunks,
    )
    names = tuple(_tensor_arg(cg, ref) for ref in refs)
    kd, qd, ak, aq, gt, values, output, state, cu_seqlens, cu_chunks = names
    _emit_module_imports(cg, plan.schedule)
    if plan.schedule == "sm100_tmem":
        output_scale = df.literal_expr(plan.output_scale)
        if not output_scale.isidentifier():
            return False
        if (
            plan.threads != _SM100_TMEM_THREADS
            or plan.device_abi != _SM100_TMEM_DEVICE_ABI
            or plan.input_stages != 8
            or plan.tma_stages != 6
            or plan.factor_tma_value_splits != 2
            or plan.output_acc_stages != 2
            or plan.output_smem_stages != 7
            or plan.output_store_wait_groups != 6
            or plan.tmem_cols != 512
        ):
            return False
        cg.cute_wrapper_plans.append(
            {
                "kind": "chunk_recurrence_sm100",
                "kd_name": kd,
                "qd_name": qd,
                "ak_name": ak,
                "aq_name": aq,
                "gt_name": gt,
                "v_name": values,
                "out_name": output,
                "state_name": state,
                "cu_seqlens_name": cu_seqlens,
                "cu_chunks_name": cu_chunks,
                "heads": plan.heads,
                "sequences": plan.sequences,
                "total_tokens": plan.total_tokens,
                "total_chunks": plan.total_chunks,
                "chunk_size": _BT,
                "key_size": _DK,
                "value_size": _DV,
                "threads": plan.threads,
                "smem_bytes": plan.smem_bytes,
                "workspace_layout_version": plan.workspace_layout_version,
                "factor_key_xor": 8,
                "outputs_scaled": False,
                "scale_name": output_scale,
                "device_abi": plan.device_abi,
                "input_stages": plan.input_stages,
                "tma_stages": plan.tma_stages,
                "factor_tma_value_splits": plan.factor_tma_value_splits,
                "output_acc_stages": plan.output_acc_stages,
                "output_smem_stages": plan.output_smem_stages,
                "output_store_wait_groups": plan.output_store_wait_groups,
                "tmem_cols": plan.tmem_cols,
                "dv_partitions": plan.dv_partitions,
            }
        )
        df.placeholder_args.update((*names, output_scale))
        df.preamble = []
        df.body = [ast.Pass()]
        cg.cute_uses_matmul = True
        return True

    if plan.schedule == "sm100_warp_dv4":
        output_scale = df.literal_expr(plan.output_scale)
        if not output_scale.isidentifier():
            return False
        if (
            plan.threads != _SM100_WARP_DV4_THREADS
            or plan.smem_bytes != _SM100_WARP_DV4_SMEM_BYTES
            or plan.device_abi != _SM100_WARP_DV4_DEVICE_ABI
            or plan.input_stages != _SM100_WARP_DV4_INPUT_STAGES
            or plan.tma_stages != 6
            or plan.factor_tma_value_splits != 1
            or plan.output_acc_stages != 2
            or plan.output_smem_stages != 3
            or plan.output_store_wait_groups != 0
            or plan.tmem_cols != 0
            or plan.dv_partitions != 4
        ):
            return False
        desc_names = tuple(
            df.new_var(name)
            for name in (
                "_chunk_recurrence_desc_factor",
                "_chunk_recurrence_desc_aq",
                "_chunk_recurrence_desc_gt",
                "_chunk_recurrence_desc_v",
                "_chunk_recurrence_desc_out",
                "_chunk_recurrence_desc_state_in",
                "_chunk_recurrence_desc_state_out",
            )
        )
        cg.cute_wrapper_plans.append(
            {
                "kind": "chunk_recurrence_warp_dv4",
                "kd_name": kd,
                "qd_name": qd,
                "ak_name": ak,
                "aq_name": aq,
                "gt_name": gt,
                "v_name": values,
                "out_name": output,
                "state_name": state,
                "cu_seqlens_name": cu_seqlens,
                "cu_chunks_name": cu_chunks,
                "desc_args": desc_names,
                "heads": plan.heads,
                "sequences": plan.sequences,
                "total_tokens": plan.total_tokens,
                "total_chunks": plan.total_chunks,
                "chunk_size": _BT,
                "key_size": _DK,
                "value_size": _DV,
                "threads": plan.threads,
                "smem_bytes": plan.smem_bytes,
                "workspace_layout_version": plan.workspace_layout_version,
                "factor_key_xor": 8,
                "outputs_scaled": False,
                "scale_name": output_scale,
                "device_abi": plan.device_abi,
                "input_stages": plan.input_stages,
                "tma_stages": plan.tma_stages,
                "factor_tma_value_splits": plan.factor_tma_value_splits,
                "output_acc_stages": plan.output_acc_stages,
                "output_smem_stages": plan.output_smem_stages,
                "output_store_wait_groups": plan.output_store_wait_groups,
                "tmem_cols": plan.tmem_cols,
                "dv_partitions": plan.dv_partitions,
            }
        )
        df.wrapper_only_params.extend(desc_names)
        df.placeholder_args.update((*names, output_scale))
        df.preamble = []
        df.body = [ast.Pass()]
        cg.cute_uses_matmul = True
        return True

    return False


__all__ = [
    "CuteChunkRecurrencePlan",
    "_select_sm100_dv_partitions",
    "codegen_chunk_recurrence",
    "detect_chunk_recurrence_search_geometry",
    "plan_chunk_recurrence",
]
