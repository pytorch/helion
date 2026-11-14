"""Analysis: FX matchers for the fused attention roots and their search surfaces.

Role: analysis.  Everything here reads the device IR (FX graphs,
``CompileEnvironment`` block facts) and returns frozen facts --
``AttentionSoftmaxPattern`` for one QK->softmax->PV loop body,
``FlashSearchSurface`` for the config-independent autotune surface of a dense,
causal, score-modified or FlexAttention kernel -- or ``None`` when the kernel
is outside the fused tcgen05 envelope.  The matchers never mutate the graphs;
the one side effect is ``flex_search_surface`` registering the flex aux-SMEM
candidates on the config spec.

The flash emitters (``cute_flash``, ``cute_flash_bwd``, ``cute_flash_gated``)
and ``CuteBackend`` import the ``_attention_*`` helpers from here; the gate
functions ``_attention_flash_gate_enabled`` / ``_attention_flash_supported``
are read through this module's globals at call time, so tests patch them on
this module.  Moved out of ``helion/_compiler/backend.py``.
"""

from __future__ import annotations

import dataclasses
import math
import operator
import os
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import Sequence

import torch

from ...utils import is_full_slice
from ..attention_plan import ALIBI_BIAS_KIND
from ..attention_plan import CAUSAL_MASK_KIND
from ..attention_plan import DOCUMENT_MASK_KIND
from ..attention_plan import PREFIX_LM_MASK_KIND
from ..attention_plan import RELATIVE_BIAS_KIND
from ..attention_plan import ROW_KV_BEGIN_MASK_KIND
from ..attention_plan import SLIDING_WINDOW_MASK_KIND
from ..attention_plan import SOFTCAP_KIND
from ..attention_plan import TENSOR_BIAS_KIND
from ..attention_plan import AttentionScoreModifier
from ..attention_plan import AttentionScorePlan
from ..attention_plan import flash_head_dims_supported
from .match_common import FlattenedRowOffset
from .match_common import flattened_row_offset
from .match_common import host_tensor_name
from .match_common import is_call
from .match_common import is_neg_inf
from .match_common import new_var_source
from .match_common import packed_f32x2_multiply
from .match_common import root_grid_origin_is_canonical
from .match_common import scalar_value
from .match_common import scale_operand
from .match_common import unit_fixed_block

if TYPE_CHECKING:
    import sympy

    from ...compile_environment import CompileEnvironment
    from ...device_ir import DeviceIR
    from ..flex_match import FlexMatch


class FlashSearchSurface(NamedTuple):
    head_dim: int
    num_kv: int
    num_bh: int
    tensor_4d_heads: int | None
    io_dtype: torch.dtype
    block_size_targets: dict[int, int]
    is_causal: bool
    has_kv_tile_pruning: bool
    requires_ws_overlap: bool
    small_biased_candidate: bool
    standard_dense_output: bool
    standard_causal_output: bool
    output_requires_tma: bool
    supports_tensor_4d_tma: bool
    has_row_epilogue: bool
    plain_row_body: bool
    has_score_modifiers: bool
    score_modifier_kinds: tuple[str, ...]
    tmem_rowmax_compatible: bool = False
    causal_resident_compatible: bool = False
    # Dense attention whose epilogue is ``O`` plus the LSE row and nothing
    # else (``flash_graph_output_facts(...).dense_lse_output``; any softmax
    # scale): the lowerings that store the LSE from the final row statistics
    # are offered on it as on the output-only dense surface.
    dense_lse_output: bool = False
    # V/O head dim when it differs from the Q/K ``head_dim`` (MLA prefill).
    head_dim_v: int | None = None
    # Grouped-query divisor (``heads_q // heads_kv``; 1 for plain attention).
    kv_group: int = 1
    # Flex envelope facts (``AttentionScorePlan.is_flex``): a data-dependent
    # BlockMask list walk (variable trip count driven by tables), whether the
    # captured score / mask regions exist, and the separate Q / KV lengths
    # (which need not be equal nor multiples of 128).
    has_list_walk: bool = False
    flex_has_score_region: bool = False
    flex_has_mask_region: bool = False
    flex_trivial_walk: bool = False
    q_seq: int | None = None
    kv_seq: int | None = None
    # Padded sizes of the flex hook's shared-memory aux staging candidates
    # (``flash_score_mod.flex_aux_smem_candidates``, plan order): the budget
    # knob's search choices and default resolve against them.
    flex_aux_smem_candidates: tuple[int, ...] = ()
    # Registers of the hook's prefetchable per-chunk table loads at 32
    # columns (``flash_score_mod.flex_prefetch_regs``): the prefetch budget
    # knob's choices resolve against them.
    flex_prefetch_regs: tuple[int, ...] = ()
    # Bytes per column of the hook's coalescible row-table runs
    # (``flash_score_mod.flex_coalesce_steps``): the cooperative request
    # width knob's choices resolve against them.
    flex_coalesce_steps: tuple[int, ...] = ()
    # Per-element operations of the score hook
    # (``flash_score_mod.flex_hook_ops``): the chunk-loop roll's default.
    flex_hook_ops: int = 0


class AttentionSoftmaxPattern(NamedTuple):
    score_plan: AttentionScorePlan
    io_dtype: torch.dtype
    # The score tile's rank: 3 for the ``bmm`` form over ``q[tile_b, tile_m, :]``
    # (the root the fused bodies are generated for), 2 for the ``hl.dot`` form
    # over ``q[bh, tile_m, :]`` with a scalar batch-head index.
    score_rank: int = 3
    # The block the K/V loads' scalar batch index belongs to (the 2-D form:
    # ``tile.begin`` of a unit tile or an ``hl.grid`` index); None when the
    # batch index is the inner batch tile itself (the 3-D form). A surface
    # admits the body only when it is the root batch block the Q tile is
    # read at (``_attention_pattern_fits_root``).
    batch_block_id: int | None = None
    # The flattened-rows form (``_attention_flattened_kv_load_indices``):
    # the kernel tiles ``B*H*M`` query rows on one root axis and reads K/V
    # at ``tile_n.index + tile_m.begin // M * N``; ``batch_block_id`` is
    # that root block. None for the (batch, query) roots.
    flattened: FlattenedRowOffset | None = None

    @property
    def head_dim(self) -> int:
        return self.score_plan.head_dim

    @property
    def v_k_major(self) -> bool:
        return self.score_plan.v_k_major

    @property
    def head_dim_v(self) -> int:
        return self.score_plan.value_head_dim

    @property
    def is_causal(self) -> bool:
        return self.score_plan.is_causal


def _attention_flash_gate_enabled() -> bool:
    """Gate for the fused tcgen05 flash-attention path.

    The fused QK->softmax->PV tcgen05 codegen (mirrors
    ``.notes/spikes/fa_tcgen05_spike.py``) is ON by default: a detected fp16/bf16,
    128x128, dense or canonical-causal, seq%128==0 attention kernel runs on the
    tensor cores. Set ``HELION_CUTE_FLASH=0`` to force the scalar-fallback path.
    The detector (``_attention_loop_shape`` plus graph-metadata output
    validation) is strict, so any config outside the envelope (fp32, unsupported
    score masks, head_dim not in {64,128}, non-128 tiles, seq%128!=0) falls back
    to the scalar path with no behavior change.
    """
    return os.environ.get("HELION_CUTE_FLASH", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _attention_flash_supported() -> bool:
    from ..mma_support import get_cute_mma_support

    return bool(get_cute_mma_support().tcgen05_f16bf16)


def _attention_loop_carried_arg(
    graph: torch.fx.Graph,
    index: int,
) -> torch.fx.Node | None:
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    if index >= len(placeholders):
        return None
    return placeholders[index]


def _attention_is_loop_carried_value(
    node: torch.fx.Node,
    placeholder: torch.fx.Node,
) -> bool:
    return node is placeholder or new_var_source(node) is placeholder


def _attention_is_block_symnode(node: torch.fx.Node, block_id: int) -> bool:
    from ....language._tracing_ops import _get_symnode

    return bool(
        node.op == "call_function"
        and node.target is _get_symnode
        and len(node.args) >= 1
        and isinstance(node.args[0], str)
        and node.args[0] == f"block_size_{block_id}"
    )


def _attention_is_inner_batch_index(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops.aten.sym_size.int
        and len(node.args) >= 2
        and isinstance(node.args[1], int)
        and node.args[1] == 0
    )


def _attention_scalar_batch_block_id(node: object) -> int | None:
    """Block id of a scalar batch index, or None.

    The 2-D attention form (``q[bh, tile_m, :]`` with 2-D ``hl.dot`` scores)
    indexes the collapsed batch-head axis with a scalar: ``tile.begin`` of a
    tile (``tile_begin(block_size_<id>)``) or an ``hl.grid`` index (an
    unbacked symbol whose origin is the grid axis, ``GridOrigin``). The block
    must be pinned to one element (``_attention_unit_fixed_block``): only
    then does the scalar address the whole tile, so the fused body's one
    batch head per program is the kernel's.
    """
    from ....language import tile_ops
    from ....language._tracing_ops import _get_symnode
    from ...host_function import HostFunction
    from ...variable_origin import GridOrigin

    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return None
    block_id: int | None = None
    if node.target is tile_ops.tile_begin:
        source = node.args[0] if node.args else None
        if (
            not isinstance(source, torch.fx.Node)
            or source.op != "call_function"
            or source.target is not _get_symnode
            or not source.args
            or not isinstance(source.args[0], str)
            or not source.args[0].startswith("block_size_")
        ):
            return None
        block_id = int(source.args[0][len("block_size_") :])
    elif node.target is _get_symnode:
        value = node.meta.get("val")
        if not isinstance(value, torch.SymInt):
            return None
        origin_info = HostFunction.current().expr_to_origin.get(value._sympy_())
        if origin_info is None or type(origin_info.origin) is not GridOrigin:
            return None
        block_id = origin_info.origin.block_id
    if block_id is None or not _attention_unit_fixed_block(block_id):
        return None
    return block_id


def _attention_tile_begin_block(symbol: sympy.Symbol) -> int | None:
    """The block whose ``tile.begin`` the host symbol stands for, or None."""
    from ...host_function import HostFunction
    from ...variable_origin import TileBeginOrigin

    origin_info = HostFunction.current().expr_to_origin.get(symbol)
    if origin_info is None or type(origin_info.origin) is not TileBeginOrigin:
        return None
    return origin_info.origin.block_id


def _attention_unit_fixed_block(block_id: int) -> bool:
    """:func:`unit_fixed_block` of the current compile environment."""
    from ...compile_environment import CompileEnvironment

    return unit_fixed_block(CompileEnvironment.current(), block_id)


def _attention_is_scalar_batch_index(node: object) -> bool:
    return _attention_scalar_batch_block_id(node) is not None


def _attention_arg_is_scaled_q(
    node: torch.fx.Node,
    expected_scale: float,
    q_placeholder: torch.fx.Node,
) -> bool:
    scale = _attention_arg_scaled_q_factor(node, q_placeholder)
    return scale is not None and math.isclose(
        scale, expected_scale, rel_tol=1e-5, abs_tol=1e-7
    )


def _attention_arg_scaled_q_factor(
    node: torch.fx.Node,
    q_placeholder: torch.fx.Node,
) -> float | None:
    if node.op != "call_function" or node.target is not torch.ops.aten.mul.Tensor:
        return None
    if len(node.args) < 2:
        return None
    tensor_arg = None
    scale_arg = None
    for arg in node.args[:2]:
        if isinstance(arg, torch.fx.Node):
            tensor_arg = arg
        elif isinstance(arg, (int, float)):
            scale_arg = float(arg)
    if (
        tensor_arg is None
        or scale_arg is None
        or not _attention_is_loop_carried_value(tensor_arg, q_placeholder)
    ):
        return None
    return scale_arg


def _attention_pv_p_arg_base(node: torch.fx.Node) -> torch.fx.Node:
    from ....language._tracing_ops import _mask_to

    while (
        node.op == "call_function"
        and node.target
        in (
            _mask_to,
            torch.ops.prims.convert_element_type.default,
        )
        and node.args
        and isinstance(node.args[0], torch.fx.Node)
    ):
        node = node.args[0]
    return node


def _attention_tile_index_source(node: object) -> torch.fx.Node | None:
    from ....language import memory_ops
    from ....language import tile_ops
    from ....language import view_ops

    while isinstance(node, torch.fx.Node):
        if node.op != "call_function":
            return None
        if node.target in (memory_ops.load, view_ops.subscript):
            if not node.args or not isinstance(node.args[0], torch.fx.Node):
                return None
            node = node.args[0]
            continue
        if node.target is tile_ops.tile_index:
            source = node.args[0] if node.args else None
            return source if isinstance(source, torch.fx.Node) else None
        return None
    return None


def _attention_is_q_tile_index(
    node: object,
    q_placeholder: torch.fx.Node,
) -> bool:
    source = _attention_tile_index_source(node)
    if source is None or len(source.args) < 2:
        return False
    dim = source.args[1]
    return (
        source.op == "call_function"
        and source.target is torch.ops.aten.sym_size.int
        and source.args[0] is q_placeholder
        and isinstance(dim, int)
        and dim == 1
    )


def _attention_is_kv_tile_index(node: object, kv_block_id: int | None) -> bool:
    source = _attention_tile_index_source(node)
    if source is None:
        return False
    if kv_block_id is None:
        return True
    return _attention_is_block_symnode(source, int(kv_block_id))


def _attention_causal_score_node(
    qk_node: torch.fx.Node,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> torch.fx.Node | None:
    """Return the canonical causal ``where(tile_m >= tile_n, qk, -inf)`` node."""
    users = [
        user
        for user in qk_node.users
        if user.op == "call_function" and user.target is torch.ops.aten.where.self
    ]
    if len(users) != 1 or set(qk_node.users) != {users[0]}:
        return None
    where_node = users[0]
    if len(where_node.args) < 3 or where_node.args[1] is not qk_node:
        return None
    if not is_neg_inf(where_node.args[2]):
        return None
    pred = where_node.args[0]
    if (
        not isinstance(pred, torch.fx.Node)
        or pred.op != "call_function"
        or pred.target is not torch.ops.aten.ge.Tensor
        or len(pred.args) < 2
    ):
        return None
    lhs, rhs = pred.args[:2]
    if not _attention_is_q_tile_index(lhs, q_placeholder):
        return None
    if not _attention_is_kv_tile_index(rhs, kv_block_id):
        return None
    return where_node


def _attention_load_host_tensor_name(node: torch.fx.Node) -> str | None:
    """The host tensor a ``load(_host_tensor(name), ...)`` node reads."""
    from ....language import memory_ops

    if not is_call(node, memory_ops.load):
        return None
    return host_tensor_name(node.args[0] if node.args else None)


def _attention_is_q_block_index(
    node: torch.fx.Node,
    q_placeholder: torch.fx.Node,
) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops.aten.sym_size.int
        and len(node.args) >= 2
        and node.args[0] is q_placeholder
        and isinstance(node.args[1], int)
        and node.args[1] == 1
    )


def _attention_tensor_bias_score_node(
    score_node: torch.fx.Node,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> tuple[torch.fx.Node, AttentionScoreModifier] | None:
    """Return ``score + bias[tile_b, tile_m, tile_n]`` as a score modifier."""
    from ....language import memory_ops

    users = [
        user
        for user in score_node.users
        if user.op == "call_function" and user.target is torch.ops.aten.add.Tensor
    ]
    if len(users) != 1 or set(score_node.users) != {users[0]}:
        return None
    add_node = users[0]
    if add_node.kwargs.get("alpha", 1) != 1 or len(add_node.args) < 2:
        return None
    if add_node.args[0] is score_node and isinstance(add_node.args[1], torch.fx.Node):
        bias_node = add_node.args[1]
    elif add_node.args[1] is score_node and isinstance(add_node.args[0], torch.fx.Node):
        bias_node = add_node.args[0]
    else:
        return None
    if bias_node.op != "call_function" or bias_node.target is not memory_ops.load:
        return None
    tensor_name = _attention_load_host_tensor_name(bias_node)
    if tensor_name is None:
        return None
    if (
        len(bias_node.args) < 4
        or bias_node.args[2] is not None
        or bias_node.args[3] is not None
    ):
        return None
    indices = bias_node.args[1]
    if not isinstance(indices, (list, tuple)) or len(indices) != 3:
        return None
    batch_idx, q_idx, kv_idx = indices
    if not isinstance(batch_idx, torch.fx.Node) or not isinstance(q_idx, torch.fx.Node):
        return None
    if not _attention_is_inner_batch_index(batch_idx):
        return None
    if not _attention_is_q_block_index(q_idx, q_placeholder):
        return None
    if not isinstance(kv_idx, torch.fx.Node):
        return None
    if kv_block_id is not None and not _attention_is_block_symnode(
        kv_idx, int(kv_block_id)
    ):
        return None
    return add_node, AttentionScoreModifier(TENSOR_BIAS_KIND, tensor_name=tensor_name)


def _attention_add_score_user(
    score_node: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.fx.Node] | None:
    users = [
        user
        for user in score_node.users
        if user.op == "call_function" and user.target is torch.ops.aten.add.Tensor
    ]
    if len(users) != 1 or set(score_node.users) != {users[0]}:
        return None
    add_node = users[0]
    if add_node.kwargs.get("alpha", 1) != 1 or len(add_node.args) < 2:
        return None
    if add_node.args[0] is score_node and isinstance(add_node.args[1], torch.fx.Node):
        return add_node, add_node.args[1]
    if add_node.args[1] is score_node and isinstance(add_node.args[0], torch.fx.Node):
        return add_node, add_node.args[0]
    return None


class _AttentionLinearIndex(NamedTuple):
    q_coeff: float
    k_coeff: float
    constant: float


def _attention_index_linear_coeffs(
    node: object,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> _AttentionLinearIndex | None:
    """Return ``query_coeff * q + key_coeff * k + constant`` for ``node``."""
    scalar = scalar_value(node)
    if scalar is not None:
        return _AttentionLinearIndex(0.0, 0.0, scalar)
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return None
    if _attention_is_q_tile_index(node, q_placeholder):
        return _AttentionLinearIndex(1.0, 0.0, 0.0)
    if _attention_is_kv_tile_index(node, kv_block_id):
        return _AttentionLinearIndex(0.0, 1.0, 0.0)
    if node.target in (torch.ops.aten.sub.Tensor, torch.ops.aten.add.Tensor):
        if len(node.args) < 2:
            return None
        lhs = _attention_index_linear_coeffs(
            node.args[0],
            q_placeholder,
            kv_block_id=kv_block_id,
        )
        rhs = _attention_index_linear_coeffs(
            node.args[1],
            q_placeholder,
            kv_block_id=kv_block_id,
        )
        if lhs is None or rhs is None:
            return None
        if node.target is torch.ops.aten.sub.Tensor:
            return _AttentionLinearIndex(
                lhs.q_coeff - rhs.q_coeff,
                lhs.k_coeff - rhs.k_coeff,
                lhs.constant - rhs.constant,
            )
        return _AttentionLinearIndex(
            lhs.q_coeff + rhs.q_coeff,
            lhs.k_coeff + rhs.k_coeff,
            lhs.constant + rhs.constant,
        )
    if node.target is torch.ops.aten.mul.Tensor and len(node.args) >= 2:
        lhs_scalar = scalar_value(node.args[0])
        rhs_scalar = scalar_value(node.args[1])
        if lhs_scalar is not None:
            rhs = _attention_index_linear_coeffs(
                node.args[1],
                q_placeholder,
                kv_block_id=kv_block_id,
            )
            return (
                None
                if rhs is None
                else _AttentionLinearIndex(
                    lhs_scalar * rhs.q_coeff,
                    lhs_scalar * rhs.k_coeff,
                    lhs_scalar * rhs.constant,
                )
            )
        if rhs_scalar is not None:
            lhs = _attention_index_linear_coeffs(
                node.args[0],
                q_placeholder,
                kv_block_id=kv_block_id,
            )
            return (
                None
                if lhs is None
                else _AttentionLinearIndex(
                    rhs_scalar * lhs.q_coeff,
                    rhs_scalar * lhs.k_coeff,
                    rhs_scalar * lhs.constant,
                )
            )
    return None


def _attention_index_delta_factor(
    node: object,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> float | None:
    """Return the coefficient of ``(query_index - key_index)`` in ``node``."""
    coeffs = _attention_index_linear_coeffs(
        node,
        q_placeholder,
        kv_block_id=kv_block_id,
    )
    if coeffs is None:
        return None
    if not math.isclose(coeffs.q_coeff + coeffs.k_coeff, 0.0, abs_tol=1e-7):
        return None
    if not math.isclose(coeffs.constant, 0.0, abs_tol=1e-7):
        return None
    return coeffs.q_coeff


def _attention_product_terms(node: object) -> list[object]:
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is torch.ops.aten.mul.Tensor
        and len(node.args) >= 2
    ):
        return _attention_product_terms(node.args[0]) + _attention_product_terms(
            node.args[1]
        )
    return [node]


class _AttentionBatchIndex(NamedTuple):
    mode: str
    divisor: int | None = None


def _attention_alibi_slope_tensor_name(
    node: object,
) -> tuple[str, _AttentionBatchIndex] | None:
    if not isinstance(node, torch.fx.Node):
        return None
    tensor_name = _attention_load_host_tensor_name(node)
    if tensor_name is None:
        return None
    if len(node.args) < 4 or node.args[2] is not None or node.args[3] is not None:
        return None
    indices = node.args[1]
    if not isinstance(indices, (list, tuple)) or len(indices) != 1:
        return None
    index = indices[0]
    index_mode = _attention_collapsed_batch_index_mode(
        index,
        allow_mod=True,
        allow_floordiv=False,
    )
    if index_mode is None:
        return None
    return tensor_name, index_mode


def _attention_positive_int_scalar(node: object) -> int | None:
    value = scalar_value(node)
    if value is None:
        return None
    int_value = int(value)
    if int_value <= 0 or not math.isclose(value, float(int_value), abs_tol=1e-7):
        return None
    return int_value


def _attention_collapsed_batch_index_mode(
    node: object,
    *,
    allow_mod: bool,
    allow_floordiv: bool,
) -> _AttentionBatchIndex | None:
    if not isinstance(node, torch.fx.Node):
        return None
    if _attention_is_inner_batch_index(node) or _attention_is_scalar_batch_index(node):
        return _AttentionBatchIndex("identity")
    source = _attention_tile_index_source(node)
    if source is not None and _attention_is_inner_batch_index(source):
        return _AttentionBatchIndex("identity")
    if node.op != "call_function" or len(node.args) < 2:
        return None
    if node.target in (
        operator.mod,
        torch.ops.aten.remainder.Scalar,
    ):
        lhs, rhs = node.args[:2]
        divisor = _attention_positive_int_scalar(rhs)
        if not allow_mod or divisor is None:
            return None
        lhs_mode = _attention_collapsed_batch_index_mode(
            lhs,
            allow_mod=False,
            allow_floordiv=False,
        )
        if lhs_mode is None or lhs_mode.mode != "identity":
            return None
        return _AttentionBatchIndex("mod", divisor)
    if node.target in (
        operator.floordiv,
        torch.ops.aten.floor_divide.default,
        torch.ops.aten.floor_divide.Scalar,
    ):
        lhs, rhs = node.args[:2]
        divisor = _attention_positive_int_scalar(rhs)
        if not allow_floordiv or divisor is None:
            return None
        lhs_mode = _attention_collapsed_batch_index_mode(
            lhs,
            allow_mod=False,
            allow_floordiv=False,
        )
        if lhs_mode is None or lhs_mode.mode != "identity":
            return None
        return _AttentionBatchIndex("floordiv", divisor)
    if (
        node.target is torch.ops.aten.div.Tensor_mode
        and node.kwargs.get("rounding_mode") == "floor"
    ):
        lhs, rhs = node.args[:2]
        divisor = _attention_positive_int_scalar(rhs)
        if not allow_floordiv or divisor is None:
            return None
        lhs_mode = _attention_collapsed_batch_index_mode(
            lhs,
            allow_mod=False,
            allow_floordiv=False,
        )
        if lhs_mode is None or lhs_mode.mode != "identity":
            return None
        return _AttentionBatchIndex("floordiv", divisor)
    return None


def _attention_alibi_bias_modifier(
    bias_node: torch.fx.Node,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> AttentionScoreModifier | None:
    terms = _attention_product_terms(bias_node)
    scalar = 1.0
    slope_name: str | None = None
    slope_index: _AttentionBatchIndex | None = None
    delta_factor: float | None = None
    for term in terms:
        term_scalar = scalar_value(term)
        if term_scalar is not None:
            scalar *= term_scalar
            continue
        term_slope = _attention_alibi_slope_tensor_name(term)
        if term_slope is not None:
            if slope_name is not None:
                return None
            slope_name, slope_index = term_slope
            continue
        term_delta = _attention_index_delta_factor(
            term,
            q_placeholder,
            kv_block_id=kv_block_id,
        )
        if term_delta is None or math.isclose(term_delta, 0.0, abs_tol=1e-12):
            return None
        if delta_factor is not None:
            return None
        delta_factor = term_delta
    if slope_name is None or delta_factor is None:
        return None
    # Runtime lowers ALiBi as ``(key - query) * slope * scale``.
    return AttentionScoreModifier(
        ALIBI_BIAS_KIND,
        tensor_name=slope_name,
        scale_log2=-scalar * delta_factor,
        index_mode=slope_index.mode if slope_index is not None else None,
        index_divisor=slope_index.divisor if slope_index is not None else None,
    )


def _attention_relative_or_alibi_score_node(
    score_node: torch.fx.Node,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> tuple[torch.fx.Node, AttentionScoreModifier] | None:
    add = _attention_add_score_user(score_node)
    if add is None:
        return None
    add_node, bias_node = add
    alibi = _attention_alibi_bias_modifier(
        bias_node,
        q_placeholder,
        kv_block_id=kv_block_id,
    )
    if alibi is not None:
        return add_node, alibi
    factor = _attention_index_delta_factor(
        bias_node,
        q_placeholder,
        kv_block_id=kv_block_id,
    )
    if factor is None or math.isclose(factor, 0.0, abs_tol=1e-12):
        return None
    return add_node, AttentionScoreModifier(RELATIVE_BIAS_KIND, scale_log2=factor)


def _attention_doc_id_load(
    node: object,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> tuple[str, str, _AttentionBatchIndex] | None:
    from ....language import view_ops

    while (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is view_ops.subscript
        and node.args
        and isinstance(node.args[0], torch.fx.Node)
    ):
        node = node.args[0]
    if not isinstance(node, torch.fx.Node):
        return None
    tensor_name = _attention_load_host_tensor_name(node)
    if tensor_name is None:
        return None
    if len(node.args) < 4 or node.args[2] is not None or node.args[3] is not None:
        return None
    indices = node.args[1]
    if not isinstance(indices, (list, tuple)) or len(indices) != 2:
        return None
    batch_idx, seq_idx = indices
    index_mode = _attention_collapsed_batch_index_mode(
        batch_idx,
        allow_mod=False,
        allow_floordiv=True,
    )
    if index_mode is None:
        return None
    if not isinstance(seq_idx, torch.fx.Node):
        return None
    if _attention_is_q_block_index(seq_idx, q_placeholder):
        return tensor_name, "q", index_mode
    if kv_block_id is not None and _attention_is_block_symnode(
        seq_idx, int(kv_block_id)
    ):
        return tensor_name, "kv", index_mode
    return None


def _attention_bool_terms(node: object, targets: frozenset[object]) -> list[object]:
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target in targets
        and len(node.args) >= 2
    ):
        return _attention_bool_terms(node.args[0], targets) + _attention_bool_terms(
            node.args[1], targets
        )
    return [node]


_ATTENTION_AND_TARGETS: frozenset[object] = frozenset(
    {
        operator.and_,
        torch.ops.aten.bitwise_and.Tensor,
        torch.ops.aten.logical_and.default,
    }
)
_ATTENTION_OR_TARGETS: frozenset[object] = frozenset(
    {
        operator.or_,
        torch.ops.aten.bitwise_or.Tensor,
        torch.ops.aten.logical_or.default,
    }
)


def _attention_is_causal_predicate(
    node: object,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> bool:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return False
    if node.target is torch.ops.aten.ge.Tensor and len(node.args) >= 2:
        lhs, rhs = node.args[:2]
        if _attention_is_q_tile_index(
            lhs, q_placeholder
        ) and _attention_is_kv_tile_index(rhs, kv_block_id):
            return True
    if node.target in (torch.ops.aten.ge.Tensor, torch.ops.aten.ge.Scalar):
        if len(node.args) < 2:
            return False
        delta = _attention_index_delta_factor(
            node.args[0],
            q_placeholder,
            kv_block_id=kv_block_id,
        )
        bound = scalar_value(node.args[1])
        return (
            delta is not None
            and bound is not None
            and math.isclose(delta, 1.0, rel_tol=1e-6, abs_tol=1e-7)
            and math.isclose(bound, 0.0, abs_tol=1e-7)
        )
    return False


def _attention_sliding_window_bound(
    node: object,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> int | None:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return None
    if node.target not in (torch.ops.aten.le.Tensor, torch.ops.aten.le.Scalar):
        return None
    if len(node.args) < 2:
        return None
    delta = _attention_index_delta_factor(
        node.args[0],
        q_placeholder,
        kv_block_id=kv_block_id,
    )
    bound = scalar_value(node.args[1])
    if (
        delta is None
        or bound is None
        or not math.isclose(delta, 1.0, rel_tol=1e-6, abs_tol=1e-7)
    ):
        return None
    window = int(bound)
    if not math.isclose(bound, float(window), abs_tol=1e-7) or window < 0:
        return None
    return window


def _attention_prefix_length(
    node: object,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> int | None:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return None
    if node.target not in (torch.ops.aten.lt.Tensor, torch.ops.aten.lt.Scalar):
        return None
    if len(node.args) < 2:
        return None
    if not _attention_is_kv_tile_index(node.args[0], kv_block_id):
        return None
    bound = scalar_value(node.args[1])
    if bound is None:
        return None
    prefix = int(bound)
    if not math.isclose(bound, float(prefix), abs_tol=1e-7) or prefix < 0:
        return None
    return prefix


def _attention_document_mask_name(
    node: object,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> tuple[str, _AttentionBatchIndex] | None:
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target is not torch.ops.aten.eq.Tensor
        or len(node.args) < 2
    ):
        return None
    lhs = _attention_doc_id_load(
        node.args[0],
        q_placeholder,
        kv_block_id=kv_block_id,
    )
    rhs = _attention_doc_id_load(
        node.args[1],
        q_placeholder,
        kv_block_id=kv_block_id,
    )
    if lhs is None or rhs is None or lhs[0] != rhs[0] or lhs[2] != rhs[2]:
        return None
    return (lhs[0], lhs[2]) if {lhs[1], rhs[1]} == {"q", "kv"} else None


def _attention_row_kv_begin_mask_name(
    node: object,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> tuple[str, _AttentionBatchIndex] | None:
    """Match ``kv_idx >= lb[batch, q_tile]`` (or ``lb[...] <= kv_idx``).

    ``lb`` is a ``[B, S]`` per-row KV lower bound loaded at the Q tile, the
    document-table index form of ``_attention_doc_id_load``.
    """
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or len(node.args) < 2
    ):
        return None
    if node.target is torch.ops.aten.ge.Tensor:
        kv_node, bound_node = node.args[:2]
    elif node.target is torch.ops.aten.le.Tensor:
        bound_node, kv_node = node.args[:2]
    else:
        return None
    if not _attention_is_kv_tile_index(kv_node, kv_block_id):
        return None
    bound = _attention_doc_id_load(
        bound_node,
        q_placeholder,
        kv_block_id=kv_block_id,
    )
    if bound is None or bound[1] != "q":
        return None
    return bound[0], bound[2]


def _attention_mask_modifier_from_predicate(
    pred: torch.fx.Node,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> AttentionScoreModifier | None:
    if _attention_is_causal_predicate(
        pred,
        q_placeholder,
        kv_block_id=kv_block_id,
    ):
        return AttentionScoreModifier(CAUSAL_MASK_KIND)

    or_terms = _attention_bool_terms(pred, _ATTENTION_OR_TARGETS)
    if len(or_terms) == 2:
        prefix = None
        has_causal = False
        for term in or_terms:
            if _attention_is_causal_predicate(
                term,
                q_placeholder,
                kv_block_id=kv_block_id,
            ):
                has_causal = True
                continue
            term_prefix = _attention_prefix_length(
                term,
                q_placeholder,
                kv_block_id=kv_block_id,
            )
            if term_prefix is not None:
                prefix = term_prefix
        if prefix is not None and has_causal:
            return AttentionScoreModifier(PREFIX_LM_MASK_KIND, prefix_length=prefix)

    and_terms = _attention_bool_terms(pred, _ATTENTION_AND_TARGETS)
    if len(and_terms) >= 2:
        has_causal = False
        window = None
        document_name = None
        document_index: _AttentionBatchIndex | None = None
        row_kv_begin_name = None
        row_kv_begin_index: _AttentionBatchIndex | None = None
        for term in and_terms:
            if _attention_is_causal_predicate(
                term,
                q_placeholder,
                kv_block_id=kv_block_id,
            ):
                has_causal = True
                continue
            term_window = _attention_sliding_window_bound(
                term,
                q_placeholder,
                kv_block_id=kv_block_id,
            )
            if term_window is not None:
                if window is not None:
                    return None
                window = term_window
                continue
            term_document = _attention_document_mask_name(
                term,
                q_placeholder,
                kv_block_id=kv_block_id,
            )
            if term_document is not None:
                if document_name is not None:
                    return None
                document_name, document_index = term_document
                continue
            term_row_kv_begin = _attention_row_kv_begin_mask_name(
                term,
                q_placeholder,
                kv_block_id=kv_block_id,
            )
            if term_row_kv_begin is not None:
                if row_kv_begin_name is not None:
                    return None
                row_kv_begin_name, row_kv_begin_index = term_row_kv_begin
                continue
            return None
        if row_kv_begin_name is not None:
            if not has_causal or window is not None or document_name is not None:
                return None
            return AttentionScoreModifier(
                ROW_KV_BEGIN_MASK_KIND,
                tensor_name=row_kv_begin_name,
                index_mode=row_kv_begin_index.mode
                if row_kv_begin_index is not None
                else None,
                index_divisor=row_kv_begin_index.divisor
                if row_kv_begin_index is not None
                else None,
            )
        if has_causal and window is not None and document_name is None:
            return AttentionScoreModifier(
                SLIDING_WINDOW_MASK_KIND,
                window_size=window,
            )
        if has_causal and document_name is not None and window is None:
            return AttentionScoreModifier(
                DOCUMENT_MASK_KIND,
                tensor_name=document_name,
                index_mode=document_index.mode if document_index is not None else None,
                index_divisor=document_index.divisor
                if document_index is not None
                else None,
            )
    return None


def _attention_mask_score_node(
    score_node: torch.fx.Node,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> tuple[torch.fx.Node, AttentionScoreModifier] | None:
    users = [
        user
        for user in score_node.users
        if user.op == "call_function" and user.target is torch.ops.aten.where.self
    ]
    if len(users) != 1 or set(score_node.users) != {users[0]}:
        return None
    where_node = users[0]
    if len(where_node.args) < 3 or where_node.args[1] is not score_node:
        return None
    if not is_neg_inf(where_node.args[2]):
        return None
    pred = where_node.args[0]
    if not isinstance(pred, torch.fx.Node):
        return None
    modifier = _attention_mask_modifier_from_predicate(
        pred,
        q_placeholder,
        kv_block_id=kv_block_id,
    )
    if modifier is None:
        return None
    return where_node, modifier


def _attention_softcap_score_node(
    score_node: torch.fx.Node,
) -> tuple[torch.fx.Node, AttentionScoreModifier] | None:
    div_users = [
        user
        for user in score_node.users
        if user.op == "call_function" and user.target is torch.ops.aten.div.Tensor
    ]
    if len(div_users) != 1 or set(score_node.users) != {div_users[0]}:
        return None
    div_node = div_users[0]
    if len(div_node.args) < 2 or div_node.args[0] is not score_node:
        return None
    softcap = scalar_value(div_node.args[1])
    if softcap is None or softcap <= 0.0:
        return None
    tanh_users = [
        user
        for user in div_node.users
        if user.op == "call_function" and user.target is torch.ops.aten.tanh.default
    ]
    if len(tanh_users) != 1 or set(div_node.users) != {tanh_users[0]}:
        return None
    tanh_node = tanh_users[0]
    mul_users = [
        user
        for user in tanh_node.users
        if user.op == "call_function" and user.target is torch.ops.aten.mul.Tensor
    ]
    if len(mul_users) != 1 or set(tanh_node.users) != {mul_users[0]}:
        return None
    mul_node = mul_users[0]
    if len(mul_node.args) < 2:
        return None
    if mul_node.args[0] is tanh_node:
        cap_arg = mul_node.args[1]
    elif mul_node.args[1] is tanh_node:
        cap_arg = mul_node.args[0]
    else:
        return None
    if not math.isclose(
        scalar_value(cap_arg) or float("nan"),
        softcap,
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        return None
    return mul_node, AttentionScoreModifier(SOFTCAP_KIND, value_log2=softcap)


def _attention_score_modifiers(
    qk_node: torch.fx.Node,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> tuple[torch.fx.Node, tuple[AttentionScoreModifier, ...]] | None:
    score_node = qk_node
    modifiers: list[AttentionScoreModifier] = []
    while True:
        bias = _attention_tensor_bias_score_node(
            score_node,
            q_placeholder,
            kv_block_id=kv_block_id,
        )
        if bias is not None:
            score_node, modifier = bias
            modifiers.append(modifier)
            continue
        index_bias = _attention_relative_or_alibi_score_node(
            score_node,
            q_placeholder,
            kv_block_id=kv_block_id,
        )
        if index_bias is not None:
            score_node, modifier = index_bias
            modifiers.append(modifier)
            continue
        softcap = _attention_softcap_score_node(score_node)
        if softcap is not None:
            score_node, modifier = softcap
            modifiers.append(modifier)
            continue
        mask = _attention_mask_score_node(
            score_node,
            q_placeholder,
            kv_block_id=kv_block_id,
        )
        if mask is not None:
            score_node, modifier = mask
            modifiers.append(modifier)
            continue
        return score_node, tuple(modifiers)


class _AttentionKVLoadIndices(NamedTuple):
    """The ``(batch, kv_tile)`` indices of a canonical K/V load.

    ``kv_group`` is the grouped-query divisor: the batch index is
    ``batch_index // kv_group`` (``1`` for the plain ``tile_b`` form, where
    ``batch_index`` is the index node itself). ``k_major`` marks the
    pre-transposed ``[batch, :, tile_n]`` form (V only; see
    ``AttentionScorePlan.v_k_major``).
    """

    batch_index: torch.fx.Node
    kv_tile: torch.fx.Node
    kv_group: int
    k_major: bool = False
    # The block of a scalar ``batch_index`` (``_attention_scalar_batch_block_id``,
    # the 2-D form); None when it is the inner batch tile itself (the 3-D
    # form, tied to the Q tile by its leading dim). A surface compares it with
    # the root batch block: nothing else ties a scalar to the Q tile's row.
    batch_block_id: int | None = None
    # The flattened-rows form's offset (``batch_index`` is its symnode and
    # ``batch_block_id`` the root block it is taken from); None otherwise.
    flattened: FlattenedRowOffset | None = None


def _attention_kv_batch_index_group(
    node: torch.fx.Node,
) -> tuple[torch.fx.Node, int] | None:
    """Return ``(dividend, divisor)`` of a K/V batch index, or None.

    Accepts the inner batch tile index itself (divisor 1) or its floor
    division by a positive constant (``tile_b.index // group``), the form a
    grouped-query kernel uses to map query heads onto shared K/V heads.
    """
    index_mode = _attention_collapsed_batch_index_mode(
        node,
        allow_mod=False,
        allow_floordiv=True,
    )
    if index_mode is None:
        return None
    if index_mode.mode == "identity":
        return node, 1
    assert index_mode.mode == "floordiv" and index_mode.divisor is not None
    dividend = node.args[0]
    assert isinstance(dividend, torch.fx.Node)
    return dividend, index_mode.divisor


def _attention_canonical_kv_load_indices(
    node: torch.fx.Node,
    *,
    kv_block_id: int | None,
    allow_k_major: bool = False,
) -> _AttentionKVLoadIndices | None:
    from ....language import memory_ops

    if node.op != "call_function" or node.target is not memory_ops.load:
        return None
    if len(node.args) < 4 or node.args[2] is not None or node.args[3] is not None:
        return None
    indices = node.args[1]
    if not isinstance(indices, (list, tuple)):
        return None
    if len(indices) == 2:
        return _attention_flattened_kv_load_indices(indices, kv_block_id=kv_block_id)
    if len(indices) != 3:
        return None
    if not isinstance(indices[0], torch.fx.Node):
        return None
    # ``[batch, tile_n, :]`` (rows of keys / values, the head dim contiguous)
    # or, when admitted, the pre-transposed ``[batch, :, tile_n]`` form (the KV
    # axis contiguous: the K-major layout 8-bit tcgen05 operands need).
    if isinstance(indices[1], torch.fx.Node) and is_full_slice(indices[2]):
        kv_tile = indices[1]
        k_major = False
    elif (
        allow_k_major
        and is_full_slice(indices[1])
        and isinstance(indices[2], torch.fx.Node)
    ):
        kv_tile = indices[2]
        k_major = True
    else:
        return None
    batch_index = _attention_kv_batch_index_group(indices[0])
    if batch_index is None:
        return None
    if kv_block_id is not None and not _attention_is_block_symnode(
        kv_tile, int(kv_block_id)
    ):
        return None
    dividend, kv_group = batch_index
    return _AttentionKVLoadIndices(
        dividend,
        kv_tile,
        kv_group,
        k_major,
        batch_block_id=_attention_scalar_batch_block_id(dividend),
    )


def _attention_flattened_kv_load_indices(
    indices: Sequence[object], *, kv_block_id: int | None
) -> _AttentionKVLoadIndices | None:
    """The flattened-rows K/V load ``k[tile_n.index + offset, :]`` with
    ``offset = tile_m.begin // M * N`` folded into one symnode
    (``match_common.flattened_row_offset``): the kernel tiles ``B*H*M`` query
    rows on one root axis and reads its batch head's ``N`` key rows through
    the offset.  The offset node stands as the batch index (K and V share
    it), the KV loop's block symnode as the KV tile and the root block the
    offset is taken from as the batch block."""
    from ....language import tile_ops

    row, last = indices
    if not isinstance(row, torch.fx.Node) or not is_full_slice(last):
        return None
    if row.op != "call_function" or row.target is not torch.ops.aten.add.Tensor:
        return None
    if len(row.args) < 2 or row.kwargs:
        return None
    for tile_arg, offset_arg in (row.args[:2], row.args[1::-1]):
        if not is_call(tile_arg, tile_ops.tile_index):
            continue
        assert isinstance(tile_arg, torch.fx.Node)
        kv_tile = tile_arg.args[0] if tile_arg.args else None
        flattened = flattened_row_offset(offset_arg, _attention_tile_begin_block)
        if isinstance(kv_tile, torch.fx.Node) and flattened is not None:
            break
    else:
        return None
    if kv_block_id is not None and not _attention_is_block_symnode(
        kv_tile, int(kv_block_id)
    ):
        return None
    assert isinstance(offset_arg, torch.fx.Node)
    return _AttentionKVLoadIndices(
        offset_arg,
        kv_tile,
        1,
        batch_block_id=flattened.block_id,
        flattened=flattened,
    )


def _attention_transposed_source(
    node: torch.fx.Node, rank: int
) -> torch.fx.Node | None:
    """Source of a swap of the last two dims of a ``rank``-D tensor, or None:
    ``permute([0, 2, 1])`` / ``transpose(1, 2)`` at rank 3 (the ``bmm`` form),
    ``permute([1, 0])`` / ``transpose(0, 1)`` / ``t()`` at rank 2 (the 2-D
    ``hl.dot`` form)."""
    if node.op != "call_function" or not node.args:
        return None
    if node.target is torch.ops.aten.permute.default:
        if len(node.args) < 2 or node.args[1] != [*range(rank - 2), rank - 1, rank - 2]:
            return None
    elif node.target is torch.ops.aten.transpose.int:
        if len(node.args) < 3 or {node.args[1], node.args[2]} != {rank - 2, rank - 1}:
            return None
    elif node.target is torch.ops.aten.t.default:
        if rank != 2:
            return None
    else:
        return None
    source = node.args[0]
    return source if isinstance(source, torch.fx.Node) else None


def _attention_k_load_indices(
    node: torch.fx.Node,
    *,
    kv_block_id: int | None,
    rank: int = 3,
) -> _AttentionKVLoadIndices | None:
    source = _attention_transposed_source(node, rank)
    if source is None:
        return None
    return _attention_canonical_kv_load_indices(source, kv_block_id=kv_block_id)


def _attention_k_load_head_dim(node: torch.fx.Node) -> int | None:
    """Return K's static head dim from the host tensor behind ``K^T``.

    ``node`` is the transposed K operand of the score matmul; its source is the
    canonical ``k_view[b, tile_n, :]`` load whose host tensor keeps the
    specialized head dim (the loop graph's operand metadata does not).
    """
    if node.op != "call_function" or not node.args:
        return None
    source = node.args[0]
    if not isinstance(source, torch.fx.Node):
        return None
    return _attention_load_head_dim(source)


def _attention_load_head_dim(source: torch.fx.Node) -> int | None:
    """Return the static head dim of the host tensor behind a canonical
    ``x_view[b, tile, :]`` load, or the flattened form's ``x[rows, :]`` (the
    loop graph's operand metadata may carry the width as an unbacked symbol;
    the host tensor keeps it specialized)."""
    from ....language import memory_ops
    from ....language._tracing_ops import _host_tensor

    if source.op != "call_function" or source.target is not memory_ops.load:
        return None
    tensor = source.args[0] if source.args else None
    if (
        not isinstance(tensor, torch.fx.Node)
        or tensor.op != "call_function"
        or tensor.target is not _host_tensor
    ):
        return None
    value = tensor.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.ndim not in (2, 3):
        return None
    head_dim = value.shape[-1]
    return head_dim if isinstance(head_dim, int) else None


def _attention_v_load_indices(
    node: torch.fx.Node,
    *,
    kv_block_id: int | None,
    rank: int = 3,
) -> _AttentionKVLoadIndices | None:
    """The V operand of the PV matmul: the canonical ``v[b, tile_n, :]`` load,
    or the transpose of a pre-transposed ``v[b, :, tile_n]`` load (recorded as
    ``k_major``)."""
    canonical = _attention_canonical_kv_load_indices(node, kv_block_id=kv_block_id)
    if canonical is not None:
        return canonical
    source = _attention_transposed_source(node, rank)
    if source is None:
        return None
    indices = _attention_canonical_kv_load_indices(
        source, kv_block_id=kv_block_id, allow_k_major=True
    )
    if indices is None or not indices.k_major:
        return None
    return indices


def _attention_v_load_head_dim(node: torch.fx.Node, rank: int) -> int | None:
    """V's static head dim from the host tensor behind the PV matmul's V
    operand (the canonical load's last dim, the K-major load's middle one)."""
    from ....language import memory_ops
    from ....language._tracing_ops import _host_tensor

    source = node
    k_major = False
    if source.op != "call_function" or source.target is not memory_ops.load:
        transposed = _attention_transposed_source(node, rank)
        if transposed is None:
            return None
        source = transposed
        k_major = True
    if source.op != "call_function" or source.target is not memory_ops.load:
        return None
    tensor = source.args[0] if source.args else None
    if (
        not isinstance(tensor, torch.fx.Node)
        or tensor.op != "call_function"
        or tensor.target is not _host_tensor
    ):
        return None
    value = tensor.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.ndim not in (2, 3):
        return None
    if k_major:
        if value.ndim != 3:
            return None
        head_dim = value.shape[1]
    else:
        head_dim = value.shape[-1]
    return head_dim if isinstance(head_dim, int) else None


def _attention_matmul_form(node: torch.fx.Node) -> str | None:
    """``"qk"`` / ``"pv"`` for the two matmul forms the matcher reads: the
    3-D ``aten.bmm.dtype`` / ``aten.baddbmm.default`` pair and the 2-D
    ``hl.dot(a, b)`` / ``hl.dot(p, v, acc=acc)`` pair; None otherwise."""
    from ....language import matmul_ops

    if node.op != "call_function":
        return None
    if node.target is torch.ops.aten.bmm.dtype:
        return "qk"
    if node.target is torch.ops.aten.baddbmm.default:
        return "pv"
    if node.target is matmul_ops.dot and len(node.args) >= 3:
        return "pv" if isinstance(node.args[2], torch.fx.Node) else "qk"
    return None


def _attention_pv_operands(
    pv_node: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node] | None:
    """``(acc, p, v)`` of a PV matmul: ``baddbmm(acc, p, v)`` or
    ``dot(p, v, acc)``."""
    from ....language import matmul_ops

    if pv_node.target is torch.ops.aten.baddbmm.default:
        if len(pv_node.args) < 3:
            return None
        acc, p, v = pv_node.args[:3]
    elif pv_node.target is matmul_ops.dot:
        if len(pv_node.args) < 3:
            return None
        p, v, acc = pv_node.args[:3]
    else:
        return None
    if not (
        isinstance(acc, torch.fx.Node)
        and isinstance(p, torch.fx.Node)
        and isinstance(v, torch.fx.Node)
    ):
        return None
    return acc, p, v


class _AttentionPostDotScale(NamedTuple):
    """The post-dot form of the softmax scale (``_attention_post_dot_scale``)."""

    # ``qk * scale``: the score root the modifiers and the chain read.
    score_node: torch.fx.Node
    # A Python number, or the host SymFloat of a runtime scale.
    scale: float | sympy.Expr
    # The deferred form's ``amax(_mask_to(qk, -inf)) * scale``: the running
    # max reads the unscaled scores' row max and scales it (``amax(qk) * s ==
    # amax(qk * s)`` for ``s > 0``); None when it reads the scaled score.
    deferred_max: torch.fx.Node | None


def _attention_other_operand(node: torch.fx.Node, operand: torch.fx.Node) -> object:
    """The other of a binary node's two operands when ``operand`` is one of
    them; None otherwise."""
    if len(node.args) < 2:
        return None
    lhs, rhs = node.args[:2]
    if lhs is operand:
        return rhs
    return lhs if rhs is operand else None


def _attention_scale_value(node: object) -> float | sympy.Expr | None:
    """A softmax scale operand: a Python number, or a host SymFloat (a kernel
    scalar the body reads at runtime; ``AttentionScorePlan.runtime_scale``)."""
    from ...host_function import HostFunction

    scale = scale_operand(node)
    if scale is None:
        return None
    value = scale[0]
    if isinstance(value, float):
        return value
    origin_info = HostFunction.current().expr_to_origin.get(value)
    if origin_info is None or not origin_info.origin.is_host():
        return None
    return value


def _attention_post_dot_scale(
    qk_node: torch.fx.Node,
) -> _AttentionPostDotScale | None:
    """``qk * scale`` as the score root when the matmul's users are that
    scalar multiply alone (the fused body scales the fp32 scores either way),
    or, in the deferred form, the multiply plus ``_mask_to(qk, -inf) ->
    amax([-1]) -> * scale`` feeding the running max.  The deferred row max
    equals the scaled scores' for a positive scale: a literal is checked
    here, a runtime scale by the generated host function before the launch
    (``cute_flash._flash_emit_runtime_scale_host_assert``).
    """
    from ....language._tracing_ops import _mask_to

    users = list(qk_node.users)
    scaled_users = [
        user
        for user in users
        if user.op == "call_function" and user.target is torch.ops.aten.mul.Tensor
    ]
    if len(scaled_users) != 1 or len(users) > 2:
        return None
    (scaled,) = scaled_users
    scale_node = _attention_other_operand(scaled, qk_node)
    scale = _attention_scale_value(scale_node)
    if scale is None:
        return None
    if len(users) == 1:
        return _AttentionPostDotScale(scaled, scale, None)
    (masked,) = (user for user in users if user is not scaled)
    if not (
        is_call(masked, _mask_to)
        and len(masked.args) >= 2
        and masked.args[0] is qk_node
        and masked.args[1] == float("-inf")
    ):
        return None
    amax_users = list(masked.users)
    if len(amax_users) != 1 or not is_call(amax_users[0], torch.ops.aten.amax.default):
        return None
    (amax_node,) = amax_users
    if (
        len(amax_node.args) < 2
        or amax_node.args[1] != [-1]
        or (len(amax_node.args) >= 3 and bool(amax_node.args[2]))
    ):
        return None
    deferred_users = list(amax_node.users)
    if len(deferred_users) != 1 or not is_call(
        deferred_users[0], torch.ops.aten.mul.Tensor
    ):
        return None
    (deferred,) = deferred_users
    if _attention_scale_value(_attention_other_operand(deferred, amax_node)) != scale:
        return None
    if isinstance(scale, float) and scale <= 0:
        return None
    return _AttentionPostDotScale(scaled, scale, deferred)


def _attention_rescale_operands(
    node: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.fx.Node] | None:
    """The two tensor operands of the accumulator rescale ``prev_acc *
    alpha[..., None]``: an ``aten.mul.Tensor`` over two nodes, or the packed
    ``mul.f32x2`` inline asm ``examples/blackwell_attention.py`` spells it
    with (``match_common.packed_f32x2_multiply``): the same round-to-nearest
    multiply under another name."""
    if node.op == "call_function" and node.target is torch.ops.aten.mul.Tensor:
        if len(node.args) < 2:
            return None
        lhs, rhs = node.args[:2]
        if isinstance(lhs, torch.fx.Node) and isinstance(rhs, torch.fx.Node):
            return lhs, rhs
        return None
    return packed_f32x2_multiply(node)


def _attention_exact_online_softmax_chain(
    score_node: torch.fx.Node,
    pv_node: torch.fx.Node,
    *,
    deferred_max: torch.fx.Node | None = None,
) -> bool:
    """Whether ``score_node`` (the scaled, modified score tile) feeds the
    exact online-softmax chain the fused bodies compute: ``m = maximum(prev_m,
    amax(_mask_to(score, -inf)))``, ``p = exp2(score - m)``, ``alpha =
    exp2(prev_m - m)``, ``l = prev_l * alpha + sum(_mask_to(p, 0))``, ``acc =
    pv(p, v, prev_acc * alpha)``, the loop returning ``(m, l, acc)`` in the
    order the three statistics are carried.  The carried roles are read from
    their uses (the running max feeds ``maximum``, the row sum the ``l``
    update, the accumulator the rescale), not from their positions; the
    loop's other inputs (tiles read outside the KV loop) are not carried and
    take no part.  ``deferred_max`` is the post-dot form's ``amax(_mask_to(qk, -inf)) *
    scale`` node (``_attention_post_dot_scale``): the running max then reads
    it instead of the scaled score's own masked ``amax``.
    """
    from ....language import view_ops
    from ....language._tracing_ops import _mask_to

    # The score tile's rank: 3 for the ``bmm`` form, 2 for the ``hl.dot`` form.
    # Row vectors broadcast back over it as ``[:, ..., None]``.
    score_val = score_node.meta.get("val")
    score_rank = score_val.ndim if isinstance(score_val, torch.Tensor) else 3
    graph = score_node.graph

    def broadcast_row_vector_source(node: torch.fx.Node) -> torch.fx.Node | None:
        if node.op != "call_function" or node.target is not view_ops.subscript:
            return None
        if not node.args or not isinstance(node.args[0], torch.fx.Node):
            return None
        indices = node.args[1] if len(node.args) > 1 else None
        if not isinstance(indices, (list, tuple)) or len(indices) != score_rank:
            return None
        if indices[-1] is not None or not all(
            is_full_slice(index) for index in indices[:-1]
        ):
            return None
        return node.args[0]

    def is_broadcast_row_vector(node: torch.fx.Node, source: torch.fx.Node) -> bool:
        return broadcast_row_vector_source(node) is source

    def is_squeeze_last_dim(node: torch.fx.Node, source: torch.fx.Node) -> bool:
        dim = node.args[1] if len(node.args) >= 2 else None
        return (
            node.op == "call_function"
            and node.target is torch.ops.aten.squeeze.dim
            and len(node.args) >= 2
            and node.args[0] is source
            and isinstance(dim, int)
            and dim == -1
        )

    def binary_node_has_args(
        node: torch.fx.Node,
        target: object,
        lhs: torch.fx.Node,
        rhs: torch.fx.Node,
        *,
        ordered: bool,
    ) -> bool:
        if node.op != "call_function" or node.target is not target:
            return False
        if len(node.args) < 2:
            return False
        if ordered:
            return node.args[0] is lhs and node.args[1] is rhs
        return (node.args[0] is lhs and node.args[1] is rhs) or (
            node.args[0] is rhs and node.args[1] is lhs
        )

    def unwrap_new_var(node: object) -> object:
        if isinstance(node, torch.fx.Node):
            return new_var_source(node) or node
        return node

    # The Q tile, then the loop's other inputs in the kernel's order: the
    # three carried statistics (named below by their uses) and any tile read
    # outside the KV loop; the loop returns the carried values alone, in
    # their input order.
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    candidates = placeholders[1:]
    if len(candidates) < 3:
        return False

    def carried_placeholder(node: object) -> torch.fx.Node | None:
        node = unwrap_new_var(node)
        return node if isinstance(node, torch.fx.Node) and node in candidates else None

    def loop_body_returns(results: dict[torch.fx.Node, torch.fx.Node]) -> bool:
        carried = [placeholder for placeholder in candidates if placeholder in results]
        for node in graph.nodes:
            if node.op != "output":
                continue
            if len(node.args) != 1 or not isinstance(node.args[0], (list, tuple)):
                return False
            outputs = node.args[0]
            return len(outputs) == len(carried) and all(
                unwrap_new_var(output) is results[placeholder]
                for output, placeholder in zip(outputs, carried, strict=True)
            )
        return False

    qk_sub_users = [
        user
        for user in score_node.users
        if user.op == "call_function" and user.target is torch.ops.aten.sub.Tensor
    ]
    if len(qk_sub_users) != 1:
        return False
    if deferred_max is None:
        qk_mask_users = [
            user
            for user in score_node.users
            if user.op == "call_function" and user.target is _mask_to
        ]
        if len(qk_mask_users) != 1:
            return False
        if set(score_node.users) != {qk_mask_users[0], qk_sub_users[0]}:
            return False

        qk_mask = qk_mask_users[0]
        if len(qk_mask.args) < 2 or qk_mask.args[0] is not score_node:
            return False
        if qk_mask.args[1] != float("-inf"):
            return False
        amax_users = [
            user
            for user in qk_mask.users
            if user.op == "call_function" and user.target is torch.ops.aten.amax.default
        ]
        if len(amax_users) != 1 or set(qk_mask.users) != {amax_users[0]}:
            return False
        max_input = amax_users[0]
        if len(max_input.args) < 2 or max_input.args[1] != [-1]:
            return False
        amax_keepdim = len(max_input.args) >= 3 and bool(max_input.args[2])
    else:
        if set(score_node.users) != {qk_sub_users[0]}:
            return False
        max_input = deferred_max
        amax_keepdim = False
    maximum_users = [
        user
        for user in max_input.users
        if user.op == "call_function" and user.target is torch.ops.aten.maximum.default
    ]
    if len(maximum_users) != 1:
        return False
    max_node = maximum_users[0]
    if len(max_node.args) < 2:
        return False
    if max_node.args[0] is max_input and isinstance(max_node.args[1], torch.fx.Node):
        prev_m_arg = max_node.args[1]
    elif max_node.args[1] is max_input and isinstance(max_node.args[0], torch.fx.Node):
        prev_m_arg = max_node.args[0]
    else:
        return False
    if amax_keepdim:
        prev_m = broadcast_row_vector_source(prev_m_arg)
        if prev_m is None:
            return False
        max_state_users = [
            user for user in max_node.users if is_squeeze_last_dim(user, max_node)
        ]
        if len(max_state_users) != 1:
            return False
        max_state_node = max_state_users[0]
    else:
        prev_m = prev_m_arg
        max_state_node = max_node
    m_placeholder = carried_placeholder(prev_m)
    if m_placeholder is None:
        return False

    centered = qk_sub_users[0]
    if len(centered.args) < 2 or centered.args[0] is not score_node:
        return False
    max_view = centered.args[1]
    if not isinstance(max_view, torch.fx.Node):
        return False
    if amax_keepdim:
        if max_view is not max_node:
            return False
    else:
        if not is_broadcast_row_vector(max_view, max_node):
            return False

    p_users = [
        user
        for user in centered.users
        if user.op == "call_function"
        and user.target in (torch.ops.aten.exp.default, torch.ops.aten.exp2.default)
    ]
    if len(p_users) != 1 or set(centered.users) != {p_users[0]}:
        return False
    p_node = p_users[0]

    p_mask_users = [
        user
        for user in p_node.users
        if user.op == "call_function"
        and user.target is _mask_to
        and len(user.args) >= 2
        and user.args[0] is p_node
        and user.args[1] == 0
    ]
    p_sum_users = [
        user
        for p_mask in p_mask_users
        for user in p_mask.users
        if user.op == "call_function"
        and user.target is torch.ops.aten.sum.dim_IntList
        and len(user.args) >= 2
        and user.args[1] == [-1]
    ]
    if len(p_sum_users) != 1:
        return False
    l_ij = p_sum_users[0]

    alpha_sub_users = [
        user
        for user in max_state_node.users
        if binary_node_has_args(
            user,
            torch.ops.aten.sub.Tensor,
            prev_m,
            max_state_node,
            ordered=True,
        )
    ]
    if len(alpha_sub_users) != 1:
        return False
    alpha_sub = alpha_sub_users[0]
    alpha_users = [
        user
        for user in alpha_sub.users
        if user.op == "call_function" and user.target is p_node.target
    ]
    if len(alpha_users) != 1 or set(alpha_sub.users) != {alpha_users[0]}:
        return False
    alpha = alpha_users[0]

    # ``l = prev_l * alpha + l_ij``: the carried operand of the multiply
    # names the row sum.
    l_updates = [
        (l_placeholder, user)
        for l_mul in alpha.users
        if l_mul.op == "call_function"
        and l_mul.target is torch.ops.aten.mul.Tensor
        and len(l_mul.args) >= 2
        and alpha in l_mul.args[:2]
        for l_placeholder in (
            carried_placeholder(arg)
            for arg in l_mul.args[:2]
            if carried_placeholder(arg) not in (None, m_placeholder)
        )
        for user in l_mul.users
        if binary_node_has_args(
            user,
            torch.ops.aten.add.Tensor,
            l_mul,
            l_ij,
            ordered=False,
        )
    ]
    if len(l_updates) != 1:
        return False
    l_placeholder, l_update = l_updates[0]
    assert l_placeholder is not None

    alpha_views = [user for user in alpha.users if is_broadcast_row_vector(user, alpha)]
    if len(alpha_views) != 1:
        return False
    alpha_view = alpha_views[0]
    # ``acc = prev_acc * alpha[..., None]``: the carried operand of the
    # rescale names the accumulator.
    acc_rescales = [
        (acc_placeholder, user)
        for user in alpha_view.users
        for operands in (_attention_rescale_operands(user),)
        if operands is not None and alpha_view in operands
        for acc_placeholder in (
            carried_placeholder(arg)
            for arg in operands
            if carried_placeholder(arg) not in (None, m_placeholder, l_placeholder)
        )
    ]
    if len(acc_rescales) != 1:
        return False
    acc_placeholder, acc_rescaled = acc_rescales[0]
    assert acc_placeholder is not None

    pv_operands = _attention_pv_operands(pv_node)
    if pv_operands is None:
        return False
    acc_arg, p_arg, _ = pv_operands
    if acc_arg is not acc_rescaled:
        return False
    if _attention_pv_p_arg_base(p_arg) is not p_node:
        return False
    return loop_body_returns(
        {
            m_placeholder: max_state_node,
            l_placeholder: l_update,
            acc_placeholder: pv_node,
        }
    )


def _attention_online_softmax_exp_base(score_node: torch.fx.Node) -> str | None:
    for centered in score_node.users:
        if (
            centered.op != "call_function"
            or centered.target is not torch.ops.aten.sub.Tensor
            or len(centered.args) < 2
            or centered.args[0] is not score_node
        ):
            continue
        for user in centered.users:
            if user.op != "call_function":
                continue
            if user.target is torch.ops.aten.exp2.default:
                return "exp2"
            if user.target is torch.ops.aten.exp.default:
                return "exp"
    return None


def _attention_softmax_pattern_head_dim(
    graph: torch.fx.Graph,
    *,
    kv_block_id: int | None = None,
) -> AttentionSoftmaxPattern | None:
    """Config-independent online-softmax attention detector.

    Returns the score plan (head dim, scale, modifiers, operand layout) and the
    operand dtype when ``graph`` is a flash body: a QK matmul + PV matmul
    feeding an ``amax``/``exp2``/``sum`` online softmax with either no score
    masking or exactly ``where(tile_m.index >= tile_n.index, qk, -inf)``. Two
    source forms are read: the 3-D ``bmm`` + ``baddbmm`` pair over
    ``q[tile_b, tile_m, :]`` tiles, and the 2-D ``hl.dot`` pair over
    ``q[bh, tile_m, :]`` tiles with a scalar batch-head index. The softmax
    scale is either on Q (``(q * scale) @ k^T``) or on the scores
    (``(q @ k^T) * scale``); V is either the canonical ``v[b, tile_n, :]``
    tile or the pre-transposed ``v[b, :, tile_n]`` one (recorded as
    ``AttentionScorePlan.v_k_major``). Operands are fp16, bf16 or fp8 (e4m3);
    which of these facts a fused body hosts, and whether the 2-D form's
    scalar batch block is the root's, is the surface's question
    (``_attention_pattern_fits_root``), not the matcher's. Shared by
    ``_attention_loop_shape`` (the codegen detector) and the autotuner flash
    seed heuristic so the two never diverge on what counts as a flash kernel.
    """
    from ..cute_flash import FLASH_MATCHED_IO_DTYPES

    qk_nodes: list[torch.fx.Node] = []
    pv_nodes: list[torch.fx.Node] = []
    has_amax = has_exp = has_sum = False
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target
        matmul_form = _attention_matmul_form(node)
        if matmul_form == "qk":
            qk_nodes.append(node)
        elif matmul_form == "pv":
            pv_nodes.append(node)
        elif target is torch.ops.aten.amax.default:
            has_amax = True
        elif target in (torch.ops.aten.exp.default, torch.ops.aten.exp2.default):
            has_exp = True
        elif target is torch.ops.aten.sum.dim_IntList:
            has_sum = True
    if len(qk_nodes) != 1 or len(pv_nodes) != 1:
        return None
    qk_node = qk_nodes[0]
    pv_node = pv_nodes[0]
    # One source form: both matmuls 3-D aten ops, or both 2-D ``hl.dot``.
    if (qk_node.target is torch.ops.aten.bmm.dtype) != (
        pv_node.target is torch.ops.aten.baddbmm.default
    ):
        return None
    score_rank = 3 if qk_node.target is torch.ops.aten.bmm.dtype else 2
    if not (has_amax and has_exp and has_sum):
        return None
    qk_val = qk_node.meta.get("val")
    pv_val = pv_node.meta.get("val")
    if not isinstance(qk_val, torch.Tensor) or not isinstance(pv_val, torch.Tensor):
        return None
    # QK score block: ([batch,] tile_m, tile_n); PV output: ([batch,] tile_m, head_dim).
    if qk_val.ndim != score_rank or pv_val.ndim != score_rank:
        return None
    if qk_val.dtype != torch.float32 or pv_val.dtype != torch.float32:
        return None
    pv_operands = _attention_pv_operands(pv_node)
    if pv_operands is None:
        return None
    _, p_operand, v_operand = pv_operands
    # The Q/K/V operands share one dtype (fp16, bf16 or fp8). Checked via FX
    # node metadata (available at detection time, unlike the device-function
    # arguments).
    operand_nodes = [qk_node.args[0], qk_node.args[1], v_operand]
    operand_dtype: torch.dtype | None = None
    for arg in operand_nodes:
        if not isinstance(arg, torch.fx.Node):
            return None
        operand_val = arg.meta.get("val")
        if (
            not isinstance(operand_val, torch.Tensor)
            or operand_val.dtype not in FLASH_MATCHED_IO_DTYPES
        ):
            return None
        if operand_dtype is None:
            operand_dtype = operand_val.dtype
        elif operand_val.dtype != operand_dtype:
            return None
    if score_rank == 2:
        # ``hl.dot`` takes P at whatever dtype the kernel converts it to; the
        # fused body stages P in the operand dtype, so require that form.
        p_val = p_operand.meta.get("val")
        if not isinstance(p_val, torch.Tensor) or p_val.dtype != operand_dtype:
            return None
    head_dim_v = pv_val.shape[-1]
    if not isinstance(head_dim_v, int):
        # The loop graph's PV output carries the V/O width as an unbacked
        # symbol for the widths the shape environment does not keep concrete
        # (64 and 128 stay ints, 96 does not); V's host tensor keeps the
        # specialized width, as K's does for the Q/K width below.
        head_dim_v = _attention_v_load_head_dim(v_operand, score_rank)
        if head_dim_v is None:
            return None
    if not isinstance(qk_node.args[1], torch.fx.Node):
        return None
    k_indices = _attention_k_load_indices(
        qk_node.args[1], kv_block_id=kv_block_id, rank=score_rank
    )
    if k_indices is None:
        return None
    # The Q/K head dim comes from K's host tensor: the loop graph's operand
    # metadata carries the contraction as an unbacked symbol, while the host
    # tensor keeps the specialized width. A K load without a host tensor keeps
    # the historical symmetric reading (the scale check below still pins it).
    head_dim = _attention_k_load_head_dim(qk_node.args[1])
    if head_dim is None:
        head_dim = head_dim_v
    if not flash_head_dims_supported(head_dim, head_dim_v):
        return None
    v_indices = _attention_v_load_indices(
        v_operand, kv_block_id=kv_block_id, rank=score_rank
    )
    if v_indices is None:
        return None
    # K and V must read the same KV tile and the same (grouped) batch head:
    # one ``kv_b = tile_b.index // group`` reused for both, or the plain index.
    if (
        k_indices.batch_index is not v_indices.batch_index
        or k_indices.kv_tile is not v_indices.kv_tile
        or k_indices.kv_group != v_indices.kv_group
    ):
        return None
    # The 2-D form reads K/V at a scalar of one unit block (the surface
    # checks it is the root batch block, ``_attention_pattern_fits_root``);
    # the 3-D form at the inner batch tile the Q tile's leading dim is.
    if score_rank == 2 and k_indices.batch_block_id is None:
        return None
    q_placeholder = _attention_loop_carried_arg(graph, 0)
    if q_placeholder is None:
        return None
    q_arg = qk_node.args[0]
    if not isinstance(q_arg, torch.fx.Node):
        return None
    score_root = qk_node
    deferred_max: torch.fx.Node | None = None
    q_scale: float | sympy.Expr | None = _attention_arg_scaled_q_factor(
        q_arg, q_placeholder
    )
    if q_scale is None:
        # The post-dot form: an unscaled Q and ``qk * scale`` as the score
        # root. The fused body scales the fp32 scores in both forms.
        if not _attention_is_loop_carried_value(q_arg, q_placeholder):
            return None
        peeled = _attention_post_dot_scale(qk_node)
        if peeled is None:
            return None
        score_root, q_scale, deferred_max = peeled
    modifiers_result = _attention_score_modifiers(
        score_root,
        q_placeholder,
        kv_block_id=kv_block_id,
    )
    if modifiers_result is None:
        return None
    score_node, modifiers = modifiers_result
    # The deferred row max reads the unmodified scores, and a runtime scale
    # leaves the bias scales (relative to it) unknown at compile time: both
    # forms are admitted on the plain dense body only.
    if modifiers and (deferred_max is not None or not isinstance(q_scale, float)):
        return None
    if not _attention_exact_online_softmax_chain(
        score_node, pv_node, deferred_max=deferred_max
    ):
        return None
    exp_base = _attention_online_softmax_exp_base(score_node)
    if exp_base == "exp2":
        expected_scale = math.log2(math.e) / math.sqrt(head_dim)
        scale_multiplier = 1.0
        qk_scale_log2 = expected_scale
        bias_scale_log2 = 1.0
        lse_scale = 1.0
    elif exp_base == "exp":
        expected_scale = 1.0 / math.sqrt(head_dim)
        scale_multiplier = math.log2(math.e)
        qk_scale_log2 = math.log2(math.e) / math.sqrt(head_dim)
        bias_scale_log2 = math.log2(math.e)
        lse_scale = math.log(2.0)
    else:
        return None
    runtime_scale: sympy.Expr | None = None
    if isinstance(q_scale, float):
        if not math.isclose(q_scale, expected_scale, rel_tol=1e-5, abs_tol=1e-7):
            return None
    else:
        # A runtime scale: the body multiplies the scores by the kernel
        # scalar; the plan's log2 scale is the factor that puts the product
        # in the exp2 domain.
        runtime_scale = q_scale
        qk_scale_log2 = scale_multiplier
    additive_modifier_kinds = {
        TENSOR_BIAS_KIND,
        RELATIVE_BIAS_KIND,
        ALIBI_BIAS_KIND,
    }
    from ...compile_environment import CompileEnvironment

    fast_math = CompileEnvironment.current().settings.fast_math
    scaled_modifiers = tuple(
        dataclasses.replace(
            modifier,
            scale_log2=modifier.scale_log2 * bias_scale_log2,
        )
        if modifier.kind in additive_modifier_kinds
        else dataclasses.replace(
            modifier,
            value_log2=(
                None
                if modifier.value_log2 is None
                else modifier.value_log2 * bias_scale_log2
            ),
            fast_math=fast_math,
        )
        if modifier.kind == SOFTCAP_KIND
        else modifier
        for modifier in modifiers
    )
    score_plan = AttentionScorePlan(
        head_dim=head_dim,
        qk_scale_log2=qk_scale_log2,
        lse_scale=lse_scale,
        modifiers=scaled_modifiers,
        kv_group=k_indices.kv_group,
        head_dim_v=None if head_dim_v == head_dim else head_dim_v,
        v_k_major=v_indices.k_major,
        runtime_scale=runtime_scale,
    )
    if not score_plan.has_lowering():
        return None
    assert operand_dtype is not None
    return AttentionSoftmaxPattern(
        score_plan=score_plan,
        io_dtype=operand_dtype,
        score_rank=score_rank,
        batch_block_id=k_indices.batch_block_id,
        flattened=k_indices.flattened,
    )


def _flash_block_sizes_reachable(
    env: CompileEnvironment,
    targets: dict[int, int],
    *,
    overshoot_dimension: bool = False,
) -> bool:
    """True when every block id's fragment range can reach its flash target.

    ``overshoot_dimension``: the fused body tiles past the dimension's extent
    (ragged query rows), so the extent's cap on the range (``update_hint``)
    does not count against the target; only a declared ceiling
    (``BlockSizeSpec.declared_max_size``) does.
    """
    from ....autotuner.config_fragment import BlockSizeFragment

    if set(env.config_spec.block_sizes.valid_block_ids()) != set(targets):
        return False
    for block_id, target in targets.items():
        block_spec = env.config_spec.block_sizes.block_id_lookup(block_id)
        fragment = block_spec._fragment(env.config_spec)
        assert isinstance(fragment, BlockSizeFragment)
        high = fragment.high
        if overshoot_dimension:
            declared = block_spec.declared_max_size
            high = max(high, target if declared is None else declared)
        if not fragment.low <= target <= high:
            return False
    return True


def _attention_flash_block_size_targets(
    root_block_ids: Sequence[int], kv_block_id: int, *, tile: int
) -> dict[int, int]:
    """The block sizes a fused body runs: one batch head per program and
    ``tile``-row query and KV tiles. A root batch block pinned to one element
    (the 2-D form's ``hl.tile(..., block_size=1)`` or ``hl.grid`` axis) has no
    searchable fragment and is left out, as the flex surface leaves out its
    unit b/h axes; the 3-D form's autotuned batch tile is targeted at 1. The
    flattened-rows form's one root axis is its query tile.
    """
    targets: dict[int, int] = {}
    if len(root_block_ids) == 2 and not _attention_unit_fixed_block(
        int(root_block_ids[0])
    ):
        targets[int(root_block_ids[0])] = 1
    targets[int(root_block_ids[-1])] = tile
    targets[int(kv_block_id)] = tile
    return targets


def _attention_kv_tile_count(kv_block_id: int) -> int | None:
    """The 128-row KV tile count of the KV loop's block (None unless static).

    Counted at the default ``cute_flash_kv_tile_n`` whatever width a config
    picks: the fp8 parity gate (``flash_operand_facts_lowered``) runs on this
    count, and the other searched width (160 rows) is offered only to extents
    it divides, so with an even 128-row count (``seq % 256 == 0``) that
    extent is a multiple of 1280 and its 160-row count is even too.
    """
    from ...compile_environment import CompileEnvironment

    kv_seq = CompileEnvironment.current().block_sizes[kv_block_id].size
    if not isinstance(kv_seq, int):
        return None
    return (kv_seq + 127) // 128


def _attention_pattern_fits_root(
    pattern: AttentionSoftmaxPattern,
    root_block_ids: Sequence[int],
    *,
    kv_block_id: int | None = None,
) -> bool:
    """Whether a matched attention body may lower to a fused flash body under
    the root ``root_block_ids``: its operand facts have a body
    (``cute_flash.flash_operand_facts_lowered``, which for fp8 operands also
    needs the KV loop's tile count, read from ``kv_block_id``) and, in the
    2-D form, the K/V loads' scalar batch block is the root batch block the
    Q tile is read at (the 3-D form ties them through the Q tile's leading
    dim). The flattened-rows form tiles its rows on a one-axis root, which
    its K/V offset is taken from; the other forms read the Q tile under a
    (batch, query) root. Shared by the search surface and the codegen
    detector so the two never diverge.
    """
    from ..cute_flash import flash_operand_facts_lowered

    if not flash_operand_facts_lowered(
        pattern.io_dtype,
        v_k_major=pattern.v_k_major,
        score_rank=pattern.score_rank,
        num_kv=None if kv_block_id is None else _attention_kv_tile_count(kv_block_id),
    ):
        return False
    if (pattern.flattened is not None) != (len(root_block_ids) == 1):
        return False
    return pattern.batch_block_id is None or pattern.batch_block_id == int(
        root_block_ids[0]
    )


class _AttentionGeometry(NamedTuple):
    num_bh: int
    q_seq: int
    kv_seq: int


def _attention_pattern_geometry(
    pattern: AttentionSoftmaxPattern,
    root_block_ids: Sequence[int],
    kv_block_id: int,
) -> _AttentionGeometry | None:
    """The static batch-head count and query / KV lengths of a matched body
    under its root: the (batch, query) root's two blocks and the KV loop's,
    or, for the flattened-rows form, the root's ``B*H*M`` rows split by the
    K/V offset's ``M`` and the KV loop's extent, which the offset must
    stride K/V by. None unless every size is concrete."""
    from ...compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    rows = env.block_sizes[int(root_block_ids[0])].size
    kv_seq = env.block_sizes[int(kv_block_id)].size
    if not isinstance(rows, int) or not isinstance(kv_seq, int):
        return None
    if pattern.flattened is None:
        q_seq = env.block_sizes[int(root_block_ids[1])].size
        if not isinstance(q_seq, int):
            return None
        return _AttentionGeometry(rows, q_seq, kv_seq)
    q_seq = pattern.flattened.rows_per_group
    if rows % q_seq != 0 or pattern.flattened.group_stride != kv_seq:
        return None
    return _AttentionGeometry(rows // q_seq, q_seq, kv_seq)


def detect_flash_search_surface(device_ir: DeviceIR) -> FlashSearchSurface | None:
    """Config-independent flash detector for the autotune search surface.

    Returns the flash head_dim, KV tile count, and block-size targets when
    ``device_ir`` is inside the same static shape envelope as the fused tcgen05
    path: square fp16/bf16 self-attention, concrete ``seq % 128 == 0``, and reachable
    ``block_sizes=[1, 128, 128]`` for ``(tile_b, tile_m, tile_n)`` (``[128, 128]``
    for the flattened-rows form's ``(tile_m, tile_n)``). Keeping this strict
    prevents the autotuner from benchmarking configs that can only fall back
    to the scalar path after the flash knobs have been added.
    """
    from ...compile_environment import CompileEnvironment
    from ...device_ir import ForLoopGraphInfo

    if not _attention_flash_gate_enabled() or not _attention_flash_supported():
        return None
    if len(device_ir.grid_block_ids) != 1 or not root_grid_origin_is_canonical(
        device_ir
    ):
        return None
    root_grid_ids = device_ir.grid_block_ids[0]
    env = CompileEnvironment.current()
    # The FlexAttention envelope binds to a 3-axis root grid (tile_b, tile_h,
    # tile_m) with unit b/h tiles; the collapsed envelope below to a 2-axis
    # grid (tile_b, tile_m), or the flattened-rows form's one axis over
    # ``B*H*M`` rows, plus one inner tile_n device loop.
    if len(root_grid_ids) == 3:
        return _detect_flex_search_surface(device_ir, root_grid_ids, env)
    if len(root_grid_ids) not in (1, 2):
        return None

    def block_sizes_reachable(targets: dict[int, int]) -> bool:
        return _flash_block_sizes_reachable(env, targets)

    flash_surface: FlashSearchSurface | None = None
    generic_fallback_required = False
    generic_fallback_targets: dict[int, int] | None = None
    for graph_info in device_ir.graphs:
        if not isinstance(graph_info, ForLoopGraphInfo):
            continue
        block_ids = graph_info.block_ids
        if len(block_ids) != 1 or any(bid in root_grid_ids for bid in block_ids):
            continue
        pattern = _attention_softmax_pattern_head_dim(
            graph_info.graph,
            kv_block_id=block_ids[0],
        )
        if pattern is None:
            continue
        # Operand facts no fused body hosts (fp8 operands on an odd KV tile
        # count or with the canonical V) or a K/V batch index off the root:
        # the generic lowering keeps the kernel, the codegen detector
        # (``_attention_loop_shape``) declines the same way.
        if not _attention_pattern_fits_root(
            pattern, root_grid_ids, kv_block_id=block_ids[0]
        ):
            continue
        from ..cute_flash import _flash_output_requires_tma
        from ..cute_flash import flash_graph_output_facts
        from ..cute_flash import flash_tmem_rowmax_score_plan_supported

        facts = flash_graph_output_facts(
            device_ir.graphs,
            root_block_ids=root_grid_ids,
            kv_block_id=block_ids[0],
            score_plan=pattern.score_plan,
        )
        if facts is None or not facts.lse_plan_valid:
            continue
        has_row_epilogue = facts.has_row_epilogue
        tensor_4d_batch_heads = facts.tensor_4d_batch_heads
        geometry = _attention_pattern_geometry(pattern, root_grid_ids, block_ids[0])
        if geometry is None or geometry.q_seq != geometry.kv_seq:
            continue
        num_bh, q_seq, kv_seq = geometry
        if q_seq % 128 != 0:
            continue
        num_kv = (kv_seq + 127) // 128
        output_requires_tma = _flash_output_requires_tma(
            num_bh, q_seq, pattern.head_dim_v
        )
        # Only the plain FA4 topology (two 128-row query tiles per CTA, direct
        # output store: staged O does not fit next to 2 x 192-wide Q tiles)
        # hosts the asymmetric head-dim pair.
        if pattern.score_plan.asymmetric_head_dims and (
            num_kv % 2 != 0
            or pattern.score_plan.requires_ws_overlap
            or output_requires_tma
        ):
            continue
        # TMA output is currently FA4-only. Do not expose a flash search whose
        # only legal output path is unavailable; the generic CuTe search can
        # still handle these uncommon large-output shapes.
        if output_requires_tma and (
            num_kv % 2 != 0 or pattern.score_plan.requires_ws_overlap
        ):
            generic_fallback_required = True
            fallback_targets = _attention_flash_block_size_targets(
                root_grid_ids, block_ids[0], tile=64
            )
            if generic_fallback_targets is None and block_sizes_reachable(
                fallback_targets
            ):
                generic_fallback_targets = fallback_targets
            continue
        block_size_targets = _attention_flash_block_size_targets(
            root_grid_ids, block_ids[0], tile=128
        )
        if not block_sizes_reachable(block_size_targets):
            continue
        if flash_surface is None:
            flash_surface = FlashSearchSurface(
                head_dim=pattern.head_dim,
                num_kv=num_kv,
                num_bh=num_bh,
                tensor_4d_heads=(
                    tensor_4d_batch_heads[1]
                    if tensor_4d_batch_heads is not None
                    else None
                ),
                io_dtype=pattern.io_dtype,
                block_size_targets=block_size_targets,
                is_causal=pattern.is_causal,
                has_kv_tile_pruning=pattern.score_plan.has_kv_tile_pruning,
                requires_ws_overlap=pattern.score_plan.requires_ws_overlap,
                small_biased_candidate=facts.small_biased_candidate,
                standard_dense_output=facts.standard_dense_output,
                dense_lse_output=facts.dense_lse_output,
                standard_causal_output=facts.standard_causal_output,
                output_requires_tma=output_requires_tma,
                supports_tensor_4d_tma=facts.supports_tensor_4d_tma,
                has_row_epilogue=has_row_epilogue,
                # The 64-row query tile exists only for plain rows.
                plain_row_body=(
                    not pattern.score_plan.modifiers and not has_row_epilogue
                ),
                # The row programs take a fused row epilogue but no modifier.
                has_score_modifiers=bool(pattern.score_plan.modifiers),
                # Equality and full 128-row coverage were established above.
                causal_resident_compatible=q_seq % 256 == 0,
                tmem_rowmax_compatible=flash_tmem_rowmax_score_plan_supported(
                    pattern.score_plan
                ),
                score_modifier_kinds=pattern.score_plan.modifier_kinds,
                head_dim_v=pattern.score_plan.head_dim_v,
                kv_group=pattern.score_plan.kv_group,
                has_list_walk=pattern.score_plan.has_list_walk,
            )
    if generic_fallback_required:
        # Imported here: ``cute/backend.py`` imports this module at import time,
        # before the autotune surface (which pulls in the runtime) can load.
        from ..autotune.flash_surface import enable_cute_attention_generic_fallback

        enable_cute_attention_generic_fallback(
            env.config_spec, block_size_targets=generic_fallback_targets
        )
        return None
    if flash_surface is None:
        flash_surface = _detect_flex_search_surface(device_ir, root_grid_ids, env)
    return flash_surface


def _detect_flex_search_surface(
    device_ir: DeviceIR,
    root_grid_ids: Sequence[int],
    env: CompileEnvironment,
) -> FlashSearchSurface | None:
    """The flex-envelope surface: the whole-graph-set matcher on
    ``device_ir.graphs`` (never cache its node identities; codegen re-runs
    it on the graph copies)."""
    from ..flex_match import FLEX_PLAN_ENABLED
    from ..flex_match import match_flex_envelope

    if not FLEX_PLAN_ENABLED:
        return None
    match = match_flex_envelope(device_ir.graphs, root_grid_ids, env=env)
    if match is None:
        return None
    return flex_search_surface(match, env)


def flex_search_surface(
    match: FlexMatch, env: CompileEnvironment
) -> FlashSearchSurface | None:
    """Build the autotune surface of a matched flex kernel.

    Only the tunable (tile_m, tile_n) ids enter ``block_size_targets`` (the
    unit b/h axes are ``FixedBlockSizeSource(1)``, which the matcher already
    verified; ``block_id_lookup`` has no entry for them). ``num_kv`` is the
    upper bound ``cdiv(kv_seq, 128)`` of the list walk. The 128-row pins
    hold for every ``q_seq >= 1``: the emitter row-guards a ragged last tile,
    so a query length below 128 (which caps the generic range at its extent)
    is not a decline; a declared ceiling below 128 is.
    """
    from ..cute_flash import _flash_flex_coalesce_steps
    from ..cute_flash import _flash_flex_hook_ops
    from ..cute_flash import _flash_flex_prefetch_regs
    from ..cute_flash import _flash_output_requires_tma
    from ..cute_flash import flash_tmem_rowmax_score_plan_supported
    from ..flash_score_mod import flex_aux_smem_candidates

    plan = match.plan
    num_bh = match.num_bh
    # LSE uses a direct layout with signed Int32 element coordinates.
    if match.lse_name is not None and num_bh * match.q_seq > 1 << 31:
        return None
    block_size_targets = {match.q_block_id: 128, match.kv_block_id: 128}
    if not _flash_block_sizes_reachable(
        env, block_size_targets, overshoot_dimension=True
    ):
        return None
    if plan.flex_dense:
        return _flex_dense_search_surface(match, block_size_targets)
    return FlashSearchSurface(
        head_dim=match.head_dim,
        num_kv=match.num_kv,
        num_bh=num_bh,
        tensor_4d_heads=match.heads,
        io_dtype=match.io_dtype,
        block_size_targets=block_size_targets,
        is_causal=False,
        has_kv_tile_pruning=False,
        requires_ws_overlap=False,
        small_biased_candidate=False,
        standard_dense_output=False,
        dense_lse_output=False,
        standard_causal_output=False,
        output_requires_tma=_flash_output_requires_tma(
            num_bh, match.q_seq, match.head_dim_v
        ),
        # The flex emitter reads its 4-D operands directly; the collapsed
        # path's reshape-view TMA proof does not apply.
        supports_tensor_4d_tma=False,
        # The hardware row max reduces the list walk's full tiles (raw
        # scores); a score region runs the hook on every tile.
        tmem_rowmax_compatible=flash_tmem_rowmax_score_plan_supported(plan),
        has_row_epilogue=False,
        # The hook bodies exist only in the 128-row bodies.
        plain_row_body=False,
        # Mask-only plans keep the dense body on full tiles: only a score
        # region (or an unpeeled scale) makes every tile run the hook.
        has_score_modifiers=plan.flex_has_score_region,
        score_modifier_kinds=plan.modifier_kinds,
        head_dim_v=plan.head_dim_v,
        kv_group=plan.kv_group,
        has_list_walk=plan.has_list_walk,
        flex_has_score_region=plan.flex_has_score_region,
        flex_has_mask_region=plan.flex_has_mask_region,
        flex_trivial_walk=not plan.has_list_walk,
        q_seq=match.q_seq,
        kv_seq=match.kv_seq,
        flex_aux_smem_candidates=flex_aux_smem_candidates(plan),
        flex_prefetch_regs=_flash_flex_prefetch_regs(
            plan, num_bh, match.heads or num_bh, match.num_kv
        ),
        flex_coalesce_steps=_flash_flex_coalesce_steps(
            plan, num_bh, match.heads or num_bh, match.num_kv
        ),
        flex_hook_ops=_flash_flex_hook_ops(
            plan, num_bh, match.heads or num_bh, match.num_kv
        ),
    )


def _flex_dense_search_surface(
    match: FlexMatch, block_size_targets: dict[int, int]
) -> FlashSearchSurface:
    """The surface of a dense flex plan (``AttentionScorePlan.flex_dense``):
    the dense attention surface over the flex kernel's two tunable tiles.

    The plan is dense attention (the trivial walk, no regions, the dense
    geometry), so it takes the dense facts the collapsed attention kernels
    report -- every family and knob of the dense surface, the row programs
    on small grids, the hardware row max -- and none of the flex envelope's
    (no list walk, no trivial-walk pins, no hook facts).  The facts the dense
    bodies cannot prove for the flex operands stay off, like the codegen
    detector's: the 4-D TMA family (the flex emitter reads its 4-D operands
    directly), the fused row epilogue and the small biased candidate; the
    standard output-only lowerings need no LSE and the standard scale
    (``_standard_dense_score_plan_supported``), the same test the codegen
    applies; the flex kernel returns the LSE and applies its scale as a
    scalar argument, so it is the dense LSE surface (``dense_lse_output``,
    scale-agnostic) the resident value graph is offered on.
    """
    from ..cute_flash import _flash_output_requires_tma
    from ..cute_flash import _standard_dense_score_plan_supported
    from ..cute_flash import flash_tmem_rowmax_score_plan_supported

    plan = match.plan
    num_bh = match.num_bh
    return FlashSearchSurface(
        head_dim=match.head_dim,
        num_kv=match.num_kv,
        num_bh=num_bh,
        tensor_4d_heads=match.heads,
        io_dtype=match.io_dtype,
        block_size_targets=block_size_targets,
        is_causal=False,
        has_kv_tile_pruning=False,
        requires_ws_overlap=False,
        small_biased_candidate=False,
        standard_dense_output=(
            match.lse_name is None and _standard_dense_score_plan_supported(plan)
        ),
        # The flex kernel applies its scale as a scalar argument, so the
        # output-only fact's standard-scale test fails here by construction;
        # the LSE fact asks only for the dense plan and the ``O`` + LSE epilogue.
        dense_lse_output=match.lse_name is not None,
        standard_causal_output=False,
        output_requires_tma=_flash_output_requires_tma(
            num_bh, match.q_seq, match.head_dim_v
        ),
        supports_tensor_4d_tma=False,
        has_row_epilogue=False,
        plain_row_body=True,
        has_score_modifiers=False,
        causal_resident_compatible=False,
        tmem_rowmax_compatible=flash_tmem_rowmax_score_plan_supported(plan),
        score_modifier_kinds=plan.modifier_kinds,
        head_dim_v=plan.head_dim_v,
        kv_group=plan.kv_group,
        q_seq=match.q_seq,
        kv_seq=match.kv_seq,
    )
