"""CuTe flash-attention BACKWARD path: pattern matching and emission.

Recognizes the fused attention-backward Helion kernel and replaces the
generic scalar lowering with an FA4-style fused tcgen05 kernel
(flash-attention main's ``flash_bwd_sm100`` schedule).

The matched Helion source shape (see ``benchmarks/cute/attnbwd_kernels.py``
``attention_bwd`` / ``attention_bwd_causal``):

- Root grid: one tile over the flattened KV rows (``[G*n_dim]``), loading a
  K tile and a V tile, carrying fp32 dK/dV accumulators through an inner
  ``_for_loop`` over the matching Q rows, then storing dV and dK*scale in
  the io dtype.
- Inner loop body: five ``hl.dot`` ops —
  ``S^T = K @ Q^T`` and ``dP^T = V @ dO^T`` (fp32 out),
  ``dV += P^T @ dO`` and ``dK += dS^T? @ Q`` (accumulating), and
  ``dQ = dS^T @ K`` — plus the softmax recompute
  ``P = exp2(S^T * qk_scale2 - lse2[None, :])`` (lse2 = base-2 LSE row
  vector), ``dS = (P * (dP^T - delta[None, :])).to(io)``, and an
  ``hl.atomic_add(dq_fp32, [tile_m, :], dQ * dq_scale)``; the root stores
  ``dK * dk_scale``. Each scale is a host scalar or a Python number, and may
  be omitted (1): e.g. examples/attention.py pre-scales K by
  ``sm_scale * log2(e)`` and rescales dQ by ``ln(2)``.
- Causal variants mask ``S^T`` with
  ``where(q_pos[None, :] >= kv_pos[:, None], s, -inf)`` where
  ``q_pos = tile_index(tile_m) % m_dim`` / ``kv_pos = tile_index(tile_n) %
  n_dim``, and start the inner loop at the (block-floored) diagonal.
  GQA arrives pre-reshaped: the Q rows of one KV group are ``group`` blocks
  of ``m_dim`` rows each, so the same mask modulus covers grouped queries.

Emission follows flash-attention main's SM100 backward: 512 threads /
16 warps (4 dQ-reduce, 8 compute, 1 MMA, 1 load), TMEM-resident dK/dV,
P/dS written back over S/dP in TMEM, and dQ accumulated into the row-major
fp32 buffer with ``cp.reduce.async.bulk``.

``cute_flash_bwd_family`` = "row_mma" (``cute_flash_bwd_row_mma.py``) swaps
this body for register-MMA row programs on small grids, where the per-CTA
fixed cost of the tcgen05 kernel dominates; the surface, matcher and host
tensors are shared.
"""

from __future__ import annotations

import dataclasses
import os
import typing
from typing import TYPE_CHECKING

import torch

from ..emitter_protocols import source_adapter
from .analysis.match_common import host_tensor_load
from .analysis.match_common import is_call
from .analysis.match_common import split_scale
from .analysis.match_common import strip_new_var
from .autotune.flash_surface import enable_cute_flash_bwd_search
from .cute_flash_bwd_common import FBWD_2CTA_DS_OFFSETS
from .cute_flash_bwd_common import FBWD_2CTA_SEND
from .cute_flash_bwd_common import FBWD_2CTA_SMEM_VIEWS
from .cute_flash_bwd_common import FBWD_CLC_CONSUME
from .cute_flash_bwd_common import FBWD_COMPUTE_LOAD_VIEWS
from .cute_flash_bwd_common import FBWD_MBAR_INITS
from .cute_flash_bwd_common import FBWD_SMEM_VIEWS
from .cute_flash_bwd_common import FLASH_BWD_EXP2_F32_KEY
from .cute_flash_bwd_common import FLASH_BWD_PERSISTENT_KEY
from .cute_flash_bwd_common import FLASH_BWD_TWO_CTA_KEY
from .cute_flash_bwd_common import fbwd_2cta_mma_setup
from .cute_flash_bwd_common import fbwd_2cta_operand_partitions
from .cute_flash_bwd_common import fbwd_clc_consume_only
from .cute_flash_bwd_common import fbwd_clc_scheduler
from .cute_flash_bwd_common import fbwd_clc_setup
from .cute_flash_bwd_common import fbwd_compute_tmem_views
from .cute_flash_bwd_common import fbwd_dq_reduce_partitions
from .cute_flash_bwd_common import fbwd_dq_reduce_views
from .cute_flash_bwd_common import fbwd_epilogue_block
from .cute_flash_bwd_common import fbwd_mma_descriptor_setup
from .cute_flash_bwd_common import fbwd_mma_fragments
from .cute_flash_bwd_common import fbwd_mma_step_loop_head
from .cute_flash_bwd_common import fbwd_mma_tile_entry
from .cute_flash_bwd_common import fbwd_p_store_views
from .cute_flash_bwd_common import fbwd_tile_loop
from .cute_flash_bwd_common import fbwd_tmem_warp
from .state import cute_codegen_state

if TYPE_CHECKING:
    import sympy

    from ..device_function import TensorArg
    from ..device_ir import DeviceIR
    from ..generate_ast import GenerateAST
    from .flex_bwd_match import FlexBwdPlan
    from .flex_bwd_match import FlexBwdTmemPlan

__all__ = [
    "AttentionBwdMatch",
    "AttentionBwdPlan",
    "detect_flash_bwd_search_surface",
    "match_attention_bwd",
]


def _flash_bwd_gate_enabled() -> bool:
    return os.environ.get("HELION_CUTE_FLASH_BWD", "1") == "1"


@dataclasses.dataclass(frozen=True)
class AttentionBwdPlan:
    """Config-independent facts about a matched backward-attention kernel."""

    head_dim: int
    io_dtype: torch.dtype
    causal: bool
    m_dim: int  # causal mask modulus (query rows per group repeat)
    mm_dim: int  # query rows per KV group (group * m_dim)
    n_dim: int  # KV rows per group
    total_kv_rows: int
    qk_scale_expr: str  # host expression (or literal) for the exp2-domain scale
    out_scale_expr: str  # host expression (or literal) for the dQ output scale
    dk_scale_expr: str  # host expression (or literal) for the dK output scale

    @property
    def group(self) -> int:
        return self.mm_dim // self.m_dim

    @property
    def num_groups(self) -> int:
        return self.total_kv_rows // self.n_dim


@dataclasses.dataclass(frozen=True)
class AttentionBwdMatch:
    """A matched kernel: the plan plus the host-tensor roles for codegen."""

    plan: AttentionBwdPlan
    # Host-tensor names (as seen by ``_host_tensor``) for each operand role.
    q_name: str
    k_name: str
    v_name: str
    do_name: str
    lse_name: str
    delta_name: str
    dq_name: str
    dk_name: str
    dv_name: str
    # The runtime scales (exp2 domain, dQ, dK): sympy expressions (host
    # symbols) captured from the traced SymFloat metadata so symbol identity is
    # preserved for ``DeviceFunction.sympy_expr`` rendering at codegen time, or
    # Python floats for constants in the graph (an omitted scale is 1.0).
    qk_scale_sym: sympy.Expr | float | None = None
    out_scale_sym: sympy.Expr | float | None = None
    dk_scale_sym: sympy.Expr | float | None = None


def _node_target_name(node: torch.fx.Node) -> str:
    target = node.target
    return getattr(target, "__name__", str(target))


def _is_permute_of(node: object, source: torch.fx.Node) -> bool:
    return (
        is_call(node, torch.ops.aten.permute.default)
        and node.args[0] is source  # type: ignore[union-attr]
        and node.args[1] == [1, 0]  # type: ignore[union-attr]
    )


def _match_causal_mask(
    node: torch.fx.Node,
) -> tuple[torch.fx.Node, int, int] | None:
    """Match ``where(q_pos[None, :] >= kv_pos[:, None], s, -inf)``.

    Returns (masked score source node, m_dim, n_dim) or None.
    """
    from ...language import tile_ops
    from ...language import view_ops

    if node.target is not torch.ops.aten.where.self:
        return None
    cond, score, fill = node.args
    if not (
        isinstance(fill, torch.fx.Node)
        and fill.target is torch.ops.aten.scalar_tensor.default
        and fill.args[0] == float("-inf")
    ):
        return None
    if not (
        isinstance(cond, torch.fx.Node) and cond.target is torch.ops.aten.ge.Tensor
    ):
        return None
    lhs, rhs = cond.args

    def _pos_modulus(sub: object, index: list[object]) -> int | None:
        if not (
            isinstance(sub, torch.fx.Node)
            and sub.target is view_ops.subscript
            and sub.args[1] == index
        ):
            return None
        rem = sub.args[0]
        if not (
            isinstance(rem, torch.fx.Node)
            and rem.target is torch.ops.aten.remainder.Scalar
            and isinstance(rem.args[1], int)
        ):
            return None
        idx = rem.args[0]
        if not (isinstance(idx, torch.fx.Node) and idx.target is tile_ops.tile_index):
            return None
        return typing.cast("int", rem.args[1])

    m_dim = _pos_modulus(lhs, [None, slice(None)])
    n_dim = _pos_modulus(rhs, [slice(None), None])
    if m_dim is None or n_dim is None:
        return None
    if not isinstance(score, torch.fx.Node):
        return None
    return score, m_dim, n_dim


def _match_bwd_inner_graph(
    graph: torch.fx.Graph,
) -> dict[str, object] | None:
    """Match the inner Q-loop body of the fused backward kernel.

    Returns a dict of structural facts or None. Keys: q_name, do_name,
    lse_name, delta_name, dq_name, io_dtype, head_dim, qk_scale_expr,
    out_scale_expr, causal, m_dim, n_dim.
    """
    from ...language import matmul_ops
    from ...language.atomic_ops import atomic_add

    dot_nodes = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target is matmul_ops.dot
    ]
    if len(dot_nodes) != 5:
        return None
    atomic_nodes = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target is atomic_add
    ]
    if len(atomic_nodes) != 1:
        return None
    atomic_node = atomic_nodes[0]

    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    if len(placeholders) < 4:
        return None
    # Loop-carried args: (k_j, v_j, dv_acc, dk_acc) wrapped in _new_var.
    k_arg, v_arg, dv_arg, dk_arg = placeholders[:4]

    def _uses_placeholder(node: object, placeholder: torch.fx.Node) -> bool:
        return strip_new_var(node) is placeholder

    # Classify dots.
    qk_dot = dp_dot = dv_dot = dq_dot = dk_dot = None
    for node in dot_nodes:
        a, b, acc = node.args[0], node.args[1], node.args[2]
        if acc is not None:
            if _uses_placeholder(acc, dv_arg):
                dv_dot = node
            elif _uses_placeholder(acc, dk_arg):
                dk_dot = node
            continue
        if _uses_placeholder(a, k_arg) and node.args[3] == torch.float32:
            if (
                isinstance(b, torch.fx.Node)
                and b.target is torch.ops.aten.permute.default
            ):
                qk_dot = node
        elif _uses_placeholder(a, v_arg) and node.args[3] == torch.float32:
            dp_dot = node
        elif (
            isinstance(a, torch.fx.Node)
            and a.target is torch.ops.aten.permute.default
            and _uses_placeholder(b, k_arg)
            and node.args[3] == torch.float32
        ):
            dq_dot = node
    if None in (qk_dot, dp_dot, dv_dot, dq_dot, dk_dot):
        return None
    assert qk_dot is not None and dp_dot is not None and dv_dot is not None
    assert dq_dot is not None and dk_dot is not None

    # Q and dO tile loads feed the permutes of qk/dp.
    q_load = host_tensor_load(
        typing.cast("torch.fx.Node", qk_dot.args[1]).args[0], index_len=2
    )
    dp_permute = dp_dot.args[1]
    if not (
        isinstance(dp_permute, torch.fx.Node)
        and dp_permute.target is torch.ops.aten.permute.default
    ):
        return None
    do_load = host_tensor_load(dp_permute.args[0], index_len=2)
    if q_load is None or do_load is None:
        return None
    q_name, _ = q_load
    do_name, _ = do_load

    # Softmax recompute chain: p_t = exp2(score * qk_scale - lse[None, :]).
    p_t = None
    for node in graph.nodes:
        if node.op == "call_function" and node.target is torch.ops.aten.exp2.default:
            p_t = node
            break
    if p_t is None:
        return None
    sub_node = p_t.args[0]
    if not (
        isinstance(sub_node, torch.fx.Node)
        and sub_node.target is torch.ops.aten.sub.Tensor
    ):
        return None
    scaled_score, lse_bcast = sub_node.args
    split = split_scale(scaled_score)
    if split is None:
        return None
    score_node, (qk_scale_sympy, qk_scale_expr) = split
    causal = False
    m_dim = n_dim = None
    if isinstance(score_node, torch.fx.Node) and score_node is not qk_dot:
        masked = _match_causal_mask(score_node)
        if masked is None:
            return None
        score_node, m_dim, n_dim = masked
        causal = True
    if score_node is not qk_dot:
        return None
    # lse row broadcast: subscript(load(lse, [tile_m]), [None, :]).
    if not isinstance(lse_bcast, torch.fx.Node):
        return None
    from ...language import view_ops

    if lse_bcast.target is not view_ops.subscript or lse_bcast.args[1] != [
        None,
        slice(None),
    ]:
        return None
    lse_load = host_tensor_load(lse_bcast.args[0], index_len=1)
    if lse_load is None:
        return None
    lse_name, _ = lse_load

    # dS chain: ds = convert(p_t * (dp_t - delta[None, :]), io_dtype).
    ds_node = None
    delta_name = None
    for node in graph.nodes:
        if (
            node.op == "call_function"
            and node.target is torch.ops.aten.mul.Tensor
            and node.args[0] is p_t
        ):
            inner_sub = node.args[1]
            if not (
                isinstance(inner_sub, torch.fx.Node)
                and inner_sub.target is torch.ops.aten.sub.Tensor
                and inner_sub.args[0] is dp_dot
            ):
                continue
            delta_bcast = inner_sub.args[1]
            if not (
                isinstance(delta_bcast, torch.fx.Node)
                and delta_bcast.target is view_ops.subscript
                and delta_bcast.args[1] == [None, slice(None)]
            ):
                continue
            delta_load = host_tensor_load(delta_bcast.args[0], index_len=1)
            if delta_load is None:
                continue
            delta_name = delta_load[0]
            for user in node.users:
                if user.target is torch.ops.prims.convert_element_type.default:
                    ds_node = user
                    break
            break
    if ds_node is None or delta_name is None:
        return None
    io_dtype = ds_node.args[1]
    if io_dtype not in (torch.float16, torch.bfloat16):
        return None

    # dV dot consumes convert(p_t) and do_i; dK dot consumes ds and q_i;
    # dQ dot consumes permute(ds).
    dv_a = dv_dot.args[0]
    if not (
        isinstance(dv_a, torch.fx.Node)
        and dv_a.target is torch.ops.prims.convert_element_type.default
        and dv_a.args[0] is p_t
        and dv_a.args[1] == io_dtype
    ):
        return None
    if dk_dot.args[0] is not ds_node:
        return None
    if not _is_permute_of(dq_dot.args[0], ds_node):
        return None

    # atomic_add(dq, [tile_m, :], dq_dot * out_scale, 'relaxed')
    from ...language._tracing_ops import _host_tensor

    dq_base = atomic_node.args[0]
    if not (isinstance(dq_base, torch.fx.Node) and dq_base.target is _host_tensor):
        return None
    dq_name = dq_base.args[0]
    assert isinstance(dq_name, str)
    index = atomic_node.args[1]
    if not (
        isinstance(index, (list, tuple)) and len(index) == 2 and index[1] == slice(None)
    ):
        return None
    split = split_scale(atomic_node.args[2])
    if split is None or split[0] is not dq_dot:
        return None
    out_scale_sympy, out_scale_expr = split[1]

    # head_dim from the dv dot output metadata.
    dv_val = dv_dot.meta.get("val")
    if not isinstance(dv_val, torch.Tensor) or dv_val.ndim != 2:
        return None
    head_dim = dv_val.shape[1]
    if not isinstance(head_dim, int) or head_dim not in (64, 128):
        return None

    return {
        "q_name": q_name,
        "do_name": do_name,
        "lse_name": lse_name,
        "delta_name": delta_name,
        "dq_name": dq_name,
        "io_dtype": io_dtype,
        "head_dim": head_dim,
        "qk_scale_expr": qk_scale_expr,
        "out_scale_expr": out_scale_expr,
        "qk_scale_sym": qk_scale_sympy,
        "out_scale_sym": out_scale_sympy,
        "causal": causal,
        "m_dim": m_dim,
        "n_dim": n_dim,
    }


def _match_bwd_root_graph(
    graph: torch.fx.Graph,
    *,
    inner_facts: dict[str, object],
) -> dict[str, object] | None:
    """Match the root KV-grid graph. Returns k/v/dk/dv names + loop dims."""
    from ...language import memory_ops
    from ...language._tracing_ops import _for_loop
    from ...language._tracing_ops import _host_tensor
    from ...language._tracing_ops import _phi
    from ...language.creation_ops import full
    from ...language.tile_ops import tile_begin

    for_nodes = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target is _for_loop
    ]
    if len(for_nodes) != 1:
        return None
    for_node = for_nodes[0]
    begins, ends, carried = for_node.args[1], for_node.args[2], for_node.args[3]
    if not (
        isinstance(begins, (list, tuple))
        and isinstance(ends, (list, tuple))
        and isinstance(carried, (list, tuple))
        and len(begins) == 1
        and len(ends) == 1
        and len(carried) == 4
    ):
        return None
    k_load_node, v_load_node, dv_full, dk_full = carried

    k_load = host_tensor_load(k_load_node, index_len=2)
    v_load = host_tensor_load(v_load_node, index_len=2)
    if k_load is None or v_load is None:
        return None
    k_name, k_block = k_load
    v_name, _ = v_load
    for acc in (dv_full, dk_full):
        if not (isinstance(acc, torch.fx.Node) and acc.target is full):
            return None
        if acc.args[2] is not torch.float32:
            return None

    # Loop bound arithmetic:
    #   base = tile_begin // n_dim * mm_dim ; end = base + mm_dim
    #   begin = end - mm_dim (dense) or base + (tile_begin % n_dim // bs * bs)
    end_node = ends[0]
    if not (
        isinstance(end_node, torch.fx.Node)
        and _node_target_name(end_node) == "add"
        and isinstance(end_node.args[1], int)
    ):
        return None
    mm_dim = end_node.args[1]
    base_node = end_node.args[0]
    if not (
        isinstance(base_node, torch.fx.Node)
        and _node_target_name(base_node) == "mul"
        and base_node.args[1] == mm_dim
    ):
        return None
    div_node = base_node.args[0]
    if not (
        isinstance(div_node, torch.fx.Node)
        and _node_target_name(div_node) == "floordiv"
        and isinstance(div_node.args[1], int)
    ):
        return None
    n_dim = div_node.args[1]
    begin_source = div_node.args[0]
    if not (
        isinstance(begin_source, torch.fx.Node) and begin_source.target is tile_begin
    ):
        return None

    begin_node = begins[0]
    causal_begin = False
    if begin_node is base_node:
        pass  # dense: starts at the group base
    elif (
        isinstance(begin_node, torch.fx.Node)
        and _node_target_name(begin_node) == "add"
        and begin_node.args[0] is base_node
    ):
        # diag = base + (tile_begin % n_dim) // block_m * block_m
        diag = begin_node.args[1]
        if not (isinstance(diag, torch.fx.Node) and _node_target_name(diag) == "mul"):
            return None
        causal_begin = True
    else:
        return None

    # dv / dk stores: store(dv, [kv_block, :], convert(phi(dv_acc))) and
    # store(dk, [kv_block, :], convert(phi(dk_acc) * dk_scale)). The emitted
    # epilogue scales dK only, so each store must read its own accumulator.
    store_nodes = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target is memory_ops.store
    ]
    if len(store_nodes) != 2:
        return None
    dv_name = dk_name = None
    dk_scale = None
    for store_node in store_nodes:
        base = store_node.args[0]
        if not (isinstance(base, torch.fx.Node) and base.target is _host_tensor):
            return None
        value = store_node.args[2]
        if not (
            isinstance(value, torch.fx.Node)
            and value.target is torch.ops.prims.convert_element_type.default
        ):
            return None
        split = split_scale(value.args[0])
        if split is None:
            return None
        phi, scale = split
        if not (isinstance(phi, torch.fx.Node) and phi.target is _phi):
            return None
        if phi.args[0] is dv_full and scale[0] == 1.0:
            dv_name = base.args[0]
        elif phi.args[0] is dk_full:
            dk_name = base.args[0]
            dk_scale = scale
        else:
            return None
    if not (
        isinstance(dv_name, str) and isinstance(dk_name, str) and dk_scale is not None
    ):
        return None

    inner_causal = bool(inner_facts["causal"])
    if inner_causal != causal_begin:
        return None
    if inner_causal:
        if inner_facts["n_dim"] != n_dim:
            return None
        if (
            not isinstance(inner_facts["m_dim"], int)
            or mm_dim % typing.cast("int", inner_facts["m_dim"]) != 0
        ):
            return None

    return {
        "k_name": k_name,
        "v_name": v_name,
        "dk_name": dk_name,
        "dv_name": dv_name,
        "mm_dim": mm_dim,
        "n_dim": n_dim,
        "dk_scale_sym": dk_scale[0],
        "dk_scale_expr": dk_scale[1],
    }


def match_attention_bwd(device_ir: DeviceIR) -> AttentionBwdMatch | None:
    """Config-independent matcher for the fused attention-backward kernel."""
    from ..compile_environment import CompileEnvironment
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import RootGraphInfo

    if not _flash_bwd_gate_enabled():
        return None
    if len(device_ir.grid_block_ids) != 1 or len(device_ir.grid_block_ids[0]) != 1:
        return None
    root_graph = None
    inner_graph = None
    for graph_info in device_ir.graphs:
        if isinstance(graph_info, RootGraphInfo):
            root_graph = graph_info
        elif isinstance(graph_info, ForLoopGraphInfo):
            if inner_graph is not None:
                return None
            inner_graph = graph_info
    if root_graph is None or inner_graph is None:
        return None
    if len(inner_graph.block_ids) != 1:
        return None

    inner_facts = _match_bwd_inner_graph(inner_graph.graph)
    if inner_facts is None:
        return None
    root_facts = _match_bwd_root_graph(root_graph.graph, inner_facts=inner_facts)
    if root_facts is None:
        return None

    env = CompileEnvironment.current()
    kv_block_id = device_ir.grid_block_ids[0][0]
    total_kv_rows = env.block_sizes[kv_block_id].size
    if not isinstance(total_kv_rows, int):
        return None
    mm_dim = typing.cast("int", root_facts["mm_dim"])
    n_dim = typing.cast("int", root_facts["n_dim"])
    m_dim = (
        typing.cast("int", inner_facts["m_dim"]) if inner_facts["causal"] else mm_dim
    )
    if total_kv_rows % n_dim != 0:
        return None
    if n_dim % 128 != 0 or mm_dim % 128 != 0 or m_dim % 128 != 0:
        return None

    plan = AttentionBwdPlan(
        head_dim=typing.cast("int", inner_facts["head_dim"]),
        io_dtype=typing.cast("torch.dtype", inner_facts["io_dtype"]),
        causal=bool(inner_facts["causal"]),
        m_dim=m_dim,
        mm_dim=mm_dim,
        n_dim=n_dim,
        total_kv_rows=total_kv_rows,
        qk_scale_expr=typing.cast("str", inner_facts["qk_scale_expr"]),
        out_scale_expr=typing.cast("str", inner_facts["out_scale_expr"]),
        dk_scale_expr=typing.cast("str", root_facts["dk_scale_expr"]),
    )
    return AttentionBwdMatch(
        plan=plan,
        q_name=typing.cast("str", inner_facts["q_name"]),
        k_name=typing.cast("str", root_facts["k_name"]),
        v_name=typing.cast("str", root_facts["v_name"]),
        do_name=typing.cast("str", inner_facts["do_name"]),
        lse_name=typing.cast("str", inner_facts["lse_name"]),
        delta_name=typing.cast("str", inner_facts["delta_name"]),
        dq_name=typing.cast("str", inner_facts["dq_name"]),
        dk_name=typing.cast("str", root_facts["dk_name"]),
        dv_name=typing.cast("str", root_facts["dv_name"]),
        qk_scale_sym=typing.cast("sympy.Expr | float", inner_facts["qk_scale_sym"]),
        out_scale_sym=typing.cast("sympy.Expr | float", inner_facts["out_scale_sym"]),
        dk_scale_sym=typing.cast("sympy.Expr | float", root_facts["dk_scale_sym"]),
    )


def _flash_bwd_wg_compute_block(
    *,
    h: int,
    plan: AttentionBwdPlan,
    io_dtype: str,
    scale2_expr: str,
    dk_scale_expr: str,
    n_tiles: int,
    m_mod_tiles: int,
    steps_expr: str,
    dk_off: int,
    kv_stage: int,
    persistent: bool = False,
    exp2_f32: bool = False,
) -> str:
    """One compute warpgroup's body (h = column half 0/1, warps 4-7 / 8-11):
    dK accumulates at TMEM column ``dk_off``; dV stages through the tile's
    slot of the ``kv_stage``-deep V ring."""
    d = plan.head_dim
    warp_lo, warp_hi = (4, 8) if h == 0 else (8, 12)
    mask_arg = "fbwd_mask_lim" if plan.causal else "None"
    m_tile_expr = (
        f"fbwd_i // fbwd_seg * {plan.m_dim // 128} + fbwd_n_tile + fbwd_i % fbwd_seg"
        if plan.causal
        else "fbwd_m_start + fbwd_i"
    )
    # LSE/Delta staging slot. The two slots are separated by the group's S-t2r
    # barrier (3 + h) of the iteration in between, so the slot must alternate on the RUNNING
    # iteration count. Persistent: `fbwd_i` restarts per tile, and with an odd
    # `fbwd_steps` (causal walks) the next tile's first write would land in
    # the slot the other warps of the group may still be reading (Delta of the
    # previous tile's last iteration); `fbwd_phase` toggles once per iteration
    # across tiles, so it is exactly that running parity.
    par_expr = "fbwd_phase * 128" if persistent else "(fbwd_i % 2) * 128"

    def p_pass(mask: str, ind: str) -> str:
        return (
            f"{ind}for fbwd_ch in cutlass.range_constexpr(2):\n"
            f"{ind}    fbwd_lse_frg = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)\n"
            f"{ind}    fbwd_lse_v = cute.make_tensor(fbwd_sLSE.iterator + fbwd_par + {64 * h} + fbwd_ch * 32, cute.make_layout(32))\n"
            f"{ind}    cute.autovec_copy(fbwd_lse_v, fbwd_lse_frg)\n"
            f"{ind}    _helion_flash_rt.fbwd_p_pairs_packed(tLDrS, fbwd_lse_frg, fbwd_ch * 32, 32, fbwd_scale2, {mask}, fbwd_ch * 32, exp2_f32={exp2_f32})\n"
        )

    # Causal: only the tiles that intersect the diagonal (m rows below
    # kv_base + 128) run the masked softmax; every other iteration takes the
    # mask-free path (FA4 does the same), saving the per-pair compare+select.
    if plan.causal:
        p_passes = (
            f"            if (fbwd_qbase - {64 * h}) < fbwd_kv_base + 128:\n"
            + p_pass(mask_arg, " " * 16)
            + "            else:\n"
            + p_pass("None", " " * 16)
        )
    else:
        p_passes = p_pass("None", " " * 12)
    m_tile_first = m_tile_expr.replace("fbwd_i", "cutlass.Int32(0)")
    epilogue_1 = fbwd_epilogue_block(
        h=h,
        rows=128,
        d=d,
        epi_off=str(256 if h == 0 else dk_off),
        epi_scale=(
            "_helion_flash_rt._scale_fragment_packed_f32x2(fbwd_reg, cutlass.Float32("
            + dk_scale_expr
            + "))"
            if h == 1
            else "pass"
        ),
        kv_tile="fbwd_tile_id",
        io_dtype=io_dtype,
        dkv_wait="fbwd_tile_it % 2",
        v_empty_expr=f"storage.v_mbar_ptr.data_ptr() + {kv_stage} + fbwd_tile_it % {kv_stage}",
        kv_slot=f"fbwd_tile_it % {kv_stage}",
    )
    # Next tile of the walk without a division: +1 inside a stripe, and a jump
    # to the next stripe's diagonal when the stripe ends (causal GQA walk).
    m_tile_next = (
        f"fbwd_m_tile + 1 + cutlass.Int32((fbwd_m_tile % {plan.m_dim // 128}) == {plan.m_dim // 128 - 1}) * fbwd_n_tile"
        if plan.causal
        else "fbwd_m_tile + 1"
    )
    tile_region = _fbwd_persistent_tiles(
        f"""        fbwd_stage_col0 = fbwd_q_row_base + ({m_tile_first}) * 128 + {64 * h} + fbwd_stat_j
        if fbwd_local_tidx < 64:
            _helion_flash_rt.cp_async_ca_4(fbwd_sLSE.iterator + (fbwd_phase * 128 + {64 * h} + fbwd_stat_j), _fbwd_mLSE.iterator + fbwd_stage_col0)
        else:
            _helion_flash_rt.cp_async_ca_4(fbwd_sDelta.iterator + (fbwd_phase * 128 + {64 * h} + fbwd_stat_j), _fbwd_mDelta.iterator + fbwd_stage_col0)
        cute.arch.cp_async_commit_group()
        fbwd_m_tile = {m_tile_first}
        for fbwd_i in cutlass.range({steps_expr}, unroll=1):
            fbwd_qbase = (fbwd_m_tile % {m_mod_tiles}) * 128 + {64 * h}
            fbwd_row_base = fbwd_q_row_base + fbwd_m_tile * 128
            fbwd_mask_lim = fbwd_kv_base + fbwd_thr_row - fbwd_qbase
            fbwd_par = {par_expr}
            _helion_flash_rt.mbar_spin_wait(fbwd_s_full_ptr, fbwd_phase, 10000000)
            tLDrS = cute.make_rmem_tensor(tLDcS.shape, cutlass.Float32)
            cute.copy(fbwd_tiled_ld, tLDtS, tLDrS)
            cute.arch.fence_view_async_tmem_load()
            cute.arch.cp_async_wait_group(0)
            _helion_flash_rt.named_barrier_wait_unaligned({3 + h}, 128)
            fbwd_m_tile_nxt = {m_tile_next}
            if fbwd_i + 1 < {steps_expr}:
                fbwd_stage_col = fbwd_q_row_base + fbwd_m_tile_nxt * 128 + {64 * h} + fbwd_stat_j
                if fbwd_local_tidx < 64:
                    _helion_flash_rt.cp_async_ca_4(fbwd_sLSE.iterator + (128 - fbwd_par + {64 * h} + fbwd_stat_j), _fbwd_mLSE.iterator + fbwd_stage_col)
                else:
                    _helion_flash_rt.cp_async_ca_4(fbwd_sDelta.iterator + (128 - fbwd_par + {64 * h} + fbwd_stat_j), _fbwd_mDelta.iterator + fbwd_stage_col)
                cute.arch.cp_async_commit_group()
{p_passes}            tSTrP = cute.make_rmem_tensor(tSTcP.shape, cutlass.Float32)
            tSTrP_e = cute.make_tensor(cute.recast_ptr(tSTrP.iterator, dtype={io_dtype}), tLDrS.layout)
            tSTrP_e.store(tLDrS.load().to({io_dtype}))
            cute.copy(fbwd_tiled_st, tSTrP, tSTtP)
            cute.arch.fence_view_async_tmem_store()
            _helion_flash_rt.mbarrier_arrive(fbwd_p_full_ptr)
            _helion_flash_rt.mbar_spin_wait(fbwd_dp_full_ptr, fbwd_phase, 10000000)
            if (fbwd_i > 0) | (fbwd_tile_it > 0):
                _helion_flash_rt.mbar_spin_wait(fbwd_dq_full_ptr, fbwd_dqf_phase, 10000000)
                fbwd_dqf_phase ^= 1
            for fbwd_ch in cutlass.range_constexpr(2):
                tLDrdP = cute.make_rmem_tensor(tLDcS[None, None, 0].shape, cutlass.Float32)
                cute.copy(fbwd_tiled_ld, tLDtdP[None, None, fbwd_ch], tLDrdP)
                fbwd_dlt_frg = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)
                fbwd_dlt_v = cute.make_tensor(fbwd_sDelta.iterator + fbwd_par + {64 * h} + fbwd_ch * 32, cute.make_layout(32))
                cute.autovec_copy(fbwd_dlt_v, fbwd_dlt_frg)
                cute.arch.fence_view_async_tmem_load()
                _helion_flash_rt.fbwd_ds_pairs_packed(tLDrdP, tLDrS, fbwd_ch * 32, fbwd_dlt_frg, 32)
                fbwd_ds_bf = cute.make_rmem_tensor(tLDrdP.layout, {io_dtype})
                fbwd_ds_bf.store(tLDrdP.load().to({io_dtype}))
                fbwd_ds_flat = cute.make_tensor(fbwd_ds_bf.iterator, cute.make_layout(32))
                if fbwd_ch == 0:
                    cute.autovec_copy(fbwd_ds_flat, fbwd_sdS_c0)
                else:
                    cute.autovec_copy(fbwd_ds_flat, fbwd_sdS_c1)
            cute.arch.fence_view_async_shared()
            _helion_flash_rt.mbarrier_arrive(fbwd_ds_full_ptr)
            fbwd_phase ^= 1
            fbwd_m_tile = fbwd_m_tile_nxt
{epilogue_1}""",
        plan,
        persistent,
    )
    return f"""
    if (warp_idx >= {warp_lo}) & (warp_idx < {warp_hi}):
        cute.arch.setmaxregister_increase(136)
{fbwd_compute_tmem_views(rows=128, s_col=64 * h, dp_col=128 + 64 * h)}        cS = cute.make_identity_tensor((128, 128))
{FBWD_COMPUTE_LOAD_VIEWS}        # P (bf16) lands in this group's OWN S half (cols 64h..64h+31): the dV gemm
        # reads it with gapped k-chunk offsets, and the two groups never touch
        # each other's TMEM, so the S-t2r barrier below is group-local.
{fbwd_p_store_views(p_col=64 * h)}        fbwd_sdS_mnc = cute.make_tensor(fbwd_sdS_mn.iterator, cute.composition(fbwd_sdS_mn.layout, cute.make_layout((32, 2, 2, 128))))
        fbwd_crow = cutlass.Int32(tLDcS[0][0])
        fbwd_sdS_c0 = fbwd_sdS_mnc[None, 0, {h}, fbwd_crow]
        fbwd_sdS_c1 = fbwd_sdS_mnc[None, 1, {h}, fbwd_crow]
        fbwd_thr_row = cutlass.Int32(tLDcS[0][0])
        fbwd_phase = cutlass.Int32(0)
        fbwd_dqf_phase = cutlass.Int32(0)
        fbwd_scale2 = cutlass.Float32({scale2_expr})
        # LSE/Delta staging by 4-byte cp.async one iteration ahead (threads
        # 0-63: LSE, 64-127: Delta): slot (i+1)%2 was last read in iteration
        # i-1 and every thread of the group has passed its S-t2r barrier of
        # iteration i before the copies are issued, so the global-load latency never sits on the
        # softmax critical path. The first tile iteration is staged here into
        # the current parity slot (persistent tiles carry ``fbwd_phase``).
        fbwd_stat_j = fbwd_local_tidx % 64
{tile_region}        _helion_flash_rt.named_barrier_arrive_unaligned(2, 13 * 32)
"""


def _fbwd_tile_setup_lines(plan: AttentionBwdPlan, indent: str) -> str:
    n_tiles = plan.n_dim // 128
    m_tiles = plan.mm_dim // 128
    causal_start = "fbwd_n_tile" if plan.causal else "cutlass.Int32(0)"
    m_mod_tiles = plan.m_dim // 128
    group = plan.group
    lines = [
        f"fbwd_bh = fbwd_tile_id // {n_tiles}",
        f"fbwd_n_tile = fbwd_tile_id % {n_tiles}",
        f"fbwd_m_start = {causal_start}",
    ]
    if plan.causal:
        lines += [
            f"fbwd_seg = {m_mod_tiles} - fbwd_n_tile",
            f"fbwd_steps = {group} * fbwd_seg",
        ]
    else:
        lines += [f"fbwd_steps = {m_tiles} - fbwd_m_start"]
    lines += [
        f"fbwd_q_tile_base = fbwd_bh * {m_tiles}",
        f"fbwd_q_row_base = fbwd_bh * {plan.mm_dim}",
        "fbwd_kv_base = fbwd_n_tile * 128",
    ]
    return "".join(indent + line + "\n" for line in lines)


def _fbwd_persistent_tiles(
    region: str, plan: AttentionBwdPlan, persistent: bool
) -> str:
    """A warp role's per-tile region of the 1-CTA body (8-space indentation):
    itself for the one-tile CTA; under the persistent walk, the loop over the
    CTA's tiles around it. Dynamic schedule: the first tile is this CTA's
    blockIdx, later tiles come from the CLC response ring (warp 15 steals
    not-yet-launched CTAs), so the hardware's launch order and load balance
    are preserved. The gate variable ``fbwd_tile_it`` is loop-carried;
    barrier phase counters live outside the loops so parities continue
    across tiles.
    """
    if not persistent:
        return region
    ind = " " * 8
    head = (
        f"{ind}fbwd_work = fbwd_clc_hw.initial_work_tile_info()\n"
        f"{ind}fbwd_tile_it = cutlass.Int32(0)\n"
        f"{ind}while fbwd_work.is_valid_tile:\n"
        f"{ind}    fbwd_tile_id = cutlass.Int32(fbwd_work.tile_idx[0])\n"
        + _fbwd_tile_setup_lines(plan, ind + "    ")
    )
    tail = FBWD_CLC_CONSUME.format(ind=ind + "    ") + (
        f"{ind}    fbwd_tile_it = fbwd_tile_it + 1\n"
    )
    return fbwd_tile_loop(region, head=head, tail=tail)


def emit_flash_bwd_device_body(
    *,
    plan: AttentionBwdPlan,
    io_dtype: str,
    scale2_expr: str,
    out_scale_expr: str,
    dk_scale_expr: str,
    q_stage: int,
    do_stage: int,
    kv_stage: int = 1,
    persistent: bool = False,
    total_tiles: int = 0,
    exp2_f32: bool = False,
) -> str:
    """The fused backward device body (v1: in-order MMA, 1-CTA, 512 threads).

    Warp roles: 0-3 dQ reduce, 4-7 / 8-11 compute (column halves of the
    128x128 S/dP tiles), 12 MMA (all five gemms via cute.gemm), 13 load
    (TMA), 14/15 idle. TMEM: S@0 (P bf16 aliases S), dP@128, dV@256,
    dK@{dk_off}, dQ@{dq_off} (aliases dP when head_dim == 128). dS lives
    only in smem (two major-mode views feed the dK and dQ gemms). dQ is
    staged row-major in smem and accumulated into the fp32 dq buffer with
    cp.reduce.async.bulk per 32-row warp chunk.
    """
    d = plan.head_dim
    dk_off = 256 + (64 if d == 64 else 128)
    dq_off = 384 if d == 64 else 128
    n_tiles = plan.n_dim // 128
    m_tiles = plan.mm_dim // 128
    m_mod_tiles = plan.m_dim // 128
    steps_expr = "fbwd_steps"
    causal_start = "fbwd_n_tile" if plan.causal else "cutlass.Int32(0)"
    group = plan.group
    if plan.causal:
        # Segmented walk: skip fully-masked m-tiles entirely (they contribute
        # zero); each of the `group` stripes starts at its own diagonal.
        seg_line = f"\n    fbwd_seg = {m_mod_tiles} - fbwd_n_tile"
        steps_expr2 = f"{group} * fbwd_seg"
        m_tile_expr = (
            f"fbwd_i // fbwd_seg * {m_mod_tiles} + fbwd_n_tile + fbwd_i % fbwd_seg"
        )
    else:
        seg_line = ""
        steps_expr2 = f"{m_tiles} - fbwd_m_start"
        m_tile_expr = "fbwd_m_start + fbwd_i"

    # dQ TMEM reuse gates. D=128 aliases dQ over the dP columns, so dP(i+1)
    # must wait for the reduce warps to release dQ(i) (unconditional; dQ(i)
    # was committed earlier in the same body). D=64 gives dQ its own
    # columns, so only the NEXT dQ write needs the release of the previous.
    if d == 128:
        dq_empty_wait_g2 = """
            _helion_flash_rt.mbar_spin_wait(fbwd_dq_empty_ptr, fbwd_dqe_phase, 10000000)
            fbwd_dqe_phase ^= 1"""
        dq_empty_wait_g5 = ""
        dq_empty_wait_tail = ""
        dq_empty_wait_prologue = """
        if fbwd_tile_it > 0:
            _helion_flash_rt.mbar_spin_wait(fbwd_dq_empty_ptr, fbwd_dqe_phase, 10000000)
            fbwd_dqe_phase ^= 1"""
    else:
        dq_empty_wait_g2 = ""
        dq_empty_wait_prologue = ""
        dq_empty_wait_g5 = """
            if (fbwd_i > 0) | (fbwd_tile_it > 0):
                _helion_flash_rt.mbar_spin_wait(fbwd_dq_empty_ptr, fbwd_dqe_phase, 10000000)
                fbwd_dqe_phase ^= 1"""
        # Persistent: a later tile with fbwd_steps == 1 skips the loop, so
        # its only dQ gemm must still wait for the release of the previous
        # tile's last dQ (one wait per dQ gemm except the kernel's first).
        tail_cond = (
            "(fbwd_steps > 1) | (fbwd_tile_it > 0)" if persistent else "fbwd_steps > 1"
        )
        dq_empty_wait_tail = f"""
        if {tail_cond}:
            _helion_flash_rt.mbar_spin_wait(fbwd_dq_empty_ptr, fbwd_dqe_phase, 10000000)
            fbwd_dqe_phase ^= 1"""

    # dQ drain: each lane owns one row of the 128x{d} fp32 tile. Per 32-column
    # chunk: t2r (32 regs live), 8x16B stores into a SWIZZLE_128B 32x32 box
    # (conflict-free, no padding), then ONE 2D TMA tensor reduce-add of the box
    # into row-major dQ. Per-warp 2-deep box ring (2 x 4KB x 4 warps = 32KB).
    n_chunks = d // 32
    reduce_body = f"""            _helion_flash_rt.mbar_spin_wait(fbwd_dq_full_ptr, fbwd_phase, 10000000)
            fbwd_phase ^= 1
            fbwd_my_row = cutlass.Int32(tRDcDQ[0][0]) % 32
            fbwd_gdq_row = fbwd_row_base // 32 + warp_idx
            tRDrDQ = cute.make_rmem_tensor(tRDcDQ.shape, cutlass.Float32)
            cute.copy(fbwd_tiled_dq_ld, tRDtDQ, tRDrDQ)
            cute.arch.fence_view_async_tmem_load()
            _helion_flash_rt.mbarrier_arrive(fbwd_dq_empty_ptr)
            _helion_flash_rt._scale_fragment_packed_f32x2(tRDrDQ, fbwd_out_scale)
            for fbwd_c in cutlass.range_constexpr({n_chunks}):
                fbwd_dq_flat = cute.make_tensor(tRDrDQ[None, fbwd_c, 0, 0].iterator, cute.make_layout(32))
                fbwd_sdq_box = fbwd_sdq[None, None, warp_idx * 2 + fbwd_c % 2]
                cute.arch.cp_async_bulk_wait_group(1, read=True)
                cute.arch.sync_warp()
                cute.autovec_copy(fbwd_dq_flat, fbwd_sdq_box[fbwd_my_row, None])
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                fbwd_gdq_box = fbwd_gdq[None, None, fbwd_gdq_row, fbwd_c]
                tDQsDQ, tDQgDQ = cute_cpasync_flash.tma_partition(_fbwd_tma_dq, 0, cute.make_layout(1), cute.group_modes(fbwd_sdq_box, 0, 2), cute.group_modes(fbwd_gdq_box, 0, 2))
                cute.copy(_fbwd_tma_dq, tDQsDQ, tDQgDQ)
                cute.arch.cp_async_bulk_commit_group()"""

    # Persistent: the MMA warp's last ds_full wait of a tile must flip the
    # parity like every other one, or the next tile's first wait is one phase
    # behind: it passes early (dK/dQ read a stale dS) and, as soon as the
    # compute warps get one more ds_full arrival ahead, it waits on the
    # current phase forever (the deadlock: MMA @ds_full, compute @dp_full,
    # reduce @dq_full, load @q_empty).
    ds_tail_toggle = "\n        fbwd_dsf_phase ^= 1" if persistent else ""

    compute_blocks = "".join(
        _flash_bwd_wg_compute_block(
            h=h,
            plan=plan,
            io_dtype=io_dtype,
            scale2_expr=scale2_expr,
            dk_scale_expr=dk_scale_expr,
            n_tiles=n_tiles,
            m_mod_tiles=m_mod_tiles,
            steps_expr=steps_expr,
            dk_off=dk_off,
            kv_stage=kv_stage,
            persistent=persistent,
            exp2_f32=exp2_f32,
        )
        for h in (0, 1)
    )
    # The CLC scheduler (persistent walk): its state at kernel entry, the
    # response-consuming warp 14 and the stealing warp 15.
    if persistent:
        clc_setup = fbwd_clc_setup(total_tiles, 1, "cute.make_layout((1, 1, 1, 1))")
        clc_warps = (
            "    if warp_idx == 14:\n"
            + fbwd_clc_consume_only(" " * 8)
            + "    if warp_idx == 15:\n"
            + fbwd_clc_scheduler(" " * 8, None)
        )
    else:
        clc_setup = clc_warps = ""

    # The warp roles' per-tile regions: the load warp's K / V loads and Q / dO
    # ring, the MMA warp's gemm chain from the K wait to the K release, the
    # reduce warps' dQ drain loop and (in the compute blocks) the statistics
    # staging through the epilogue; the persistent walk loops each of them
    # over the CTA's tiles.
    load_region = _fbwd_persistent_tiles(
        f"""        fbwd_ke = fbwd_k_prod.acquire_and_advance()
        cute.copy(_fbwd_tma_k, tKgK[None, fbwd_tile_id], tKsK[None, fbwd_ke.index], tma_bar_ptr=fbwd_ke.barrier)
        fbwd_ve = fbwd_v_prod.acquire_and_advance()
        cute.copy(_fbwd_tma_v, tVgV[None, fbwd_tile_id], tVsV[None, fbwd_ve.index], tma_bar_ptr=fbwd_ve.barrier)
        for fbwd_i in cutlass.range(fbwd_steps, unroll=1):
            fbwd_q_tile = fbwd_q_tile_base + ({m_tile_expr})
            fbwd_qe = fbwd_q_prod.acquire_and_advance()
            cute.copy(_fbwd_tma_q, tQgQ[None, fbwd_q_tile], tQsQ[None, fbwd_qe.index], tma_bar_ptr=fbwd_qe.barrier)
            fbwd_doe = fbwd_do_prod.acquire_and_advance()
            cute.copy(_fbwd_tma_do, tDOgDO[None, fbwd_q_tile], tDOsDO[None, fbwd_doe.index], tma_bar_ptr=fbwd_doe.barrier)
""",
        plan,
        persistent,
    )
    mma_region = _fbwd_persistent_tiles(
        fbwd_mma_tile_entry("        ")
        + f"""        # Prologue: S(0), dP(0), dV(0). FA4's 1-iteration software skew: each
        # loop body issues dK/dQ of iteration i and S/dP/dV of iteration i+1,
        # so the compute warps' softmax/dS work overlaps the tensor pipe.
        fbwd_qe = fbwd_q_cons.wait_and_advance()
        fbwd_doe = fbwd_do_cons.wait_and_advance()
        fbwd_ve = fbwd_v_cons.wait_and_advance()
        fbwd_mb[1] = _helion_flash_ptx.make_smem_desc_start_addr(sV[None, None, None, fbwd_ve.index].iterator)
        fbwd_st_v = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 1)
        _helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_s_addr, fbwd_st_k, fbwd_k_base, tSrK[None, None, None, 0].layout, _helion_flash_ptx.make_smem_desc_start_addr(sQ[None, None, None, fbwd_qe.index].iterator), fbwd_q_base, tSrQ[None, None, None, 0].layout, "helion_fbwd_ss_idesc", zero_init=True)
        with cute.arch.elect_one():
            cute_tcgen05_flash.commit(fbwd_s_full_ptr){dq_empty_wait_prologue}
        _helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_dp_addr, fbwd_st_v, fbwd_v_base, tSrV[None, None, None, 0].layout, _helion_flash_ptx.make_smem_desc_start_addr(sdO[None, None, None, fbwd_doe.index].iterator), fbwd_do_base, tSrDO[None, None, None, 0].layout, "helion_fbwd_ss_idesc", zero_init=True)
        with cute.arch.elect_one():
            cute_tcgen05_flash.commit(fbwd_dp_full_ptr)
        _helion_flash_rt.mbar_spin_wait(fbwd_p_full_ptr, fbwd_pf_phase, 10000000)
        fbwd_pf_phase ^= 1
        _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(fbwd_dv_addr, fbwd_s_addr, _helion_flash_ptx.make_smem_desc_start_addr(sdOt[None, None, None, fbwd_doe.index].iterator), fbwd_dot_base, tTSrP[None, None, None, 0].layout, tTSrDOt[None, None, None, 0].layout, "helion_fbwd_ts_idesc", zero_init=fbwd_dv_zero, a_offsets={FBWD_2CTA_DS_OFFSETS})
{fbwd_mma_step_loop_head("        ")}            _helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_s_addr, fbwd_st_k, fbwd_k_base, tSrK[None, None, None, 0].layout, _helion_flash_ptx.make_smem_desc_start_addr(sQ[None, None, None, fbwd_qe2.index].iterator), fbwd_q_base, tSrQ[None, None, None, 0].layout, "helion_fbwd_ss_idesc", zero_init=True)
            with cute.arch.elect_one():
                cute_tcgen05_flash.commit(fbwd_s_full_ptr)
            _helion_flash_rt.mbar_spin_wait(fbwd_ds_full_ptr, fbwd_dsf_phase, 10000000)
            fbwd_dsf_phase ^= 1
            _helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_dk_addr, fbwd_st_dsnk, fbwd_dsnk_base, tKSrDS[None, None, None, 0].layout, _helion_flash_ptx.make_smem_desc_start_addr(sQt[None, None, None, fbwd_qe.index].iterator), fbwd_qt_base, tKSrQt[None, None, None, 0].layout, "helion_fbwd_dsk_idesc", zero_init=fbwd_dk_zero)
            fbwd_dk_zero = cutlass.Boolean(False){dq_empty_wait_g5}
            _helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_dqa_addr, fbwd_st_dsmn, fbwd_dsmn_base, tDQrDS[None, None, None, 0].layout, fbwd_st_kt, fbwd_kt_base, tDQrKt[None, None, None, 0].layout, "helion_fbwd_dq_idesc", zero_init=True)
            with cute.arch.elect_one():
                cute_tcgen05_flash.commit(fbwd_dq_full_ptr)
            fbwd_qe.release()
            fbwd_qe = fbwd_qe2{dq_empty_wait_g2}
            fbwd_doe = fbwd_do_cons.wait_and_advance()
            _helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_dp_addr, fbwd_st_v, fbwd_v_base, tSrV[None, None, None, 0].layout, _helion_flash_ptx.make_smem_desc_start_addr(sdO[None, None, None, fbwd_doe.index].iterator), fbwd_do_base, tSrDO[None, None, None, 0].layout, "helion_fbwd_ss_idesc", zero_init=True)
            with cute.arch.elect_one():
                cute_tcgen05_flash.commit(fbwd_dp_full_ptr)
            _helion_flash_rt.mbar_spin_wait(fbwd_p_full_ptr, fbwd_pf_phase, 10000000)
            fbwd_pf_phase ^= 1
            _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(fbwd_dv_addr, fbwd_s_addr, _helion_flash_ptx.make_smem_desc_start_addr(sdOt[None, None, None, fbwd_doe.index].iterator), fbwd_dot_base, tTSrP[None, None, None, 0].layout, tTSrDOt[None, None, None, 0].layout, "helion_fbwd_ts_idesc", zero_init=False, a_offsets={FBWD_2CTA_DS_OFFSETS})
            fbwd_doe.release()
        # dV is complete once the loop's last dV gemm retires: let WG0 start
        # its epilogue under the tail dK/dQ gemms.
        with cute.arch.elect_one():
            cute_tcgen05_flash.commit(fbwd_dv_done_ptr)
        # Tail: dK/dQ of the last iteration.
        fbwd_st_k = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 0)
        fbwd_st_v = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 1)
        fbwd_st_dsnk = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 2)
        fbwd_st_dsmn = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 3)
        fbwd_st_kt = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 4)
        _helion_flash_rt.mbar_spin_wait(fbwd_ds_full_ptr, fbwd_dsf_phase, 10000000){ds_tail_toggle}
        _helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_dk_addr, fbwd_st_dsnk, fbwd_dsnk_base, tKSrDS[None, None, None, 0].layout, _helion_flash_ptx.make_smem_desc_start_addr(sQt[None, None, None, fbwd_qe.index].iterator), fbwd_qt_base, tKSrQt[None, None, None, 0].layout, "helion_fbwd_dsk_idesc", zero_init=fbwd_dk_zero){dq_empty_wait_tail}
        _helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_dqa_addr, fbwd_st_dsmn, fbwd_dsmn_base, tDQrDS[None, None, None, 0].layout, fbwd_st_kt, fbwd_kt_base, tDQrKt[None, None, None, 0].layout, "helion_fbwd_dq_idesc", zero_init=True)
        with cute.arch.elect_one():
            cute_tcgen05_flash.commit(fbwd_dq_full_ptr)
        fbwd_qe.release()
        with cute.arch.elect_one():
            cute_tcgen05_flash.commit(fbwd_dkv_done_ptr)
        # K is free once the tail dQ retired (dK is staged through sdS); the V
        # slot doubles as the dV staging and is released by WG0's epilogue.
        fbwd_ke.release()
""",
        plan,
        persistent,
    )
    reduce_region = _fbwd_persistent_tiles(
        f"""        for fbwd_i in cutlass.range(fbwd_steps, unroll=1):
            fbwd_m_tile = {m_tile_expr}
            fbwd_row_base = fbwd_q_row_base + fbwd_m_tile * 128
{reduce_body}
""",
        plan,
        persistent,
    )
    mma_warp = fbwd_tmem_warp(
        fbwd_mma_fragments(
            "        ", dp_off=128, dv_off=256, dk_off=dk_off, dq_off=dq_off
        )
        + fbwd_mma_descriptor_setup(
            "        ", ts_op="_fbwd_ts_mma", dsk_op="_fbwd_dsk_mma"
        )
        + """        fbwd_dsmn_base = _helion_flash_ptx.smem_desc_base_from_tensor(fbwd_sdS_mn, _helion_flash_ptx.Major.MN)
        fbwd_kt_base = _helion_flash_ptx.smem_desc_base_from_tensor(sKt, _helion_flash_ptx.Major.MN)
        _helion_flash_ptx.declare_ptx_idesc(_fbwd_dq_mma.op, "helion_fbwd_dq_idesc")
        fbwd_mb = storage.mma_base.get_tensor(cute.make_layout(8))
        fbwd_mb[2] = _helion_flash_ptx.make_smem_desc_start_addr(fbwd_sdS_nk[None, None, None, 0].iterator)
        fbwd_mb[3] = _helion_flash_ptx.make_smem_desc_start_addr(fbwd_sdS_mn[None, None, None, 0].iterator)
        fbwd_pf_phase = cutlass.Int32(0)
        fbwd_dsf_phase = cutlass.Int32(0)
        fbwd_dqe_phase = cutlass.Int32(0)
"""
        + mma_region,
        dims=(d, d, d),
    )
    return f"""
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    fbwd_local_tidx = tidx % 128
    fbwd_tile_id = cutlass.Int32(cute.arch.block_idx()[0])
    fbwd_bh = fbwd_tile_id // {n_tiles}
    fbwd_n_tile = fbwd_tile_id % {n_tiles}
    fbwd_m_start = {causal_start}{seg_line}
    fbwd_steps = {steps_expr2}
    fbwd_q_tile_base = fbwd_bh * {m_tiles}
    fbwd_q_row_base = fbwd_bh * {plan.mm_dim}
    fbwd_kv_base = fbwd_n_tile * 128
    fbwd_tile_it = cutlass.Int32(0)
    fbwd_grid_dim = cutlass.Int32(cute.arch.grid_dim()[0])
    if warp_idx == 0:
        cute_cpasync_flash.prefetch_descriptor(_fbwd_tma_q)
        cute_cpasync_flash.prefetch_descriptor(_fbwd_tma_k)
        cute_cpasync_flash.prefetch_descriptor(_fbwd_tma_v)
        cute_cpasync_flash.prefetch_descriptor(_fbwd_tma_do)
        cute_cpasync_flash.prefetch_descriptor(_fbwd_tma_dq)
    _fbwd_storage_cls = _helion_flash_rt.flash_bwd_shared_storage({d}, {q_stage}, {do_stage}, {kv_stage}, {io_dtype})
    smem = cutlass_utils_flash.SmemAllocator()
    storage = smem.allocate(_fbwd_storage_cls)
{FBWD_SMEM_VIEWS}    fbwd_sdS_flat = cute.make_tensor(fbwd_sdS_mn.iterator, cute.composition(fbwd_sdS_mn.layout, cute.make_layout((128, 128))))
    fbwd_sLSE = storage.sLSE.get_tensor(cute.make_layout(2 * 128))
    fbwd_sDelta = storage.sDelta.get_tensor(cute.make_layout(2 * 128))
    fbwd_sdq = storage.sdQaccum.get_tensor(_fbwd_dqsl.outer, swizzle=_fbwd_dqsl.inner)
    fbwd_s_full_ptr = storage.s_full_mbar.data_ptr()
    fbwd_dp_full_ptr = storage.dp_full_mbar.data_ptr()
    fbwd_p_full_ptr = storage.p_full_mbar.data_ptr()
    fbwd_ds_full_ptr = storage.ds_full_mbar.data_ptr()
    fbwd_dq_full_ptr = storage.dq_full_mbar.data_ptr()
    fbwd_dq_empty_ptr = storage.dq_empty_mbar.data_ptr()
    fbwd_dkv_done_ptr = storage.dkv_done_mbar.data_ptr()
    fbwd_dv_done_ptr = storage.dv_done_mbar.data_ptr()
    fbwd_tmem_dealloc_ptr = storage.tmem_dealloc_mbar.data_ptr()
{FBWD_MBAR_INITS}    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    fbwd_tmem_user_bar = cutlass_pipeline_flash.NamedBarrier(barrier_id=2, num_threads=13 * 32)
    fbwd_tmem = cutlass_utils_flash.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=fbwd_tmem_user_bar, allocator_warp_id=12, is_two_cta=False, two_cta_tmem_dealloc_mbar_ptr=fbwd_tmem_dealloc_ptr)
    fbwd_q_bytes = cute.size_in_bytes({io_dtype}, cute.select(_fbwd_qsl, mode=[0, 1, 2]))
    fbwd_k_bytes = cute.size_in_bytes({io_dtype}, cute.select(_fbwd_ksl, mode=[0, 1, 2]))
    fbwd_q_prod, fbwd_q_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(num_stages={q_stage}, producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), tx_count=fbwd_q_bytes, barrier_storage=storage.q_mbar_ptr.data_ptr()).make_participants()
    fbwd_do_prod, fbwd_do_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(num_stages={do_stage}, producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), tx_count=fbwd_q_bytes, barrier_storage=storage.do_mbar_ptr.data_ptr()).make_participants()
    fbwd_k_prod, fbwd_k_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(num_stages={kv_stage}, producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), tx_count=fbwd_k_bytes, barrier_storage=storage.k_mbar_ptr.data_ptr()).make_participants()
    fbwd_v_prod, fbwd_v_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(num_stages={kv_stage}, producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), tx_count=fbwd_k_bytes, barrier_storage=storage.v_mbar_ptr.data_ptr()).make_participants()
{clc_setup}    fbwd_sst = _fbwd_ss_mma.get_slice(0)
    fbwd_tst = _fbwd_ts_mma.get_slice(0)
    fbwd_dskt = _fbwd_dsk_mma.get_slice(0)
    fbwd_dqt = _fbwd_dq_mma.get_slice(0)
    gK = cute.flat_divide(_fbwd_mKt, cute.select((128, 128, {d}), mode=[0, 2]))
    gQ = cute.flat_divide(_fbwd_mQt, cute.select((128, 128, {d}), mode=[1, 2]))
    gV = cute.flat_divide(_fbwd_mVt, cute.select((128, 128, {d}), mode=[0, 2]))
    gdO = cute.flat_divide(_fbwd_mdOt, cute.select((128, 128, {d}), mode=[1, 2]))
    tSgK = fbwd_sst.partition_A(gK)
    tSgQ = fbwd_sst.partition_B(gQ)
    tSgV = fbwd_sst.partition_A(gV)
    tSgdO = fbwd_sst.partition_B(gdO)
    tKsK, tKgK_kdl = cute_cpasync_flash.tma_partition(_fbwd_tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(tSgK, 0, 3))
    tQsQ, tQgQ_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(tSgQ, 0, 3))
    tVsV, tVgV_kdl = cute_cpasync_flash.tma_partition(_fbwd_tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3), cute.group_modes(tSgV, 0, 3))
    tDOsDO, tDOgDO_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_do, 0, cute.make_layout(1), cute.group_modes(sdO, 0, 3), cute.group_modes(tSgdO, 0, 3))
    if (warp_idx >= 12) & (warp_idx < 16):
        cute.arch.setmaxregister_decrease(88)
{clc_warps}    if warp_idx == 13:
        tKgK = tKgK_kdl[None, None, 0, 0]
        tQgQ = tQgQ_qdl[None, None, 0, 0]
        tVgV = tVgV_kdl[None, None, 0, 0]
        tDOgDO = tDOgDO_qdl[None, None, 0, 0]
{load_region}        fbwd_q_prod.tail()
        fbwd_do_prod.tail()
        fbwd_k_prod.tail()
        fbwd_v_prod.tail()
{mma_warp}    if warp_idx < 4:
        cute.arch.setmaxregister_increase(152)
{fbwd_dq_reduce_views(d=d, dq_off=dq_off)}{fbwd_dq_reduce_partitions("        ", d=d)}        fbwd_gdq = cute.local_tile(_fbwd_mDQt, (32, 32), (None, None))
        fbwd_out_scale = cutlass.Float32({out_scale_expr})
        fbwd_phase = cutlass.Int32(0)
{reduce_region}        cute.arch.cp_async_bulk_wait_group(0)
        _helion_flash_rt.named_barrier_arrive_unaligned(2, 13 * 32)
{compute_blocks}"""


emit_flash_bwd_body = source_adapter(emit_flash_bwd_device_body)


def _fbwd2_m_tile(plan: AttentionBwdPlan, i: str) -> str:
    """m-tile visited at step ``i`` of the 2-CTA walk.

    Causal: segmented walk over the ``group`` stripes of the (GQA-packed) Q
    axis, each starting at the cluster diagonal ``fbwd_m_start`` (fully-masked
    tiles are skipped). Non-causal: plain ``fbwd_m_start + i``.
    """
    if plan.causal:
        m_mod_tiles = plan.m_dim // 128
        return f"(({i}) // fbwd_seg * {m_mod_tiles} + fbwd_m_start + ({i}) % fbwd_seg)"
    return f"(fbwd_m_start + ({i}))"


def _fbwd2_tile_setup_lines(plan: AttentionBwdPlan, indent: str) -> str:
    """Per-tile derivations of the 2-CTA body from the cluster tile index."""
    n_tiles = plan.n_dim // 128
    m_tiles = plan.mm_dim // 128
    m_mod_tiles = plan.m_dim // 128
    causal_start = "fbwd_n_tile // 2 * 2" if plan.causal else "cutlass.Int32(0)"
    lines = [
        "fbwd_tile_id = fbwd_cl_tile * 2 + fbwd_rank",
        f"fbwd_bh = fbwd_tile_id // {n_tiles}",
        f"fbwd_n_tile = fbwd_tile_id % {n_tiles}",
        f"fbwd_m_start = {causal_start}",
    ]
    if plan.causal:
        lines += [
            f"fbwd_seg = {m_mod_tiles} - fbwd_m_start",
            f"fbwd_steps = {plan.group} * fbwd_seg",
        ]
    else:
        lines += [f"fbwd_steps = {m_tiles} - fbwd_m_start"]
    lines += [
        f"fbwd_q_tile_base = fbwd_bh * {m_tiles}",
        f"fbwd_q_row_base = fbwd_bh * {plan.mm_dim}",
        "fbwd_kv_base = fbwd_n_tile // 2 * 256",
    ]
    return "".join(indent + line + "\n" for line in lines)


def _fbwd2_persistent_tiles(
    region: str, plan: AttentionBwdPlan, persistent: bool, indent: int = 8
) -> str:
    """A warp role's per-tile region of the 2-CTA body at ``indent`` spaces
    (the leader MMA warp's sits under ``if fbwd_is_leader:``): itself for
    the one-tile cluster, else the loop over the cluster's tiles around it
    (see :func:`_fbwd_persistent_tiles`); the cluster tile is the CLC
    response's CTA index halved."""
    if not persistent:
        return region
    ind = " " * indent
    head = (
        f"{ind}fbwd_work = fbwd_clc_hw.initial_work_tile_info()\n"
        f"{ind}fbwd_tile_it = cutlass.Int32(0)\n"
        f"{ind}while fbwd_work.is_valid_tile:\n"
        f"{ind}    fbwd_cl_tile = cutlass.Int32(fbwd_work.tile_idx[0]) // 2\n"
        + _fbwd2_tile_setup_lines(plan, ind + "    ")
    )
    tail = FBWD_CLC_CONSUME.format(ind=ind + "    ") + (
        f"{ind}    fbwd_tile_it = fbwd_tile_it + 1\n"
    )
    return fbwd_tile_loop(region, head=head, tail=tail)


def _flash_bwd_2cta_wg_compute_block(
    *,
    h: int,
    plan: AttentionBwdPlan,
    io_dtype: str,
    scale2_expr: str,
    dk_scale_expr: str,
    m_mod_tiles: int,
    steps_expr: str,
    persistent: bool = False,
    exp2_f32: bool = False,
) -> str:
    """One compute warpgroup's body for the 2-CTA family (column half h)."""
    d = plan.head_dim
    warp_lo, warp_hi = (4, 8) if h == 0 else (8, 12)
    mask_arg = "fbwd_mask_lim" if plan.causal else "None"
    # Next tile of the walk without a division: +1 inside a stripe, a jump to
    # the next stripe's cluster diagonal when the stripe ends (causal GQA).
    m_tile_next = (
        f"fbwd_m_tile + 1 + cutlass.Int32((fbwd_m_tile % {m_mod_tiles}) == {m_mod_tiles - 1}) * fbwd_m_start"
        if plan.causal
        else "fbwd_m_tile + 1"
    )
    epi_scale = (
        "_helion_flash_rt._scale_fragment_packed_f32x2(fbwd_reg, cutlass.Float32("
        + dk_scale_expr
        + "))"
        if h == 1
        else "pass"
    )
    epilogue_2 = fbwd_epilogue_block(
        h=h,
        rows=256,
        d=d,
        epi_off=str(128 if h == 0 else 384),
        epi_scale=epi_scale,
        kv_tile="fbwd_cl_tile * 2 + fbwd_rank",
        io_dtype=io_dtype,
        dkv_wait="fbwd_tile_it % 2",
        v_empty_expr="storage.v_mbar_ptr.data_ptr() + 1",
    )
    s_read_arrive = (
        "            with cute.arch.elect_one():\n"
        "                _helion_flash_rt.mbarrier_arrive(fbwd_s_read_ptr)\n"
        if h == 0
        else ""
    )
    s_read_wait = (
        "            _helion_flash_rt.mbar_spin_wait(fbwd_s_read_ptr, fbwd_phase, 10000000)\n"
        if h == 1
        else ""
    )

    def p_pass(mask: str, ind: str) -> str:
        return (
            f"{ind}for fbwd_ch in cutlass.range_constexpr(4):\n"
            f"{ind}    fbwd_lse_frg = cute.make_rmem_tensor(cute.make_layout(16), cutlass.Float32)\n"
            f"{ind}    fbwd_lse_v = cute.make_tensor(fbwd_sLSE.iterator + fbwd_par + {64 * h} + fbwd_ch * 16, cute.make_layout(16))\n"
            f"{ind}    cute.autovec_copy(fbwd_lse_v, fbwd_lse_frg)\n"
            f"{ind}    _helion_flash_rt.fbwd_p_pairs_packed(tLDrS, fbwd_lse_frg, fbwd_ch * 16, 16, fbwd_scale2, {mask}, fbwd_ch * 16, exp2_f32={exp2_f32})\n"
        )

    # Causal: only the two m tiles that intersect the cluster diagonal (m rows
    # below kv_base + 256) run the masked softmax (FA4 does the same).
    if plan.causal:
        p_passes = (
            f"            if (fbwd_qbase - {64 * h}) < fbwd_kv_base + 256:\n"
            + p_pass(mask_arg, " " * 16)
            + "            else:\n"
            + p_pass("None", " " * 16)
        )
    else:
        p_passes = p_pass("None", " " * 12)
    tile_region = _fbwd2_persistent_tiles(
        f"""        fbwd_stage_col0 = fbwd_q_row_base + {_fbwd2_m_tile(plan, "0")} * 128 + {64 * h} + fbwd_local_tidx % 64
        if fbwd_local_tidx < 64:
            _helion_flash_rt.cp_async_ca_4(fbwd_sLSE.iterator + ({64 * h} + fbwd_local_tidx), _fbwd_mLSE.iterator + fbwd_stage_col0)
        else:
            _helion_flash_rt.cp_async_ca_4(fbwd_sDelta.iterator + ({64 * h} - 64 + fbwd_local_tidx), _fbwd_mDelta.iterator + fbwd_stage_col0)
        cute.arch.cp_async_commit_group()
        fbwd_m_tile = {_fbwd2_m_tile(plan, "0")}
        for fbwd_i in cutlass.range({steps_expr}, unroll=1):
            fbwd_qbase = (fbwd_m_tile % {m_mod_tiles}) * 128 + {64 * h}
            fbwd_row_base = fbwd_q_row_base + fbwd_m_tile * 128
            fbwd_mask_lim = fbwd_kv_base + fbwd_thr_row - fbwd_qbase
            fbwd_par = (fbwd_i % 2) * 128
            _helion_flash_rt.mbar_spin_wait(fbwd_s_full_ptr, fbwd_phase, 10000000)
            tLDrS = cute.make_rmem_tensor(tLDcS.shape, cutlass.Float32)
            cute.copy(fbwd_tiled_ld, tLDtS, tLDrS)
            cute.arch.fence_view_async_tmem_load()
{s_read_arrive}            cute.arch.cp_async_wait_group(0)
            _helion_flash_rt.named_barrier_wait_unaligned({3 + h}, 128)
            if fbwd_i > 0:
                with cute.arch.elect_one():
                    _helion_flash_rt.mbarrier_arrive(fbwd_ds_smem_full_ptr, cutlass.Int32(0))
            # Next iteration's LSE (threads 0-63) / Delta (64-127) go into the
            # free parity slot via cp.async: its last readers were iteration
            # i-1 and every thread of the group has passed the barrier above
            # (the last step re-copies its own row: harmless).
            # Next tile of the walk without a division: +1 inside a stripe, a
            # jump to the next stripe's cluster diagonal when the stripe ends.
            fbwd_m_tile_nxt = {m_tile_next}
            if fbwd_i + 1 < {steps_expr}:
                fbwd_stage_col = fbwd_q_row_base + fbwd_m_tile_nxt * 128 + {64 * h} + fbwd_local_tidx % 64
                if fbwd_local_tidx < 64:
                    _helion_flash_rt.cp_async_ca_4(fbwd_sLSE.iterator + (128 - fbwd_par + {64 * h} + fbwd_local_tidx), _fbwd_mLSE.iterator + fbwd_stage_col)
                else:
                    _helion_flash_rt.cp_async_ca_4(fbwd_sDelta.iterator + (128 - fbwd_par + {64 * h} - 64 + fbwd_local_tidx), _fbwd_mDelta.iterator + fbwd_stage_col)
                cute.arch.cp_async_commit_group()
{p_passes}            tSTrP = cute.make_rmem_tensor(tSTcP.shape, cutlass.Float32)
            tSTrP_e = cute.make_tensor(cute.recast_ptr(tSTrP.iterator, dtype={io_dtype}), tLDrS.layout)
            tSTrP_e.store(tLDrS.load().to({io_dtype}))
{s_read_wait}            cute.copy(fbwd_tiled_st, tSTrP, tSTtP)
            cute.arch.fence_view_async_tmem_store()
            with cute.arch.elect_one():
                _helion_flash_rt.mbarrier_arrive(fbwd_p_full_ptr, cutlass.Int32(0))
            _helion_flash_rt.mbar_spin_wait(fbwd_dp_full_ptr, fbwd_phase, 10000000)
            for fbwd_ch in cutlass.range_constexpr(2):
                tLDrdP = cute.make_rmem_tensor(tLDcS[None, None, 0].shape, cutlass.Float32)
                cute.copy(fbwd_tiled_ld, tLDtdP[None, None, fbwd_ch], tLDrdP)
                cute.arch.fence_view_async_tmem_load()
                for fbwd_hh in cutlass.range_constexpr(2):
                    fbwd_dlt_frg = cute.make_rmem_tensor(cute.make_layout(16), cutlass.Float32)
                    fbwd_dlt_v = cute.make_tensor(fbwd_sDelta.iterator + fbwd_par + {64 * h} + fbwd_ch * 32 + fbwd_hh * 16, cute.make_layout(16))
                    cute.autovec_copy(fbwd_dlt_v, fbwd_dlt_frg)
                    _helion_flash_rt.fbwd_ds_pairs_packed_off(tLDrdP, fbwd_hh * 16, tLDrS, fbwd_ch * 32 + fbwd_hh * 16, fbwd_dlt_frg, 16)
                fbwd_ds_bf = cute.make_rmem_tensor(tLDrdP.layout, {io_dtype})
                fbwd_ds_bf.store(tLDrdP.load().to({io_dtype}))
                fbwd_ds_w = cute.make_tensor(cute.recast_ptr(fbwd_ds_bf.iterator, dtype=cutlass.Float32), cute.make_layout(tSTcP[None, None, 0].shape))
                cute.copy(fbwd_tiled_st, fbwd_ds_w, tSTtDS[None, None, fbwd_ch])
                fbwd_ds_flat = cute.make_tensor(fbwd_ds_bf.iterator, cute.make_layout(32))
                if fbwd_ch == 0:
                    # sdS / the exchange staging are the dQ(i-1) operands: gate
                    # only the first smem write on the dQ gemm (FA4), not the
                    # dP pass (a later tile's first write waits for the
                    # previous tile's last dQ).
                    if (fbwd_i > 0) | (fbwd_tile_it > 0):
                        _helion_flash_rt.mbar_spin_wait(fbwd_dq_full_ptr, fbwd_dqf_phase, 10000000)
                        fbwd_dqf_phase ^= 1
                if fbwd_rank == {h}:
                    cute.autovec_copy(fbwd_ds_flat, fbwd_sdS_kv4[None, fbwd_ch, fbwd_rank, fbwd_crow])
                else:
                    cute.autovec_copy(fbwd_ds_flat, fbwd_sdSx_v2[None, fbwd_ch, fbwd_crow])
            cute.arch.fence_view_async_tmem_store()
            with cute.arch.elect_one():
                _helion_flash_rt.mbarrier_arrive(fbwd_ds_tmem_full_ptr, cutlass.Int32(0))
            cute.arch.fence_view_async_shared()
            # The group that wrote the exchange staging publishes it to the
            # peer once all of its lanes are done (group-local barrier); the
            # other group's smem writes are certified by its deferred
            # ds_smem_full arrive after next iteration's barrier.
            if fbwd_rank != {h}:
                _helion_flash_rt.named_barrier_wait_unaligned({3 + h}, 128)
{FBWD_2CTA_SEND}
            fbwd_phase ^= 1
            fbwd_m_tile = fbwd_m_tile_nxt
        _helion_flash_rt.named_barrier_wait_unaligned({3 + h}, 128)
        with cute.arch.elect_one():
            _helion_flash_rt.mbarrier_arrive(fbwd_ds_smem_full_ptr, cutlass.Int32(0))
{epilogue_2}""",
        plan,
        persistent,
    )
    return f"""
    if (warp_idx >= {warp_lo}) & (warp_idx < {warp_hi}):
        cute.arch.setmaxregister_increase(136)
{fbwd_compute_tmem_views(rows=256, s_col=64 * h, dp_col=256 + 64 * h)}        cS = cute.make_identity_tensor((256, 128))
{FBWD_COMPUTE_LOAD_VIEWS}        # P (bf16, cols 32h..32h+31) aliases the S columns dQ does NOT reuse
        # (dQ lives at 64..127), so group 1's P lands in group 0's S half: group
        # 0 publishes 's_read' once its S t2r has landed and only group 1 waits
        # on it (one-directional, no 256-thread barrier).
{fbwd_p_store_views(p_col=32 * h, ds_col=256 + 64 * h)}        fbwd_sdS_kv4 = cute.make_tensor(fbwd_sdS_mn.iterator, cute.make_layout((32, 2, 2, 128), stride=(1, 32, 8192, 64)))
        fbwd_crow = cutlass.Int32(tLDcS[0][0]) % 128
        fbwd_sdSx_v2 = cute.make_tensor(fbwd_sdSx_ptr, cute.make_layout((32, 2, 128), stride=(1, 32, 64)))
        fbwd_thr_row = cutlass.Int32(tLDcS[0][0])
        fbwd_phase = cutlass.Int32(0)
        fbwd_dqf_phase = cutlass.Int32(0)
        fbwd_scale2 = cutlass.Float32({scale2_expr})
        # LSE/Delta staging runs one iteration ahead: slot (i+1)%2 is free once
        # every thread of the group has passed the S-t2r barrier of iteration i
        # (its last reader was iteration i-1), so the global loads have a whole
        # iteration to land instead of sitting on the softmax critical path.
{tile_region}        _helion_flash_rt.named_barrier_arrive_unaligned(2, 13 * 32)
"""


def emit_flash_bwd_2cta_device_body(
    *,
    plan: AttentionBwdPlan,
    io_dtype: str,
    scale2_expr: str,
    out_scale_expr: str,
    dk_scale_expr: str,
    persistent: bool = False,
    total_pairs: int = 0,
    exp2_f32: bool = False,
) -> str:
    """2-CTA (cluster (2,1,1)) fused backward device body for head_dim 128."""
    d = plan.head_dim
    m_mod_tiles = plan.m_dim // 128
    mt0 = _fbwd2_m_tile(plan, "0")
    mti = _fbwd2_m_tile(plan, "fbwd_i")
    mti1 = _fbwd2_m_tile(plan, "fbwd_i + 1")
    mtl = _fbwd2_m_tile(plan, "fbwd_steps - 1")
    compute_blocks = "".join(
        _flash_bwd_2cta_wg_compute_block(
            h=h,
            plan=plan,
            io_dtype=io_dtype,
            scale2_expr=scale2_expr,
            dk_scale_expr=dk_scale_expr,
            m_mod_tiles=m_mod_tiles,
            steps_expr="fbwd_steps",
            persistent=persistent,
            exp2_f32=exp2_f32,
        )
        for h in (0, 1)
    )
    # The CLC scheduler (persistent walk): its state at kernel entry, warp 15
    # stealing cluster tiles on the leader CTA and consuming on the follower,
    # and the follower's MMA warp consuming the responses it has no work for.
    if persistent:
        clc_setup = fbwd_clc_setup(total_pairs * 2, 2, "_fbwd_cluster_vmnk")
        clc_warps = (
            "    if warp_idx == 15:\n        if fbwd_is_leader:\n"
            + fbwd_clc_scheduler(" " * 12, None)
            + "        else:\n"
            + fbwd_clc_consume_only(" " * 12)
        )
        clc_mma_nonleader = "        if fbwd_rank != cutlass.Int32(0):\n" + (
            fbwd_clc_consume_only(" " * 12)
        )
    else:
        clc_setup = clc_warps = clc_mma_nonleader = ""

    # The warp roles' per-tile regions (see the 1-CTA body): the relay warp's
    # dS-exchange relay, the load warp's loads, the leader MMA warp's gemm
    # chain (inside its ``if fbwd_is_leader:``), the reduce warps' drain loop.
    relay_region = _fbwd2_persistent_tiles(
        """        for fbwd_i in cutlass.range(fbwd_steps, unroll=1):
            _helion_flash_rt.mbar_spin_wait(fbwd_ds_cl_full_ptr, fbwd_rphase, 10000000)
            fbwd_rphase ^= 1
            with cute.arch.elect_one():
                _helion_flash_rt.mbarrier_arrive(fbwd_ds_cl_leader_ptr, cutlass.Int32(0))
""",
        plan,
        persistent,
    )
    load_region = _fbwd2_persistent_tiles(
        f"""        fbwd_ke = fbwd_k_prod.acquire_and_advance()
        cute.copy(_fbwd_tma_k, tKgK[None, fbwd_cl_tile], tKsK[None, fbwd_ke.index], tma_bar_ptr=fbwd_ke.barrier)
        fbwd_q_tile = fbwd_q_tile_base + {mt0}
        fbwd_qe = fbwd_q_prod.acquire_and_advance()
        cute.copy(_fbwd_tma_q, tQgQ[None, fbwd_q_tile], tQsQ[None, fbwd_qe.index], tma_bar_ptr=fbwd_qe.barrier)
        fbwd_ve = fbwd_v_prod.acquire_and_advance()
        cute.copy(_fbwd_tma_v, tVgV[None, fbwd_cl_tile], tVsV[None, fbwd_ve.index], tma_bar_ptr=fbwd_ve.barrier)
        fbwd_doe = fbwd_do_prod.acquire_and_advance()
        cute.copy(_fbwd_tma_dot, tDOTgDOT[None, fbwd_q_tile], tDOTsDOT[None, fbwd_doe.index], tma_bar_ptr=fbwd_doe.barrier)
        cute.copy(_fbwd_tma_do2, tDOgDO[None, fbwd_q_tile], tDOsDO[None, fbwd_doe.index], tma_bar_ptr=fbwd_doe.barrier)
        fbwd_kte = fbwd_kt_prod.acquire_and_advance()
        cute.copy(_fbwd_tma_kt, tKTgKT[None, fbwd_cl_tile], tKTsKT[None, fbwd_kte.index], tma_bar_ptr=fbwd_kte.barrier)
        for fbwd_i in cutlass.range(fbwd_steps - 1, unroll=1):
            fbwd_q_tile = fbwd_q_tile_base + {mti}
            fbwd_q_tile_n = fbwd_q_tile_base + {mti1}
            fbwd_qte = fbwd_qt_prod.acquire_and_advance()
            cute.copy(_fbwd_tma_qt, tQTgQT[None, fbwd_q_tile], tQTsQT[None, fbwd_qte.index], tma_bar_ptr=fbwd_qte.barrier)
            fbwd_qe = fbwd_q_prod.acquire_and_advance()
            cute.copy(_fbwd_tma_q, tQgQ[None, fbwd_q_tile_n], tQsQ[None, fbwd_qe.index], tma_bar_ptr=fbwd_qe.barrier)
            fbwd_doe = fbwd_do_prod.acquire_and_advance()
            cute.copy(_fbwd_tma_dot, tDOTgDOT[None, fbwd_q_tile_n], tDOTsDOT[None, fbwd_doe.index], tma_bar_ptr=fbwd_doe.barrier)
            cute.copy(_fbwd_tma_do2, tDOgDO[None, fbwd_q_tile_n], tDOsDO[None, fbwd_doe.index], tma_bar_ptr=fbwd_doe.barrier)
        fbwd_q_tile = fbwd_q_tile_base + {mtl}
        fbwd_qte = fbwd_qt_prod.acquire_and_advance()
        cute.copy(_fbwd_tma_qt, tQTgQT[None, fbwd_q_tile], tQTsQT[None, fbwd_qte.index], tma_bar_ptr=fbwd_qte.barrier)
""",
        plan,
        persistent,
    )
    mma_region = _fbwd2_persistent_tiles(
        f"""            fbwd_ke = fbwd_k_cons.wait_and_advance()
            if fbwd_tile_it > 0:
                # dQ of the previous tile aliases S columns 64-127: wait for
                # the reduce warps to have read it before this tile's S(0).
                _helion_flash_rt.mbar_spin_wait(fbwd_dq_empty_ptr, fbwd_dqe_phase, 10000000)
                fbwd_dqe_phase ^= 1
            fbwd_dk_zero = cutlass.Boolean(True)
            fbwd_st_k = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 0)
            fbwd_st_v = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 1)
            fbwd_st_dsmn = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 2)
            fbwd_st_q = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 3)
            fbwd_st_dot = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 4)
            fbwd_st_do = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 5)
            fbwd_st_qt = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 6)
            fbwd_st_kt = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 7)
            fbwd_qe = fbwd_q_cons.wait_and_advance()
            _helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_s_addr, fbwd_st_k, fbwd_k_base, tSrK[None, None, None, 0].layout, fbwd_st_q, fbwd_q_base, tSrQ[None, None, None, 0].layout, "helion_fbwd_ss_idesc", zero_init=True, cta_group=2)
            with cute.arch.elect_one():
                cute_tcgen05_flash.commit(fbwd_s_full_ptr, fbwd_mcast, cute_tcgen05_flash.CtaGroup.TWO)
            fbwd_qe.release()
            fbwd_doe = fbwd_do_cons.wait_and_advance()
            fbwd_ve = fbwd_v_cons.wait_and_advance()
            _helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_dp_addr, fbwd_st_v, fbwd_v_base, tSrV[None, None, None, 0].layout, fbwd_st_dot, fbwd_dot_base, tSrDOt[None, None, None, 0].layout, "helion_fbwd_ss_idesc", zero_init=True, cta_group=2)
            with cute.arch.elect_one():
                cute_tcgen05_flash.commit(fbwd_dp_full_ptr, fbwd_mcast, cute_tcgen05_flash.CtaGroup.TWO)
            _helion_flash_rt.mbar_spin_wait(fbwd_p_full_ptr, fbwd_pf_phase, 10000000)
            fbwd_pf_phase ^= 1
            _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(fbwd_dv_addr, fbwd_s_addr, fbwd_st_do, fbwd_do2_base, tTSrP[None, None, None, 0].layout, tTSrDO[None, None, None, 0].layout, "helion_fbwd_ts_idesc", zero_init=True, cta_group=2)
            fbwd_doe.release()
            fbwd_kte = fbwd_kt_cons.wait_and_advance()
            for fbwd_i in cutlass.range(fbwd_steps - 1, unroll=1):
                fbwd_st_k = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 0)
                fbwd_st_v = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 1)
                fbwd_st_dsmn = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 2)
                fbwd_st_q = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 3)
                fbwd_st_dot = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 4)
                fbwd_st_do = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 5)
                fbwd_st_qt = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 6)
                fbwd_st_kt = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 7)
                if fbwd_i > 0:
                    _helion_flash_rt.mbar_spin_wait(fbwd_dq_empty_ptr, fbwd_dqe_phase, 10000000)
                    fbwd_dqe_phase ^= 1
                fbwd_qe = fbwd_q_cons.wait_and_advance()
                _helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_s_addr, fbwd_st_k, fbwd_k_base, tSrK[None, None, None, 0].layout, fbwd_st_q, fbwd_q_base, tSrQ[None, None, None, 0].layout, "helion_fbwd_ss_idesc", zero_init=True, cta_group=2)
                with cute.arch.elect_one():
                    cute_tcgen05_flash.commit(fbwd_s_full_ptr, fbwd_mcast, cute_tcgen05_flash.CtaGroup.TWO)
                fbwd_qe.release()
                fbwd_qte = fbwd_qt_cons.wait_and_advance()
                _helion_flash_rt.mbar_spin_wait(fbwd_ds_tmem_full_ptr, fbwd_dst_phase, 10000000)
                fbwd_dst_phase ^= 1
                _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(fbwd_dk_addr, fbwd_dp_addr, fbwd_st_qt, fbwd_qt_base, tTSrDS[None, None, None, 0].layout, tTSrQt[None, None, None, 0].layout, "helion_fbwd_ts_idesc", zero_init=fbwd_dk_zero, cta_group=2, a_offsets={FBWD_2CTA_DS_OFFSETS})
                fbwd_dk_zero = cutlass.Boolean(False)
                fbwd_qte.release()
                fbwd_doe = fbwd_do_cons.wait_and_advance()
                _helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_dp_addr, fbwd_st_v, fbwd_v_base, tSrV[None, None, None, 0].layout, fbwd_st_dot, fbwd_dot_base, tSrDOt[None, None, None, 0].layout, "helion_fbwd_ss_idesc", zero_init=True, cta_group=2)
                with cute.arch.elect_one():
                    cute_tcgen05_flash.commit(fbwd_dp_full_ptr, fbwd_mcast, cute_tcgen05_flash.CtaGroup.TWO)
                _helion_flash_rt.mbar_spin_wait(fbwd_ds_smem_full_ptr, fbwd_dss_phase, 10000000)
                fbwd_dss_phase ^= 1
                _helion_flash_rt.mbar_spin_wait(fbwd_ds_cl_leader_ptr, fbwd_dcl_phase, 10000000)
                fbwd_dcl_phase ^= 1
                _helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_dqa_addr, fbwd_st_dsmn, fbwd_dsmn_base, tDQrDS[None, None, None, 0].layout, fbwd_st_kt, fbwd_kt_base, tDQrKt[None, None, None, 0].layout, "helion_fbwd_dq_idesc", zero_init=True, cta_group=2)
                with cute.arch.elect_one():
                    cute_tcgen05_flash.commit(fbwd_dq_full_ptr, fbwd_mcast, cute_tcgen05_flash.CtaGroup.TWO)
                _helion_flash_rt.mbar_spin_wait(fbwd_p_full_ptr, fbwd_pf_phase, 10000000)
                fbwd_pf_phase ^= 1
                _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(fbwd_dv_addr, fbwd_s_addr, fbwd_st_do, fbwd_do2_base, tTSrP[None, None, None, 0].layout, tTSrDO[None, None, None, 0].layout, "helion_fbwd_ts_idesc", zero_init=False, cta_group=2)
                fbwd_doe.release()
            fbwd_st_k = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 0)
            fbwd_st_v = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 1)
            fbwd_st_dsmn = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 2)
            fbwd_st_q = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 3)
            fbwd_st_dot = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 4)
            fbwd_st_do = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 5)
            fbwd_st_qt = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 6)
            fbwd_st_kt = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 7)
            with cute.arch.elect_one():
                cute_tcgen05_flash.commit(fbwd_dv_done_ptr, fbwd_mcast, cute_tcgen05_flash.CtaGroup.TWO)
            fbwd_qte = fbwd_qt_cons.wait_and_advance()
            _helion_flash_rt.mbar_spin_wait(fbwd_ds_tmem_full_ptr, fbwd_dst_phase, 10000000)
            fbwd_dst_phase ^= 1
            _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(fbwd_dk_addr, fbwd_dp_addr, fbwd_st_qt, fbwd_qt_base, tTSrDS[None, None, None, 0].layout, tTSrQt[None, None, None, 0].layout, "helion_fbwd_ts_idesc", zero_init=fbwd_dk_zero, cta_group=2, a_offsets={FBWD_2CTA_DS_OFFSETS})
            fbwd_qte.release()
            if fbwd_steps > 1:
                _helion_flash_rt.mbar_spin_wait(fbwd_dq_empty_ptr, fbwd_dqe_phase, 10000000)
                fbwd_dqe_phase ^= 1
            _helion_flash_rt.mbar_spin_wait(fbwd_ds_smem_full_ptr, fbwd_dss_phase, 10000000)
            fbwd_dss_phase ^= 1
            _helion_flash_rt.mbar_spin_wait(fbwd_ds_cl_leader_ptr, fbwd_dcl_phase, 10000000)
            fbwd_dcl_phase ^= 1
            _helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_dqa_addr, fbwd_st_dsmn, fbwd_dsmn_base, tDQrDS[None, None, None, 0].layout, fbwd_st_kt, fbwd_kt_base, tDQrKt[None, None, None, 0].layout, "helion_fbwd_dq_idesc", zero_init=True, cta_group=2)
            with cute.arch.elect_one():
                cute_tcgen05_flash.commit(fbwd_dq_full_ptr, fbwd_mcast, cute_tcgen05_flash.CtaGroup.TWO)
            with cute.arch.elect_one():
                cute_tcgen05_flash.commit(fbwd_dkv_done_ptr, fbwd_mcast, cute_tcgen05_flash.CtaGroup.TWO)
            fbwd_ke.release()
            fbwd_kte.release()
""",
        plan,
        persistent,
        indent=12,
    )
    reduce_region = _fbwd2_persistent_tiles(
        f"""        for fbwd_i in cutlass.range(fbwd_steps, unroll=1):
            fbwd_m_tile = {mti}
            fbwd_row_base = fbwd_q_row_base + fbwd_m_tile * 128
            _helion_flash_rt.mbar_spin_wait(fbwd_dq_full_ptr, fbwd_phase, 10000000)
            fbwd_phase ^= 1
            fbwd_gdq_row = fbwd_row_base // 32 + fbwd_my_m // 32
            tRDrDQ = cute.make_rmem_tensor(tRDcDQ.shape, cutlass.Float32)
            cute.copy(fbwd_tiled_dq_ld, tRDtDQ, tRDrDQ)
            cute.arch.fence_view_async_tmem_load()
            with cute.arch.elect_one():
                _helion_flash_rt.mbarrier_arrive(fbwd_dq_empty_ptr, cutlass.Int32(0))
            _helion_flash_rt._scale_fragment_packed_f32x2(tRDrDQ, fbwd_out_scale)
            for fbwd_c in cutlass.range_constexpr({d // 64}):
                fbwd_dq_flat = cute.make_tensor(tRDrDQ[None, fbwd_c, 0, 0].iterator, cute.make_layout(32))
                fbwd_sdq_box = fbwd_sdq[None, None, warp_idx]
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.sync_warp()
                cute.autovec_copy(fbwd_dq_flat, fbwd_sdq_box[fbwd_my_row, None])
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                fbwd_gdq_box = fbwd_gdq[None, None, fbwd_gdq_row, warp_idx // 2 * {d // 64} + fbwd_c]
                tDQsDQ, tDQgDQ = cute_cpasync_flash.tma_partition(_fbwd_tma_dq, 0, cute.make_layout(1), cute.group_modes(fbwd_sdq_box, 0, 2), cute.group_modes(fbwd_gdq_box, 0, 2))
                cute.copy(_fbwd_tma_dq, tDQsDQ, tDQgDQ)
                cute.arch.cp_async_bulk_commit_group()
""",
        plan,
        persistent,
    )
    mma_warp = fbwd_tmem_warp(
        "        if fbwd_is_leader:\n"
        + fbwd_2cta_mma_setup(
            d=d, s_base="fbwd_tmem_ptr", dp_off=256, dv_off=128, dk_off=384, dq_off=64
        )
        + mma_region
        + clc_mma_nonleader
    )
    return f"""
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    fbwd_local_tidx = tidx % 128
    fbwd_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    fbwd_is_leader = fbwd_rank == cutlass.Int32(0)
    fbwd_tile_it = cutlass.Int32(0)
    fbwd_num_clusters = cutlass.Int32(cute.arch.grid_dim()[0]) // 2
    fbwd_cl_tile = cutlass.Int32(cute.arch.block_idx()[0]) // 2
{_fbwd2_tile_setup_lines(plan, "    ")}    if warp_idx == 0:
        cute_cpasync_flash.prefetch_descriptor(_fbwd_tma_q)
        cute_cpasync_flash.prefetch_descriptor(_fbwd_tma_k)
        cute_cpasync_flash.prefetch_descriptor(_fbwd_tma_v)
        cute_cpasync_flash.prefetch_descriptor(_fbwd_tma_dot)
        cute_cpasync_flash.prefetch_descriptor(_fbwd_tma_do2)
        cute_cpasync_flash.prefetch_descriptor(_fbwd_tma_qt)
        cute_cpasync_flash.prefetch_descriptor(_fbwd_tma_kt)
        cute_cpasync_flash.prefetch_descriptor(_fbwd_tma_dq)
    _fbwd_storage_cls = _helion_flash_rt.flash_bwd_2cta_shared_storage({d}, {io_dtype})
    smem = cutlass_utils_flash.SmemAllocator()
    storage = smem.allocate(_fbwd_storage_cls)
{FBWD_2CTA_SMEM_VIEWS}    fbwd_s_full_ptr = storage.s_full_mbar.data_ptr()
    fbwd_dp_full_ptr = storage.dp_full_mbar.data_ptr()
    fbwd_p_full_ptr = storage.p_full_mbar.data_ptr()
    fbwd_ds_tmem_full_ptr = storage.ds_tmem_full_mbar.data_ptr()
    fbwd_ds_smem_full_ptr = storage.ds_smem_full_mbar.data_ptr()
    fbwd_dq_full_ptr = storage.dq_full_mbar.data_ptr()
    fbwd_dq_empty_ptr = storage.dq_empty_mbar.data_ptr()
    fbwd_dkv_done_ptr = storage.dkv_done_mbar.data_ptr()
    fbwd_dv_done_ptr = storage.dv_done_mbar.data_ptr()
    fbwd_ds_cl_full_ptr = storage.ds_cluster_full_mbar.data_ptr()
    fbwd_ds_cl_leader_ptr = storage.ds_cluster_leader_mbar.data_ptr()
    fbwd_s_read_ptr = storage.s_read_mbar.data_ptr()
    fbwd_tmem_dealloc_ptr = storage.tmem_dealloc_mbar.data_ptr()
    if tidx == 0:
        cute.arch.mbarrier_init(fbwd_s_full_ptr, 1)
        cute.arch.mbarrier_init(fbwd_dp_full_ptr, 1)
        cute.arch.mbarrier_init(fbwd_p_full_ptr, 16)
        cute.arch.mbarrier_init(fbwd_ds_tmem_full_ptr, 16)
        cute.arch.mbarrier_init(fbwd_ds_smem_full_ptr, 16)
        cute.arch.mbarrier_init(fbwd_dq_full_ptr, 1)
        cute.arch.mbarrier_init(fbwd_dq_empty_ptr, 8)
        cute.arch.mbarrier_init(fbwd_dkv_done_ptr, 1)
        cute.arch.mbarrier_init(fbwd_dv_done_ptr, 1)
        cute.arch.mbarrier_init(fbwd_ds_cl_full_ptr, 1)
        cute.arch.mbarrier_init(fbwd_ds_cl_leader_ptr, 2)
        cute.arch.mbarrier_init(fbwd_s_read_ptr, 4)
    cute.arch.mbarrier_init_fence()
    fbwd_tmem_user_bar = cutlass_pipeline_flash.NamedBarrier(barrier_id=2, num_threads=13 * 32)
    fbwd_tmem = cutlass_utils_flash.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=fbwd_tmem_user_bar, allocator_warp_id=12, is_two_cta=True, two_cta_tmem_dealloc_mbar_ptr=fbwd_tmem_dealloc_ptr)
    fbwd_q_bytes = cute.size_in_bytes({io_dtype}, cute.select(_fbwd_qsl, mode=[0, 1, 2])) * 2
    fbwd_k_bytes = cute.size_in_bytes({io_dtype}, cute.select(_fbwd_ksl, mode=[0, 1, 2])) * 2
    fbwd_kt_bytes = cute.size_in_bytes({io_dtype}, cute.select(_fbwd_ktl, mode=[0, 1, 2])) * 2
    fbwd_do_bytes = fbwd_q_bytes * 2
    fbwd_q_prod, fbwd_q_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(num_stages=1, producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), tx_count=fbwd_q_bytes, barrier_storage=storage.q_mbar_ptr.data_ptr(), cta_layout_vmnk=_fbwd_cluster_vmnk).make_participants()
    fbwd_qt_prod, fbwd_qt_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(num_stages=1, producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), tx_count=fbwd_q_bytes, barrier_storage=storage.qt_mbar_ptr.data_ptr(), cta_layout_vmnk=_fbwd_cluster_vmnk).make_participants()
    fbwd_kt_prod, fbwd_kt_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(num_stages=1, producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), tx_count=fbwd_kt_bytes, barrier_storage=storage.kt_mbar_ptr.data_ptr(), cta_layout_vmnk=_fbwd_cluster_vmnk).make_participants()
    fbwd_k_prod, fbwd_k_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(num_stages=1, producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), tx_count=fbwd_k_bytes, barrier_storage=storage.k_mbar_ptr.data_ptr(), cta_layout_vmnk=_fbwd_cluster_vmnk).make_participants()
    fbwd_v_prod, fbwd_v_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(num_stages=1, producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), tx_count=fbwd_k_bytes, barrier_storage=storage.v_mbar_ptr.data_ptr(), cta_layout_vmnk=_fbwd_cluster_vmnk).make_participants()
    fbwd_do_prod, fbwd_do_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(num_stages=1, producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), tx_count=fbwd_do_bytes, barrier_storage=storage.do_mbar_ptr.data_ptr(), cta_layout_vmnk=_fbwd_cluster_vmnk).make_participants()
    cutlass_pipeline_flash.pipeline_init_arrive(cluster_shape_mn=_fbwd_cluster_vmnk, is_relaxed=True)
    cutlass_pipeline_flash.pipeline_init_wait(cluster_shape_mn=_fbwd_cluster_vmnk)
{clc_setup}{fbwd_2cta_operand_partitions(d)}    if (warp_idx >= 12) & (warp_idx < 16):
        cute.arch.setmaxregister_decrease(104)
{clc_warps}    if warp_idx == 14:
        fbwd_rphase = cutlass.Int32(0)
{relay_region}    if warp_idx == 13:
        tKgK = tKgK_kdl[None, None, 0, 0]
        tVgV = tVgV_kdl[None, None, 0, 0]
        tQgQ = tQgQ_qdl[None, None, 0, 0]
        tDOTgDOT = tDOTgDOT_qdl[None, None, 0, 0]
        tDOgDO = tDOgDO_qdl[None, 0, None, 0]
        tQTgQT = tQTgQT_qdl[None, 0, None, 0]
        tKTgKT = tKTgKT_kdl[None, 0, None, 0]
        # Issue order follows first use: K + Q(0) feed S(0), V + dO(0) feed
        # dP(0), Kt is first needed by dQ(0) two gemms later.
{load_region}        fbwd_q_prod.tail()
        fbwd_qt_prod.tail()
        fbwd_do_prod.tail()
        fbwd_k_prod.tail()
        fbwd_v_prod.tail()
        fbwd_kt_prod.tail()
{mma_warp}    if warp_idx < 4:
        cute.arch.setmaxregister_increase(136)
{fbwd_dq_reduce_views(d=d, dq_off=64)}{fbwd_dq_reduce_partitions("        ", d=d)}        fbwd_gdq = cute.local_tile(_fbwd_mDQt, (32, 32), (None, None))
        fbwd_out_scale = cutlass.Float32({out_scale_expr})
        fbwd_my_m = cutlass.Int32(tRDcDQ[0][0])
        fbwd_my_row = fbwd_my_m % 32
        fbwd_phase = cutlass.Int32(0)
{reduce_region}        cute.arch.cp_async_bulk_wait_group(0)
        _helion_flash_rt.named_barrier_arrive_unaligned(2, 13 * 32)
{compute_blocks}"""


emit_flash_bwd_2cta_body = source_adapter(emit_flash_bwd_2cta_device_body)


_FLASH_BWD_2CTA_KERNEL_PARAMS = (
    "_fbwd_ss_mma",
    "_fbwd_ts_mma",
    "_fbwd_dq_mma",
    "_fbwd_cluster_vmnk",
    "_fbwd_tma_q",
    "_fbwd_mQt",
    "_fbwd_tma_k",
    "_fbwd_mKt",
    "_fbwd_tma_v",
    "_fbwd_mVt",
    "_fbwd_tma_dot",
    "_fbwd_mdOtN",
    "_fbwd_tma_do2",
    "_fbwd_mdOtT",
    "_fbwd_tma_qt",
    "_fbwd_mQtT",
    "_fbwd_tma_kt",
    "_fbwd_mKtT",
    "_fbwd_qsl",
    "_fbwd_ksl",
    "_fbwd_vsl",
    "_fbwd_dosl",
    "_fbwd_ptl",
    "_fbwd_dotl",
    "_fbwd_qtl",
    "_fbwd_ktl",
    "_fbwd_dssl",
    "_fbwd_mLSE",
    "_fbwd_mDelta",
    "_fbwd_mDQ",
    "_fbwd_tma_dq",
    "_fbwd_mDQt",
    "_fbwd_dqsl",
    "_fbwd_mdK",
    "_fbwd_mdV",
    "_fbwd_epil",
    "_fbwd_tma_dv",
    "_fbwd_mdVt",
    "_fbwd_tma_dk",
    "_fbwd_mdKt",
)


_FLASH_BWD_KERNEL_PARAMS = (
    "_fbwd_ss_mma",
    "_fbwd_ts_mma",
    "_fbwd_dsk_mma",
    "_fbwd_dq_mma",
    "_fbwd_tma_q",
    "_fbwd_mQt",
    "_fbwd_tma_k",
    "_fbwd_mKt",
    "_fbwd_tma_v",
    "_fbwd_mVt",
    "_fbwd_tma_do",
    "_fbwd_mdOt",
    "_fbwd_qsl",
    "_fbwd_ksl",
    "_fbwd_vsl",
    "_fbwd_dosl",
    "_fbwd_ptl",
    "_fbwd_qtl",
    "_fbwd_dotl",
    "_fbwd_ktl",
    "_fbwd_dssl",
    "_fbwd_dsnk",
    "_fbwd_mLSE",
    "_fbwd_mDelta",
    "_fbwd_mDQ",
    "_fbwd_tma_dq",
    "_fbwd_mDQt",
    "_fbwd_dqsl",
    "_fbwd_mdK",
    "_fbwd_mdV",
    "_fbwd_epil",
    "_fbwd_tma_dv",
    "_fbwd_mdVt",
    "_fbwd_tma_dk",
    "_fbwd_mdKt",
)


def codegen_attention_flash_bwd(cg: GenerateAST) -> bool:
    """Replace the device body with the fused tcgen05 backward kernel.

    Mirrors ``codegen_attention_flash``: returns True when the kernel was
    emitted (the FX-derived body is skipped), False to fall back (the caller
    raises BackendUnsupported).
    """

    from ..compile_environment import CompileEnvironment
    from .cute_flash import emit_flash_module_statements

    df = cg.device_function
    match = df.cute_state.flash.bwd_match
    from .flex_bwd_match import FlexBwdMatch

    if isinstance(match, FlexBwdMatch):
        from .cute_flash_bwd_flex import codegen_attention_flash_bwd_flex

        return codegen_attention_flash_bwd_flex(cg, match)
    if not isinstance(match, AttentionBwdMatch):
        return False
    plan: AttentionBwdPlan = match.plan

    from ..device_function import TensorArg
    from .cute_flash import _flash_graph_host_tensors

    host_tensors = _flash_graph_host_tensors(df.codegen.codegen_graphs)
    role_names = {
        "q": match.q_name,
        "k": match.k_name,
        "v": match.v_name,
        "do": match.do_name,
        "lse": match.lse_name,
        "delta": match.delta_name,
        "dq": match.dq_name,
        "dk": match.dk_name,
        "dv": match.dv_name,
    }
    if any(name not in host_tensors for name in role_names.values()):
        return False
    for name in dict.fromkeys(role_names.values()):
        df.tensor_arg(host_tensors[name], prefer_name=name)
    tensor_args_by_name = {a.name: a for a in df.arguments if isinstance(a, TensorArg)}
    args = {role: tensor_args_by_name.get(name) for role, name in role_names.items()}
    if any(arg is None for arg in args.values()):
        return False

    total_q_rows = plan.num_groups * plan.mm_dim
    d = plan.head_dim
    for role in ("q", "do"):
        fake = args[role].fake_value  # type: ignore[union-attr]
        if (
            fake.ndim != 2
            or fake.dtype != plan.io_dtype
            or not fake.is_contiguous()
            or int(fake.shape[0]) != total_q_rows
            or int(fake.shape[1]) != d
        ):
            return False
    for role in ("k", "v", "dk", "dv"):
        fake = args[role].fake_value  # type: ignore[union-attr]
        if (
            fake.ndim != 2
            or fake.dtype != plan.io_dtype
            or not fake.is_contiguous()
            or int(fake.shape[0]) != plan.total_kv_rows
            or int(fake.shape[1]) != d
        ):
            return False
    for role in ("lse", "delta"):
        fake = args[role].fake_value  # type: ignore[union-attr]
        if (
            fake.ndim != 1
            or fake.dtype != torch.float32
            or not fake.is_contiguous()
            or int(fake.shape[0]) != total_q_rows
        ):
            return False
    dq_fake = args["dq"].fake_value  # type: ignore[union-attr]
    if (
        dq_fake.ndim != 2
        or dq_fake.dtype != torch.float32
        or not dq_fake.is_contiguous()
        or int(dq_fake.shape[0]) != total_q_rows
        or int(dq_fake.shape[1]) != d
    ):
        return False

    qk_scale, dq_scale, dk_scale = (
        match.qk_scale_sym,
        match.out_scale_sym,
        match.dk_scale_sym,
    )
    if qk_scale is None or dq_scale is None or dk_scale is None:
        return False

    def render(scale: sympy.Expr | float) -> str:
        return repr(scale) if isinstance(scale, float) else df.sympy_expr(scale)

    scale2_expr = render(qk_scale)
    out_scale_expr = render(dq_scale)
    dk_scale_expr = render(dk_scale)

    env = CompileEnvironment.current()
    io_dtype_str = env.backend.dtype_str(plan.io_dtype)
    from .cute_flash_bwd_row_mma import FLASH_BWD_FAMILY_KEY
    from .cute_flash_bwd_row_mma import FLASH_BWD_ROW_FAMILY

    if df.config.config.get(FLASH_BWD_FAMILY_KEY) == FLASH_BWD_ROW_FAMILY:
        # The register-MMA row programs (``cute_flash_bwd_family`` =
        # "row_mma"): one CTA per (group, kv_rows) tile, no TMA / TMEM. A
        # knob triple the plan cannot run is declined (fail closed; the
        # normalizer keeps searched configs legal).
        return _codegen_flash_bwd_row_mma(
            cg,
            match,
            args=args,  # type: ignore[arg-type]
            io_dtype=io_dtype_str,
            scale2_expr=scale2_expr,
            out_scale_expr=out_scale_expr,
            dk_scale_expr=dk_scale_expr,
        )
    # The 2-CTA cluster family (FA4's d128 configuration): a cluster of two
    # CTAs shares one 256-row KV tile, each CTA loads half of Q/dO and drains
    # half of dQ, and the dS halves are exchanged over DSMEM. An autotuner
    # knob (``cute_flash_bwd_two_cta``) where the shape allows it.
    two_cta = _flash_bwd_two_cta_allowed(plan) and bool(
        df.config.config.get(FLASH_BWD_TWO_CTA_KEY, 0)
    )
    # Softmax exp2 flavour (``cute_flash_bwd_exp2_f32``): two f32 MUFU exp2 per
    # pair, or one packed f16x2 exp2 plus the f16 pack/unpack ALU work.
    exp2_f32 = bool(df.config.config.get(FLASH_BWD_EXP2_F32_KEY, 0))
    # FA4 1-CTA staging: Q double-buffered (spans two skewed iterations);
    # TMA ring depths, bounded by the 227KB smem budget: head_dim 128 fits
    # Q x2 + dO x1 next to K/V/dS/dQ staging; head_dim 64 has room for 4-deep
    # Q and dO rings, which hide the load latency under the dQ reduce traffic
    # (-5% at s8192/d64 vs 2/2).
    q_stage = 2 if d == 128 else 4
    do_stage = 1 if d == 128 else 4
    total_tiles = plan.total_kv_rows // 128
    # Persistent tile scheduler (cute_flash_bwd_persistent=1): the full grid is
    # launched and running CTAs steal not-yet-launched tiles through cluster
    # launch control (warp 15), so the assignment stays dynamic (a static
    # round-robin walk was 1-5% slower than the plain launch).
    persistent = bool(df.config.config.get(FLASH_BWD_PERSISTENT_KEY, 0))
    # K/V ring depth. Double-buffering K/V for persistent tiles (3-deep Q/dO
    # rings to pay for it) measured ~1% slower than a single slot under the
    # dynamic scheduler, so the ring stays single-stage.
    kv_stage = 1

    emit_flash_module_statements(cg)

    wrapper_plan: dict[str, object] = {
        "kind": "helion_flash_bwd",
        "q_name": args["q"].name,  # type: ignore[union-attr]
        "k_name": args["k"].name,  # type: ignore[union-attr]
        "v_name": args["v"].name,  # type: ignore[union-attr]
        "do_name": args["do"].name,  # type: ignore[union-attr]
        "lse_name": args["lse"].name,  # type: ignore[union-attr]
        "delta_name": args["delta"].name,  # type: ignore[union-attr]
        "dq_name": args["dq"].name,  # type: ignore[union-attr]
        "dk_name": args["dk"].name,  # type: ignore[union-attr]
        "dv_name": args["dv"].name,  # type: ignore[union-attr]
        "head_dim": d,
        "dtype": io_dtype_str,
        "total_q_rows": total_q_rows,
        "total_kv_rows": plan.total_kv_rows,
        "q_stage": q_stage,
        "do_stage": do_stage,
        "kv_stage": kv_stage,
        "two_cta": two_cta,
        "persistent": persistent,
        # "fa4" makes the launcher pass min_blocks_per_mp=1; without it ptxas
        # silently drops every setmaxnreg and all 16 warps run at a flat 128
        # registers (no USETMAXREG in the cubin). With the TMA dQ drain the
        # honored 80/152/128 split is ~4% faster at s8192/d128.
        "topology": "fa4",
    }
    codegen_state = cute_codegen_state(cg)
    codegen_state.wrapper_plans.append(wrapper_plan)
    if two_cta:
        df.wrapper_only_params.extend(_FLASH_BWD_2CTA_KERNEL_PARAMS)
        df.cute_state.tcgen05.cluster_shape = (2, 1, 1)
    else:
        df.wrapper_only_params.extend(_FLASH_BWD_KERNEL_PARAMS)
    df.placeholder_args.update(a.name for a in args.values())  # type: ignore[union-attr]
    codegen_state.declare_matmul()
    df.cute_state.flash.threads = 512

    if two_cta:
        emitted = emit_flash_bwd_2cta_body(
            plan=plan,
            io_dtype=io_dtype_str,
            scale2_expr=scale2_expr,
            out_scale_expr=out_scale_expr,
            dk_scale_expr=dk_scale_expr,
            persistent=persistent,
            total_pairs=plan.total_kv_rows // 256,
            exp2_f32=exp2_f32,
        )
    else:
        emitted = emit_flash_bwd_body(
            plan=plan,
            io_dtype=io_dtype_str,
            scale2_expr=scale2_expr,
            out_scale_expr=out_scale_expr,
            dk_scale_expr=dk_scale_expr,
            q_stage=q_stage,
            do_stage=do_stage,
            kv_stage=kv_stage,
            persistent=persistent,
            total_tiles=total_tiles,
            exp2_f32=exp2_f32,
        )
    df.body = emitted.statements
    df.preamble = []
    return True


def _codegen_flash_bwd_row_mma(
    cg: GenerateAST,
    match: AttentionBwdMatch,
    *,
    args: dict[str, TensorArg],
    io_dtype: str,
    scale2_expr: str,
    out_scale_expr: str,
    dk_scale_expr: str,
) -> bool:
    """Emit the register-MMA row-program backward (``cute_flash_bwd_row_mma``)."""

    from .cute_flash_bwd_row_mma import FLASH_BWD_ROW_PLAN_KIND
    from .cute_flash_bwd_row_mma import emit_flash_bwd_row_mma_body
    from .cute_flash_bwd_row_mma import emit_flash_bwd_row_mma_module_statements
    from .cute_flash_bwd_row_mma import flash_bwd_row_facts
    from .cute_flash_bwd_row_mma import flash_bwd_row_knobs
    from .cute_flash_bwd_row_mma import flash_bwd_row_plan
    from .cute_flash_bwd_row_mma import flash_bwd_row_warps

    df = cg.device_function
    plan = match.plan
    facts = flash_bwd_row_facts(plan)
    kv_rows, q_split, q_chunk = flash_bwd_row_knobs(df.config.config)
    stages = flash_bwd_row_plan(
        facts, kv_rows=kv_rows, q_split=q_split, q_chunk=q_chunk
    )
    if stages is None:
        return False
    warps = flash_bwd_row_warps(kv_rows, q_split)
    emit_flash_bwd_row_mma_module_statements(cg)
    names = {role: arg.name for role, arg in args.items()}
    total_tiles = plan.num_groups * (plan.n_dim // kv_rows)
    codegen_state = cute_codegen_state(cg)
    codegen_state.wrapper_plans.append(
        {
            "kind": FLASH_BWD_ROW_PLAN_KIND,
            **{f"{role}_name": name for role, name in names.items()},
            "head_dim": plan.head_dim,
            "dtype": io_dtype,
            "kv_rows": kv_rows,
            "q_split": q_split,
            "q_chunk": q_chunk,
            "stages": stages,
            "row_warps": warps,
            "total_tiles": total_tiles,
        }
    )
    df.placeholder_args.update(names.values())
    codegen_state.declare_matmul()
    df.cute_state.flash.threads = 32 * warps
    # The launcher sizes the grid from the plan (one CTA per KV tile), not
    # from the 128-row program-id grid: the dependent-launch release rule
    # counts the launch's CTAs from here.
    df.cute_state.lanes.launch_ctas = total_tiles
    emitted = emit_flash_bwd_row_mma_body(
        plan=plan,
        io_dtype=io_dtype,
        scale2_expr=scale2_expr,
        out_scale_expr=out_scale_expr,
        dk_scale_expr=dk_scale_expr,
        kv_rows=kv_rows,
        q_split=q_split,
        q_chunk=q_chunk,
        stages=stages,
        **{f"{role}_name": name for role, name in names.items()},
    )
    df.body = emitted.statements
    df.preamble = []
    return True


def _flash_bwd_two_cta_allowed(plan: AttentionBwdPlan) -> bool:
    """Shapes the cluster-of-2 family supports: head_dim 128 with 256-row
    cluster KV tiles that never straddle a (batch, head) group; causal walks
    need square per-group q/kv extents (the segmented walk starts every stripe
    at the cluster diagonal)."""
    return (
        plan.head_dim == 128
        and plan.n_dim % 256 == 0
        and plan.total_kv_rows % 256 == 0
        and (not plan.causal or plan.m_dim == plan.n_dim)
    )


def detect_flash_bwd_search_surface(device_ir: DeviceIR) -> bool:
    """Enable the bwd flash search surface (pins 128x128 block sizes)."""
    from ..compile_environment import CompileEnvironment
    from .analysis.attention_match import _attention_flash_supported

    if not _attention_flash_supported():
        return False
    match = match_attention_bwd(device_ir)
    if match is None:
        return _detect_flex_bwd_search_surface(device_ir)
    env = CompileEnvironment.current()
    kv_block_id = device_ir.grid_block_ids[0][0]
    inner_block_id = None
    from ..device_ir import ForLoopGraphInfo

    for graph_info in device_ir.graphs:
        if isinstance(graph_info, ForLoopGraphInfo):
            inner_block_id = graph_info.block_ids[0]
    assert inner_block_id is not None
    targets = {kv_block_id: 128, inner_block_id: 128}
    from .analysis.attention_match import _flash_block_sizes_reachable

    if not _flash_block_sizes_reachable(env, targets):
        return False
    from .cute_flash_bwd_row_mma import flash_bwd_row_facts

    enable_cute_flash_bwd_search(
        env.config_spec,
        block_size_targets=targets,
        two_cta_allowed=_flash_bwd_two_cta_allowed(match.plan),
        row_mma_facts=flash_bwd_row_facts(match.plan),
    )
    return True


def _flex_bwd_kv_stage_allowed(
    plan: FlexBwdPlan, device: torch.device, tmem: FlexBwdTmemPlan | None
) -> bool:
    """Whether the flex backward's persistent walk can run two K / V ring
    slots on this plan: not the pair or alias_s plans (their own boundary
    schedules and staging), and only where ``_fit_smem_budget`` holds the
    second slot beside the statistics-MMA form (the deepest searched storage;
    at d128 the one-slot q1 / do1 storage is already 201 KB of 227)."""
    from .cute_flash_bwd_flex import _fbwd_fit_kwargs
    from .cute_flash_bwd_flex import _fit_smem_budget
    from .cute_flash_bwd_flex import fbwd_aux_smem_candidates

    if tmem is not None and tmem.mode in ("pair", "alias_s"):
        return False
    candidates = fbwd_aux_smem_candidates(plan)
    return (
        _fit_smem_budget(
            plan,
            stats_mma=True,
            raw_score=False,
            device=device,
            persistent=True,
            aux_smem=max(candidates) if candidates else 0,
            kv_stage=2,
            **_fbwd_fit_kwargs(plan),  # type: ignore[arg-type]
        )
        is not None
    )


def _detect_flex_bwd_search_surface(device_ir: DeviceIR) -> bool:
    """Enable the bwd flash search surface for the FlexAttention backward
    envelope (pins the KV tile and the Q tile to 128 rows; the list walk has
    a CLC persistent variant but no 2-CTA one)."""
    from ..compile_environment import CompileEnvironment
    from .analysis.attention_match import _flash_block_sizes_reachable
    from .flex_bwd_match import match_flex_attention_bwd
    from .flex_match import FLEX_PLAN_ENABLED

    if not FLEX_PLAN_ENABLED or not _flash_bwd_gate_enabled():
        return False
    if len(device_ir.grid_block_ids) != 1 or len(device_ir.grid_block_ids[0]) != 3:
        return False
    env = CompileEnvironment.current()
    match = match_flex_attention_bwd(
        device_ir.graphs, device_ir.grid_block_ids[0], env=env
    )
    if match is None:
        return False
    targets = {match.kv_block_id: 128, match.q_block_id: 128}
    if not _flash_block_sizes_reachable(env, targets):
        return False
    from .cute_flash_bwd_flex import additive_bias_region
    from .cute_flash_bwd_flex import fbwd_aux_smem_candidates
    from .cute_flash_bwd_flex import fbwd_table_step_choices
    from .cute_flash_bwd_flex import fbwd_tile_group_choices
    from .cute_flash_bwd_flex_2cta import flex_bwd_two_cta_allowed
    from .cute_flash_bwd_flex_2cta import flex_bwd_two_cta_table_steps
    from .flex_bwd_match import flex_bwd_tmem_plan

    tmem = flex_bwd_tmem_plan(match.plan.head_dim, match.plan.head_dim_v)
    # The 2-CTA pair kernel (head dim 256) has no persistent walk, statistics
    # MMAs, staged aux tables or chunked step table.
    pair = tmem is not None and tmem.mode == "pair"
    # The cluster-of-2 family at head dims (128, 128) (``cute_flash_bwd_flex_2cta``).
    two_cta = not pair and flex_bwd_two_cta_allowed(match.plan, env.device)
    enable_cute_flash_bwd_search(
        env.config_spec,
        block_size_targets=targets,
        two_cta_allowed=two_cta,
        two_cta_table_steps=flex_bwd_two_cta_table_steps(match.plan, env.device)
        if two_cta
        else 0,
        persistent_allowed=not pair,
        flex=True,
        tile_groups=fbwd_tile_group_choices(match.plan),
        fast_math=bool(env.settings.fast_math),
        aux_smem_candidates=() if pair else fbwd_aux_smem_candidates(match.plan),
        # dS^T through TMEM needs dQ's own columns (the plain TMEM plan).
        ds_tmem_allowed=tmem is not None and tmem.mode == "plain",
        stats_mma_allowed=not pair,
        # The first-step order knob: the plain forms (the alias_s plan has
        # its own measured orders, the pair kernel none of this).
        s_first_allowed=not pair and (tmem is None or tmem.mode != "alias_s"),
        # Two K / V ring slots: the plain forms' persistent walk, where the
        # storage holds them beside the deepest searched statistics form.
        kv_stage_allowed=_flex_bwd_kv_stage_allowed(match.plan, env.device, tmem),
        table_step_choices=(0,)
        if pair
        else fbwd_table_step_choices(match.plan, env.device),
        has_mask=match.plan.mask_region is not None,
        has_score=match.plan.score_region is not None,
        additive_bias=additive_bias_region(match.plan) is not None,
    )
    return True
