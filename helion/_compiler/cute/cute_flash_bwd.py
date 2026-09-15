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
  ``hl.atomic_add(dq_fp32, [tile_m, :], dQ * out_scale)``.
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
"""

from __future__ import annotations

import dataclasses
import os
import typing
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import sympy

    from ..device_ir import DeviceIR
    from ..generate_ast import GenerateAST

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
    qk_scale_expr: str  # sympy expr (host symbols) for the exp2-domain scale
    out_scale_expr: str  # sympy expr for the dK/dQ output scale

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
    # Sympy expressions (host symbols) for the two runtime scales, captured
    # from the traced SymFloat metadata so symbol identity is preserved for
    # ``DeviceFunction.sympy_expr`` rendering at codegen time.
    qk_scale_sym: sympy.Expr | None = None
    out_scale_sym: sympy.Expr | None = None


def _node_target_name(node: torch.fx.Node) -> str:
    target = node.target
    return getattr(target, "__name__", str(target))


def _is_host_tensor_load(
    node: object, *, index_len: int
) -> tuple[str, torch.fx.Node] | None:
    """Match ``load(_host_tensor(name), [block, (:,)?], None, None)``.

    Returns (host tensor name, block-index node) or None.
    """
    from ...language import memory_ops
    from ...language._tracing_ops import _host_tensor

    if not isinstance(node, torch.fx.Node) or node.target is not memory_ops.load:
        return None
    base, index = node.args[0], node.args[1]
    if not (isinstance(base, torch.fx.Node) and base.target is _host_tensor):
        return None
    if not isinstance(index, (list, tuple)) or len(index) != index_len:
        return None
    if index_len == 2 and index[1] != slice(None):
        return None
    block = index[0]
    if not isinstance(block, torch.fx.Node):
        return None
    name = base.args[0]
    assert isinstance(name, str)
    return name, block


def _symnode_expr(node: object) -> str | None:
    from ...language._tracing_ops import _get_symnode

    if isinstance(node, torch.fx.Node) and node.target is _get_symnode:
        expr = node.args[0]
        if isinstance(expr, str):
            return expr
    return None


def _symnode_sympy(node: object) -> sympy.Expr | None:
    """The traced SymInt/SymFloat's sympy expression, or None."""
    if not isinstance(node, torch.fx.Node):
        return None
    val = node.meta.get("val")
    node_attr = getattr(val, "node", None)
    return getattr(node_attr, "expr", None)


def _through_new_var(node: object) -> object:
    from ...language._tracing_ops import _new_var

    while isinstance(node, torch.fx.Node) and node.target is _new_var:
        node = node.args[0]
    return node


def _is_permute_of(node: object, source: torch.fx.Node) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.target is torch.ops.aten.permute.default
        and node.args[0] is source
        and node.args[1] == [1, 0]
    )


def _is_row_broadcast_subscript(node: object, source: torch.fx.Node) -> bool:
    """Match ``subscript(source, [None, :])`` (a row vector broadcast)."""
    from ...language import view_ops

    return (
        isinstance(node, torch.fx.Node)
        and node.target is view_ops.subscript
        and node.args[0] is source
        and node.args[1] == [None, slice(None)]
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
        return _through_new_var(node) is placeholder

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
    q_load = _is_host_tensor_load(
        typing.cast("torch.fx.Node", qk_dot.args[1]).args[0], index_len=2
    )
    dp_permute = dp_dot.args[1]
    if not (
        isinstance(dp_permute, torch.fx.Node)
        and dp_permute.target is torch.ops.aten.permute.default
    ):
        return None
    do_load = _is_host_tensor_load(dp_permute.args[0], index_len=2)
    if q_load is None or do_load is None:
        return None
    q_name, _ = q_load
    do_name, _ = do_load

    # Softmax recompute chain: p_t = exp2(mul(score, scale_sym) - lse[None, :]).
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
    mul_node, lse_bcast = sub_node.args
    if not (
        isinstance(mul_node, torch.fx.Node)
        and mul_node.target is torch.ops.aten.mul.Tensor
    ):
        return None
    score_node, scale_sym = mul_node.args
    qk_scale_expr = _symnode_expr(scale_sym)
    if qk_scale_expr is None:
        return None
    qk_scale_sympy = _symnode_sympy(scale_sym)
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
    lse_load = _is_host_tensor_load(lse_bcast.args[0], index_len=1)
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
            delta_load = _is_host_tensor_load(delta_bcast.args[0], index_len=1)
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
    value = atomic_node.args[2]
    if not (
        isinstance(value, torch.fx.Node)
        and value.target is torch.ops.aten.mul.Tensor
        and value.args[0] is dq_dot
    ):
        return None
    out_scale_expr = _symnode_expr(value.args[1])
    if out_scale_expr is None:
        return None
    out_scale_sympy = _symnode_sympy(value.args[1])

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

    k_load = _is_host_tensor_load(k_load_node, index_len=2)
    v_load = _is_host_tensor_load(v_load_node, index_len=2)
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

    # dv / dk stores: store(dv, [kv_block, :], convert(phi0)) and
    # store(dk, [kv_block, :], convert(phi1 * out_scale)).
    store_nodes = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target is memory_ops.store
    ]
    if len(store_nodes) != 2:
        return None
    dv_name = dk_name = None
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
        inner = value.args[0]
        if isinstance(inner, torch.fx.Node) and inner.target is _phi:
            dv_name = base.args[0]
        elif (
            isinstance(inner, torch.fx.Node)
            and inner.target is torch.ops.aten.mul.Tensor
            and _symnode_expr(inner.args[1]) is not None
        ):
            dk_name = base.args[0]
        else:
            return None
    if not (isinstance(dv_name, str) and isinstance(dk_name, str)):
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
        qk_scale_sym=typing.cast("sympy.Expr | None", inner_facts["qk_scale_sym"]),
        out_scale_sym=typing.cast("sympy.Expr | None", inner_facts["out_scale_sym"]),
    )


def _fbwd_epilogue_block(
    *,
    h: int,
    rows: int,
    d: int,
    epi_off: str,
    epi_scale: str,
    kv_tile: str,
    io_dtype: str,
    dkv_wait: str,
    v_empty_expr: str,
) -> str:
    """dK/dV epilogue of compute warpgroup ``h``: TMEM -> registers -> a free
    smem buffer as (128, 64) SWIZZLE_128B boxes -> one TMA store per box.

    dV (h=0) stages through this tile's V operand slot (free once dP(n-1)
    retired, i.e. at dv_done) and hands the slot back to the load warp with an
    arrive on the V ring's empty barrier (``v_empty_expr``) after the store has
    completed; dK (h=1) stages through sdS (free after the tail dQ, i.e. at
    dkv_done), so K/Kt can be released by the MMA warp at the tail and the next
    tile's K loads overlap the epilogue."""
    buf = "sV" if h == 0 else "sdS"
    tma = "_fbwd_tma_dv" if h == 0 else "_fbwd_tma_dk"
    gm = "_fbwd_mdVt" if h == 0 else "_fbwd_mdKt"
    warp_lo = 4 if h == 0 else 8
    chunks = "".join(
        (
            f"""
        fbwd_row64 = cute.make_rmem_tensor(cute.make_layout(64), {io_dtype})"""
            if c % 4 == 0
            else ""
        )
        + f"""
        fbwd_reg = cute.make_rmem_tensor(tEPIcC[None, 0, None].shape, cutlass.Float32)
        cute.copy(fbwd_tiled_epi_ld, tEPItT[None, {c}, None], fbwd_reg)
        {epi_scale}
        fbwd_rego = cute.make_tensor(fbwd_row64.iterator + {16 * (c % 4)}, cute.make_layout(tEPIcC[None, 0, None].shape))
        fbwd_rego.store(fbwd_reg.load().to({io_dtype}))"""
        + (
            f"""
        cute.autovec_copy(fbwd_row64, fbwd_sEpi[fbwd_local_tidx, None, fbwd_epi_slot0 + {c // 4}])
        cute.arch.fence_view_async_shared()
        _helion_flash_rt.named_barrier_wait_unaligned({5 + h}, 128)
        if warp_idx == {warp_lo}:
            tEPIsS{c // 4}, tEPIgS{c // 4} = cute_cpasync_flash.tma_partition({tma}, 0, cute.make_layout(1), cute.group_modes(fbwd_sEpi[None, None, fbwd_epi_slot0 + {c // 4}], 0, 2), cute.group_modes(fbwd_gEpi[None, None, {kv_tile}, {c // 4}], 0, 2))
            cute.copy({tma}, tEPIsS{c // 4}, tEPIgS{c // 4})
            cute.arch.cp_async_bulk_commit_group()"""
            if c % 4 == 3
            else ""
        )
        for c in range(d // 16)
    )
    hs = (
        """
            with cute.arch.elect_one():
                _helion_flash_rt.mbarrier_arrive(fbwd_kv_empty_ptr)"""
        if h == 0
        else ""
    )
    kv_empty = (
        f"""
        fbwd_kv_empty_ptr = {v_empty_expr}
"""
        if h == 0
        else ""
    )
    done_ptr = "fbwd_dv_done_ptr" if h == 0 else "fbwd_dkv_done_ptr"
    return f"""        _helion_flash_rt.mbar_spin_wait({done_ptr}, {dkv_wait}, 10000000)
        fbwd_dkv_shape = fbwd_tst.partition_shape_C(({rows}, {d}))
        tDKV_frag = fbwd_tst.make_fragment_C(fbwd_dkv_shape)
        tEPI_t = cute.make_tensor(fbwd_tmem_ptr + {epi_off}, tDKV_frag.layout)
        fbwd_epi_tiler = ((cute.size(tEPI_t, mode=[0, 0]), cute.size(tEPI_t, mode=[0, 1])),)
        tEPI_epi = cute.zipped_divide(tEPI_t, fbwd_epi_tiler)
        fbwd_epi_ld_atom = cute.make_copy_atom(cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(16)), cutlass.Float32)
        fbwd_tiled_epi_ld = cute_tcgen05_flash.make_tmem_copy(fbwd_epi_ld_atom, tEPI_epi[None, 0])
        fbwd_thr_epi_ld = fbwd_tiled_epi_ld.get_slice(fbwd_local_tidx)
        tEPItT = fbwd_thr_epi_ld.partition_S(tEPI_epi)
        fbwd_cepi = cute.make_identity_tensor(({rows}, {d}))
        fbwd_c_epi = cute.zipped_divide(fbwd_tst.partition_C(fbwd_cepi), fbwd_epi_tiler)
        tEPIcC = fbwd_thr_epi_ld.partition_D(fbwd_c_epi)
        # Epilogue: the tile is staged through the (now free) {buf} operand buffer
        # as (128, 64) SWIZZLE_128B boxes -- 16 columns per t2r chunk, one row
        # per lane -- and each complete box is TMA-stored by the group's first
        # warp (warp-collective issue).
        fbwd_sEpi = storage.{buf}.get_tensor(_fbwd_epil.outer, swizzle=_fbwd_epil.inner)
        fbwd_epi_slot0 = {f"{d // 64} * (FBWD_KV_SLOT)" if h == 0 else "0"}
        fbwd_gEpi = cute.local_tile({gm}, (128, 64), (None, None)){kv_empty}{chunks}
        if warp_idx == {warp_lo}:
            cute.arch.cp_async_bulk_wait_group(0){hs}
"""


def _flash_bwd_wg_compute_block(
    *,
    h: int,
    plan: AttentionBwdPlan,
    io_dtype: str,
    scale2_expr: str,
    out_scale_expr: str,
    n_tiles: int,
    m_mod_tiles: int,
    steps_expr: str,
    persistent: bool = False,
) -> str:
    """One compute warpgroup's body (h = column half 0/1, warps 4-7 / 8-11)."""
    d = plan.head_dim
    warp_lo, warp_hi = (4, 8) if h == 0 else (8, 12)
    mask_arg = "fbwd_mask_lim" if plan.causal else "None"
    m_tile_expr = (
        f"fbwd_i // fbwd_seg * {plan.m_dim // 128} + fbwd_n_tile + fbwd_i % fbwd_seg"
        if plan.causal
        else "fbwd_m_start + fbwd_i"
    )
    # LSE/Delta staging slot. The two slots are separated by named barrier 3
    # of the iteration in between, so the slot must alternate on the RUNNING
    # iteration count. Persistent: `fbwd_i` restarts per tile, and with an odd
    # `fbwd_steps` (causal walks) the next tile's first write would land in
    # the slot the other warps of the group may still be reading (Delta of the
    # previous tile's last iteration); `fbwd_phase` toggles once per iteration
    # across tiles, so it is exactly that running parity.
    par_expr = "fbwd_phase * 128" if persistent else "(fbwd_i % 2) * 128"
    p_pass_call = (
        "\n                _helion_flash_rt.fbwd_p_pairs_packed(tLDrS, "
        "fbwd_lse_frg, fbwd_ch * 32, 32, fbwd_scale2, " + mask_arg + ", fbwd_ch * 32)"
    )
    m_tile_first = m_tile_expr.replace("fbwd_i", "cutlass.Int32(0)")
    epilogue_1 = _fbwd_epilogue_block(
        h=h,
        rows=128,
        d=d,
        epi_off=str(256 if h == 0 else "FBWD_DK_OFF"),
        epi_scale=(
            "_helion_flash_rt._scale_fragment_packed_f32x2(fbwd_reg, cutlass.Float32("
            + out_scale_expr
            + "))"
            if h == 1
            else "pass"
        ),
        kv_tile="fbwd_tile_id",
        io_dtype=io_dtype,
        dkv_wait="fbwd_tile_it % 2",
        v_empty_expr="storage.v_mbar_ptr.data_ptr() + FBWD_KV_STAGE + fbwd_tile_it % FBWD_KV_STAGE",
    ).replace("FBWD_KV_SLOT", "fbwd_tile_it % FBWD_KV_STAGE")
    # Next tile of the walk without a division: +1 inside a stripe, and a jump
    # to the next stripe's diagonal when the stripe ends (causal GQA walk).
    m_tile_next = (
        f"fbwd_m_tile + 1 + cutlass.Int32((fbwd_m_tile % {plan.m_dim // 128}) == {plan.m_dim // 128 - 1}) * fbwd_n_tile"
        if plan.causal
        else "fbwd_m_tile + 1"
    )
    return f"""
    if (warp_idx >= {warp_lo}) & (warp_idx < {warp_hi}):
        cute.arch.setmaxregister_increase(136)
        fbwd_s_shape = fbwd_sst.partition_shape_C((128, 128))
        tStS_frag = fbwd_sst.make_fragment_C(fbwd_s_shape)
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem_ptr = fbwd_tmem.retrieve_ptr(cutlass.Float32)
        fbwd_half_layout = cute.composition(tStS_frag.layout, cute.make_layout((128, 64)))
        tS_h = cute.make_tensor(fbwd_tmem_ptr + {64 * h}, fbwd_half_layout)
        tdP_h = cute.make_tensor(fbwd_tmem_ptr + {128 + 64 * h}, fbwd_half_layout)
        cS = cute.make_identity_tensor((128, 128))
        tScS = fbwd_sst.partition_C(cS)
        fbwd_cs_half_layout = cute.composition(tScS.layout, cute.make_layout((128, 64)))
        tScS_h = cute.make_tensor(tScS.iterator, fbwd_cs_half_layout)
        fbwd_ld_atom = cute.make_copy_atom(cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)
        fbwd_tiled_ld = cute_tcgen05_flash.make_tmem_copy(fbwd_ld_atom, tS_h)
        fbwd_thr_ld = fbwd_tiled_ld.get_slice(fbwd_local_tidx)
        tLDtS = fbwd_thr_ld.partition_S(tS_h)
        tLDtdP = fbwd_thr_ld.partition_S(tdP_h)
        tLDcS = fbwd_thr_ld.partition_D(tScS_h)
        fbwd_p_half_layout = cute.composition(tStS_frag.layout, cute.make_layout((128, 32)))
        tP_h = cute.make_tensor(fbwd_tmem_ptr + {32 * h}, fbwd_p_half_layout)
        fbwd_cp_half_layout = cute.composition(tScS.layout, cute.make_layout((128, 32)))
        tPcS_h = cute.make_tensor(tScS.iterator, fbwd_cp_half_layout)
        fbwd_st_atom = cute.make_copy_atom(cute_tcgen05_flash.St32x32bOp(cute_tcgen05_flash.Repetition(16)), cutlass.Float32)
        fbwd_tiled_st = cute_tcgen05_flash.make_tmem_copy(fbwd_st_atom, tP_h)
        fbwd_thr_st = fbwd_tiled_st.get_slice(fbwd_local_tidx)
        tSTtP = fbwd_thr_st.partition_D(tP_h)
        tSTcP = fbwd_thr_st.partition_S(tPcS_h)
        fbwd_sdS_mnc = cute.make_tensor(fbwd_sdS_mn.iterator, cute.composition(fbwd_sdS_mn.layout, cute.make_layout((32, 2, 2, 128))))
        fbwd_crow = cutlass.Int32(tLDcS[0][0])
        fbwd_sdS_c0 = fbwd_sdS_mnc[None, 0, {h}, fbwd_crow]
        fbwd_sdS_c1 = fbwd_sdS_mnc[None, 1, {h}, fbwd_crow]
        fbwd_thr_row = cutlass.Int32(tLDcS[0][0])
        fbwd_phase = cutlass.Int32(0)
        fbwd_dqf_phase = cutlass.Int32(0)
        fbwd_scale2 = cutlass.Float32({scale2_expr})
        # LSE/Delta staging by 4-byte cp.async one iteration ahead (threads
        # 0-63: LSE, 64-127: Delta): slot (i+1)%2 was last read in iteration
        # i-1 and every thread has passed barrier 3 of iteration i before the
        # copies are issued, so the global-load latency never sits on the
        # softmax critical path. The first tile iteration is staged here into
        # the current parity slot (persistent tiles carry ``fbwd_phase``).
        fbwd_stat_j = fbwd_local_tidx % 64
        fbwd_stage_col0 = fbwd_q_row_base + ({m_tile_first}) * 128 + {64 * h} + fbwd_stat_j
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
            _helion_flash_rt.named_barrier_wait_unaligned(3, 256)
            fbwd_m_tile_nxt = {m_tile_next}
            if fbwd_i + 1 < {steps_expr}:
                fbwd_stage_col = fbwd_q_row_base + fbwd_m_tile_nxt * 128 + {64 * h} + fbwd_stat_j
                if fbwd_local_tidx < 64:
                    _helion_flash_rt.cp_async_ca_4(fbwd_sLSE.iterator + (128 - fbwd_par + {64 * h} + fbwd_stat_j), _fbwd_mLSE.iterator + fbwd_stage_col)
                else:
                    _helion_flash_rt.cp_async_ca_4(fbwd_sDelta.iterator + (128 - fbwd_par + {64 * h} + fbwd_stat_j), _fbwd_mDelta.iterator + fbwd_stage_col)
                cute.arch.cp_async_commit_group()
            for fbwd_ch in cutlass.range_constexpr(2):
                fbwd_lse_frg = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)
                fbwd_lse_v = cute.make_tensor(fbwd_sLSE.iterator + fbwd_par + {64 * h} + fbwd_ch * 32, cute.make_layout(32))
                cute.autovec_copy(fbwd_lse_v, fbwd_lse_frg){p_pass_call}
            tSTrP = cute.make_rmem_tensor(tSTcP.shape, cutlass.Float32)
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
{epilogue_1}        _helion_flash_rt.named_barrier_arrive_unaligned(2, 13 * 32)
"""


_FBWD_CLC_SETUP = """
    # CLC dynamic tile scheduler: warp 15 steals not-yet-launched tiles and the
    # 16-byte responses flow through a 2-stage ring every warp consumes. All
    # 512 threads of every CTA in the cluster release into the leader's empty
    # barrier, so its arrive count scales with the cluster size.
    fbwd_clc_params = cutlass_utils_flash.ClcDynamicPersistentTileSchedulerParams(problem_shape_ntile_mnl=(TOTAL_CTAS, 1, 1), cluster_shape_mnk=(CLUSTER_N, 1, 1))
    fbwd_clc_hw = cutlass_utils_flash.ClcDynamicPersistentTileScheduler.create(fbwd_clc_params, cute.arch.block_idx(), cute.arch.grid_dim(), storage.clc_response.data_ptr())
    fbwd_clc_base = storage.clc_response.data_ptr()
    fbwd_clc_pipe = cutlass_pipeline_flash.PipelineClcFetchAsync.create(barrier_storage=storage.clc_mbar_ptr.data_ptr(), num_stages=2, producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread, CONSUMERS), tx_count=16, cta_layout_vmnk=CTA_LAYOUT)
    fbwd_clc_cstate = cutlass_pipeline_flash.make_pipeline_state(cutlass_pipeline_flash.PipelineUserType.Consumer, 2)
"""

_FBWD_CLC_CONSUME = (
    "{ind}fbwd_clc_pipe.consumer_wait(fbwd_clc_cstate)\n"
    "{ind}fbwd_clc_rptr = (fbwd_clc_base + fbwd_clc_cstate.index * cutlass.Int32(4)).align(16)\n"
    "{ind}fbwd_work = fbwd_clc_hw.work_tile_info_from_clc_response(fbwd_clc_rptr)\n"
    "{ind}fbwd_clc_pipe.consumer_release(fbwd_clc_cstate)\n"
    "{ind}fbwd_clc_cstate.advance()\n"
)


def _fbwd_clc_setup(total_ctas: int, cluster: int, cta_layout: str) -> str:
    return (
        _FBWD_CLC_SETUP.replace("TOTAL_CTAS", str(total_ctas))
        .replace("CLUSTER_N", str(cluster))
        .replace("CONSUMERS", str(512 * cluster))
        .replace("CTA_LAYOUT", cta_layout)
    )


def _fbwd_clc_consume_only(ind: str) -> str:
    """A warp with no per-tile work still consumes every CLC response."""
    return (
        f"{ind}fbwd_work = fbwd_clc_hw.initial_work_tile_info()\n"
        f"{ind}while fbwd_work.is_valid_tile:\n" + _FBWD_CLC_CONSUME.format(ind=ind + "    ")
    )


def _fbwd_clc_scheduler(ind: str, leader_guard: str | None) -> str:
    """Scheduler warp: steal the next tile (leader CTA only in a cluster; the
    multicast response reaches every CTA) and consume like everyone else."""
    issue = (
        f"{ind}    with cute.arch.elect_one():\n"
        f"{ind}        cute.arch.issue_clc_query(fbwd_clc_mbar, fbwd_clc_rptr)\n"
    )
    if leader_guard is not None:
        issue = f"{ind}    if {leader_guard}:\n" + "".join(
            "    " + line + "\n" for line in issue.splitlines()
        )
    # The query is issued only once the current tile's dV is complete (dv_done):
    # reserving the next tile at tile start would commit it a whole tile early
    # and, with causal tiles of very different lengths, leave the tail badly
    # balanced (gqa-causal lost 12% that way); the epilogue still hides the
    # response latency.
    return (
        f"{ind}fbwd_clc_pstate = cutlass_pipeline_flash.make_pipeline_state(cutlass_pipeline_flash.PipelineUserType.Producer, 2)\n"
        f"{ind}fbwd_clc_tile_it = cutlass.Int32(0)\n"
        f"{ind}fbwd_work = fbwd_clc_hw.initial_work_tile_info()\n"
        f"{ind}while fbwd_work.is_valid_tile:\n"
        f"{ind}    _helion_flash_rt.mbar_spin_wait(fbwd_dv_done_ptr, fbwd_clc_tile_it % 2, 10000000)\n"
        f"{ind}    fbwd_clc_tile_it = fbwd_clc_tile_it + 1\n"
        f"{ind}    fbwd_clc_pipe.producer_acquire(fbwd_clc_pstate)\n"
        f"{ind}    fbwd_clc_mbar = fbwd_clc_pipe.producer_get_barrier(fbwd_clc_pstate)\n"
        f"{ind}    fbwd_clc_rptr = (fbwd_clc_base + fbwd_clc_pstate.index * cutlass.Int32(4)).align(16)\n"
        + issue
        + f"{ind}    fbwd_clc_pstate.advance()\n"
        + _FBWD_CLC_CONSUME.format(ind=ind + "    ")
        + f"{ind}fbwd_clc_pipe.producer_tail(fbwd_clc_pstate)\n"
    )


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


def _fbwd_wrap_persistent(body: str, plan: AttentionBwdPlan, total_tiles: int) -> str:
    """Wrap each role's per-tile region in a persistent while-loop.

    Regions are located by rendered-text anchors; each is indented one level
    and preceded by the tile-id loop head plus the per-tile derivations.
    The gate variables (fbwd_tile_it) are loop-carried; barrier phase
    counters live outside the loops so parities continue across tiles.
    """
    lines = body.splitlines(keepends=True)

    def find(anchor: str, start: int = 0) -> int:
        for i in range(start, len(lines)):
            if lines[i].rstrip("\n") == anchor or lines[i].startswith(anchor):
                return i
        raise AssertionError(f"anchor not found: {anchor[:80]}")

    regions = []
    # load warp
    s = find("        fbwd_ke = fbwd_k_prod.acquire_and_advance()")
    e = find("            cute.copy(_fbwd_tma_do, tDOgDO[None, fbwd_q_tile]", s)
    regions.append((s, e))
    # MMA warp: K/V waits .. ve release
    s = find("        fbwd_ke = fbwd_k_cons.wait_and_advance()")
    e = find("        fbwd_ke.release()", s)
    regions.append((s, e))
    # reduce warps: locate its for-loop after the tRDcDQ partition
    anchor = find("        tRDcDQ = fbwd_thr_dq_ld.partition_D(tDQcDQ)")
    s = find("        for fbwd_i in cutlass.range(fbwd_steps, unroll=1):", anchor)
    e = s
    while e + 1 < len(lines) and (
        lines[e + 1].startswith("            ") or not lines[e + 1].strip()
    ):
        e += 1
    while not lines[e].strip():
        e -= 1
    regions.append((s, e))
    # compute warpgroups (2 blocks): for-loop .. epilogue store
    pos = 0
    for _ in range(2):
        blk = find("        fbwd_scale2 = cutlass.Float32(", pos)
        s = find("        fbwd_stage_col0 = ", blk)
        e = find("            cute.arch.cp_async_bulk_wait_group(0)", s)
        while e + 1 < len(lines) and lines[e + 1].startswith("            "):
            e += 1  # WG0's V-slot release sits under the same if
        regions.append((s, e))
        pos = e + 1

    setup = _fbwd_tile_setup_lines(plan, " " * 12)
    # Dynamic schedule: the first tile is this CTA's blockIdx, later tiles
    # come from the CLC response ring (warp 15 steals not-yet-launched CTAs),
    # so the hardware's launch order and load balance are preserved.
    head = (
        "        fbwd_work = fbwd_clc_hw.initial_work_tile_info()\n"
        "        fbwd_tile_it = cutlass.Int32(0)\n"
        "        while fbwd_work.is_valid_tile:\n"
        "            fbwd_tile_id = cutlass.Int32(fbwd_work.tile_idx[0])\n" + setup
    )
    tail = _FBWD_CLC_CONSUME.format(ind=" " * 12) + "            fbwd_tile_it = fbwd_tile_it + 1\n"
    out = []
    prev = 0
    for s, e in sorted(regions):
        assert s >= prev, "overlapping persistent regions"
        out.extend(lines[prev:s])
        out.append(head)
        for line in lines[s : e + 1]:
            out.append("    " + line if line.strip() else line)
        out.append(tail)
        prev = e + 1
    out.extend(lines[prev:])
    return "".join(out)


def emit_flash_bwd_device_body(
    *,
    plan: AttentionBwdPlan,
    io_dtype: str,
    scale2_expr: str,
    out_scale_expr: str,
    q_stage: int,
    do_stage: int,
    kv_stage: int = 1,
    persistent: bool = False,
    total_tiles: int = 0,
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
            out_scale_expr=out_scale_expr,
            n_tiles=n_tiles,
            m_mod_tiles=m_mod_tiles,
            steps_expr=steps_expr,
            persistent=persistent,
        ).replace("FBWD_DK_OFF", str(dk_off))
        for h in (0, 1)
    )

    body = f"""
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
    sQ = storage.sQ.get_tensor(_fbwd_qsl.outer, swizzle=_fbwd_qsl.inner)
    sdO = storage.sdO.get_tensor(_fbwd_dosl.outer, swizzle=_fbwd_dosl.inner)
    sK = storage.sK.get_tensor(_fbwd_ksl.outer, swizzle=_fbwd_ksl.inner)
    sV = storage.sV.get_tensor(_fbwd_vsl.outer, swizzle=_fbwd_vsl.inner)
    sQt = cute.make_tensor(cute.recast_ptr(sQ.iterator, _fbwd_qtl.inner), _fbwd_qtl.outer)
    sdOt = cute.make_tensor(cute.recast_ptr(sdO.iterator, _fbwd_dotl.inner), _fbwd_dotl.outer)
    sKt = cute.make_tensor(cute.recast_ptr(sK.iterator, _fbwd_ktl.inner), _fbwd_ktl.outer)
    fbwd_sdS_mn = storage.sdS.get_tensor(_fbwd_dssl.outer, swizzle=_fbwd_dssl.inner)
    fbwd_sdS_nk = cute.make_tensor(cute.recast_ptr(fbwd_sdS_mn.iterator, _fbwd_dsnk.inner), _fbwd_dsnk.outer)
    fbwd_sdS_flat = cute.make_tensor(fbwd_sdS_mn.iterator, cute.composition(fbwd_sdS_mn.layout, cute.make_layout((128, 128))))
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
    if tidx == 0:
        cute.arch.mbarrier_init(fbwd_s_full_ptr, 1)
        cute.arch.mbarrier_init(fbwd_dp_full_ptr, 1)
        cute.arch.mbarrier_init(fbwd_p_full_ptr, 256)
        cute.arch.mbarrier_init(fbwd_ds_full_ptr, 256)
        cute.arch.mbarrier_init(fbwd_dq_full_ptr, 1)
        cute.arch.mbarrier_init(fbwd_dq_empty_ptr, 128)
        cute.arch.mbarrier_init(fbwd_dkv_done_ptr, 1)
        cute.arch.mbarrier_init(fbwd_dv_done_ptr, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    fbwd_tmem_user_bar = cutlass_pipeline_flash.NamedBarrier(barrier_id=2, num_threads=13 * 32)
    fbwd_tmem = cutlass_utils_flash.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=fbwd_tmem_user_bar, allocator_warp_id=12, is_two_cta=False, two_cta_tmem_dealloc_mbar_ptr=fbwd_tmem_dealloc_ptr)
    fbwd_q_bytes = cute.size_in_bytes({io_dtype}, cute.select(_fbwd_qsl, mode=[0, 1, 2]))
    fbwd_k_bytes = cute.size_in_bytes({io_dtype}, cute.select(_fbwd_ksl, mode=[0, 1, 2]))
    fbwd_q_prod, fbwd_q_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(num_stages={q_stage}, producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), tx_count=fbwd_q_bytes, barrier_storage=storage.q_mbar_ptr.data_ptr()).make_participants()
    fbwd_do_prod, fbwd_do_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(num_stages={do_stage}, producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), tx_count=fbwd_q_bytes, barrier_storage=storage.do_mbar_ptr.data_ptr()).make_participants()
    fbwd_k_prod, fbwd_k_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(num_stages={kv_stage}, producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), tx_count=fbwd_k_bytes, barrier_storage=storage.k_mbar_ptr.data_ptr()).make_participants()
    fbwd_v_prod, fbwd_v_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(num_stages={kv_stage}, producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), tx_count=fbwd_k_bytes, barrier_storage=storage.v_mbar_ptr.data_ptr()).make_participants()
{{clc_setup}}    fbwd_sst = _fbwd_ss_mma.get_slice(0)
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
{{clc_warps}}    if warp_idx == 13:
        tKgK = tKgK_kdl[None, None, 0, 0]
        tQgQ = tQgQ_qdl[None, None, 0, 0]
        tVgV = tVgV_kdl[None, None, 0, 0]
        tDOgDO = tDOgDO_qdl[None, None, 0, 0]
        fbwd_ke = fbwd_k_prod.acquire_and_advance()
        cute.copy(_fbwd_tma_k, tKgK[None, fbwd_tile_id], tKsK[None, fbwd_ke.index], tma_bar_ptr=fbwd_ke.barrier)
        fbwd_ve = fbwd_v_prod.acquire_and_advance()
        cute.copy(_fbwd_tma_v, tVgV[None, fbwd_tile_id], tVsV[None, fbwd_ve.index], tma_bar_ptr=fbwd_ve.barrier)
        for fbwd_i in cutlass.range(fbwd_steps, unroll=1):
            fbwd_q_tile = fbwd_q_tile_base + ({m_tile_expr})
            fbwd_qe = fbwd_q_prod.acquire_and_advance()
            cute.copy(_fbwd_tma_q, tQgQ[None, fbwd_q_tile], tQsQ[None, fbwd_qe.index], tma_bar_ptr=fbwd_qe.barrier)
            fbwd_doe = fbwd_do_prod.acquire_and_advance()
            cute.copy(_fbwd_tma_do, tDOgDO[None, fbwd_q_tile], tDOsDO[None, fbwd_doe.index], tma_bar_ptr=fbwd_doe.barrier)
        fbwd_q_prod.tail()
        fbwd_do_prod.tail()
        fbwd_k_prod.tail()
        fbwd_v_prod.tail()
    if warp_idx == 12:
        fbwd_tmem.allocate(512)
        fbwd_s_shape = fbwd_sst.partition_shape_C((128, 128))
        tStS_frag = fbwd_sst.make_fragment_C(fbwd_s_shape)
        fbwd_dkv_shape = fbwd_tst.partition_shape_C((128, {d}))
        tDKV_frag = fbwd_tst.make_fragment_C(fbwd_dkv_shape)
        fbwd_dk_shape = fbwd_dskt.partition_shape_C((128, {d}))
        tDK_frag = fbwd_dskt.make_fragment_C(fbwd_dk_shape)
        fbwd_dq_shape = fbwd_dqt.partition_shape_C((128, {d}))
        tDQ_frag = fbwd_dqt.make_fragment_C(fbwd_dq_shape)
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem_ptr = fbwd_tmem.retrieve_ptr(cutlass.Float32)
        tS_t = cute.make_tensor(fbwd_tmem_ptr, tStS_frag.layout)
        tdP_t = cute.make_tensor(fbwd_tmem_ptr + 128, tStS_frag.layout)
        tDV_t = cute.make_tensor(fbwd_tmem_ptr + 256, tDKV_frag.layout)
        tDK_t = cute.make_tensor(fbwd_tmem_ptr + {dk_off}, tDK_frag.layout)
        tDQ_t = cute.make_tensor(fbwd_tmem_ptr + {dq_off}, tDQ_frag.layout)
        tP = cute.make_tensor(tS_t.iterator, _fbwd_ptl.outer)
        tSrK = fbwd_sst.make_fragment_A(sK)
        tSrQ = fbwd_sst.make_fragment_B(sQ)
        tSrV = fbwd_sst.make_fragment_A(sV)
        tSrDO = fbwd_sst.make_fragment_B(sdO)
        tTSrP = fbwd_tst.make_fragment_A(tP)
        tTSrDOt = fbwd_tst.make_fragment_B(sdOt)
        tKSrDS = fbwd_dskt.make_fragment_A(fbwd_sdS_nk)
        tKSrQt = fbwd_dskt.make_fragment_B(sQt)
        tDQrDS = fbwd_dqt.make_fragment_A(fbwd_sdS_mn)
        tDQrKt = fbwd_dqt.make_fragment_B(sKt)
        fbwd_s_addr = tS_t.iterator.toint()
        fbwd_dp_addr = tdP_t.iterator.toint()
        fbwd_dv_addr = tDV_t.iterator.toint()
        fbwd_dk_addr = tDK_t.iterator.toint()
        fbwd_dqa_addr = tDQ_t.iterator.toint()
        fbwd_k_base = _helion_flash_ptx.smem_desc_base_from_tensor(sK, _helion_flash_ptx.Major.K)
        fbwd_v_base = _helion_flash_ptx.smem_desc_base_from_tensor(sV, _helion_flash_ptx.Major.K)
        _helion_flash_ptx.declare_ptx_idesc(_fbwd_ss_mma.op, "helion_fbwd_ss_idesc")
        fbwd_q_base = _helion_flash_ptx.smem_desc_base_from_tensor(sQ, _helion_flash_ptx.Major.K)
        fbwd_do_base = _helion_flash_ptx.smem_desc_base_from_tensor(sdO, _helion_flash_ptx.Major.K)
        fbwd_dot_base = _helion_flash_ptx.smem_desc_base_from_tensor(sdOt, _helion_flash_ptx.Major.MN)
        _helion_flash_ptx.declare_ptx_idesc(_fbwd_ts_mma.op, "helion_fbwd_ts_idesc")
        fbwd_dsnk_base = _helion_flash_ptx.smem_desc_base_from_tensor(fbwd_sdS_nk, _helion_flash_ptx.Major.K)
        fbwd_qt_base = _helion_flash_ptx.smem_desc_base_from_tensor(sQt, _helion_flash_ptx.Major.MN)
        _helion_flash_ptx.declare_ptx_idesc(_fbwd_dsk_mma.op, "helion_fbwd_dsk_idesc")
        fbwd_dsmn_base = _helion_flash_ptx.smem_desc_base_from_tensor(fbwd_sdS_mn, _helion_flash_ptx.Major.MN)
        fbwd_kt_base = _helion_flash_ptx.smem_desc_base_from_tensor(sKt, _helion_flash_ptx.Major.MN)
        _helion_flash_ptx.declare_ptx_idesc(_fbwd_dq_mma.op, "helion_fbwd_dq_idesc")
        fbwd_mb = storage.mma_base.get_tensor(cute.make_layout(8))
        fbwd_mb[2] = _helion_flash_ptx.make_smem_desc_start_addr(fbwd_sdS_nk[None, None, None, 0].iterator)
        fbwd_mb[3] = _helion_flash_ptx.make_smem_desc_start_addr(fbwd_sdS_mn[None, None, None, 0].iterator)
        fbwd_pf_phase = cutlass.Int32(0)
        fbwd_dsf_phase = cutlass.Int32(0)
        fbwd_dqe_phase = cutlass.Int32(0)
        fbwd_ke = fbwd_k_cons.wait_and_advance()
        fbwd_mb[0] = _helion_flash_ptx.make_smem_desc_start_addr(sK[None, None, None, fbwd_ke.index].iterator)
        fbwd_mb[4] = _helion_flash_ptx.make_smem_desc_start_addr(sKt[None, None, None, fbwd_ke.index].iterator)
        fbwd_st_k = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 0)
        fbwd_st_dsnk = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 2)
        fbwd_st_dsmn = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 3)
        fbwd_st_kt = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 4)
        fbwd_dv_zero = cutlass.Boolean(True)
        fbwd_dk_zero = cutlass.Boolean(True)
        # Prologue: S(0), dP(0), dV(0). FA4's 1-iteration software skew: each
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
        _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(fbwd_dv_addr, fbwd_s_addr, _helion_flash_ptx.make_smem_desc_start_addr(sdOt[None, None, None, fbwd_doe.index].iterator), fbwd_dot_base, tTSrP[None, None, None, 0].layout, tTSrDOt[None, None, None, 0].layout, "helion_fbwd_ts_idesc", zero_init=fbwd_dv_zero)
        fbwd_dv_zero = cutlass.Boolean(False)
        fbwd_doe.release()
        for fbwd_i in cutlass.range(fbwd_steps - 1, unroll=1):
            fbwd_st_k = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 0)
            fbwd_st_v = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 1)
            fbwd_st_dsnk = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 2)
            fbwd_st_dsmn = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 3)
            fbwd_st_kt = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 4)
            fbwd_qe2 = fbwd_q_cons.wait_and_advance()
            _helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_s_addr, fbwd_st_k, fbwd_k_base, tSrK[None, None, None, 0].layout, _helion_flash_ptx.make_smem_desc_start_addr(sQ[None, None, None, fbwd_qe2.index].iterator), fbwd_q_base, tSrQ[None, None, None, 0].layout, "helion_fbwd_ss_idesc", zero_init=True)
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
            _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(fbwd_dv_addr, fbwd_s_addr, _helion_flash_ptx.make_smem_desc_start_addr(sdOt[None, None, None, fbwd_doe.index].iterator), fbwd_dot_base, tTSrP[None, None, None, 0].layout, tTSrDOt[None, None, None, 0].layout, "helion_fbwd_ts_idesc", zero_init=False)
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
        fbwd_tmem.relinquish_alloc_permit()
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem.free(fbwd_tmem_ptr)
    if warp_idx < 4:
        cute.arch.setmaxregister_increase(152)
        fbwd_dq_shape = fbwd_dqt.partition_shape_C((128, {d}))
        tDQ_frag = fbwd_dqt.make_fragment_C(fbwd_dq_shape)
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem_ptr = fbwd_tmem.retrieve_ptr(cutlass.Float32)
        tDQ_t = cute.make_tensor(fbwd_tmem_ptr + {dq_off}, tDQ_frag.layout)
        cDQ = cute.make_identity_tensor((128, {d}))
        tDQcDQ = fbwd_dqt.partition_C(cDQ)
        fbwd_dq_ld_atom = cute.make_copy_atom(cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)
        fbwd_tiled_dq_ld = cute_tcgen05_flash.make_tmem_copy(fbwd_dq_ld_atom, tDQ_t)
        fbwd_thr_dq_ld = fbwd_tiled_dq_ld.get_slice(fbwd_local_tidx)
        tRDtDQ = fbwd_thr_dq_ld.partition_S(tDQ_t)
        tRDcDQ = fbwd_thr_dq_ld.partition_D(tDQcDQ)
        fbwd_gdq = cute.local_tile(_fbwd_mDQt, (32, 32), (None, None))
        fbwd_out_scale = cutlass.Float32({out_scale_expr})
        fbwd_phase = cutlass.Int32(0)
        for fbwd_i in cutlass.range(fbwd_steps, unroll=1):
            fbwd_m_tile = {m_tile_expr}
            fbwd_row_base = fbwd_q_row_base + fbwd_m_tile * 128
{reduce_body}
        cute.arch.cp_async_bulk_wait_group(0)
        _helion_flash_rt.named_barrier_arrive_unaligned(2, 13 * 32)
{compute_blocks}"""
    body = body.replace("FBWD_KV_STAGE", str(kv_stage))
    if persistent:
        body = body.replace(
            "{clc_setup}",
            _fbwd_clc_setup(total_tiles, 1, "cute.make_layout((1, 1, 1, 1))"),
        ).replace(
            "{clc_warps}",
            "    if warp_idx == 14:\n"
            + _fbwd_clc_consume_only(" " * 8)
            + "    if warp_idx == 15:\n"
            + _fbwd_clc_scheduler(" " * 8, None),
        )
        body = _fbwd_wrap_persistent(body, plan, total_tiles)
    else:
        body = body.replace("{clc_setup}", "").replace("{clc_warps}", "")
    return body


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


def _fbwd2_wrap_persistent(body: str, plan: AttentionBwdPlan, total_pairs: int) -> str:
    """Wrap each role's per-tile region of the 2-CTA body in a cluster-tile loop.

    Same text-anchored scheme as ``_fbwd_wrap_persistent``: the regions are
    the relay warp's relay loop, the load warp's per-tile loads, the leader
    MMA warp's prologue..tail, the reduce warps' drain loop and both compute
    warpgroups' staging..epilogue; barrier phase counters stay outside so the
    parities continue across tiles.
    """
    lines = body.splitlines(keepends=True)

    def find(anchor: str, start: int = 0) -> int:
        for i in range(start, len(lines)):
            if lines[i].startswith(anchor):
                return i
        raise AssertionError(f"anchor not found: {anchor[:80]}")

    def block_end(start: int, indent: str) -> int:
        e = start
        while e + 1 < len(lines) and (
            lines[e + 1].startswith(indent) or not lines[e + 1].strip()
        ):
            e += 1
        while not lines[e].strip():
            e -= 1
        return e

    regions = []
    # relay warp 14: its per-tile relay loop
    r14 = find("    if warp_idx == 14:")
    s = find("        for fbwd_i in cutlass.range(fbwd_steps, unroll=1):", r14)
    regions.append((s, block_end(s, "            ")))
    # load warp 13: K load .. tail Qt load
    r13 = find("    if warp_idx == 13:")
    s = find("        fbwd_ke = fbwd_k_prod.acquire_and_advance()", r13)
    e = find("        fbwd_q_prod.tail()", s) - 1
    while not lines[e].strip():
        e -= 1
    regions.append((s, e))
    # leader MMA warp: K wait .. Kt release
    s = find("            fbwd_ke = fbwd_k_cons.wait_and_advance()")
    e = find("            fbwd_kte.release()", s)
    regions.append((s, e))
    # reduce warps: their drain loop
    anchor = find("        tRDcDQ = fbwd_thr_dq_ld.partition_D(tDQcDQ)")
    s = find("        for fbwd_i in cutlass.range(fbwd_steps, unroll=1):", anchor)
    regions.append((s, block_end(s, "            ")))
    # compute warpgroups (2 blocks): LSE staging .. epilogue
    pos = 0
    for _ in range(2):
        blk = find("        fbwd_scale2 = cutlass.Float32(", pos)
        s = find("        fbwd_stage_col0 = ", blk)
        e = find("            cute.arch.cp_async_bulk_wait_group(0)", s)
        while e + 1 < len(lines) and lines[e + 1].startswith("            "):
            e += 1
        regions.append((s, e))
        pos = e + 1

    out = []
    prev = 0
    for s, e in sorted(regions):
        assert s >= prev, "overlapping persistent regions"
        ind = " " * (len(lines[s]) - len(lines[s].lstrip(" ")))
        head = (
            f"{ind}fbwd_work = fbwd_clc_hw.initial_work_tile_info()\n"
            f"{ind}fbwd_tile_it = cutlass.Int32(0)\n"
            f"{ind}while fbwd_work.is_valid_tile:\n"
            f"{ind}    fbwd_cl_tile = cutlass.Int32(fbwd_work.tile_idx[0]) // 2\n"
            + _fbwd2_tile_setup_lines(plan, ind + "    ")
        )
        tail = _FBWD_CLC_CONSUME.format(ind=ind + "    ") + f"{ind}    fbwd_tile_it = fbwd_tile_it + 1\n"
        out.extend(lines[prev:s])
        out.append(head)
        for line in lines[s : e + 1]:
            out.append("    " + line if line.strip() else line)
        out.append(tail)
        prev = e + 1
    out.extend(lines[prev:])
    return "".join(out)


def _flash_bwd_2cta_wg_compute_block(
    *,
    h: int,
    plan: AttentionBwdPlan,
    io_dtype: str,
    scale2_expr: str,
    out_scale_expr: str,
    m_mod_tiles: int,
    steps_expr: str,
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
        + out_scale_expr
        + "))"
        if h == 1
        else "pass"
    )
    epilogue_2 = _fbwd_epilogue_block(
        h=h,
        rows=256,
        d=d,
        epi_off=str(128 if h == 0 else 384),
        epi_scale=epi_scale,
        kv_tile="fbwd_cl_tile * 2 + fbwd_rank",
        io_dtype=io_dtype,
        dkv_wait="fbwd_tile_it % 2",
        v_empty_expr="storage.v_mbar_ptr.data_ptr() + 1",
    ).replace("FBWD_KV_SLOT", "0")
    return f"""
    if (warp_idx >= {warp_lo}) & (warp_idx < {warp_hi}):
        cute.arch.setmaxregister_increase(136)
        fbwd_s_shape = fbwd_sst.partition_shape_C((256, 128))
        tStS_frag = fbwd_sst.make_fragment_C(fbwd_s_shape)
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem_ptr = fbwd_tmem.retrieve_ptr(cutlass.Float32)
        fbwd_half_layout = cute.composition(tStS_frag.layout, cute.make_layout((128, 64)))
        tS_h = cute.make_tensor(fbwd_tmem_ptr + {64 * h}, fbwd_half_layout)
        tdP_h = cute.make_tensor(fbwd_tmem_ptr + {256 + 64 * h}, fbwd_half_layout)
        cS = cute.make_identity_tensor((256, 128))
        tScS = fbwd_sst.partition_C(cS)
        fbwd_cs_half_layout = cute.composition(tScS.layout, cute.make_layout((128, 64)))
        tScS_h = cute.make_tensor(tScS.iterator, fbwd_cs_half_layout)
        fbwd_ld_atom = cute.make_copy_atom(cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)
        fbwd_tiled_ld = cute_tcgen05_flash.make_tmem_copy(fbwd_ld_atom, tS_h)
        fbwd_thr_ld = fbwd_tiled_ld.get_slice(fbwd_local_tidx)
        tLDtS = fbwd_thr_ld.partition_S(tS_h)
        tLDtdP = fbwd_thr_ld.partition_S(tdP_h)
        tLDcS = fbwd_thr_ld.partition_D(tScS_h)
        fbwd_p_half_layout = cute.composition(tStS_frag.layout, cute.make_layout((128, 32)))
        tP_h = cute.make_tensor(fbwd_tmem_ptr + {32 * h}, fbwd_p_half_layout)
        tDS_h = cute.make_tensor(fbwd_tmem_ptr + {256 + 64 * h}, fbwd_p_half_layout)
        fbwd_cp_half_layout = cute.composition(tScS.layout, cute.make_layout((128, 32)))
        tPcS_h = cute.make_tensor(tScS.iterator, fbwd_cp_half_layout)
        fbwd_st_atom = cute.make_copy_atom(cute_tcgen05_flash.St32x32bOp(cute_tcgen05_flash.Repetition(16)), cutlass.Float32)
        fbwd_tiled_st = cute_tcgen05_flash.make_tmem_copy(fbwd_st_atom, tP_h)
        fbwd_thr_st = fbwd_tiled_st.get_slice(fbwd_local_tidx)
        tSTtP = fbwd_thr_st.partition_D(tP_h)
        tSTtDS = fbwd_thr_st.partition_D(tDS_h)
        tSTcP = fbwd_thr_st.partition_S(tPcS_h)
        fbwd_sdS_kv4 = cute.make_tensor(fbwd_sdS_mn.iterator, cute.make_layout((32, 2, 2, 128), stride=(1, 32, 8192, 64)))
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
        fbwd_stage_col0 = fbwd_q_row_base + {_fbwd2_m_tile(plan, "0")} * 128 + {64 * h} + fbwd_local_tidx % 64
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
            cute.arch.cp_async_wait_group(0)
            _helion_flash_rt.named_barrier_wait_unaligned(3, 256)
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
            for fbwd_ch in cutlass.range_constexpr(4):
                fbwd_lse_frg = cute.make_rmem_tensor(cute.make_layout(16), cutlass.Float32)
                fbwd_lse_v = cute.make_tensor(fbwd_sLSE.iterator + fbwd_par + {64 * h} + fbwd_ch * 16, cute.make_layout(16))
                cute.autovec_copy(fbwd_lse_v, fbwd_lse_frg)
                _helion_flash_rt.fbwd_p_pairs_packed(tLDrS, fbwd_lse_frg, fbwd_ch * 16, 16, fbwd_scale2, {mask_arg}, fbwd_ch * 16)
            tSTrP = cute.make_rmem_tensor(tSTcP.shape, cutlass.Float32)
            tSTrP_e = cute.make_tensor(cute.recast_ptr(tSTrP.iterator, dtype={io_dtype}), tLDrS.layout)
            tSTrP_e.store(tLDrS.load().to({io_dtype}))
            cute.copy(fbwd_tiled_st, tSTrP, tSTtP)
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
            _helion_flash_rt.named_barrier_wait_unaligned(3, 256)
{_FBWD_2CTA_SEND if h == 0 else ""}
            fbwd_phase ^= 1
            fbwd_m_tile = fbwd_m_tile_nxt
        with cute.arch.elect_one():
            _helion_flash_rt.mbarrier_arrive(fbwd_ds_smem_full_ptr, cutlass.Int32(0))
{epilogue_2}        _helion_flash_rt.named_barrier_arrive_unaligned(2, 13 * 32)
"""


_FBWD_2CTA_SEND = """            if fbwd_local_tidx == 0:
                fbwd_peer = fbwd_rank ^ 1
                cute.arch.mbarrier_arrive_and_expect_tx(fbwd_ds_cl_full_ptr, 16384, peer_cta_rank_in_cluster=fbwd_peer)
                _helion_flash_rt.cpasync_bulk_s2cluster(fbwd_sdSx_ptr, fbwd_sdS_mn.iterator + fbwd_rank * 8192, fbwd_ds_cl_full_ptr, 16384, fbwd_peer)"""


def emit_flash_bwd_2cta_device_body(
    *,
    plan: AttentionBwdPlan,
    io_dtype: str,
    scale2_expr: str,
    out_scale_expr: str,
    persistent: bool = False,
    total_pairs: int = 0,
) -> str:
    """2-CTA (cluster (2,1,1)) fused backward device body for head_dim 128."""
    d = plan.head_dim
    n_tiles = plan.n_dim // 128
    m_tiles = plan.mm_dim // 128
    m_mod_tiles = plan.m_dim // 128
    causal_start = "fbwd_n_tile // 2 * 2" if plan.causal else "cutlass.Int32(0)"
    if plan.causal:
        seg_lines = f"\n    fbwd_seg = {m_mod_tiles} - fbwd_m_start\n    fbwd_steps = {plan.group} * fbwd_seg"
    else:
        seg_lines = f"\n    fbwd_steps = {m_tiles} - fbwd_m_start"
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
            out_scale_expr=out_scale_expr,
            m_mod_tiles=m_mod_tiles,
            steps_expr="fbwd_steps",
        )
        for h in (0, 1)
    )

    body = f"""
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
    sQ = storage.sQ.get_tensor(_fbwd_qsl.outer, swizzle=_fbwd_qsl.inner)
    sdOt = storage.sdOt.get_tensor(_fbwd_dosl.outer, swizzle=_fbwd_dosl.inner)
    sdO = storage.sdO.get_tensor(_fbwd_dotl.outer, swizzle=_fbwd_dotl.inner)
    sQt = storage.sQt.get_tensor(_fbwd_qtl.outer, swizzle=_fbwd_qtl.inner)
    sK = storage.sK.get_tensor(_fbwd_ksl.outer, swizzle=_fbwd_ksl.inner)
    sV = storage.sV.get_tensor(_fbwd_vsl.outer, swizzle=_fbwd_vsl.inner)
    sKt = storage.sKt.get_tensor(_fbwd_ktl.outer, swizzle=_fbwd_ktl.inner)
    fbwd_sdS_mn = storage.sdS.get_tensor(_fbwd_dssl.outer, swizzle=_fbwd_dssl.inner)
    fbwd_sdSx_plain = storage.sdSx.get_tensor(cute.make_layout(8192))
    fbwd_sdSx_ptr = cute.recast_ptr(fbwd_sdSx_plain.iterator, _fbwd_dssl.inner)
    fbwd_sLSE = storage.sLSE.get_tensor(cute.make_layout(2 * 128))
    fbwd_sDelta = storage.sDelta.get_tensor(cute.make_layout(2 * 128))
    fbwd_sdq = storage.sdQaccum.get_tensor(_fbwd_dqsl.outer, swizzle=_fbwd_dqsl.inner)
    fbwd_s_full_ptr = storage.s_full_mbar.data_ptr()
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
{{clc_setup}}    fbwd_sst = _fbwd_ss_mma.get_slice(fbwd_rank)
    fbwd_tst = _fbwd_ts_mma.get_slice(fbwd_rank)
    fbwd_dqt = _fbwd_dq_mma.get_slice(fbwd_rank)
    fbwd_mcast = cutlass_pipeline_flash.PipelineUmmaAsync._compute_tmem_sync_mask(_fbwd_cluster_vmnk)
    gK = cute.flat_divide(_fbwd_mKt, cute.select((256, 128, {d}), mode=[0, 2]))
    gV = cute.flat_divide(_fbwd_mVt, cute.select((256, 128, {d}), mode=[0, 2]))
    gQ = cute.flat_divide(_fbwd_mQt, cute.select((256, 128, {d}), mode=[1, 2]))
    gdOt = cute.flat_divide(_fbwd_mdOtN, cute.select((256, 128, {d}), mode=[1, 2]))
    gdO = cute.flat_divide(_fbwd_mdOtT, cute.select((256, {d}, 128), mode=[1, 2]))
    gQt = cute.flat_divide(_fbwd_mQtT, cute.select((256, {d}, 128), mode=[1, 2]))
    gKt = cute.flat_divide(_fbwd_mKtT, cute.select((128, {d}, 256), mode=[1, 2]))
    tSgK = fbwd_sst.partition_A(gK)
    tSgV = fbwd_sst.partition_A(gV)
    tSgQ = fbwd_sst.partition_B(gQ)
    tSgDOt = fbwd_sst.partition_B(gdOt)
    tTSgDO = fbwd_tst.partition_B(gdO)
    tTSgQt = fbwd_tst.partition_B(gQt)
    tDQgKt = fbwd_dqt.partition_B(gKt)
    tKsK, tKgK_kdl = cute_cpasync_flash.tma_partition(_fbwd_tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(tSgK, 0, 3))
    tVsV, tVgV_kdl = cute_cpasync_flash.tma_partition(_fbwd_tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3), cute.group_modes(tSgV, 0, 3))
    tQsQ, tQgQ_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(tSgQ, 0, 3))
    tDOTsDOT, tDOTgDOT_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_dot, 0, cute.make_layout(1), cute.group_modes(sdOt, 0, 3), cute.group_modes(tSgDOt, 0, 3))
    tDOsDO, tDOgDO_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_do2, 0, cute.make_layout(1), cute.group_modes(sdO, 0, 3), cute.group_modes(tTSgDO, 0, 3))
    tQTsQT, tQTgQT_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_qt, 0, cute.make_layout(1), cute.group_modes(sQt, 0, 3), cute.group_modes(tTSgQt, 0, 3))
    tKTsKT, tKTgKT_kdl = cute_cpasync_flash.tma_partition(_fbwd_tma_kt, 0, cute.make_layout(1), cute.group_modes(sKt, 0, 3), cute.group_modes(tDQgKt, 0, 3))
    if (warp_idx >= 12) & (warp_idx < 16):
        cute.arch.setmaxregister_decrease(104)
{{clc_warps}}    if warp_idx == 14:
        fbwd_rphase = cutlass.Int32(0)
        for fbwd_i in cutlass.range(fbwd_steps, unroll=1):
            _helion_flash_rt.mbar_spin_wait(fbwd_ds_cl_full_ptr, fbwd_rphase, 10000000)
            fbwd_rphase ^= 1
            with cute.arch.elect_one():
                _helion_flash_rt.mbarrier_arrive(fbwd_ds_cl_leader_ptr, cutlass.Int32(0))
    if warp_idx == 13:
        tKgK = tKgK_kdl[None, None, 0, 0]
        tVgV = tVgV_kdl[None, None, 0, 0]
        tQgQ = tQgQ_qdl[None, None, 0, 0]
        tDOTgDOT = tDOTgDOT_qdl[None, None, 0, 0]
        tDOgDO = tDOgDO_qdl[None, 0, None, 0]
        tQTgQT = tQTgQT_qdl[None, 0, None, 0]
        tKTgKT = tKTgKT_kdl[None, 0, None, 0]
        # Issue order follows first use: K + Q(0) feed S(0), V + dO(0) feed
        # dP(0), Kt is first needed by dQ(0) two gemms later.
        fbwd_ke = fbwd_k_prod.acquire_and_advance()
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
        fbwd_q_prod.tail()
        fbwd_qt_prod.tail()
        fbwd_do_prod.tail()
        fbwd_k_prod.tail()
        fbwd_v_prod.tail()
        fbwd_kt_prod.tail()
    if warp_idx == 12:
        fbwd_tmem.allocate(512)
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem_ptr = fbwd_tmem.retrieve_ptr(cutlass.Float32)
        if fbwd_is_leader:
            fbwd_s_shape = fbwd_sst.partition_shape_C((256, 128))
            tStS_frag = fbwd_sst.make_fragment_C(fbwd_s_shape)
            fbwd_dkv_shape = fbwd_tst.partition_shape_C((256, {d}))
            tDKV_frag = fbwd_tst.make_fragment_C(fbwd_dkv_shape)
            fbwd_dq_shape = fbwd_dqt.partition_shape_C((128, {d}))
            tDQ_frag = fbwd_dqt.make_fragment_C(fbwd_dq_shape)
            tS_t = cute.make_tensor(fbwd_tmem_ptr, tStS_frag.layout)
            tdP_t = cute.make_tensor(fbwd_tmem_ptr + 256, tStS_frag.layout)
            tDV_t = cute.make_tensor(fbwd_tmem_ptr + 128, tDKV_frag.layout)
            tDK_t = cute.make_tensor(fbwd_tmem_ptr + 384, tDKV_frag.layout)
            tDQ_t = cute.make_tensor(fbwd_tmem_ptr + 64, tDQ_frag.layout)
            tP = cute.make_tensor(tS_t.iterator, _fbwd_ptl.outer)
            tDS = cute.make_tensor(tdP_t.iterator, _fbwd_ptl.outer)
            tSrK = fbwd_sst.make_fragment_A(sK)
            tSrQ = fbwd_sst.make_fragment_B(sQ)
            tSrV = fbwd_sst.make_fragment_A(sV)
            tSrDOt = fbwd_sst.make_fragment_B(sdOt)
            tTSrP = fbwd_tst.make_fragment_A(tP)
            tTSrDS = fbwd_tst.make_fragment_A(tDS)
            tTSrDO = fbwd_tst.make_fragment_B(sdO)
            tTSrQt = fbwd_tst.make_fragment_B(sQt)
            tDQrDS = fbwd_dqt.make_fragment_A(fbwd_sdS_mn)
            tDQrKt = fbwd_dqt.make_fragment_B(sKt)
            fbwd_s_addr = tS_t.iterator.toint()
            fbwd_dp_addr = tdP_t.iterator.toint()
            fbwd_dv_addr = tDV_t.iterator.toint()
            fbwd_dk_addr = tDK_t.iterator.toint()
            fbwd_dqa_addr = tDQ_t.iterator.toint()
            fbwd_k_base = _helion_flash_ptx.smem_desc_base_from_tensor(sK, _helion_flash_ptx.Major.K)
            fbwd_v_base = _helion_flash_ptx.smem_desc_base_from_tensor(sV, _helion_flash_ptx.Major.K)
            _helion_flash_ptx.declare_ptx_idesc(_fbwd_ss_mma.op, "helion_fbwd_ss_idesc")
            fbwd_q_base = _helion_flash_ptx.smem_desc_base_from_tensor(sQ, _helion_flash_ptx.Major.K)
            fbwd_dot_base = _helion_flash_ptx.smem_desc_base_from_tensor(sdOt, _helion_flash_ptx.Major.K)
            fbwd_do2_base = _helion_flash_ptx.smem_desc_base_from_tensor(sdO, _helion_flash_ptx.Major.MN)
            fbwd_qt_base = _helion_flash_ptx.smem_desc_base_from_tensor(sQt, _helion_flash_ptx.Major.MN)
            _helion_flash_ptx.declare_ptx_idesc(_fbwd_ts_mma.op, "helion_fbwd_ts_idesc")
            fbwd_dsmn_base = _helion_flash_ptx.smem_desc_base_from_tensor(fbwd_sdS_mn, _helion_flash_ptx.Major.MN)
            fbwd_kt_base = _helion_flash_ptx.smem_desc_base_from_tensor(sKt, _helion_flash_ptx.Major.MN)
            _helion_flash_ptx.declare_ptx_idesc(_fbwd_dq_mma.op, "helion_fbwd_dq_idesc")
            # Operand start addresses parked in smem; re-read with a volatile
            # load per iteration so the per-k descriptors are never hoisted.
            fbwd_mb = storage.mma_base.get_tensor(cute.make_layout(8))
            fbwd_mb[0] = _helion_flash_ptx.make_smem_desc_start_addr(sK[None, None, None, 0].iterator)
            fbwd_mb[1] = _helion_flash_ptx.make_smem_desc_start_addr(sV[None, None, None, 0].iterator)
            fbwd_mb[2] = _helion_flash_ptx.make_smem_desc_start_addr(fbwd_sdS_mn[None, None, None, 0].iterator)
            fbwd_mb[3] = _helion_flash_ptx.make_smem_desc_start_addr(sQ[None, None, None, 0].iterator)
            fbwd_mb[4] = _helion_flash_ptx.make_smem_desc_start_addr(sdOt[None, None, None, 0].iterator)
            fbwd_mb[5] = _helion_flash_ptx.make_smem_desc_start_addr(sdO[None, None, None, 0].iterator)
            fbwd_mb[6] = _helion_flash_ptx.make_smem_desc_start_addr(sQt[None, None, None, 0].iterator)
            fbwd_mb[7] = _helion_flash_ptx.make_smem_desc_start_addr(sKt[None, None, None, 0].iterator)
            fbwd_pf_phase = cutlass.Int32(0)
            fbwd_dst_phase = cutlass.Int32(0)
            fbwd_dss_phase = cutlass.Int32(0)
            fbwd_dcl_phase = cutlass.Int32(0)
            fbwd_dqe_phase = cutlass.Int32(0)
            fbwd_ke = fbwd_k_cons.wait_and_advance()
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
                _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(fbwd_dk_addr, fbwd_dp_addr, fbwd_st_qt, fbwd_qt_base, tTSrDS[None, None, None, 0].layout, tTSrQt[None, None, None, 0].layout, "helion_fbwd_ts_idesc", zero_init=fbwd_dk_zero, cta_group=2, a_offsets={_FBWD_2CTA_DS_OFFSETS})
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
            _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(fbwd_dk_addr, fbwd_dp_addr, fbwd_st_qt, fbwd_qt_base, tTSrDS[None, None, None, 0].layout, tTSrQt[None, None, None, 0].layout, "helion_fbwd_ts_idesc", zero_init=fbwd_dk_zero, cta_group=2, a_offsets={_FBWD_2CTA_DS_OFFSETS})
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
{{clc_mma_nonleader}}        fbwd_tmem.relinquish_alloc_permit()
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem.free(fbwd_tmem_ptr)
    if warp_idx < 4:
        cute.arch.setmaxregister_increase(136)
        fbwd_dq_shape = fbwd_dqt.partition_shape_C((128, {d}))
        tDQ_frag = fbwd_dqt.make_fragment_C(fbwd_dq_shape)
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem_ptr = fbwd_tmem.retrieve_ptr(cutlass.Float32)
        tDQ_t = cute.make_tensor(fbwd_tmem_ptr + 64, tDQ_frag.layout)
        cDQ = cute.make_identity_tensor((128, {d}))
        tDQcDQ = fbwd_dqt.partition_C(cDQ)
        fbwd_dq_ld_atom = cute.make_copy_atom(cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)
        fbwd_tiled_dq_ld = cute_tcgen05_flash.make_tmem_copy(fbwd_dq_ld_atom, tDQ_t)
        fbwd_thr_dq_ld = fbwd_tiled_dq_ld.get_slice(fbwd_local_tidx)
        tRDtDQ = fbwd_thr_dq_ld.partition_S(tDQ_t)
        tRDcDQ = fbwd_thr_dq_ld.partition_D(tDQcDQ)
        fbwd_gdq = cute.local_tile(_fbwd_mDQt, (32, 32), (None, None))
        fbwd_out_scale = cutlass.Float32({out_scale_expr})
        fbwd_my_m = cutlass.Int32(tRDcDQ[0][0])
        fbwd_my_row = fbwd_my_m % 32
        fbwd_phase = cutlass.Int32(0)
        for fbwd_i in cutlass.range(fbwd_steps, unroll=1):
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
        cute.arch.cp_async_bulk_wait_group(0)
        _helion_flash_rt.named_barrier_arrive_unaligned(2, 13 * 32)
{compute_blocks}"""
    if persistent:
        body = body.replace(
            "{clc_setup}", _fbwd_clc_setup(total_pairs * 2, 2, "_fbwd_cluster_vmnk")
        ).replace(
            "{clc_warps}",
            "    if warp_idx == 15:\n        if fbwd_is_leader:\n"
            + _fbwd_clc_scheduler(" " * 12, None)
            + "        else:\n"
            + _fbwd_clc_consume_only(" " * 12),
        ).replace(
            "{clc_mma_nonleader}",
            "        if fbwd_rank != cutlass.Int32(0):\n" + _fbwd_clc_consume_only(" " * 12),
        )
        body = _fbwd2_wrap_persistent(body, plan, total_pairs)
    else:
        body = (
            body.replace("{clc_setup}", "")
            .replace("{clc_warps}", "")
            .replace("{clc_mma_nonleader}", "")
        )
    return body


# dS (bf16-packed, 64 words per compute warpgroup) sits in its OWN dP column
# half (words 0-31 of 256 + 64*h): a warpgroup's dS stores then only overwrite
# dP columns it has already loaded, so the dP/dS pass runs in two 32-column
# chunks without cross-warpgroup barriers. The dK gemm's A operand is thus
# gapped: k-tiles 0-3 at words 0-31, k-tiles 4-7 at words 64-95.
_FBWD_2CTA_DS_OFFSETS = tuple((k // 4) * 64 + (k % 4) * 8 for k in range(8))

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
    import ast as ast_module

    from ..compile_environment import CompileEnvironment
    from .cute_flash import emit_flash_module_statements

    df = cg.device_function
    match = df.cute_state.attention_flash_bwd_match
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

    if match.qk_scale_sym is None or match.out_scale_sym is None:
        return False
    scale2_expr = df.sympy_expr(match.qk_scale_sym)  # pyrefly: ignore [bad-argument-type]
    out_scale_expr = df.sympy_expr(match.out_scale_sym)  # pyrefly: ignore [bad-argument-type]

    env = CompileEnvironment.current()
    io_dtype_str = env.backend.dtype_str(plan.io_dtype)
    # The 2-CTA cluster family (FA4's d128 configuration): a cluster of two
    # CTAs shares one 256-row KV tile, each CTA loads half of Q/dO and drains
    # half of dQ, and the dS halves are exchanged over DSMEM. An autotuner
    # knob (``cute_flash_bwd_two_cta``) where the shape allows it;
    # HELION_CUTE_FLASH_BWD_2CTA=0/1 overrides for bring-up/debugging.
    two_cta_env = os.environ.get("HELION_CUTE_FLASH_BWD_2CTA")
    two_cta = _flash_bwd_two_cta_allowed(plan) and (
        two_cta_env == "1"
        if two_cta_env is not None
        else bool(df.config.config.get("cute_flash_bwd_two_cta", 0))
    )
    # FA4 1-CTA staging: Q double-buffered (spans two skewed iterations);
    # TMA ring depths, bounded by the 227KB smem budget: head_dim 128 fits
    # Q x2 + dO x1 next to K/V/dS/dQ staging; head_dim 64 has room for 4-deep
    # Q and dO rings, which hide the load latency under the dQ reduce traffic
    # (-5% at s8192/d64 vs 2/2).
    q_stage = 2 if d == 128 else 4
    do_stage = 1 if d == 128 else 4
    total_tiles = plan.total_kv_rows // 128
    # Persistent tile scheduler (cute_flash_bwd_persistent=1): each CTA walks
    # tile_id += grid_dim. The former fresh-compile deadlock was the MMA warp's
    # tail ds_full wait not flipping its parity (see ds_tail_toggle). It is
    # still slower than the non-persistent launch (d128 +0.5%, d128-causal and
    # s4k-b8 +8%: static round-robin tiles, no inter-tile overlap), so the
    # autotuner is expected to keep picking 0.
    persistent = bool(df.config.config.get("cute_flash_bwd_persistent", 0))
    # Persistent tiles at head_dim 64 double-buffer K/V (the next tile's
    # operands land while this tile runs; the epilogue staging keeps its own
    # slot), paid for by 3-deep instead of 4-deep Q/dO rings.
    kv_stage = 2 if (persistent and d == 64) else 1
    if "HELION_FBWD_KV_STAGE" in os.environ:  # experiment override
        kv_stage = int(os.environ["HELION_FBWD_KV_STAGE"])
    if kv_stage == 2:
        q_stage = 3
        do_stage = 3

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
    cg.cute_wrapper_plans.append(wrapper_plan)  # type: ignore[attr-defined]
    if two_cta:
        df.wrapper_only_params.extend(_FLASH_BWD_2CTA_KERNEL_PARAMS)
        df.cute_state.cluster_shape = (2, 1, 1)
    else:
        df.wrapper_only_params.extend(_FLASH_BWD_KERNEL_PARAMS)
    df.placeholder_args.update(a.name for a in args.values())  # type: ignore[union-attr]
    cg.cute_uses_matmul = True  # type: ignore[attr-defined]
    df.cute_state.attention_flash_threads = 512

    if two_cta:
        body_src = emit_flash_bwd_2cta_device_body(
            plan=plan,
            io_dtype=io_dtype_str,
            scale2_expr=scale2_expr,
            out_scale_expr=out_scale_expr,
            persistent=persistent,
            total_pairs=plan.total_kv_rows // 256,
        )
    else:
        body_src = emit_flash_bwd_device_body(
            plan=plan,
            io_dtype=io_dtype_str,
            scale2_expr=scale2_expr,
            out_scale_expr=out_scale_expr,
            q_stage=q_stage,
            do_stage=do_stage,
            kv_stage=kv_stage,
            persistent=persistent,
            total_tiles=total_tiles,
        )
    wrapped = ast_module.parse("if True:\n" + body_src)
    assert isinstance(wrapped.body[0], ast_module.If)
    df.body = list(wrapped.body[0].body)
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
    from ..backend import _attention_flash_supported
    from ..compile_environment import CompileEnvironment

    if not _attention_flash_supported():
        return False
    match = match_attention_bwd(device_ir)
    if match is None:
        return False
    env = CompileEnvironment.current()
    kv_block_id = device_ir.grid_block_ids[0][0]
    inner_block_id = None
    from ..device_ir import ForLoopGraphInfo

    for graph_info in device_ir.graphs:
        if isinstance(graph_info, ForLoopGraphInfo):
            inner_block_id = graph_info.block_ids[0]
    assert inner_block_id is not None
    targets = {kv_block_id: 128, inner_block_id: 128}
    from ..backend import _flash_block_sizes_reachable

    if not _flash_block_sizes_reachable(env, targets):
        return False
    env.config_spec.enable_cute_flash_bwd_search(
        block_size_targets=targets,
        two_cta_allowed=_flash_bwd_two_cta_allowed(match.plan),
    )
    return True
