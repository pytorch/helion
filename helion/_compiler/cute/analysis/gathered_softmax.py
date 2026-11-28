"""Recognize a complete gathered QK, online softmax and PV page transition."""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ....language import _tracing_ops
from ....language import matmul_ops
from ....language import view_ops
from ..register_loads import host_load_is_readonly
from .gathered_bf16_mma import GatheredBf16Mma
from .gathered_bf16_mma import prove_gathered_bf16_mma
from .gathered_bf16_mma import unmasked_host_load
from .match_common import MatchFailure as _Mismatch
from .match_common import input_closure
from .match_common import is_call
from .match_common import match_args as _args
from .match_common import node_tensor
from .match_common import require_match as _require
from .match_common import scalar_value
from .match_common import strip_new_var

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...compile_environment import CompileEnvironment
    from ...device_ir import GraphInfo


@dataclass(frozen=True)
class GatheredSoftmax:
    qk: GatheredBf16Mma
    value: Node
    value_bid: int
    logical: Node
    scale: Node
    limit: Node
    previous_maximum: Node
    previous_denominator: Node
    previous_numerator: Node


def _row(node: object) -> object:
    source, index = _args(node, view_ops.subscript, 2)
    _require(index == [slice(None), None])
    return source


def _capture(node: object, dtype: torch.dtype, rank: int) -> Node:
    source = strip_new_var(node)
    t = node_tensor(source)
    _require(isinstance(source, Node) and source.op == "placeholder")
    _require(t is not None and t.dtype == dtype and t.ndim == rank)
    return cast("Node", source)


def prove_gathered_softmax(
    graph: torch.fx.Graph, graphs: Sequence[GraphInfo], env: CompileEnvironment
) -> GatheredSoftmax | None:
    """Require exact dataflow through both MMAs and all three returned carries."""
    try:
        aten = torch.ops.aten
        output = graph.find_nodes(op="output")[0]
        out = output.args[0]
        _require(isinstance(out, (tuple, list)) and len(out) == 3)
        maximum, denominator, numerator = cast("tuple[Node, Node, Node]", out)
        maximum = cast("Node", strip_new_var(maximum))
        old_m, reduced_m = _args(maximum, aten.maximum.default, 2)
        folded_m, axis = _args(reduced_m, aten.amax.default, 2)
        scores, identity = _args(folded_m, _tracing_ops._mask_to, 2)
        _require(axis == [-1] and identity == -float("inf"))
        mask, scaled, invalid = _args(scores, aten.where.self, 3)
        _require(scalar_value(invalid) == -float("inf"))
        qk_phi, scale = _args(scaled, aten.mul.Tensor, 2)
        _require(
            is_call(scale, _tracing_ops._get_symnode)
            and isinstance(cast("Node", scale).meta.get("val"), torch.SymFloat)
        )
        seed, extraction = _args(qk_phi, _tracing_ops._phi, 2)
        kfold, index = _args(extraction, operator.getitem, 2)
        _require(isinstance(kfold, Node) and index == 0)
        qk = prove_gathered_bf16_mma(cast("Node", kfold), graphs, env)
        if qk is None or qk.seed is not seed:
            return None
        # Reductions below exclude no physical keys beyond the proved row mask.
        pos2d, limits = _args(mask, aten.lt.Tensor, 2)
        pos, index = _args(pos2d, view_ops.subscript, 2)
        _require(index == [None, slice(None)])
        limit0 = _row(limits)
        limit = strip_new_var(limit0)
        ltensor = node_tensor(limit)
        _require(isinstance(limit, Node) and limit.op == "placeholder")
        _require(
            ltensor is not None
            and ltensor.shape == (qk.rows,)
            and ltensor.dtype in (torch.int32, torch.int64)
        )
        iota, offset = _args(pos, aten.add.Tensor, 2)
        logical, page_len = _args(offset, operator.mul, 2)
        table = unmasked_host_load(qk.page)
        _require(
            table is not None and table[1][-1] is logical and page_len == qk.columns
        )
        _require(isinstance(iota, Node) and is_call(iota, torch.ops.prims.iota.default))
        arange = cast("Node", iota)
        _require(
            arange.args == (qk.columns,)
            and arange.kwargs.get("start") == 0
            and arange.kwargs.get("step") == 1
        )
        _require(
            arange.kwargs.get("dtype") == torch.int32
            and not arange.kwargs.get("requires_grad", False)
        )
        lhs_cast, v, correction, precision = _args(numerator, matmul_ops.dot, 4)
        _require(precision is None)
        p32, cast_type = _args(
            lhs_cast, torch.ops.prims.convert_element_type.default, 2
        )
        _require(cast_type == torch.bfloat16)
        (diff,) = _args(p32, aten.exp2.default, 1)
        scores2, broadcast_m = _args(diff, aten.sub.Tensor, 2)
        _require(scores2 is scores)
        safe_m = _row(broadcast_m)
        eq, zero, same_max = _args(safe_m, aten.where.self, 3)
        eq_max, eq_inf = _args(eq, aten.eq.Scalar, 2)
        _require(
            eq_max is maximum
            and same_max is maximum
            and eq_inf == -float("inf")
            and scalar_value(zero) == 0
        )
        old_acc, broadcast_alpha = _args(correction, aten.mul.Tensor, 2)
        alpha = _row(broadcast_alpha)
        (adiff,) = _args(alpha, aten.exp2.default, 1)
        _require(_args(adiff, aten.sub.Tensor, 2) == (old_m, safe_m))
        scaled_old, reduced_l = _args(denominator, aten.add.Tensor, 2)
        old_l, lalpha = _args(scaled_old, aten.mul.Tensor, 2)
        folded_p, axis = _args(reduced_l, aten.sum.dim_IntList, 2)
        _require(
            lalpha is alpha
            and axis == [-1]
            and _args(folded_p, _tracing_ops._mask_to, 2) == (p32, 0)
        )
        # FX scalar promotion and the reduction dtype are part of the
        # recurrence. In particular a widened or pre-rounded denominator
        # is not the FP32 sum of the unrounded exponentials.
        for value32 in (
            scaled,
            scores,
            maximum,
            safe_m,
            alpha,
            p32,
            denominator,
            numerator,
        ):
            tensor32 = node_tensor(value32)
            _require(tensor32 is not None and tensor32.dtype == torch.float32)
        pos1d = node_tensor(pos)
        _require(pos1d is not None and pos1d.dtype == torch.int32)
        m = _capture(old_m, torch.float32, 1)
        denom = _capture(old_l, torch.float32, 1)
        a = _capture(old_acc, torch.float32, 2)
        _require(len({m, denom, a, limit}) == 4)
        vm = unmasked_host_load(v)
        if vm is None:
            return None
        value, vidx = vm
        _require(len(vidx) == 3 and vidx[0] is qk.page and vidx[1] == slice(None))
        vid = vidx[2]
        _require(_args(vid, aten.sym_size.int, 2) == (a, 1))
        sym = cast("Node", vid).meta.get("val")
        bid = env.resolve_block_id(sym)
        _require(
            isinstance(sym, torch.SymInt)
            and bid is not None
            and sym._sympy_() == env.block_sizes[bid].var._sympy_()
        )
        vt = node_tensor(value)
        _require(
            vt is not None
            and vt.dtype == torch.bfloat16
            and vt.ndim == 3
            and env.known_equal(vt.size(1), qk.columns)
        )
        _require(
            host_load_is_readonly(cast("Node", v), env, graphs)
            and host_load_is_readonly(qk.page, env, graphs)
        )
        for state in (m, denom, a):
            st = node_tensor(state)
            _require(st is not None and env.known_equal(st.size(0), qk.rows))
        # The omitted body is exactly the ancestors of its outputs. Never drop a
        # store, assertion, barrier, atomic, or any other unused executable node.
        live = input_closure(output)
        _require(all(n.op == "placeholder" or n in live for n in graph.nodes))
        return GatheredSoftmax(
            qk,
            value,
            cast("int", bid),
            cast("Node", logical),
            cast("Node", scale),
            cast("Node", limit),
            m,
            denom,
            a,
        )
    except _Mismatch:
        return None
