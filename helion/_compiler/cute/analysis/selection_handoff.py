"""Prove a disjoint four-row FP32 selection and its flattened, sole-CTA merge."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import prod
import operator
from typing import TYPE_CHECKING

import sympy
import torch
from torch.fx import Node

from ....language import _tracing_ops as tr
from ....language import memory_ops
from ....language import tile_ops
from ..fragment_vector_io import static_int
from ..fragment_vector_io import static_strides
from ..promote_output_axis import _fresh_tensors
from .match_common import host_tensor_name
from .match_common import node_tensor
from .match_common import unit_fixed_block

if TYPE_CHECKING:
    from ...compile_environment import CompileEnvironment
    from ...device_ir import DeviceIR


@dataclass(frozen=True)
class SelectionHandoff:
    producer: int
    consumer: int
    logits: torch.Tensor
    values: torch.Tensor
    ids: torch.Tensor
    flat_values: torch.Tensor
    flat_ids: torch.Tensor


def _op(n: object, target: object) -> Node | None:
    return (
        n
        if isinstance(n, Node) and n.op == "call_function" and n.target is target
        else None
    )


def _at(n: object, target: object, args: tuple[object, ...]) -> bool:
    return (
        isinstance(n, Node) and n.target is target and n.args == args and not n.kwargs
    )


def _tensor(
    env: CompileEnvironment, n: object, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor | None:
    t = node_tensor(n)
    if (
        host_tensor_name(n) is None
        or t is None
        or t.dtype != dtype
        or tuple(map(static_int, t.shape)) != shape
    ):
        return None
    strides = static_strides(env, t)
    if strides is None or any(
        sz > 1 and s != prod(shape[i + 1 :])
        for i, (sz, s) in enumerate(zip(shape, strides, strict=True))
    ):
        return None
    return t


def _axis(env: CompileEnvironment, node: object) -> int | None:
    begin = _op(node, tile_ops.tile_begin)
    if begin is None or len(begin.args) != 1 or begin.kwargs:
        return None
    symbol = _op(begin.args[0], tr._get_symnode)
    if symbol is None:
        return None
    val = symbol.meta.get("val")
    return env.resolve_block_id(val) if isinstance(val, torch.SymInt) else None


def _same_span(a: torch.Tensor, b: torch.Tensor) -> bool:
    ao, bo = static_int(a.storage_offset()), static_int(b.storage_offset())
    return (
        ao is not None
        and ao == bo
        and a.untyped_storage()._cdata == b.untyped_storage()._cdata
    )


def selection_structure(ir: DeviceIR) -> bool:
    """Keep the shape-independent schema stable across a multi-shape binding."""
    flags = [
        any(n.target is torch.ops.aten.topk.default for n in ir.graphs[i].graph.nodes)
        for i in ir.root_ids
    ]
    return any(a and b for a, b in pairwise(flags))


def match_selection_handoff(
    env: CompileEnvironment, ir: DeviceIR
) -> SelectionHandoff | None:
    if (
        env.process_group_name is not None
        or env.config_spec.target_device_capability is None
        or env.config_spec.target_device_capability[0] != 10
        or ir.host_function is None
        or not selection_structure(ir)
    ):
        return None
    fresh = _fresh_tensors(ir.host_function)
    pairs = []
    for i, (p, c) in enumerate(pairwise(ir.root_ids)):
        # The intervening full joins must be left intact. In particular the prior
        # root may publish other input data, not just the logits.
        if not i or i + 2 >= len(ir.root_ids):
            continue
        axes = ir.task_families[i].logical_axis_order
        next_axes = ir.task_families[i + 1].logical_axis_order
        if (
            len(axes) != 2
            or len(next_axes) != 1
            or not all(unit_fixed_block(env, ax) for ax in (*axes, *next_axes))
        ):
            continue
        families = (ir.task_families[i], ir.task_families[i + 1])
        if any(
            a.block_id in ir.noncanonical_task_origin_block_ids
            or not a.canonical_origin
            for family in families
            for a in family.axes
        ):
            continue
        if tuple(
            tuple(
                int(a.extent) if isinstance(a.extent, (int, sympy.Integer)) else None
                for a in f.axes
            )
            for f in families
        ) != (
            (1, 4),
            (1,),
        ):
            continue
        if ir.phases[ir.phase_for_root(i)].roots != [i] or ir.phases[
            ir.phase_for_root(i + 1)
        ].roots != [i + 1]:
            continue
        if ir.phase_for_root(i) + 1 != ir.phase_for_root(i + 1):
            continue
        pn = list(ir.graphs[p].graph.nodes)
        cn = list(ir.graphs[c].graph.nodes)
        top = [n for n in pn if n.target is torch.ops.aten.topk.default]
        loads = [n for n in pn if n.target is memory_ops.load]
        stores = [n for n in pn if n.target is memory_ops.store]
        if len(top) != 1 or len(loads) != 1 or len(stores) != 2:
            continue
        (tk,) = top
        (rd,) = loads
        if (
            not _at(tk, torch.ops.aten.topk.default, (rd, 8))
            or len(rd.args) != 4
            or rd.args[2:] != (None, None)
            or rd.kwargs
        ):
            continue
        inds = rd.args[1]
        if not isinstance(inds, (list, tuple)) or len(inds) != 2:
            continue
        row, j = inds
        add = _op(j, torch.ops.aten.add.Tensor)
        if add is None or len(add.args) != 2 or add.kwargs:
            continue
        lanes, base = add.args
        lane = _op(lanes, torch.ops.prims.iota.default)
        mul = _op(base, operator.mul)
        if (
            lane is None
            or lane.args != (32,)
            or lane.kwargs
            != {
                "start": 0,
                "step": 1,
                "dtype": torch.int32,
                "device": env.device,
                "requires_grad": False,
            }
        ):
            continue
        if mul is None or len(mul.args) != 2 or mul.kwargs or mul.args[1] != 32:
            continue
        group = mul.args[0]
        if (_axis(env, row), _axis(env, group)) != tuple(axes):
            continue
        logits = _tensor(env, rd.args[0], (1, 128), torch.float32)
        if logits is None:
            continue
        v, ix = sorted(
            stores,
            key=lambda s: (
                0
                if isinstance(s.args[2], Node) and s.args[2].target is operator.getitem
                else 1
            ),
        )
        if any(
            s.kwargs
            or len(s.args) != 4
            or s.args[3] is not None
            or s.args[1] != [row, group, slice(None)]
            for s in (v, ix)
        ):
            continue
        if not _at(v.args[2], operator.getitem, (tk, 0)):
            continue
        add_ids = _op(ix.args[2], torch.ops.aten.add.Tensor)
        if (
            add_ids is None
            or len(add_ids.args) != 2
            or add_ids.kwargs
            or add_ids.args[1] is not mul
        ):
            continue
        cast = _op(add_ids.args[0], torch.ops.prims.convert_element_type.default)
        if (
            cast is None
            or len(cast.args) != 2
            or cast.kwargs
            or cast.args[1] is not torch.int32
            or not _at(cast.args[0], operator.getitem, (tk, 1))
        ):
            continue
        values = _tensor(env, v.args[0], (1, 4, 8), torch.float32)
        ids = _tensor(env, ix.args[0], (1, 4, 8), torch.int32)
        if (
            values is None
            or ids is None
            or values not in fresh
            or ids not in fresh
            or len({t.untyped_storage()._cdata for t in (logits, values, ids)}) != 3
        ):
            continue
        # Reject any side effect, even a dead or unused operation in this root.
        ancestors = set[Node]()
        todo = [*stores]
        while todo:
            n = todo.pop()
            if n not in ancestors:
                ancestors.add(n)
                todo.extend(n.all_input_nodes)
        if any(n not in ancestors and n.op != "output" for n in pn):
            continue
        hits = [n for n in cn if n.target is torch.ops.aten.topk.default]
        if len(hits) != 1:
            continue
        (mt,) = hits
        if len(mt.args) != 2 or mt.args[1] != 8 or mt.kwargs:
            continue
        vr = _op(mt.args[0], memory_ops.load)
        if vr is None or len(vr.args) != 4 or vr.kwargs or vr.args[2:] != (None, None):
            continue
        vi = vr.args[1]
        if (
            not isinstance(vi, (list, tuple))
            or len(vi) != 2
            or vi[1] != slice(None)
            or _axis(env, vi[0]) != next_axes[0]
        ):
            continue
        pos = next((n for n in cn if _at(n, operator.getitem, (mt, 1))), None)
        irs = [
            n for n in cn if n.target is memory_ops.load and n.args[1] == [vi[0], pos]
        ]
        if pos is None or len(irs) != 1:
            continue
        (rid,) = irs
        if len(rid.args) != 4 or rid.kwargs or rid.args[2:] != (None, None):
            continue
        fv = _tensor(env, vr.args[0], (1, 32), torch.float32)
        fi = _tensor(env, rid.args[0], (1, 32), torch.int32)
        if (
            fv is None
            or fi is None
            or not _same_span(values, fv)
            or not _same_span(ids, fi)
        ):
            continue
        storage = {t.untyped_storage()._cdata for t in (ids, values)}
        if any(n.op != "output" and n.target not in _MERGE_OPS for n in cn):
            continue
        if any(
            (t := node_tensor(n.args[0])) is None
            or t.untyped_storage()._cdata in storage
            for n in cn
            if n.target in (memory_ops.load, memory_ops.store) and n not in (vr, rid)
        ):
            continue
        pairs.append(SelectionHandoff(p, c, logits, values, ids, fv, fi))
    return pairs[0] if len(pairs) == 1 else None


_MERGE_OPS = {
    tr._get_symnode,
    tr._host_tensor,
    tr._mask_to,
    tile_ops.tile_begin,
    memory_ops.load,
    memory_ops.store,
    operator.getitem,
    torch.ops.aten.topk.default,
    torch.ops.aten.amax.default,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.exp.default,
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.mul.Tensor,
    torch.ops.prims.convert_element_type.default,
}
