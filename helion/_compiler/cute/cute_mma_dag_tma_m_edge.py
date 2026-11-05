"""Complete, bounded proof for TMA-only shared-A DAG M edges.

This is not a tensor-owner classifier or a change to warp ragged admission.
The existing whole DAG, registered runtime facts and live source nodes remain
mandatory. Only the otherwise unseeded contiguous-M input slice is declared;
full-M and already warp-ragged programs retain their old facts and ladders.
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import TYPE_CHECKING

import torch

from ...language import creation_ops
from ...language import memory_ops
from ..runtime_tensor_guards import runtime_tensors_are_proven_disjoint
from .memory_ops import cute_tensor_base_is_aligned

if TYPE_CHECKING:
    from .cute_mma_dag import Access
    from .cute_mma_dag import MmaDagProgram

log = logging.getLogger(__name__)


def _plain_access(access: Access, axes: tuple[object, object], *, store: bool) -> bool:
    """Exact rank-two direct tensor coordinates, no mask/view/offset authority."""
    tensor = access.fake
    mask_index = 3 if store else 2
    offset = tensor.storage_offset()
    if type(offset) is not int:
        return False
    return (
        tensor.ndim == 2
        and offset == 0
        and all(type(s) is int and s > 0 for s in tensor.shape)
        and all(type(s) is int and s > 0 for s in access.strides)
        and access.store_map is None
        and tuple(access.dims) == axes
        and tuple(access.tile_dims) == axes
        and len(access.subscripts) == 2
        and all(
            sub.kind == "axis" and sub.offset is None and sub.window is None
            for sub in access.subscripts
        )
        and len(access.node.args) > mask_index
        and access.node.args[mask_index] is None
        and access.node.target is (memory_ops.store if store else memory_ops.load)
    )


def _complete_source_envelope(program: MmaDagProgram) -> bool:
    """Every live root/loop operation belongs to the same bounded dataflow.

    Use the program's actual graphs, including copies used by code generation;
    never infer membership by a name or by another graph's node metadata.
    """
    import operator

    from ...language import _tracing_ops
    from ...language.matmul_ops import dot
    from .matmul_utils import SHAPE_QUERY_TARGETS

    roots = [d.node for d in program.dots] + [s.access.node for s in program.stores]
    required: set[torch.fx.Node] = set()
    stack = list(roots)
    while stack:
        node = stack.pop()
        if node not in required:
            required.add(node)
            stack.extend(node.all_input_nodes)
    allowed = {
        _tracing_ops._host_tensor,
        _tracing_ops._get_symnode,
        _tracing_ops._new_var,
        _tracing_ops._phi,
        _tracing_ops._for_loop,
        operator.getitem,
        creation_ops.full,
        memory_ops.load,
        memory_ops.store,
        torch.ops.prims.convert_element_type.default,
        torch.ops.aten.addmm.default,
        dot,
        *SHAPE_QUERY_TARGETS,
    }
    graphs = {node.graph for node in roots}
    return all(
        node.op in ("placeholder", "output", "get_attr")
        or (node in required and node.op == "call_function" and node.target in allowed)
        for graph in graphs
        for node in graph.nodes
    )


def prove_shared_a_m_edge_axes(
    program: MmaDagProgram, *, allow_unbound: bool = False
) -> tuple[int, int, int] | None:
    """Prove the whole independent shared-A/owned-output family, or refuse.

    Registration may opt into live facts only from the existing registered
    classifiers. Code generation uses their immutable bound snapshots by
    default, and re-proves this complete conjunction on the emitted DAG.
    """
    from ..compile_environment import CompileEnvironment
    from .cute_mma_dag import AXIS_LOOP
    from .cute_mma_dag import AXIS_ROOT
    from .cute_mma_dag import Dot
    from .cute_mma_dag import Store
    from .cute_mma_dag import ragged_root_axes
    from .matmul_utils import strip_new_vars

    env = CompileEnvironment.current()
    if (
        env.config_spec.target_device_capability not in ((10, 0), (10, 3))
        or not program.simple
        or len(program.loops) != 1
        or not 2 <= len(program.dots) <= 4
        or len(program.carries) != len(program.dots)
        or len(program.slots) != len(program.dots)
        or len(program.stores) != len(program.dots)
        or program.derived
        or program.reduces
        or program.vec_ops
        or program.vec_stores
        or program.vec_leaves
        or program.vec_slots
        or program.scans
        or program.pass_chains
        or program.windowed
        or program.tile_index_loads
        or program.scalar_loads
        or program.store_aux_loads
    ):
        return None
    first = program.dots[0]
    m, n, k = first.rows, first.cols, first.k
    if (
        len(program.axes) != 3
        or {axis.kind for axis in program.axes} != {AXIS_ROOT, AXIS_LOOP}
        or m.kind != AXIS_ROOT
        or n.kind != AXIS_ROOT
        or k.kind != AXIS_LOOP
        or any(type(a.extent) is not int or a.extent <= 0 for a in (m, n, k))
        or m.extent < 64
        or m.extent % 64 == 0
        or m.index in ragged_root_axes(program)
        or n.extent % 16
        or k.extent % 16
        or first.transposed
    ):
        return None
    loop = program.loops[0]
    if (
        loop.parent != -1
        or not loop.innermost
        or loop.axes != (k,)
        or loop.tile_end is not None
        or loop.begin != 0
        or loop.end != k.extent
        or loop.vec_carries
        or len(loop.steps) != len(program.dots)
        or any(not isinstance(s, Dot) for s in loop.steps)
        or tuple(loop.steps) != program.dots
        or len(program.body) != 1 + len(program.stores)
        or program.body[0] is not loop
        or any(not isinstance(s, Store) for s in program.body[1:])
        or tuple(program.body[1:]) != program.stores
    ):
        return None
    a = first.a
    if (
        a.kind != "load"
        or a.load is None
        or a.dtype not in (torch.float16, torch.bfloat16)
        or a.load.dtype != a.dtype
        or a.pointwise
        or a.aux_loads
        or not _plain_access(a.load, (m, k), store=False)
        or tuple(a.load.fake.shape) != (m.extent, k.extent)
        # The first slice is the missing contiguous-M copy proof, not a
        # reseed of already admitted row-major/warp-ragged subjects.
        or a.load.strides != (1, m.extent)
    ):
        return None
    inputs = [a.load.fake]
    outputs = []
    for dot, carry, store in zip(
        program.dots, program.carries, program.stores, strict=True
    ):
        if (
            dot.rows != m
            or dot.cols != n
            or dot.k != k
            or dot.region != "loop"
            or dot.loop != loop.index
            or dot.transposed
            or dot.a.chain_key != a.chain_key
            or dot.a.load is None
            or dot.a.load.node is not a.load.node
            or dot.a.value is not a.value
            or dot.a.aux_loads
            or dot.a.pointwise
            or dot.slot != carry.slot
            or dot.acc_slot != carry.slot
            or dot.acc is None
            or strip_new_vars(dot.acc) is not carry.placeholder
            or program.resolve(carry.output) is not dot.node
            or carry.rows != m
            or carry.cols != n
            or carry.init is not None
            or carry.init_frag is not None
            or carry.init_value is not None
            or carry.init_aux
        ):
            return None
        init = carry.phi.args[0]
        if not isinstance(init, torch.fx.Node):
            return None
        init = strip_new_vars(init)
        if (
            init.target is not creation_ops.full
            or len(init.args) < 3
            or type(init.args[1]) not in (int, float)
            or init.args[1] != 0
            or math.copysign(1.0, init.args[1]) != 1.0
            or init.args[2] is not torch.float32
        ):
            return None
        b = dot.b
        if (
            b.kind != "load"
            or b.load is None
            or b.dtype != a.dtype
            or b.load.dtype != b.dtype
            or b.pointwise
            or b.aux_loads
            or not _plain_access(b.load, (k, n), store=False)
            or tuple(b.load.fake.shape) != (k.extent, n.extent)
            or b.load.strides not in ((n.extent, 1), (1, k.extent))
            or store.region != "epilogue"
            or store.swap
            or program.frag_slots.get(store.frag) != carry.slot
            or not _plain_access(store.access, (m, n), store=True)
            or tuple(store.access.fake.shape) != (m.extent, n.extent)
            or store.access.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or store.access.strides != (n.extent, 1)
        ):
            return None
        value = strip_new_vars(store.value)
        seen: set[torch.fx.Node] = set()
        while value is not store.frag:
            if value in seen:
                return None
            seen.add(value)
            if (
                value.target is not torch.ops.prims.convert_element_type.default
                or len(value.args) != 2
                or value.args[1] not in (torch.float16, torch.bfloat16, torch.float32)
                or not isinstance(value.args[0], torch.fx.Node)
            ):
                return None
            value = strip_new_vars(value.args[0])
        inputs.append(b.load.fake)
        outputs.append(store.access.fake)
    if not _complete_source_envelope(program):
        return None
    tensors = (*inputs, *outputs)
    if any(
        env.tensor_input_source(t) is None
        or not cute_tensor_base_is_aligned(env, t, 16)
        or any(s != 1 and (s * t.dtype.itemsize) % 16 for s in t.stride())
        for t in tensors
    ):
        return None
    if not all(
        runtime_tensors_are_proven_disjoint(
            env, left, right, allow_unbound=allow_unbound
        )
        for left, right in itertools.combinations(tensors, 2)
    ):
        return None
    return m.index, n.index, k.index


def discover_shared_a_m_edge_axes(
    program: MmaDagProgram,
) -> tuple[int, int, int] | None:
    """Only optional discovery can fail closed; selected lowering stays loud."""
    try:
        return prove_shared_a_m_edge_axes(program, allow_unbound=True)
    except Exception:
        log.debug("optional shared-A TMA M-edge discovery failed", exc_info=True)
        return None
