"""Prove one complete, warp-uniform paged NT reduction on the projected FX graph.

This is a local substitution inside an existing collective fragment root. The
parent retains its page lookup, masks, branches, consumers and memory effects.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ....language import _tracing_ops
from ....language import atomic_ops
from ....language import creation_ops
from ....language import inline_asm_ops
from ....language import matmul_ops
from ....language import memory_ops
from ....language import tile_ops
from ...device_ir import ForLoopGraphInfo
from ...device_ir import control_flow_child_graph_ids
from ..dependent_phases import local_phase_boundary
from ..register_loads import host_load_is_readonly
from ..scalar_fragment_assertion import requires_assertion_owner
from .conditional_regions import prove_conditional_regions
from .match_common import host_tensor_name
from .match_common import input_closure
from .match_common import is_call
from .match_common import node_tensor
from .match_common import strip_new_var

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...compile_environment import CompileEnvironment
    from ...device_ir import DeviceIR
    from ...device_ir import GraphInfo


@dataclass(frozen=True)
class GatheredBf16Mma:
    lhs: Node
    rhs: Node
    page: Node
    seed: Node
    rows: int
    columns: int
    depth: int


def unmasked_host_load(node: object) -> tuple[Node, tuple[object, ...]] | None:
    if not isinstance(node, Node) or not is_call(node, memory_ops.load):
        return None
    if len(node.args) != 4 or node.args[2:] != (None, None) or node.kwargs:
        return None
    host, indices = node.args[:2]
    if not isinstance(host, Node) or host_tensor_name(host) is None:
        return None
    if not isinstance(indices, (list, tuple)):
        return None
    return host, tuple(indices)


def _tile_origin(index: object, env: CompileEnvironment) -> int | None:
    if not isinstance(index, Node) or not is_call(index, _tracing_ops._get_symnode):
        return None
    size = index.meta.get("val")
    if not isinstance(size, torch.SymInt):
        return None
    bid = env.resolve_block_id(size)
    if bid is None or size._sympy_() != env.block_sizes[bid].var._sympy_():
        return None
    return bid


def prove_gathered_bf16_mma(
    node: Node, graphs: Sequence[GraphInfo], env: CompileEnvironment
) -> GatheredBf16Mma | None:
    """Match a read-only, zero-seeded K fold and its exact outer page capture."""
    if (
        not is_call(node, _tracing_ops._for_loop)
        or len(node.args) != 4
        or node.kwargs
        or node.args[1] != [0]
    ):
        return None
    # A gathered panel substitutes zero for an invalid page. A checked
    # fragment must instead prove every unmasked host load is in bounds.
    if requires_assertion_owner(env.config_spec):
        return None
    graph_id, _, stops, inputs = node.args
    body = next((g for g in graphs if g.graph_id == graph_id), None)
    if (
        not isinstance(body, ForLoopGraphInfo)
        or len(body.block_ids) != 1
        or not isinstance(stops, (tuple, list))
        or len(stops) != 1
        or type(stops[0]) is not int
        or not isinstance(inputs, (tuple, list))
        or len(inputs) != 2
    ):
        return None
    (output,) = body.graph.find_nodes(op="output")
    returns = output.args[0]
    if not isinstance(returns, (list, tuple)) or len(returns) != 1:
        return None
    dot = returns[0]
    if (
        not isinstance(dot, Node)
        or not is_call(dot, matmul_ops.dot)
        or len(dot.args) != 4
        or dot.args[3] is not None
        or dot.kwargs
    ):
        return None
    left, transpose, carry, _ = dot.args
    if (
        not isinstance(transpose, Node)
        or not is_call(transpose, torch.ops.aten.permute.default)
        or len(transpose.args) != 2
        or transpose.args[1] != [1, 0]
    ):
        return None
    left_load, right_load = (
        unmasked_host_load(left),
        unmasked_host_load(transpose.args[0]),
    )
    if left_load is None or right_load is None:
        return None
    lhs, lhs_ix = left_load
    rhs, rhs_ix = right_load
    if (
        len(lhs_ix) != 2
        or len(rhs_ix) != 3
        or lhs_ix[0] != slice(None)
        or rhs_ix[1] != slice(None)
        or lhs_ix[1] is not rhs_ix[2]
        or _tile_origin(lhs_ix[1], env) != body.block_ids[0]
    ):
        return None
    placeholders = tuple(body.graph.find_nodes(op="placeholder"))
    physical, accumulator = strip_new_var(rhs_ix[0]), strip_new_var(carry)
    if len(placeholders) != 2 or {physical, accumulator} != set(placeholders):
        return None
    # Use the call's own captures, not the source graph's pointer identity:
    # graph projection copies FX nodes before codegen.
    page = inputs[placeholders.index(cast("Node", physical))]
    seed = inputs[placeholders.index(cast("Node", accumulator))]
    if not isinstance(page, Node) or not isinstance(seed, Node):
        return None
    if not any(
        is_call(item, operator.getitem)
        and item.args == (node, 0)
        and any(
            is_call(phi, _tracing_ops._phi) and phi.args == (seed, item)
            for phi in item.users
        )
        for item in node.users
    ):
        return None
    if not all(
        host_load_is_readonly(read, env, graphs)
        for read in (cast("Node", left), cast("Node", transpose.args[0]))
    ):
        return None
    table = unmasked_host_load(page)
    scalar = node_tensor(page)
    if (
        table is None
        or scalar is None
        or scalar.ndim
        or scalar.dtype
        not in (
            torch.int32,
            torch.int64,
        )
    ):
        return None
    # One logical page per CTA. The earlier scalar table load retains its own
    # address guards; the physical value still needs bounds checks at use.
    table_indices = table[1]
    if not (
        1 <= len(table_indices) <= 2 and all(type(i) is int for i in table_indices[:-1])
    ):
        return None
    logical = table_indices[-1]
    if (
        not isinstance(logical, Node)
        or not is_call(logical, tile_ops.tile_begin)
        or len(logical.args) != 1
    ):
        return None
    if _tile_origin(logical.args[0], env) is None:
        return None
    q, keys, result = node_tensor(lhs), node_tensor(rhs), node_tensor(dot)
    if q is None or keys is None or result is None:
        return None
    if (
        q.dtype != torch.bfloat16
        or keys.dtype != torch.bfloat16
        or result.dtype != torch.float32
    ):
        return None
    if q.ndim != 2 or keys.ndim != 3 or result.ndim != 2:
        return None
    m, k = q.shape
    n = keys.size(1)
    if not all(type(i) is int for i in (m, n, k)):
        return None
    if min(m, n, k) <= 0 or m % 16 or n % 8 or k % 16 or k != stops[0]:
        return None
    if not (
        env.known_equal(keys.size(2), k)
        and env.known_equal(q.stride(1), 1)
        and env.known_equal(keys.stride(2), 1)
        and env.known_equal(result.size(0), m)
        and env.known_equal(result.size(1), n)
    ):
        return None
    if (
        not is_call(seed, creation_ops.full)
        or len(seed.args) != 4
        or seed.args[0] != [m, n]
        or type(seed.args[1]) not in (float, int)
        or math.copysign(1.0, cast("float", seed.args[1])) < 0
        or seed.args[1] != 0
        or seed.args[2:] != (torch.float32, None)
        or seed.kwargs
    ):
        return None
    # Nothing besides the two reads and the captured fold may disappear.
    if input_closure(output) != set(body.graph.nodes):
        return None
    return GatheredBf16Mma(lhs, rhs, page, seed, m, n, k)


def gathered_phase_region(
    ir: DeviceIR, env: CompileEnvironment, graphs: Sequence[GraphInfo]
) -> list[GraphInfo] | None:
    """Own only the producer of a local ordered RAW boundary.

    Both roots must keep their complete, disjoint conditional/loop trees and
    their original joins. The ordinary persistent scheduler continues to own
    both root grids, the combine's lane map and the grid-wide publication.
    """
    capability = env.config_spec.target_device_capability
    if not (
        capability is not None
        and capability[0] >= 8
        and len(ir.root_ids) == 2
        and tuple(phase.roots for phase in ir.phases) == ([0], [1])
        and ir.implicit_dependency_starts == frozenset({1})
        and ir.tile_dependency_graph is not None
        and local_phase_boundary(ir.tile_dependency_graph)
        and env.process_group_name is None
        and not env.config_spec.has_explicit_pdl
    ):
        return None
    by_id = {graph.graph_id: graph for graph in graphs}
    visited: set[int] = set()
    regions: list[list[GraphInfo]] = []
    for root in ir.root_ids:
        local: list[GraphInfo] = []
        pending = [root]
        while pending:
            index = pending.pop()
            if index in visited or index not in by_id:
                return None
            visited.add(index)
            info = by_id[index]
            local.append(info)
            for n in info.graph.nodes:
                # An opaque program can depend on the executing lane even if
                # its user declares it pure. A re-owned CTA may not evaluate
                # it on another lane or change the order of an atomic effect.
                if (
                    n.target
                    in (
                        _tracing_ops._while_loop,
                        inline_asm_ops.inline_asm_elementwise,
                    )
                    or n.target in atomic_ops.ATOMIC_OPS
                ):
                    return None
                pending.extend(control_flow_child_graph_ids(n))
        if prove_conditional_regions(local, root) is None:
            return None
        regions.append(local)
    if visited != set(by_id):
        return None
    producer = regions[0]
    if not any(
        prove_gathered_bf16_mma(n, graphs, env)
        for info in producer
        for n in info.graph.nodes
    ):
        return None
    return sorted(producer, key=lambda g: g.graph_id)
