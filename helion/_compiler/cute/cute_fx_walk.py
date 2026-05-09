"""Shared FX-graph backward walkers for the cute store path.

Two related but distinct walks read the value-chain that reaches a
``store`` codegen call. Both descend through the same Helion FX-graph
shape (matmul inside a ``_for_loop`` body subgraph; outer graph
consumes the body via ``operator.getitem(_for_loop_node, idx)``), so
the inner-outputs index and the getitem-subgraph-hop logic live here:

- :func:`reach_tcgen05_matmul_anchors` is a broad reachability walk
  used by the G3.1.0 diagnostic and the G3.1.1 store-codegen
  splice site. It descends through every input node (``_phi`` is
  transparent, both branches walked) so a value that depends on a
  tcgen05 matmul through *any* path is detected. Returns the set
  of matmul fx_nodes reached. Empty set means no matmul anchor.

- :func:`walk_carrier_to_tcgen05_matmul` is a narrow per-step walk
  used by the unary-chain classifier (:mod:`cute_epilogue`). It
  only follows ``_phi.args[1]`` (loop body output) and ``_new_var``
  / ``getitem`` passthroughs — the rest is rejected because the
  classifier needs to render the chain expression with a single
  carrier variable. ``_phi.args[0]`` (initial value, e.g.
  ``hl.zeros``) intentionally returns ``None``; the matmul is on
  the body branch.

The two semantics are genuinely distinct (reachability vs single
carrier rendering), so this module exposes both rather than
collapsing them. The shared infrastructure is the
``inner_outputs_by_graph_id`` index and the
``_for_loop`` getitem-hop helper.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


def build_inner_outputs_index(
    state: CodegenState,
) -> dict[int, tuple[torch.fx.Node | None, ...]]:
    """Index live codegen graphs by ``graph_id`` and capture each body's
    output tuple. A ``getitem(_for_loop_node, idx)`` consumer is mapped
    to the inner-graph output at index ``idx``. Inner outputs that are
    not FX nodes (constants, ``None``) are recorded as ``None`` so
    positional indexing into the tuple still works.

    Uses ``state.codegen.codegen_graphs`` (the live codegen-side graph
    list) so node identity matches the matmul fx_node that
    ``_emit_mma_pipeline`` registered. If a future codegen pass
    rewrites graphs during a single ``to_triton_code`` call, the
    registered fx_nodes would belong to the pre-rewrite copy and the
    walks would silently stop firing — today ``build_codegen_graphs``
    is the only graph-copy site and runs once per ``to_triton_code``.
    """
    inner_outputs_by_graph_id: dict[int, tuple[torch.fx.Node | None, ...]] = {}
    for graph_info in state.codegen.codegen_graphs:
        output_nodes = list(graph_info.graph.find_nodes(op="output"))
        if not output_nodes:
            continue
        (output_node,) = output_nodes
        outs = output_node.args[0] if output_node.args else ()
        if isinstance(outs, (list, tuple)):
            inner_outputs_by_graph_id[graph_info.graph_id] = tuple(
                n if isinstance(n, torch.fx.Node) else None for n in outs
            )
    return inner_outputs_by_graph_id


def _resolve_for_loop_getitem(
    node: torch.fx.Node,
    inner_outputs_by_graph_id: dict[int, tuple[torch.fx.Node | None, ...]],
) -> torch.fx.Node | None:
    """If ``node`` is ``operator.getitem(_for_loop_node, idx)`` (with
    ``idx`` a static ``int``), return the inner-graph output node at
    that index — i.e. the body's contribution to that loop output. Any
    other shape returns ``None``.

    The narrow shape is the only safe subgraph hop for both walks:
    descending into other inputs of the ``_for_loop`` call (begin / end /
    args list) would push the entire loop-arg tuple and re-enter all
    body outputs unconditionally, which is the false-positive the
    G3.1.0 cycle-9 review flagged.
    """
    from ...language import _tracing_ops

    if node.op != "call_function" or node.target is not operator.getitem:
        return None
    base = node.args[0] if node.args else None
    idx = node.args[1] if len(node.args) > 1 else None
    if (
        not isinstance(base, torch.fx.Node)
        or not isinstance(idx, int)
        or base.op != "call_function"
        or not _tracing_ops.is_for_loop_target(base.target)
    ):
        return None
    graph_id_arg = base.args[0] if base.args else None
    if not isinstance(graph_id_arg, int):
        return None
    inner_outs = inner_outputs_by_graph_id.get(graph_id_arg, ())
    if not (0 <= idx < len(inner_outs)):
        return None
    return inner_outs[idx]


def reach_tcgen05_matmul_anchors(
    state: CodegenState, value_node: torch.fx.Node
) -> set[torch.fx.Node]:
    """Return the set of registered tcgen05 matmul fx_nodes that
    ``value_node``'s FX graph transitively depends on. Empty set means
    no tcgen05 matmul is reachable.

    The walk descends through every input node (``all_input_nodes``)
    except for the ``getitem(_for_loop, idx)`` subgraph-hop branch,
    which is special-cased to descend into the matching body output
    only — descending the loop's other inputs unconditionally would
    re-enter every body output and false-positive on stores that
    consume a different loop-output mode.

    Used by the G3.1.0 diagnostic backstop in
    :mod:`memory_ops` (the ``cute_tcgen05_matmul_fx_nodes`` reach
    check at the cute store-codegen entry point) and by the G3.1.1
    splice site to recover the unique matmul anchor for the
    accepted unary chain. The classifier in :mod:`cute_epilogue` uses
    a separate :func:`walk_carrier_to_tcgen05_matmul` walker because
    it needs to commit to a single carrier path.
    """
    df = state.device_function
    target_fx_nodes = df.cute_tcgen05_matmul_fx_nodes
    if not target_fx_nodes:
        return set()

    inner_outputs_by_graph_id = build_inner_outputs_index(state)

    from ...language import _tracing_ops

    found: set[torch.fx.Node] = set()
    visited: set[torch.fx.Node] = set()
    stack: list[torch.fx.Node] = [value_node]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        if cur in target_fx_nodes:
            found.add(cur)
            continue
        # `getitem(_for_loop_node, idx)` is the subgraph hop into the
        # body's output[idx]. Descend only into the matching inner
        # output and skip the generic ``all_input_nodes`` fan-out so we
        # don't re-enter every body output via the loop-arg tuple.
        if cur.op == "call_function" and cur.target is operator.getitem:
            base = cur.args[0] if cur.args else None
            if (
                isinstance(base, torch.fx.Node)
                and base.op == "call_function"
                and _tracing_ops.is_for_loop_target(base.target)
            ):
                inner_out = _resolve_for_loop_getitem(cur, inner_outputs_by_graph_id)
                if inner_out is not None and inner_out not in visited:
                    stack.append(inner_out)
                continue
        for arg in cur.all_input_nodes:
            if arg not in visited:
                stack.append(arg)
    return found


def walk_carrier_to_tcgen05_matmul(
    node: torch.fx.Node,
    target_fx_nodes: set[torch.fx.Node],
    inner_outputs_by_graph_id: dict[int, tuple[torch.fx.Node | None, ...]],
) -> torch.fx.Node | None:
    """Walk backward from ``node`` through identity-shape passthrough
    ops until we reach a node in ``target_fx_nodes``. Returns the
    matched matmul node, or ``None`` if a non-passthrough op is
    encountered before the matmul.

    Passthrough ops are ``_phi`` (taking ``args[1]`` — the loop body
    output, since for backward analysis ``args[0]`` is the initial
    value and never contains the matmul) and ``_new_var``, plus the
    same ``getitem(_for_loop, idx)`` subgraph hop as
    :func:`reach_tcgen05_matmul_anchors`.

    Caller passes the pre-built ``inner_outputs_by_graph_id`` so a
    classifier that walks multiple times in a single store-codegen
    call (one per chain step) does not rebuild the index.
    """
    from ...language import _tracing_ops

    visited: set[torch.fx.Node] = set()
    cur: torch.fx.Node | None = node
    while cur is not None:
        if cur in visited:
            return None
        visited.add(cur)
        if cur in target_fx_nodes:
            return cur
        if cur.op != "call_function":
            return None
        target = cur.target
        if target is _tracing_ops._phi:
            arg = cur.args[1] if len(cur.args) > 1 else None
            if not isinstance(arg, torch.fx.Node):
                return None
            cur = arg
            continue
        if target is _tracing_ops._new_var:
            arg = cur.args[0] if cur.args else None
            if not isinstance(arg, torch.fx.Node):
                return None
            cur = arg
            continue
        if target is operator.getitem:
            inner_out = _resolve_for_loop_getitem(cur, inner_outputs_by_graph_id)
            if inner_out is None:
                return None
            cur = inner_out
            continue
        return None
    return None
