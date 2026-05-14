"""Shared FX-graph backward walkers for the cute store path.

Two related but distinct walks read the value-chain that reaches a
``store`` codegen call. Both descend through the same Helion FX-graph
shape (matmul inside a ``_for_loop`` body subgraph; outer graph
consumes the body via ``operator.getitem(_for_loop_node, idx)``), so
the inner-outputs index and the getitem-subgraph-hop logic live here:

- :func:`reach_tcgen05_matmul_anchors` is a broad reachability walk
  used by the loud-failure diagnostic and the store-codegen splice
  site. It descends through every input node (``_phi`` is
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
    from collections.abc import Sequence

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
    body outputs unconditionally, which would false-positive on
    stores that consume a different loop-output mode.
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

    Used by the loud-failure diagnostic backstop in
    :mod:`memory_ops` (the ``cute_tcgen05_matmul_fx_nodes`` reach
    check at the cute store-codegen entry point) and by the splice
    site to recover the unique matmul anchor for the accepted
    chain. The classifier in :mod:`cute_epilogue` uses a separate
    :func:`walk_carrier_to_tcgen05_matmul` walker because it needs
    to commit to a single carrier path.
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


def aux_tensor_load_kind(
    node: torch.fx.Node,
    *,
    carrier_tile_shape: tuple[object, ...] | None,
    carrier_tile_index_nodes: tuple[torch.fx.Node, ...] | None = None,
    carrier_global_shape: tuple[object, ...] | None = None,
) -> tuple[str, int | None] | None:
    """Classify ``node`` as a recognized auxiliary-tensor load kind, or
    return ``None`` when the load is not eligible for fusion.

    Returns one of:

    - ``("exact", None)``: the canonical 2-D ``aux[tile_m, tile_n]``
      shape — the underlying tensor's rank equals the carrier rank,
      the load result shape matches the carrier tile shape, and the
      load's index list is exactly the carrier's tile-id symbol
      nodes in the same order. The splice site uses the existing
      partition_C → flat_divide → partition_D pipeline.
    - ``("broadcast", 1)``: a rank-1 trailing-axis broadcast aux
      load (``bias[tile_n]``). The underlying tensor's rank is 1,
      the load result shape equals the carrier tile shape's
      trailing entry, and the load's single index symbol is
      exactly ``carrier_tile_index_nodes[1]``. Only the trailing
      axis is supported because PyTorch broadcasting aligns a
      rank-1 operand to the *last* dimension of a rank-2 LHS:
      ``acc + bias[tile_m]`` is either a shape error (BM ≠ BN) or
      rowvec broadcast against the trailing axis (BM == BN). A
      user wanting M-axis (column-vector) broadcasting must write
      it explicitly (``bias[tile_m][:, None]`` /
      ``.unsqueeze(-1)``); that is a separate, deferred pattern.
      See :class:`Tcgen05UnaryEpilogueChain` (``cute_epilogue.py``)
      for the splice-side broadcast-view contract.

    The classifier returns ``None`` for everything else: 3-D
    underlying tensors with a static collapse
    (``aux3d[tile_m, tile_n, 0]``), broadcast variants whose index
    is not the exact carrier tile-id symbol (e.g. ``bias[tile_n + 1]``),
    rank mismatches, kwargs, non-default ``extra_mask`` /
    ``eviction_policy`` positions, or rank-1 indexed by the
    carrier's leading-axis tile id (``bias[tile_m]`` — see above).

    Strict on ``kwargs`` (must be empty) and on the ``extra_mask``
    / ``eviction_policy`` argument positions — both must be
    ``None`` because the splice emits a plain
    ``aux_tile[indices].load()`` without forwarding either knob.

    ``carrier_tile_shape`` is the carrier's tile shape (sympy
    symbols / ints from ``meta['val'].shape``).
    ``carrier_tile_index_nodes`` is the tuple of FX symint nodes
    that index the carrier (one per tile axis) — when provided, the
    load's index list must match these node-by-node (or, for the
    broadcast kind, the single load index must equal
    ``carrier_tile_index_nodes[1]``). ``carrier_global_shape`` is
    the output tensor's full (non-tile) shape; when provided the
    broadcast classifier additionally checks that the rank-1 aux's
    extent matches the output's global size on the broadcast axis,
    closing the gap where the bias is smaller than the global axis
    but happens to be divisible by the tile extent.
    """
    from ...language.memory_ops import load as helion_load

    if node.op != "call_function" or node.target is not helion_load:
        return None
    if node.kwargs:
        return None
    args = node.args
    if len(args) < 2:
        return None
    tensor_node = args[0]
    if not isinstance(tensor_node, torch.fx.Node):
        return None
    index_list = args[1]
    if not isinstance(index_list, (list, tuple)):
        return None
    if len(args) >= 3 and args[2] is not None:
        return None  # extra_mask present.
    if len(args) >= 4 and args[3] is not None:
        return None  # eviction_policy present.
    aux_val = node.meta.get("val")
    if aux_val is None:
        return None
    aux_shape = tuple(aux_val.shape)
    aux_tensor_val = tensor_node.meta.get("val")
    if aux_tensor_val is None:
        return None
    aux_tensor_shape = tuple(aux_tensor_val.shape)

    # Defensive default when the caller cannot supply the carrier's
    # tile shape (e.g., chain entry point with unrecoverable meta).
    # Without the carrier shape we cannot disambiguate the broadcast
    # axis, so only the rank-2 exact form is accepted in this mode
    # — broadcast classification is gated on having a known carrier.
    if carrier_tile_shape is None:
        if len(aux_shape) == 2 and len(aux_tensor_shape) == 2:
            return ("exact", None)
        return None

    # The cute backend only produces rank-2 matmul carriers today.
    # Higher-rank carriers fall through to the loud-failure backstop.
    if len(carrier_tile_shape) != 2:
        return None

    if _matches_exact(
        aux_shape=aux_shape,
        aux_tensor_shape=aux_tensor_shape,
        index_list=index_list,
        carrier_tile_shape=carrier_tile_shape,
        carrier_tile_index_nodes=carrier_tile_index_nodes,
    ):
        return ("exact", None)

    return _matches_broadcast(
        aux_shape=aux_shape,
        aux_tensor_shape=aux_tensor_shape,
        aux_tensor_val=aux_tensor_val,
        index_list=index_list,
        carrier_tile_shape=carrier_tile_shape,
        carrier_tile_index_nodes=carrier_tile_index_nodes,
        carrier_global_shape=carrier_global_shape,
    )


def _matches_broadcast(
    *,
    aux_shape: tuple[object, ...],
    aux_tensor_shape: tuple[object, ...],
    aux_tensor_val: torch.Tensor,
    index_list: Sequence[object],
    carrier_tile_shape: tuple[object, ...],
    carrier_tile_index_nodes: tuple[torch.fx.Node, ...] | None,
    carrier_global_shape: tuple[object, ...] | None,
) -> tuple[str, int] | None:
    """Check whether a load matches the rank-1 trailing-axis
    (rowvec) broadcast aux pattern, and return ``("broadcast", 1)``
    on success.

    The classifier intentionally restricts to the trailing axis
    only: a bare rank-1 operand on the RHS of a rank-2 carrier
    aligns to the last dimension under PyTorch broadcasting rules,
    so ``acc + bias[tile_m]`` is not a column-vector broadcast — it
    is a shape error (BM ≠ BN) or trailing-axis broadcast (BM == BN).
    Users wanting an explicit M-axis broadcast must write it
    explicitly (``bias[tile_m][:, None]`` / ``.unsqueeze(-1)`` /
    ``.expand(...)``); that is a separate pattern handler not yet
    wired up.

    The splice site emits ``cute.make_layout((m, n),
    stride=(0, 1))`` with stride 1 hard-coded on the data axis,
    which is correct only when the underlying rank-1 tensor is
    contiguous (stride 1 on dim 0). Non-stride-1 broadcast aux
    is rejected loudly.

    When ``carrier_global_shape`` is provided, the rank-1 aux's
    extent must equal the output tensor's global size on the
    broadcast axis — not just the tile extent. The classifier
    matches the tile extent first (the ``aux[tile_n]`` user
    surface guarantees ``aux_extent == carrier_tile_shape[1]``);
    enforcing equality with ``carrier_global_shape[1]`` closes
    the gap where a smaller-than-global bias whose length happens
    to be divisible by the tile extent passes the tile check yet
    reads OOB at runtime.
    """
    if len(aux_tensor_shape) != 1 or len(aux_shape) != 1 or len(index_list) != 1:
        return None
    if carrier_tile_index_nodes is None:
        # Broadcast classification is gated on knowing the carrier's
        # tile-id symbols so we can pin which axis the load broadcasts
        # along. Without that, the unpredicated splice could broadcast
        # the wrong axis silently — bail to the loud-failure backstop.
        return None
    if len(carrier_tile_index_nodes) != 2:
        return None
    load_idx = index_list[0]
    if not isinstance(load_idx, torch.fx.Node):
        return None
    if aux_tensor_val.stride() != (1,):
        return None
    aux_extent = aux_shape[0]
    # Trailing-axis (rowvec) broadcast only — see docstring.
    axis = 1
    if load_idx is not carrier_tile_index_nodes[axis]:
        return None
    if aux_extent != carrier_tile_shape[axis]:
        return None
    if carrier_global_shape is not None:
        if len(carrier_global_shape) != 2:
            return None
        # The underlying rank-1 aux tensor's *total* length must
        # match the output's global axis size on the broadcast
        # axis. The earlier ``aux_extent == carrier_tile_shape``
        # check matches the per-load *tile* extent against the
        # carrier's tile shape; without the global-size check, an
        # aux smaller than the output axis whose length happens to
        # be divisible by the tile extent could pass classification
        # yet read OOB at runtime.
        if aux_tensor_shape[0] != carrier_global_shape[axis]:
            return None
    return ("broadcast", axis)


def _matches_exact(
    *,
    aux_shape: tuple[object, ...],
    aux_tensor_shape: tuple[object, ...],
    index_list: Sequence[object],
    carrier_tile_shape: tuple[object, ...],
    carrier_tile_index_nodes: tuple[torch.fx.Node, ...] | None,
) -> bool:
    """Check whether a load matches the exact-shape aux pattern:
    underlying tensor rank equals carrier rank, result shape equals
    carrier tile shape dim-by-dim, and the load's index list is
    exactly the carrier tile-id symbols (in order). When the
    classifier cannot recover the carrier tile-id symbols
    (``carrier_tile_index_nodes is None``), fall back to checking
    each entry of ``index_list`` is an FX symint node — used only
    for defensive paths that lose the tile-id walk.
    """
    rank = len(carrier_tile_shape)
    if (
        len(aux_shape) != rank
        or len(aux_tensor_shape) != rank
        or len(index_list) != rank
    ):
        return False
    for aux_dim, carrier_dim in zip(aux_shape, carrier_tile_shape, strict=True):
        if aux_dim != carrier_dim:
            return False
    if carrier_tile_index_nodes is not None:
        if len(carrier_tile_index_nodes) != len(index_list):
            return False
        for idx, expected in zip(index_list, carrier_tile_index_nodes, strict=True):
            if idx is not expected:
                return False
    else:
        for idx in index_list:
            if not isinstance(idx, torch.fx.Node):
                return False
    return True


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
