"""Forward FX walker that discovers auxiliary-tensor descriptors for
a tcgen05 matmul anchor.

The existing :func:`reach_tcgen05_matmul_anchors` walker in
:mod:`cute_fx_walk` walks *backward* from a store value to the set of
matmul anchors that reach it. That direction is what the store-codegen
splice needs (it has a store value, wants to know whether a tcgen05
matmul anchor exists upstream).

The productive C-input warp (``cute_plan.md`` §7.5.3.2) needs the
*opposite* direction: at MMA-codegen time, given a freshly-registered
matmul anchor, identify the auxiliary tensors that will be consumed by
downstream stores' epilogue chains. Those identities (shape, dtype, FX
node, broadcast axis) are required to allocate the SMEM ring + TMA atom
that the producer body writes into — at the same point where the
matmul plan is constructed, well before the store splice runs and
discovers the same chains backward.

This module implements that forward walk. Each descriptor records the
fields the productive-body codegen will need; multiple descriptors per
matmul are supported (chains with several aux ops, e.g.
``relu(acc + residual) + bias``).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

from ...language import memory_ops
from .cute_epilogue import _AuxiliaryTensorStep
from .cute_epilogue import analyze_tcgen05_unary_epilogue_chain

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST


@dataclasses.dataclass(frozen=True)
class Tcgen05AuxTensorDescriptor:
    """One auxiliary tensor reachable from a tcgen05 matmul anchor.

    A matmul anchor's epilogue chain (as classified by
    :func:`analyze_tcgen05_unary_epilogue_chain`) can fuse multiple
    aux loads — e.g. ``(acc + residual) * bias`` produces two
    descriptors per accepted store. Each descriptor identifies one
    aux operand by:

    - ``load_node``: the FX node for the ``helion.language.load``
      call that reads the aux tensor inside the chain. Same node
      :class:`_AuxiliaryTensorStep` already carries; preserved here
      verbatim so the productive-body splice can recover the
      per-thread index expression.
    - ``host_tensor_fx_node``: the FX node for the aux tensor itself
      (``load_node.args[0]``). The host wrapper looks this up to
      build a TMA atom keyed on the runtime tensor's stride layout.
    - ``host_tensor_val``: the ``torch.Tensor`` ``meta['val']`` on
      the host tensor's FX node — the concrete shape / dtype / device
      the producer-body codegen needs at MMA-codegen time (the TMA
      atom and SMEM ring are sized from these).
    - ``broadcast_axis``: ``None`` for the exact-shape rank-2 form
      (``residual[tile_m, tile_n]``); ``1`` for the trailing-axis
      rowvec broadcast form (``bias[tile_n]``). Matches the field on
      :class:`_AuxiliaryTensorStep` so the productive body uses the
      same axis convention as the existing per-thread splice.
    - ``store_value_node``: the FX node of the store value whose
      chain produced this descriptor. One matmul may be consumed by
      several stores (a kernel that writes the same fused result to
      multiple tensors), so the descriptor is keyed both by the aux
      operand and by which store-chain it appears in. The
      productive-body codegen can use this to pair the aux SMEM
      ring with the right store-splice consumer if a future kernel
      shape produces fan-out.

    The descriptor is intentionally minimal — just the fields the
    productive body needs. Per-aux-step splice locals (var names,
    TMEM partition state, etc.) are still owned by the store-codegen
    splice (``memory_ops._codegen_cute_store_tcgen05_tile``); the
    descriptor only carries the *identity* of the aux tensor so the
    producer-side TMA atom / SMEM ring can be allocated at MMA
    codegen time. Cross-graph FX node identity (the same load node
    visible from both the matmul-codegen pre-pass and the later
    store-codegen splice) is the link between the two sides.
    """

    load_node: torch.fx.Node
    host_tensor_fx_node: torch.fx.Node
    host_tensor_val: torch.Tensor
    broadcast_axis: int | None
    store_value_node: torch.fx.Node


def _store_value_pairs(
    cg: GenerateAST,
) -> list[tuple[torch.fx.Node, torch.fx.Node]]:
    """Enumerate ``(store_node, value_node)`` pairs for every
    ``memory_ops.store`` call across the live codegen graphs.

    Pairs (rather than values alone) so each store's per-store
    output tensor shape stays directly accessible: the chain
    analyzer needs ``output_global_shape`` to validate rowvec
    broadcast aux's global extent, and inferring it from
    ``value_node.users`` after the fact returns ``None`` whenever
    the value fans out to multiple stores — which silently
    disables the broadcast classifier's global-extent check.

    A ``store`` FX node has the shape
    ``call_function(memory_ops.store, (tensor, index, value, extra_mask))``
    (the ``store`` helper in :mod:`helion.language.memory_ops`).
    ``args[2]`` is the value being stored — the entry point the
    chain analyzer walks upstream from.
    """
    pairs: list[tuple[torch.fx.Node, torch.fx.Node]] = []
    for graph_info in cg.codegen_graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or node.target is not memory_ops.store:
                continue
            if len(node.args) < 3:
                continue
            value = node.args[2]
            if isinstance(value, torch.fx.Node):
                pairs.append((node, value))
    return pairs


def _output_global_shape_from_store(
    store_node: torch.fx.Node,
) -> tuple[object, ...] | None:
    """Return the store's output-tensor full shape, or ``None`` when
    it is not recoverable from the FX graph.

    Reads ``args[0].meta['val']`` (the destination tensor's
    ``torch.Tensor`` val) and returns its ``shape`` as a tuple.
    Used to thread per-store global shape into the chain analyzer
    so the rowvec-broadcast classifier can reject auxes whose
    extent matches the tile but not the global axis.
    """
    if not store_node.args:
        return None
    tensor_arg = store_node.args[0]
    if not isinstance(tensor_arg, torch.fx.Node):
        return None
    tensor_val = tensor_arg.meta.get("val")
    if not isinstance(tensor_val, torch.Tensor):
        return None
    return tuple(tensor_val.shape)


def _aux_descriptor_from_step(
    step: _AuxiliaryTensorStep, store_value_node: torch.fx.Node
) -> Tcgen05AuxTensorDescriptor:
    """Build a :class:`Tcgen05AuxTensorDescriptor` from one
    :class:`_AuxiliaryTensorStep` plus the store-value node whose
    chain produced it.

    Asserts on every field rather than silently returning ``None``:
    the analyzer that produced ``step`` has already classified the
    load against the same FX shape we read here (rank-1 / rank-2
    aux operand with a recoverable ``meta['val']``), so any
    mismatch is an internal invariant break — silent drops would
    mask it and surface as a confusing downstream codegen error
    (the productive-body SMEM ring is sized by descriptor count,
    so a dropped descriptor would under-allocate).
    """
    load_node = step.load_node
    assert load_node.args, "_AuxiliaryTensorStep.load_node missing args"
    host_tensor_fx_node = load_node.args[0]
    assert isinstance(host_tensor_fx_node, torch.fx.Node), (
        "_AuxiliaryTensorStep.load_node.args[0] is not an FX node"
    )
    host_tensor_val = host_tensor_fx_node.meta.get("val")
    assert isinstance(host_tensor_val, torch.Tensor), (
        "_AuxiliaryTensorStep host-tensor FX node has no torch.Tensor meta"
    )
    return Tcgen05AuxTensorDescriptor(
        load_node=load_node,
        host_tensor_fx_node=host_tensor_fx_node,
        host_tensor_val=host_tensor_val,
        broadcast_axis=step.broadcast_axis,
        store_value_node=store_value_node,
    )


def discover_tcgen05_aux_tensor_descriptors(
    cg: GenerateAST,
    matmul_fx_node: torch.fx.Node,
) -> tuple[Tcgen05AuxTensorDescriptor, ...]:
    """Forward walk from a tcgen05 matmul anchor to its downstream
    stores; return the auxiliary-tensor descriptors visible in those
    stores' epilogue chains.

    Implementation: enumerate every ``store`` call across the live
    codegen graphs as ``(store_node, value_node)`` pairs, dedup on
    the value node (so a value fed to multiple stores is analyzed
    once and contributes its descriptors once), run the existing
    chain analyzer on each unique value, keep the chains whose
    anchor is the matmul we were asked about, and extract a
    descriptor per accepted aux step.

    The walker is intentionally not a node-by-node forward DFS
    through ``node.users``. The analyzer is the source of truth for
    "what shapes are accepted as epilogue chains rooted at this
    matmul"; reimplementing its acceptance rules forward would
    duplicate the whitelist semantics and risk drift. Iterating
    candidate stores and running the analyzer per-value keeps the
    walker thin and correct: the only forward fact we need is the
    enumeration of stores, which is trivial across the live graphs.

    The matmul fx_node *must* be registered in
    ``DeviceFunction.cute_tcgen05_matmul_fx_nodes`` before the
    walker is called — the analyzer's carrier walker
    (:func:`walk_carrier_to_tcgen05_matmul`) reads that set to know
    where to stop. The MMA-codegen call site registers the fx_node
    first, then invokes this walker, then constructs the matmul
    plan with the descriptors.

    Returns an empty tuple when:

    - The kernel has no store invocations (unusual but possible —
      a kernel that returns its output via a path the walker does
      not see).
    - No store's chain reaches this matmul (pure matmul kernels
      without an aux fusion; the chain analyzer rejects identity
      stores and unary-only chains contribute zero descriptors).
    - The chain analyzer rejects every store's chain (loud-failure
      shapes like 3-D collapse aux, unsupported broadcast,
      reductions). The store-codegen splice will surface the same
      rejection at codegen time; the walker stays silent so an
      unrelated matmul in the same kernel still gets its
      descriptors.

    Multiple descriptors are returned in iteration order over the
    unique value nodes then iteration order within each chain's
    aux steps. The productive-body codegen consumes the tuple as
    an ordered list keyed by aux index (mirroring the existing
    ``_AuxStepRecord``-per-aux-step indexing in
    ``memory_ops._codegen_cute_store_tcgen05_tile``).
    """
    if matmul_fx_node not in cg.device_function.cute_tcgen05_matmul_fx_nodes:
        return ()

    # Local import for ``CodegenState``: the natural top-level
    # import would create a cycle (``inductor_lowering`` ->
    # ``aten_lowering`` -> ``cute.cute_mma`` -> ``cute.aux_tensor``
    # -> back). ``CodegenState`` is a ``NamedTuple`` whose
    # ``dataclasses.field`` defaults are not honored by the tuple
    # constructor, so every required field is provided
    # positionally. Today the analyzer only reads ``codegen`` and
    # ``device_function`` (via the ``codegen.device_function``
    # property), so the empty dict/list defaults below are unused;
    # supplying them explicitly keeps the construction robust
    # against any future analyzer change that touches the optional
    # state fields.
    from ..inductor_lowering import CodegenState

    stub_state = CodegenState(
        codegen=cg, fx_node=None, env={}, proxy_args=[], ast_args=[]
    )

    descriptors: list[Tcgen05AuxTensorDescriptor] = []
    seen_values: set[torch.fx.Node] = set()
    for store_node, store_value in _store_value_pairs(cg):
        # Dedup: a value fed to multiple stores must contribute its
        # descriptors once. Walking the same chain N times would
        # emit N copies and the productive-body SMEM ring (keyed
        # by descriptor position) would silently double-allocate.
        if store_value in seen_values:
            continue
        seen_values.add(store_value)

        output_global_shape = _output_global_shape_from_store(store_node)
        analyzed = analyze_tcgen05_unary_epilogue_chain(
            stub_state, store_value, output_global_shape=output_global_shape
        )
        if analyzed is None:
            continue
        chain, anchor = analyzed
        if anchor is not matmul_fx_node:
            continue
        for step in chain.auxiliary_tensor_steps:
            descriptors.append(_aux_descriptor_from_step(step, store_value))
    return tuple(descriptors)
