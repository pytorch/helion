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

from ...language import matmul_ops
from ...language import memory_ops
from ..compile_environment import CompileEnvironment
from .cute_epilogue import _AuxiliaryTensorStep
from .cute_epilogue import analyze_tcgen05_unary_epilogue_chain
from .cute_fx_walk import build_inner_outputs_index_from_graphs
from .cute_fx_walk import walk_carrier_to_tcgen05_matmul

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from ..host_function import HostFunction


# Mirrors ``cute_mma._TRACE_THROUGH_TARGETS`` — the data-preserving
# wrappers the MMA codegen walks through when resolving operand
# loads (``_trace_to_load`` in ``cute_mma``). Duplicated here
# rather than imported because ``cute_mma`` imports from this
# module, creating a circular dependency. The single
# ``convert_element_type`` target is the canonical operand-cast
# wrapper Helion emits for ``lhs.to(dtype) @ rhs.to(dtype)``
# patterns; keep this set in sync with ``cute_mma`` if either
# side adds a new wrapper.
_MMA_OPERAND_TRACE_THROUGH_TARGETS = (torch.ops.prims.convert_element_type.default,)
# Include ``matmul_ops.dot`` because Helion's public ``hl.dot`` API lowers to
# the same tcgen05 MMA path as the aten matmul-family calls.
_TCGEN05_AUX_DETECTOR_MMA_TARGETS = (
    torch.ops.aten.addmm.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.baddbmm.default,
    matmul_ops.dot,
)


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
    return [
        pair
        for graph_info in cg.codegen_graphs
        for pair in _store_value_pairs_from_graph(graph_info.graph)
    ]


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
    tensor_val = _output_tensor_from_store_node(store_node)
    return tuple(tensor_val.shape) if tensor_val is not None else None


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
    ``DeviceFunction.cute_state.matmul_fx_nodes`` before the
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
    if matmul_fx_node not in cg.device_function.cute_state.matmul_fx_nodes:
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


def host_function_has_tcgen05_aux_kernel_pattern(
    host_function: HostFunction,
) -> bool:
    """Conservative pre-codegen detector for kernels whose tcgen05
    matmul is followed by an aux-fused store
    (``out[tile] = (acc + residual[tile]).to(...)`` and the
    bias / rowvec variants — see :class:`_AuxiliaryTensorStep`
    for the accepted forms).

    Runs at autotune-surface configuration time (post-FX-graph
    build, pre-autotune-sample) where the
    :func:`discover_tcgen05_aux_tensor_descriptors` walker is
    unavailable — that walker requires a populated
    ``DeviceFunction.cute_state.matmul_fx_nodes`` set, which
    is only registered at MMA-codegen time per config sample.
    The aux pipeline only fires when the productive-body gate
    sees aux descriptors, so the autotune surface only needs
    to widen ``tcgen05_warp_spec_c_input_warps`` when this
    detector returns True; otherwise the C-input warp is
    inert and sampling ``c_input_warps=1`` would be a strict
    resource cost for pure-matmul kernels (the inert warp
    occupies an SM slot without delivering work).

    Implementation: walk every FX graph in
    ``host_function.device_ir.graphs`` and look for the
    coarse pattern "the kernel has at least one MMA-anchor
    call AND at least one ``memory_ops.load`` call whose
    result is NOT used as one of the MMA operands". MMA
    anchors include both the aten paths
    (``aten.addmm`` / ``aten.baddbmm`` / ``aten.mm`` /
    ``aten.bmm``) AND the ``hl.dot`` HOP — both ends up
    in the tcgen05 MMA lowering, and the canonical Helion
    residual kernel uses ``hl.dot``. Operand resolution
    traces through the same ``convert_element_type``
    wrappers the MMA codegen accepts
    (``cute_mma._trace_to_load`` /
    ``_TRACE_THROUGH_TARGETS``); without this trace,
    kernels written as ``lhs.to(dtype) @ rhs.to(dtype)``
    would have their operand loads classified as aux and
    the detector would over-fire on pure matmul.

    The detector is **conservative**: a false positive
    only widens the autotune search by one enum value
    (the productive-body safety gates handle the resulting
    invalid combinations at codegen time and the
    inert-body fallback handles invalid runs); a false
    negative would miss the residual c_input=1 win. The
    chain analyzer at codegen time
    (``analyze_tcgen05_unary_epilogue_chain``) remains the
    source of truth for which aux shapes are *accepted*
    by the productive-body codegen; this detector
    intentionally over-approximates so the autotune surface
    admits the productive shape whenever any aux load is
    plausibly involved.

    Caught patterns include:

    - exact-shape residual: ``out[tile] = (acc + residual[tile]).to(...)``
    - rowvec broadcast: ``out[tile] = (acc + bias[tile_n]).to(...)``
    - chained: ``(acc + residual)*bias``, ``relu(acc + residual)``, etc.
    - any of the above using ``hl.dot(...)`` instead of
      ``torch.addmm`` / ``torch.matmul``.
    """
    device_ir = host_function.device_ir
    graphs = device_ir.graphs
    if not graphs:
        return False

    mma_nodes, operand_load_nodes = _tcgen05_aux_detector_mma_facts(graphs)
    if not mma_nodes:
        return False
    # Second pass: any ``memory_ops.load`` call whose result
    # is NOT a resolved MMA-operand load is treated as aux.
    # Operand loads feed the dot directly (after optional
    # casts); aux loads feed an arithmetic op that combines
    # with the dot's result on the way to a store.
    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            if (
                node.op == "call_function"
                and node.target is memory_ops.load
                and node not in operand_load_nodes
            ):
                return True
    return False


def host_function_has_tcgen05_exact_shape_aux_kernel_pattern(
    host_function: HostFunction,
) -> bool:
    """Return True when a tcgen05 aux pattern has a rank-2 exact-shape input.

    The aux-TMA path requires the staged aux tensor to match the output dtype,
    so this detector requires both exact output shape and matching dtype.
    Row-vector broadcast aux inputs intentionally stay on the direct SIMT path
    because the TMA path has no useful tile to bulk-load for them. Rank-3 and
    other batched aux patterns are also excluded: the current aux-TMA wrapper
    constructs a rank-2 TMA tensor for a single output tile.

    This is still a pre-codegen detector. It reuses the epilogue-chain
    analyzer so exact-shape aux loads must pass the same whitelist and index
    checks as codegen. It then requires exactly one compatible store value,
    and every exact-shape aux descriptor on that store must match the output
    shape and dtype, mirroring the codegen-time TMA descriptor dtype gate.
    Any exact-shape aux shape or dtype mismatch in an analyzed store disables
    the detector for the whole kernel rather than trying to partially admit
    aux-TMA.
    """

    device_ir = host_function.device_ir
    graphs = device_ir.graphs
    if not graphs:
        return False

    mma_nodes, _operand_load_nodes = _tcgen05_aux_detector_mma_facts(graphs)
    if not mma_nodes:
        return False

    store_outputs: list[tuple[torch.fx.Node, torch.Tensor]] = []
    for graph_info in graphs:
        for store_node, store_value in _store_value_pairs_from_graph(graph_info.graph):
            tensor_val = _output_tensor_from_store_node(store_node)
            if tensor_val is not None:
                store_outputs.append((store_value, tensor_val))
    if not store_outputs:
        return False

    return _has_tma_compatible_analyzed_aux_store(
        store_outputs,
        inner_outputs_by_graph_id=build_inner_outputs_index_from_graphs(graphs),
        target_fx_nodes=mma_nodes,
    )


def host_function_has_tcgen05_identity_matmul_store_pattern(
    host_function: HostFunction,
) -> bool:
    """Return True only for a single identity store of a tcgen05 matmul result."""

    device_ir = host_function.device_ir
    graphs = device_ir.graphs
    if not graphs:
        return False

    mma_nodes, _operand_load_nodes = _tcgen05_aux_detector_mma_facts(graphs)
    if not mma_nodes:
        return False

    store_outputs: list[tuple[torch.fx.Node, torch.Tensor]] = []
    for graph_info in graphs:
        for store_node, store_value in _store_value_pairs_from_graph(graph_info.graph):
            tensor_val = _output_tensor_from_store_node(store_node)
            if tensor_val is not None:
                store_outputs.append((store_value, tensor_val))
    if len(store_outputs) != 1:
        return False

    store_value, _output = store_outputs[0]
    if (
        store_value.op != "call_function"
        or store_value.target is not torch.ops.prims.convert_element_type.default
        or store_value.kwargs
    ):
        return False
    cast_input = store_value.args[0] if store_value.args else None
    if not isinstance(cast_input, torch.fx.Node):
        return False
    return (
        walk_carrier_to_tcgen05_matmul(
            cast_input,
            mma_nodes,
            build_inner_outputs_index_from_graphs(graphs),
        )
        is not None
    )


def _tcgen05_aux_detector_mma_facts(
    graphs: Sequence[GraphInfo],
) -> tuple[set[torch.fx.Node], set[torch.fx.Node]]:
    # MMA anchor targets: both the aten paths and the ``hl.dot`` HOP
    # (the canonical Helion API entrypoint).
    mma_nodes: set[torch.fx.Node] = set()
    operand_load_nodes: set[torch.fx.Node] = set()
    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            if (
                node.op != "call_function"
                or node.target not in _TCGEN05_AUX_DETECTOR_MMA_TARGETS
            ):
                continue
            mma_nodes.add(node)
            # Treat every positional arg that resolves through the same operand
            # trace accepted by MMA codegen as an operand load.
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    load_node = _trace_to_load_through_casts(arg)
                    if load_node is not None:
                        operand_load_nodes.add(load_node)
    return mma_nodes, operand_load_nodes


def _store_value_pairs_from_graph(
    graph: torch.fx.Graph,
) -> list[tuple[torch.fx.Node, torch.fx.Node]]:
    pairs: list[tuple[torch.fx.Node, torch.fx.Node]] = []
    for node in graph.nodes:
        if node.op != "call_function" or node.target is not memory_ops.store:
            continue
        if len(node.args) < 3:
            continue
        value = node.args[2]
        if isinstance(value, torch.fx.Node):
            pairs.append((node, value))
    return pairs


def _output_tensor_from_store_node(store_node: torch.fx.Node) -> torch.Tensor | None:
    if not store_node.args:
        return None
    tensor_arg = store_node.args[0]
    if not isinstance(tensor_arg, torch.fx.Node):
        return None
    tensor_val = tensor_arg.meta.get("val")
    return tensor_val if isinstance(tensor_val, torch.Tensor) else None


def _same_static_shape(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if lhs.ndim != rhs.ndim:
        return False
    # Use the active compile environment so unbacked symbolic dimensions
    # conservatively return false instead of raising during bind.
    env = CompileEnvironment.current()
    for lhs_size, rhs_size in zip(lhs.shape, rhs.shape, strict=True):
        if not env.known_equal(lhs_size, rhs_size):
            return False
    return True


def _has_tma_compatible_analyzed_aux_store(
    store_outputs: list[tuple[torch.fx.Node, torch.Tensor]],
    *,
    inner_outputs_by_graph_id: dict[int, tuple[torch.fx.Node | None, ...]],
    target_fx_nodes: set[torch.fx.Node],
) -> bool:
    # Dedup by store value so repeated stores of the same computed chain are
    # treated as one aux-TMA candidate at detector time.
    compatible_store_values: set[torch.fx.Node] = set()
    for store_value, output in store_outputs:
        analyzed = analyze_tcgen05_unary_epilogue_chain(
            None,
            store_value,
            output_global_shape=tuple(output.shape),
            target_fx_nodes=target_fx_nodes,
            inner_outputs_by_graph_id=inner_outputs_by_graph_id,
        )
        if analyzed is None:
            continue
        chain, _anchor = analyzed
        exact_steps = [
            step for step in chain.auxiliary_tensor_steps if step.broadcast_axis is None
        ]
        if not exact_steps:
            continue
        for step in exact_steps:
            assert step.load_node.args, "_AuxiliaryTensorStep.load_node missing args"
            tensor_arg = step.load_node.args[0]
            assert isinstance(tensor_arg, torch.fx.Node), (
                "_AuxiliaryTensorStep.load_node.args[0] is not an FX node"
            )
            aux_tensor = tensor_arg.meta.get("val")
            assert isinstance(aux_tensor, torch.Tensor), (
                "_AuxiliaryTensorStep host-tensor FX node has no torch.Tensor meta"
            )
            if not _same_static_shape(aux_tensor, output):
                return False
            if aux_tensor.dtype != output.dtype:
                return False
        compatible_store_values.add(store_value)
        if len(compatible_store_values) > 1:
            return False
    return len(compatible_store_values) == 1


def _trace_to_load_through_casts(node: torch.fx.Node) -> torch.fx.Node | None:
    """Walk backward from ``node`` through accepted operand-cast
    wrappers to the underlying ``memory_ops.load``.

    Mirrors ``cute_mma._trace_to_load`` for the operand-trace
    contract: only data-preserving ops in
    ``_MMA_OPERAND_TRACE_THROUGH_TARGETS`` are walked
    through, matching the MMA codegen's own operand-resolution
    behavior. Returns the resolved ``memory_ops.load`` node, or
    ``None`` if a non-wrapper op (e.g. arithmetic) is
    encountered before the load — in that case the operand is
    not a pure load chain and we conservatively decline to
    treat it as an MMA operand load.
    """
    cur = node
    while cur.op == "call_function" and cur.target is not memory_ops.load:
        if cur.target not in _MMA_OPERAND_TRACE_THROUGH_TARGETS:
            return None
        input_nodes = [a for a in cur.args if isinstance(a, torch.fx.Node)]
        if len(input_nodes) != 1:
            return None
        cur = input_nodes[0]
    if cur.op != "call_function" or cur.target is not memory_ops.load:
        return None
    return cur
