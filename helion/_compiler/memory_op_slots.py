"""Single source of truth for per-memory-op tunable slot resolution (Triton backend).

Every emitted ``tl.load`` / ``tl.store`` / atomic reads its per-op config knob
(``config.indexing`` / ``config.load_eviction_policies`` / ``config.load_cache_modifiers`` /
``config.atomic_indexing``) from a slot. Historically the slot was a running emission-position
counter walked in config-dependent codegen-descent order, so the same source op landed in a
different slot under different configs (reduction rolling hoists/duplicates loads; epilogue
subtiling splits stores). See ``_lab/MEMORY_OP_IDENTITY_BRAINSTORM.md``.

``MemoryOpSlotBroker`` centralizes the four codegen-side slot namespaces so that slot resolution
has a single owner. In this (Stage 0) form it resolves **positionally** — byte-identical to the
legacy counters — but its API takes the *node* being emitted (not ``next()``/``++``), so a later
stage can flip the internals to resolve by a stable ``node.meta['mem_op_id']`` without touching any
call site.

Namespaces (densities differ — they are independent counters):
  * ``indexing``  → ``config.indexing``               (every load + store; tile.index/stack slots inert)
  * ``eviction``  → ``config.load_eviction_policies``  (loads; the codegen counter is dense over all)
  * ``cache``     → ``config.load_cache_modifiers``    (on-device loads)
  * ``atomic``    → ``config.atomic_indexing``         (every atomic; ``atomic_cas``'s slot is inert)
plus a store tally (the legacy ``device_store_index``; bookkeeping only, never a config slot).

The legacy split-store / split-atomic 1:N share (``epilogue_subtile_group_id``: the primary piece
allocates a slot, the other pieces reuse it) is reproduced here unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .device_ir import DeviceIR
    from .variable_origin import Origin


# A stable, transform-invariant identity for a source memory op:
#   (column-aware SourceLocation._key(), access-site buffer identity from origin.host_str()).
# host_str() includes the GetItem ``[key]`` so tuple-element buffers stay distinct (root_rw_name()
# collides on them); ``None`` for synthesized tensors with no host origin (tile.index / stack). All
# N copies of a source op (reduction rolling re-emit, epilogue split, static unroll) share this id
# because ``node.meta`` is preserved across every copy pass.
# loc-key matches SourceLocation._key(): (filename, lineno, colno, end_lineno, end_colno).
MemOpId = tuple["tuple[str, int | None, int, int, int] | None", "str | None"]


def _accessed_fake(node: torch.fx.Node) -> torch.Tensor | None:
    """Fake tensor of the buffer a load/store/atomic accesses (``args[0]``)."""
    arg = node.args[0] if node.args else None
    if isinstance(arg, torch.fx.Node):
        val = arg.meta.get("val")
        if isinstance(val, torch.Tensor):
            return val
    return None


def compute_mem_op_id(
    node: torch.fx.Node,
    tensor_to_origin: dict[torch.Tensor, Origin],
) -> MemOpId:
    """Derive the stable ``mem_op_id`` for a load/store/atomic node (see ``MemOpId``)."""
    loc = node.meta.get("location")
    loc_key = loc._key() if loc is not None else None
    fake = _accessed_fake(node)
    origin = tensor_to_origin.get(fake) if fake is not None else None
    buffer_key = origin.host_str() if origin is not None else None
    return (loc_key, buffer_key)


class MemoryOpSlotBroker:
    """Owns the four codegen slot namespaces for one ``DeviceFunction`` codegen run.

    Stage 0: positional resolution (reproduces the legacy emission counters byte-identically).
    """

    def __init__(
        self,
        slot_map: MemOpSlotMap | None = None,
        *,
        id_keyed: bool = False,
    ) -> None:
        # Counters (the legacy DeviceFunction.* fields now live here; DeviceFunction exposes
        # them as properties for the metal/pallas codegen paths that bump them directly).
        self.indexing_counter = 0
        self.eviction_counter = 0
        self.cache_counter = 0
        self.atomic_counter = 0
        self.store_counter = 0  # legacy device_store_index tally (not a config slot)
        # epilogue_subtile_group_id -> shared indexing/atomic slot (1:N split share).
        self.epilogue_store_slots: dict[str, int] = {}
        self.epilogue_atomic_slots: dict[str, int] = {}
        # When id_keyed, resolve each op's slot from the stable node.meta['mem_op_id'] via slot_map
        # (transform-invariant) instead of the positional counters. The counters then go unused on
        # the triton path; metal/pallas keep bumping them positionally through the properties.
        self._slot_map = slot_map
        self._id_keyed = id_keyed and slot_map is not None

    def _slot(self, kind: str, node: torch.fx.Node | None) -> int:
        assert node is not None and self._slot_map is not None
        mem_op_id = node.meta["mem_op_id"]
        table: dict[MemOpId, int] = {
            "indexing": self._slot_map.indexing,
            "eviction": self._slot_map.eviction,
            "cache": self._slot_map.cache,
            "atomic": self._slot_map.atomic,
        }[kind]
        slot = table.get(mem_op_id)
        assert slot is not None, (
            f"mem_op_id {mem_op_id!r} has no {kind} slot in the bind-time map "
            f"(node {node.name!r}); the slot map is incomplete."
        )
        return slot

    # -- loads (the triton load handler calls these in the legacy bump order:
    #    eviction (always) -> cache (on-device) -> [tile.index early-return] -> indexing) --

    def load_eviction_slot(self, node: torch.fx.Node | None) -> int:
        if self._id_keyed:
            return self._slot("eviction", node)
        idx = self.eviction_counter
        self.eviction_counter += 1
        return idx

    def load_cache_slot(self, node: torch.fx.Node | None) -> int:
        if self._id_keyed:
            return self._slot("cache", node)
        idx = self.cache_counter
        self.cache_counter += 1
        return idx

    def load_indexing_slot(self, node: torch.fx.Node | None) -> int:
        if self._id_keyed:
            return self._slot("indexing", node)
        idx = self.indexing_counter
        self.indexing_counter += 1
        return idx

    # -- stores (shared loads+stores indexing namespace; split stores share one slot) --

    def _alloc_store_indexing(self) -> int:
        self.store_counter += 1
        idx = self.indexing_counter
        self.indexing_counter += 1
        return idx

    def store_indexing_slot(self, node: torch.fx.Node | None) -> int:
        if self._id_keyed:
            # All split-store pieces share the source store's mem_op_id -> one slot (1:N share is
            # automatic; the epilogue group bookkeeping is only needed by the positional path).
            return self._slot("indexing", node)
        group_id = None if node is None else node.meta.get("epilogue_subtile_group_id")
        if group_id is None:
            return self._alloc_store_indexing()
        if node is not None and node.meta.get("epilogue_subtile_primary_output", False):
            idx = self._alloc_store_indexing()
            self.epilogue_store_slots[group_id] = idx
            return idx
        return self.epilogue_store_slots[group_id]

    # -- atomics (own namespace; split atomics share one slot; atomic_cas's slot is inert) --

    def _alloc_atomic(self) -> int:
        idx = self.atomic_counter
        self.atomic_counter += 1
        return idx

    def atomic_slot(self, node: torch.fx.Node | None) -> int:
        if self._id_keyed:
            return self._slot("atomic", node)
        group_id = None if node is None else node.meta.get("epilogue_subtile_group_id")
        if group_id is None:
            return self._alloc_atomic()
        if node is not None and node.meta.get("epilogue_subtile_primary_output", False):
            idx = self._alloc_atomic()
            self.epilogue_atomic_slots[group_id] = idx
            return idx
        return self.epilogue_atomic_slots[group_id]

    def atomic_cas_advance(self, node: torch.fx.Node | None) -> None:
        """``atomic_cas`` reads no indexing strategy, but under uniform membership it still gets an
        (inert) atomic slot in the map. Positionally, advance the counter so later atomics stay
        aligned; in id-mode the slot is fixed in the map and simply unread, so this is a no-op."""
        if self._id_keyed:
            return
        self.atomic_counter += 1


def _is_tile_index_load(node: torch.fx.Node) -> bool:
    """True if this load reads a ``tile.index`` value (materialized to arithmetic, no tl.load)."""
    from ..language import tile_index

    arg = node.args[0] if node.args else None
    return (
        isinstance(arg, torch.fx.Node)
        and arg.op == "call_function"
        and arg.target is tile_index
    )


class MemOpSlotMap:
    """Config-INDEPENDENT ``mem_op_id -> slot`` maps for the four namespaces.

    Built once at bind time by replaying the codegen slot logic over the original (pre-rolling)
    graphs in codegen-descent order. Slots are assigned **per distinct id, contiguously
    (first-occurrence)** — the counter bumps only on the FIRST emission of each id, so all N copies of
    a source op share one slot AND in-graph static-unroll duplicates collapse to one slot (no gaps).
    Codegen then resolves each emitted op's slot by ``node.meta['mem_op_id']`` instead of a per-config
    emission counter — making the slot transform-invariant. The per-namespace key count
    (``indexing_count`` etc.) sizes the config lists (Fork 1 collapse: distinct-op count, not the union).

    Densities mirror codegen exactly: ``eviction`` and ``cache`` are dense over all loads; ``indexing``
    and ``atomic`` cover EVERY source load/store and atomic under uniform source-based membership — an op
    whose lowering reads no strategy (tile.index / stack loads, stack stores, ``atomic_cas``) still gets
    a slot, but that slot is INERT (codegen never reads it). For kernels with no in-graph duplicate
    emissions (no static unroll), first-occurrence == the legacy per-emission numbering, so those slot
    VALUES are unchanged.
    """

    def __init__(self) -> None:
        self.indexing: dict[MemOpId, int] = {}
        self.eviction: dict[MemOpId, int] = {}
        self.cache: dict[MemOpId, int] = {}
        self.atomic: dict[MemOpId, int] = {}
        # Indexing slots of REAL-tensor stores (those that read an indexing strategy and can be
        # epilogue-subtiled). Under uniform source-based membership every store gets an indexing slot,
        # but only these need tensor_descriptor forcing (``store_indices``); a stack store's slot is
        # inert. Recorded here so the consumer set stays exact without re-deriving it from nodes.
        self.store_strategy_slots: set[int] = set()

    def indexing_slot(self, mem_op_id: MemOpId) -> int | None:
        return self.indexing.get(mem_op_id)

    def eviction_slot(self, mem_op_id: MemOpId) -> int | None:
        return self.eviction.get(mem_op_id)

    def cache_slot(self, mem_op_id: MemOpId) -> int | None:
        return self.cache.get(mem_op_id)

    def atomic_indexing_slot(self, mem_op_id: MemOpId) -> int | None:
        return self.atomic.get(mem_op_id)

    # Per-namespace DISTINCT-slot counts used to SIZE the config lists (Fork 1 collapse). With
    # first-occurrence numbering each namespace assigns contiguous slots ``0..K-1`` (one per distinct
    # id that reads that namespace), so the distinct count is simply the number of keys. These replace
    # the old union/facts-derived lengths so the autotuner searches exactly the real tunable surface.
    @property
    def indexing_count(self) -> int:
        return len(self.indexing)

    @property
    def eviction_count(self) -> int:
        return len(self.eviction)

    @property
    def cache_count(self) -> int:
        return len(self.cache)

    @property
    def atomic_count(self) -> int:
        return len(self.atomic)


def build_mem_op_slot_map(device_ir: DeviceIR) -> MemOpSlotMap:
    """Replay the codegen slot logic over the original (pre-rolling) graphs to produce the canonical
    ``mem_op_id -> slot`` maps.

    Walks in **codegen control-flow DESCENT order** (from ``root_ids``, recursing through
    ``_for_loop``/``_if``/``_while_loop`` into nested subgraphs) — NOT graph-list order — so the slot
    NUMBERS match what codegen emits for a non-rolled/non-subtiled config (e.g. an outer query loop's
    load precedes an inner K/V loop's loads). Reads the stamped ``node.meta['mem_op_id']`` (so it must
    run after the stamping pass) and uses a fresh positional broker; the counter bumps only on the
    FIRST occurrence of each id (so all N copies of a source op share one slot AND in-graph
    static-unroll duplicates collapse — contiguous ``0..K-1`` per namespace, Fork 1).
    """
    from ..language import _tracing_ops
    from ..language import atomic_ops
    from ..language import memory_ops

    load_op = memory_ops.load
    store_op = memory_ops.store
    atomic_targets = tuple(getattr(atomic_ops, name) for name in atomic_ops.__all__)

    graphs_by_id = {gi.graph_id: gi for gi in device_ir.graphs}
    broker = MemoryOpSlotBroker()
    smap = MemOpSlotMap()
    visited: set[int] = set()
    # Injectivity backstop (C6): a mem_op_id must name exactly ONE source buffer. Genuine 1:N copies
    # (rolling / unroll / split) all access the same buffer fake, so an id mapping to >1 distinct fake
    # means two distinct buffers collapsed to one id (e.g. a host_str collision) -> a silent mis-map.
    id_to_fakes: dict[MemOpId, set[int]] = {}

    def _track_injectivity(node: torch.fx.Node, mem_op_id: MemOpId) -> None:
        if (
            mem_op_id[1] is None
        ):  # synthesized (tile.index / stack) -> no host buffer to check
            return
        fake = _accessed_fake(node)
        if fake is not None:
            id_to_fakes.setdefault(mem_op_id, set()).add(id(fake))

    def walk(graph_id: int) -> None:
        if graph_id in visited or graph_id not in graphs_by_id:
            return
        visited.add(graph_id)
        for node in graphs_by_id[graph_id].graph.nodes:
            if node.op != "call_function":
                continue
            target = node.target
            if target is load_op:
                mem_op_id = node.meta["mem_op_id"]
                _track_injectivity(node, mem_op_id)
                # First-occurrence numbering: bump a namespace counter only the FIRST time an id reads
                # it, so repeat emissions (static unroll / rolling / split) share one contiguous slot.
                # eviction + cache are dense over ALL loads (codegen bumps them before the
                # tile.index / tuple early-returns).
                if mem_op_id not in smap.eviction:
                    smap.eviction[mem_op_id] = broker.load_eviction_slot(node)
                if mem_op_id not in smap.cache:
                    smap.cache[mem_op_id] = broker.load_cache_slot(node)
                # Source-based membership: EVERY source load gets an indexing slot, including tile.index
                # and stack loads. Their slot is INERT — codegen materializes the tile.index load to
                # arithmetic (early-return) and routes stack loads to StackIndexingStrategy, so neither
                # reads config.indexing[slot]. The uniform surface is predictable from the source; the
                # inert dim is made free by code-hash benchmark dedup.
                if mem_op_id not in smap.indexing:
                    smap.indexing[mem_op_id] = broker.load_indexing_slot(node)
            elif target is store_op:
                mem_op_id = node.meta["mem_op_id"]
                _track_injectivity(node, mem_op_id)
                # Uniform membership: every store gets an indexing slot (a stack store's is inert).
                if mem_op_id not in smap.indexing:
                    smap.indexing[mem_op_id] = broker.store_indexing_slot(node)
                # Only REAL-tensor stores read an indexing strategy / can be epilogue-subtiled, so only
                # they need tensor_descriptor forcing (store_indices). Stack stores route to
                # StackIndexingStrategy and ignore their (inert) slot.
                if _accessed_fake(node) is not None:
                    smap.store_strategy_slots.add(smap.indexing[mem_op_id])
            elif target in atomic_targets:
                mem_op_id = node.meta["mem_op_id"]
                _track_injectivity(node, mem_op_id)
                # Uniform source-based membership: EVERY source atomic gets an atomic_indexing slot,
                # including atomic_cas. atomic_cas's codegen always uses pointer indexing (it calls
                # atomic_cas_advance, never atomic_slot / get_atomic_indexing_strategy), so its slot is
                # INERT — codegen never reads it. This mirrors tile.index / stack loads getting an inert
                # indexing slot: membership is a pure function of the source ops, not of how they lower.
                if mem_op_id not in smap.atomic:
                    smap.atomic[mem_op_id] = broker.atomic_slot(node)
            elif (
                _tracing_ops.is_for_loop_target(target)
                and node.args
                and isinstance(node.args[0], int)
            ):
                walk(node.args[0])
            elif target is _tracing_ops._if and len(node.args) >= 3:
                _, if_g, else_g, *_rest = node.args
                if isinstance(if_g, int):
                    walk(if_g)
                if isinstance(else_g, int):
                    walk(else_g)
            elif target is _tracing_ops._while_loop and len(node.args) >= 2:
                cond_g, body_g, *_rest = node.args
                if isinstance(cond_g, int):
                    walk(cond_g)
                if isinstance(body_g, int):
                    walk(body_g)

    for root_id in device_ir.root_ids:
        walk(root_id)

    collisions = {k: n for k, n in id_to_fakes.items() if len(n) > 1}
    assert not collisions, (
        "mem_op_id injectivity violation: these ids each name >1 distinct buffer "
        f"(host_str collision -> silent slot mis-map): {sorted(map(repr, collisions))}"
    )
    return smap
