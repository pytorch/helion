"""Shared memory of the paired paged-attention schedule's CTA.

The paired body allocates all of its shared memory from one table: the raw
K and V rings with their scales, the query and probability panels, the row
state, the pipeline's barrier block and what a persistent work partition
adds. A CTA whose table needs more than the device grants one block does not
launch, so every choice that grows the table has to be checked against the
device before a kernel is rendered with it.

This module states the table once, without the DSL's types. The device
template allocates from it (``_paged_attention_paired_pipeline``), which
makes this file a source dependency of the paged-attention wrapper, and the
config contract, the planner and the search surface replay it against the
device's capacity (``paired_shared_memory_bytes``) to say which depths of
the raw V ring a config may name. Of the compiler it imports only the page
order's constants and the CTA's thread count, from two modules that import
nothing of it, so the template can import it.
"""

from __future__ import annotations

from typing import NamedTuple

from .paged_attention_page_order import PAGE_ORDER_BUCKETS
from .paged_attention_page_order import PAGE_ORDER_CAPACITY
from .paged_attention_registers import paired_launch_threads

# One raw tile is a page of one head: 128 keys of 128 Int8 dimensions, and
# its packed scales four bytes a key.
RAW_TILE_BYTES = 16384
SCALE_TILE_BYTES = 512
# The threads of one paired CTA: all of them scan the units' lengths.
PLAN_THREADS = paired_launch_threads()
# Words of the plan a persistent CTA's loader leaves in shared memory, of a
# compute group's walk record and of what a balanced group posts to the
# spare warp (``_paged_attention_partition`` names each word); a partition
# of whole units keeps four words of the first two.
PLAN_WORDS = 12
WALK_WORDS = 16
MAIL_WORDS = 4
WHOLE_UNIT_WORDS = 4
# Words in which the sorting warp of a bucketed page order leaves the run it
# ordered: the run's first stream step and its end.
PAGE_ORDER_SPAN_WORDS = 2
# Pages of raw K the paired body stages per head, the depth of its raw K
# ring, which no option changes. The template's K roles take their slots at
# this depth, whatever the raw V ring's is, and the table's four raw K
# tiles, scale tiles and barriers of each kind are its pages of two heads.
KEY_RING_PAGES = 2

# ``cute_mma_dag_value_ring``: the pages of raw V the paired body stages per
# head, the depth of its raw V ring. The loader requests K, K, V, V of a
# page in strict sequence, so a V request that waits for its ring slot also
# holds back the next page's K requests; one more staged page per head
# decouples them. The depth changes no result, only when a raw tile is
# requested. Two, the default and the meaning of a missing key, is the depth
# the body has always had and renders the kernel without the option. The raw
# K ring keeps two pages: a third measured nothing.
# One more page per head is two more raw tiles with their scales and two
# more pairs of ring barriers, 34,048 bytes of the CTA's shared memory, so a
# config may name a depth only where the table it gives fits the device
# (``value_ring_reason``).
MMA_DAG_VALUE_RING_KEY = "cute_mma_dag_value_ring"
PAIRED_DEFAULT_VALUE_RING = 2
MMA_DAG_VALUE_RING_CHOICES: tuple[int, ...] = (PAIRED_DEFAULT_VALUE_RING, 3)

# ``cute_mma_dag_value_ring_tiles``: the raw V ring of a persistent CTA
# counted in tiles, a tile being one head's page. The loader requests a
# page's two V tiles one after the other and the V conversion takes them in
# that order, so the ring need not hold whole pages: a ring of five tiles is
# the default's four and one more, half a page, where a third page per head
# (six tiles) does not fit a persistent CTA of 40 query columns beside its
# two query stages. Tile ``2 * step + head`` of the stream takes slot
# ``tile % tiles``. Zero, the default and the meaning of a missing key, is
# the ring counted in pages (``cute_mma_dag_value_ring``) and renders the
# kernel without the option. The count changes no result, only when a raw
# tile is requested.
#
# Measured (B200, GPU 3, the SDG eval harness, 320 requests of a 2047-key
# window on the balanced partition, us per call at two calls a sample
# against the tuned config, frozen harness, 4 legs): five tiles alone
# -0.10, on top of the newest-first walk -0.68 +- 0.09 against the walk's
# -0.58 +- 0.08, no timer step either way; with the walk and the
# range-ahead prefetch -1.16 +- 0.08, one step, where walk and prefetch
# read -0.51. It gains only in that combination, which the seeds name, and
# a search has it as one of its moves. Alone it lost up to 5 us on full
# attention (+4.96 +- 0.68 at 320 requests of 211 pages), so it is offered
# to kernels with a sliding window only.
MMA_DAG_VALUE_RING_TILES_KEY = "cute_mma_dag_value_ring_tiles"
MMA_DAG_VALUE_RING_TILES_CHOICES: tuple[int, ...] = (0, 5)


class ValueRingTiles(NamedTuple):
    """A raw V ring counted in tiles, where every function of this module
    and of the device template takes the ring's depth in pages per head:
    the paired body hands it on for ``cute_mma_dag_value_ring_tiles``."""

    tiles: int


# What names the raw V ring: its pages per head, or its tiles.
ValueRing = int | ValueRingTiles


def value_ring_tile_count(value_ring: ValueRing) -> int:
    """Raw V tiles the ring stages: two per page of a depth in pages."""
    if isinstance(value_ring, ValueRingTiles):
        return value_ring.tiles
    return 2 * value_ring


def _default_value_ring(value_ring: ValueRing) -> bool:
    return (
        not isinstance(value_ring, ValueRingTiles)
        and value_ring == PAIRED_DEFAULT_VALUE_RING
    )


class SharedArray(NamedTuple):
    """One shared-memory array of the paired body: its name there, the name
    and byte width of its element type, its element count and byte
    alignment, and the optional part of the body it belongs to (``None``:
    every body allocates it)."""

    name: str
    dtype: str
    width: int
    count: int
    alignment: int
    guard: str | None = None


def plan_geometry(units: int) -> tuple[int, int]:
    """Units per scanning thread, and the warps that scan.

    Thread ``t`` scans the ``chunk`` units from ``t * chunk``, so the first
    ``warps`` warps cover all of them and up to 640 units take one each.
    """
    assert units > 0
    chunk = -(-units // PLAN_THREADS)
    return chunk, -(-units // (chunk * 32))


def pipeline_barriers(
    value_metadata: bool,
    query_stages: int = 0,
    prefetch_signal: bool = False,
    value_ring: ValueRing = PAIRED_DEFAULT_VALUE_RING,
    page_order: bool = False,
) -> tuple[tuple[str, int, int], ...]:
    """The pipeline barriers in block order, as ``(name, width, arrivals)``.

    One barrier per raw ring slot (four of K, two per page of the V ring's
    depth, or one per tile of a ring counted in tiles) or per head (two),
    each initialized to ``arrivals``; the
    value-metadata transport adds its four. A persistent partition adds one
    barrier per query stage for the copies' completion and one for the QK
    role's release of the stage: its loader requests the queries of every
    unit but the first, so the stages' barriers are among those it
    initializes. A boundary prefetch that waits for the loader's last
    request adds the barrier of that one arrival, and a bucketed page order
    the barrier at which every lane of the sorting warp arrives for the
    loader.
    """
    ring_tiles = value_ring_tile_count(value_ring)
    assert ring_tiles >= 2 * PAIRED_DEFAULT_VALUE_RING
    barriers = (
        ("raw_k_ready", 4, 1),
        ("raw_k_free", 4, 4),
        ("raw_v_ready", ring_tiles, 1),
        ("raw_v_free", ring_tiles, 128 if value_metadata else 4),
        ("key_ready", 2, 4),
        ("key_free", 2, 1),
        ("value_ready", 2, 4),
        ("value_free", 2, 1),
        ("score_free", 2, 4),
        ("score_ready", 2, 1),
        ("probability_ready", 2, 4),
        ("pv_done", 2, 1),
    )
    assert not (value_metadata and query_stages)
    # The value-metadata actor walks the V ring at its default depth.
    assert not value_metadata or _default_value_ring(value_ring)
    # The control warp that prefetches is the value-metadata actor's, and
    # the spare one of a persistent partition.
    assert not (prefetch_signal and (value_metadata or query_stages))
    # And the one that sorts a unit's pages.
    assert not (page_order and (value_metadata or query_stages))
    if value_metadata:
        barriers += (("metadata_ready", 4, 32),)
    if query_stages:
        barriers += (
            ("query_ready", query_stages, 1),
            ("query_free", query_stages, 1),
        )
    if prefetch_signal:
        barriers += (("prefetch_signal", 1, 1),)
    if page_order:
        barriers += (("page_order_ready", 1, 32),)
    return barriers


def pipeline_barrier_plan(
    value_metadata: bool,
    query_stages: int = 0,
    prefetch_signal: bool = False,
    value_ring: ValueRing = PAIRED_DEFAULT_VALUE_RING,
    page_order: bool = False,
) -> tuple[dict[str, int], int, tuple[tuple[tuple[int, int], ...], ...]]:
    """Where each pipeline barrier lies and how the block is initialized.

    Returns the first slot of every entry of ``pipeline_barriers``, the
    block's slot count and its initialization rounds. The block is padded to
    whole rounds of 32 so that every lane of the initializing warp executes
    each round's instruction and the rounds never diverge. A round holds the
    arrival counts of its 32 slots as ``(count - 1, signed lane mask)`` terms.
    """
    offsets: dict[str, int] = {}
    counts: list[int] = []
    for name, width, arrivals in pipeline_barriers(
        value_metadata, query_stages, prefetch_signal, value_ring, page_order
    ):
        offsets[name] = len(counts)
        counts += [arrivals] * width
    counts += [1] * (-len(counts) % 32)
    rounds = []
    for start in range(0, len(counts), 32):
        terms = []
        for value in sorted(set(counts[start : start + 32]) - {1}):
            mask = sum(1 << lane for lane in range(32) if counts[start + lane] == value)
            terms.append((value - 1, mask - (1 << 32) if mask >= 1 << 31 else mask))
        rounds.append(tuple(terms))
    return offsets, len(counts), tuple(rounds)


def pipeline_shared_memory(
    query_columns: int,
    value_metadata: bool,
    query_stages: int = 0,
    units: int = 0,
    balanced: bool = False,
    prefetch_signal: bool = False,
    value_ring: ValueRing = PAIRED_DEFAULT_VALUE_RING,
    page_order: bool = False,
) -> tuple[SharedArray, ...]:
    """The paired body's shared memory, one entry per array in allocation
    order.

    Four raw K tiles of 16 KiB and two raw V tiles per page of the V ring's
    depth (``value_ring``, two by default; a ``ValueRingTiles`` is the ring
    counted in tiles) with their scales, the query and
    probability panels and the per-head state and scratch rows of both
    heads, the role slot and one block for every pipeline barrier
    (``pipeline_barrier_plan``). A query map adds its barrier and the
    value-metadata transport its flags. A persistent partition of ``units``
    units and ``query_stages`` query stages (both zero with one unit per
    CTA) has a query panel per stage and the stages' barriers in the block,
    and adds each compute group's walk record, the plan tables (a word per
    scanning thread and one per warp), the plan and, where the scan keeps
    them, the scanned lengths. The balanced partition has longer walk
    records and a longer plan, and adds what each compute group posts to
    the spare warp. A boundary prefetch that waits for the loader's last
    request (``prefetch_signal``) has the barrier of that arrival in the
    block, a deeper V ring four more ring barriers and a bucketed page order
    (``page_order``) the barrier of its handoff: each takes the block past
    its 32 slots, so it is a second round of 32 longer. The bucketed order
    also adds the order of one sort, a word per page, the buckets' counters
    and lane words and the run it ordered. The body takes every array from
    here, so a new array is a new entry, and replaying the entries
    (``pipeline_shared_bytes``) gives what the body occupies.
    """
    assert (units > 0) == (query_stages > 0) and (units > 0 or not balanced)
    panel = 2 * (query_columns * 128)
    state = 2 * (4 * query_columns)
    scanning = 32 * plan_geometry(units)[1] if units else 0
    ring_tiles = value_ring_tile_count(value_ring)
    return (
        SharedArray("raw_k", "Int8", 1, 4 * RAW_TILE_BYTES, 1024),
        SharedArray("raw_v", "Int8", 1, ring_tiles * RAW_TILE_BYTES, 1024),
        SharedArray("scale_k", "Uint8", 1, 4 * SCALE_TILE_BYTES, 128),
        SharedArray("scale_v", "Uint8", 1, ring_tiles * SCALE_TILE_BYTES, 128),
        SharedArray("shared_q_raw", "BFloat16", 2, max(query_stages, 1) * panel, 1024),
        SharedArray("probability_raw", "BFloat16", 2, panel, 1024),
        SharedArray("state_raw", "Float32", 4, state, 16),
        SharedArray("scratch_raw", "Float32", 4, state, 16),
        SharedArray("slot", "Int32", 4, 1, 4),
        SharedArray(
            "barriers",
            "Int64",
            8,
            pipeline_barrier_plan(
                value_metadata, query_stages, prefetch_signal, value_ring, page_order
            )[1],
            8,
        ),
        SharedArray(
            "walk_raw",
            "Int32",
            4,
            2 * (WALK_WORDS if balanced else WHOLE_UNIT_WORDS),
            4,
            guard="persistent",
        ),
        SharedArray("mail_raw", "Int32", 4, 2 * MAIL_WORDS, 4, guard="balanced"),
        SharedArray("plan_prefix", "Int32", 4, scanning, 4, guard="persistent"),
        SharedArray("plan_totals", "Int32", 4, 32, 4, guard="persistent"),
        SharedArray(
            "plan",
            "Int32",
            4,
            PLAN_WORDS if balanced else WHOLE_UNIT_WORDS,
            4,
            guard="persistent",
        ),
        SharedArray("plan_lengths", "Int32", 4, scanning, 4, guard="scanned_lengths"),
        SharedArray("query_ready", "Int64", 8, 1, 8, guard="query_map"),
        SharedArray("metadata_flags", "Int32", 4, 4, 16, guard="value_metadata"),
        SharedArray(
            "page_order", "Int32", 4, PAGE_ORDER_CAPACITY, 4, guard="page_order"
        ),
        SharedArray(
            "page_order_buckets", "Int32", 4, PAGE_ORDER_BUCKETS, 4, guard="page_order"
        ),
        SharedArray(
            "page_order_members", "Int32", 4, PAGE_ORDER_BUCKETS, 4, guard="page_order"
        ),
        SharedArray(
            "page_order_span", "Int32", 4, PAGE_ORDER_SPAN_WORDS, 4, guard="page_order"
        ),
    )


def pipeline_parts(
    query_map: bool,
    value_metadata: bool,
    units: int = 0,
    scanned_bounds: bool = False,
    balanced: bool = False,
    page_order: bool = False,
) -> frozenset[str]:
    """The optional parts of a body, as ``SharedArray.guard`` names them.

    ``units`` is the unit count of a persistent partition, zero with one
    unit per CTA. Its query barriers lie in the barrier block, so its body
    has no query-map barrier of its own. Its plan scan keeps the lengths it
    loaded where a thread scans one unit and the prelude has the twin that
    takes a scanned length (``scanned_bounds``). ``balanced``: the
    persistent partition that splits units. ``page_order``: a one-unit
    body that streams its pages in bucketed order.
    """
    persistent = units > 0
    assert persistent or not balanced
    assert not (page_order and (persistent or value_metadata))
    return frozenset(
        name
        for name, present in (
            ("query_map", query_map and not persistent),
            ("value_metadata", value_metadata),
            ("persistent", persistent),
            ("balanced", balanced),
            (
                "scanned_lengths",
                persistent and scanned_bounds and plan_geometry(units)[0] == 1,
            ),
            ("page_order", page_order),
        )
        if present
    )


def pipeline_shared_bytes(
    arrays: tuple[SharedArray, ...], parts: frozenset[str]
) -> int:
    """Bytes of shared memory a body with the optional ``parts`` occupies:
    the DSL's allocator places each array of ``arrays`` at the next offset
    of its alignment, at least its element's width."""
    offset = 0
    for array in arrays:
        if array.guard is not None and array.guard not in parts:
            continue
        alignment = max(array.alignment, array.width)
        offset = -(-offset // alignment) * alignment + array.width * array.count
    return offset


def paired_shared_memory_bytes(
    query_columns: int,
    *,
    value_ring: ValueRing = PAIRED_DEFAULT_VALUE_RING,
    value_metadata: bool = False,
    query_map: bool = False,
    query_stages: int = 0,
    units: int = 0,
    balanced: bool = False,
    scanned_bounds: bool = False,
    prefetch_signal: bool = False,
    page_order: bool = False,
) -> int:
    """Bytes of shared memory one CTA of the paired body occupies, for the
    arguments its table and its optional parts take."""
    return pipeline_shared_bytes(
        pipeline_shared_memory(
            query_columns,
            value_metadata,
            query_stages,
            units,
            balanced,
            prefetch_signal,
            value_ring,
            page_order,
        ),
        pipeline_parts(
            query_map, value_metadata, units, scanned_bounds, balanced, page_order
        ),
    )


def value_ring_reason(
    value_ring: ValueRing,
    capacity: int,
    query_columns: int,
    *,
    query_stages: int = 0,
    units: int = 0,
    balanced: bool = False,
) -> str | None:
    """Why a paired CTA cannot stage ``value_ring`` pages of raw V per head
    on a device that grants one CTA ``capacity`` bytes of shared memory, or
    None.

    ``query_stages`` and ``units`` are those of the config's persistent
    partition, zero with one unit per CTA, and ``balanced`` says whether it
    splits units. The table is replayed with every other optional part such
    a body can have: one unit per CTA with the query map's barrier, the
    boundary prefetch's and a bucketed page order's tables, a persistent
    CTA with the scanned lengths. Whether a config's body has them depends
    on other knobs and on the kernel's prelude, they come to 1,936 bytes
    with one unit per CTA or to a word per scanning thread, and a depth
    admitted here fits whatever those turn out to be. A capacity of
    zero is an unknown one (a render without a device), which admits every
    depth as it admits the schedule.

    A ``ValueRingTiles`` is a ring counted in tiles
    (``cute_mma_dag_value_ring_tiles``), which only a persistent CTA has:
    the same replay with that many raw V tiles.
    """
    if _default_value_ring(value_ring):
        return None
    persistent = units > 0
    if isinstance(value_ring, ValueRingTiles):
        described = f"{value_ring.tiles} tiles"
        if value_ring.tiles not in MMA_DAG_VALUE_RING_TILES_CHOICES[1:]:
            return (
                f"a raw V ring of {value_ring.tiles!r} tiles is not one of "
                f"{MMA_DAG_VALUE_RING_TILES_CHOICES[1:]!r}"
            )
        if not persistent:
            return (
                f"a raw V ring of {described} belongs to a persistent partition: "
                "with one unit per CTA the ring is counted in pages"
            )
    else:
        described = f"{value_ring} pages per head"
        if value_ring not in MMA_DAG_VALUE_RING_CHOICES:
            return (
                f"a raw V ring of {value_ring!r} pages per head is not one of "
                f"{MMA_DAG_VALUE_RING_CHOICES!r}"
            )
    needed = paired_shared_memory_bytes(
        query_columns,
        value_ring=value_ring,
        query_map=not persistent,
        query_stages=query_stages,
        units=units,
        balanced=balanced,
        scanned_bounds=persistent,
        prefetch_signal=not persistent,
        page_order=not persistent,
    )
    if capacity and needed > capacity:
        stages = (
            f" with {query_stages} query stage{'s' if query_stages != 1 else ''}"
            if persistent
            else ""
        )
        return (
            f"a raw V ring of {described} needs {needed} bytes "
            f"of shared memory per CTA{stages} and the device grants {capacity}"
        )
    return None
