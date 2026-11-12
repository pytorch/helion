"""Find the tile stores that transpose a tile another access reads.

Role: analysis.

``out_t[tile_v, tile_r] = x.permute(1, 0)`` beside ``x = hl.load(src, [tile_r,
tile_v])`` coalesces only one side: ``src`` wants consecutive threads along
``tile_v``, ``out_t`` along ``tile_r``.  ``transposed_stores`` is the
condition the seeds of ``autotune/heuristics/cute_transposed_store_seeds.py``
key on, read from ``ConfigSpec.memory_op_facts``: a rank-2 store indexed by
two tiles, one on its contiguous dim, beside another rank-2 access of the same
graph indexed by the same two tiles the other way round.  A tile stored both
ways (``out[tile_r, tile_v]`` beside ``out_t``) makes each store the other's
witness; the store a load disagrees with comes first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple

if TYPE_CHECKING:
    from ...autotuner.config_spec import MemoryOpFact
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


class TransposedStore(NamedTuple):
    """A store ``transposed_stores`` found: the block of its other dim, the
    block indexing its contiguous dim, and its element size."""

    other: int
    contiguous: int
    itemsize: int


def _tile_dims(fact: MemoryOpFact) -> tuple[int, int] | None:
    """``(contiguous, other)``: the blocks of a rank-2 access indexed by two
    tiles, the first on its stride-1 dim, else ``None``."""
    blocks = fact.subscript_block_ids
    if fact.ndim != 2 or len(blocks) != 2 or blocks[0] == blocks[1]:
        return None
    first, second = blocks
    if first is None or second is None or 0 in fact.subscript_strides:
        # A broadcast dim (stride 0) reads one element for its whole tile:
        # nothing along it is laid out, let alone transposed.
        return None
    unit = [
        block
        for block, stride in zip(blocks, fact.subscript_strides, strict=True)
        if stride == 1
    ]
    if unit == [first]:
        return first, second
    if unit == [second]:
        return second, first
    return None


def transposed_stores(
    env: CompileEnvironment, device_ir: DeviceIR
) -> list[TransposedStore]:
    """The rank-2 stores indexed by two tiles that another rank-2 access of
    their graph indexes the other way round (the store's other tile on its
    contiguous dim).  Stores whose other tile a load reads contiguously come
    first, so a tile also stored untransposed (``out[tile_r, tile_v]`` beside
    ``out_t[tile_v, tile_r]``, each the other's witness) does not steer the
    seeds against the loads."""
    spec = env.config_spec
    if spec.matmul_facts or len(device_ir.phases) > 1:
        return []
    graphs: dict[int, list[tuple[MemoryOpFact, tuple[int, int] | None]]] = {}
    for fact in spec.memory_op_facts:
        graphs.setdefault(fact.graph_id, []).append((fact, _tile_dims(fact)))
    against_loads: list[TransposedStore] = []
    against_stores: list[TransposedStore] = []
    for accesses in graphs.values():
        for fact, dims in accesses:
            if fact.kind != "store" or dims is None:
                continue
            # A rank-2 access has a fake value, hence a dtype.
            assert fact.dtype is not None
            contiguous, other = dims
            witnesses = {
                candidate.kind
                for candidate, candidate_dims in accesses
                if candidate is not fact and candidate_dims == (other, contiguous)
            }
            if witnesses:
                found = TransposedStore(other, contiguous, fact.dtype.itemsize)
                (against_loads if "load" in witnesses else against_stores).append(found)
    return [*against_loads, *against_stores]
