"""Seed planning for the ``cute_mma_dag`` search surface.

:class:`MmaDagSeedPlanner` turns the matched program's facts into the ranked
seed configs the autotuner starts from (the first seed is the promoted
default on the validated GPU generations).  Every seed is one
:class:`MmaDagSeed` record; a twin is its seed with one knob changed, so the
knobs travel with the record through the later insertions and the ranking
needs no rank-indexed bookkeeping.  The steps run in a fixed order (the tile
seeds of both modes, then the twins behind them, each step's docstring
carrying the measurement that put it there) and
:meth:`MmaDagSeedPlanner.configs` renders the records as configs.  The tile
ladders and ranks themselves live with their modes
(``mma_dag_seed_configs`` in cute_mma_dag.py, ``tcgen05_seed_configs`` in
cute_mma_dag_tcgen05.py).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import cast

from ...runtime.config import Config
from .cute_mma_dag import AXIS_LEAD
from .cute_mma_dag import AXIS_LOOP
from .cute_mma_dag import AXIS_ROOT
from .cute_mma_dag import MMA_DAG_AUX_KEY
from .cute_mma_dag import MMA_DAG_AUX_REGISTER
from .cute_mma_dag import MMA_DAG_AUX_RING
from .cute_mma_dag import MMA_DAG_CLUSTER_KEY
from .cute_mma_dag import MMA_DAG_CLUSTER_MAX
from .cute_mma_dag import MMA_DAG_DERIVED_EXCHANGE
from .cute_mma_dag import MMA_DAG_DERIVED_KEY
from .cute_mma_dag import MMA_DAG_DERIVED_PASS
from .cute_mma_dag import MMA_DAG_EARLY_TMEM_RETIREMENT_KEY
from .cute_mma_dag import MMA_DAG_EPILOGUE_KEY
from .cute_mma_dag import MMA_DAG_EPILOGUE_ROW
from .cute_mma_dag import MMA_DAG_EPILOGUE_SPLIT
from .cute_mma_dag import MMA_DAG_EPILOGUE_STREAM_COLS
from .cute_mma_dag import MMA_DAG_EPILOGUE_STREAMED
from .cute_mma_dag import MMA_DAG_EXCHANGE_AUTO
from .cute_mma_dag import MMA_DAG_EXCHANGE_KEY
from .cute_mma_dag import MMA_DAG_KEY
from .cute_mma_dag import MMA_DAG_LOADS_KEY
from .cute_mma_dag import MMA_DAG_LOADS_LANE
from .cute_mma_dag import MMA_DAG_LOADS_TMA
from .cute_mma_dag import MMA_DAG_LOOKAHEAD_KEY
from .cute_mma_dag import MMA_DAG_LOOKAHEAD_OFF
from .cute_mma_dag import MMA_DAG_LOOKAHEAD_ON
from .cute_mma_dag import MMA_DAG_PASS_WARP_STAGES
from .cute_mma_dag import MMA_DAG_PASS_WARPS_ATTENTION_PAIRED
from .cute_mma_dag import MMA_DAG_PASS_WARPS_ATTENTION_SCOPED
from .cute_mma_dag import MMA_DAG_PASS_WARPS_KEY
from .cute_mma_dag import MMA_DAG_PASS_WARPS_MMA
from .cute_mma_dag import MMA_DAG_PASS_WARPS_OFF
from .cute_mma_dag import MMA_DAG_PASS_WARPS_ON
from .cute_mma_dag import MMA_DAG_PERSISTENT_KEY
from .cute_mma_dag import MMA_DAG_PROBABILITY_SCHEDULE_KEY
from .cute_mma_dag import MMA_DAG_PROBABILITY_SCHEDULE_SERIAL
from .cute_mma_dag import MMA_DAG_QUERY_TMA_KEY
from .cute_mma_dag import MMA_DAG_SLEEP_WAITS_KEY
from .cute_mma_dag import MMA_DAG_SPLIT_CLUSTER
from .cute_mma_dag import MMA_DAG_SPLIT_FIXUP
from .cute_mma_dag import MMA_DAG_SPLIT_KEY
from .cute_mma_dag import MMA_DAG_STAGES_KEY
from .cute_mma_dag import MMA_DAG_STORE_DIRECT
from .cute_mma_dag import MMA_DAG_STORE_KEY
from .cute_mma_dag import MMA_DAG_STORE_STAGED
from .cute_mma_dag import MMA_DAG_TCGEN05
from .cute_mma_dag import MMA_DAG_V_LAG_KEY
from .cute_mma_dag import MMA_DAG_VALUE_METADATA_READ_OVERLAP_KEY
from .cute_mma_dag import MMA_DAG_VALUE_METADATA_SKIP_KEY
from .cute_mma_dag import MMA_DAG_WARP
from .cute_mma_dag import MMA_DAG_WARP_CHOICES
from .cute_mma_dag import MMA_DAG_WARPS_KEY
from .cute_mma_dag import fixup_admitted
from .cute_mma_dag import mma_dag_config_reason
from .cute_mma_dag import mma_dag_cta_count
from .cute_mma_dag import mma_dag_pass_warps_lead
from .cute_mma_dag import mma_dag_pass_warps_reason
from .cute_mma_dag import mma_dag_seed_configs
from .cute_mma_dag import mma_dag_seed_variants
from .cute_mma_dag import paged_attention_pass_warp_choices
from .cute_mma_dag import paged_attention_probability_choices
from .cute_mma_dag import paged_attention_v_lag_choices
from .cute_mma_dag_tcgen05 import tcgen05_corner_seeds
from .cute_mma_dag_tcgen05 import tcgen05_lookahead_stages
from .cute_mma_dag_tcgen05 import tcgen05_persistent_ctas
from .cute_mma_dag_tcgen05 import tcgen05_seed_configs
from .cute_mma_dag_tcgen05 import tcgen05_split_seed
from .cute_mma_dag_tcgen05 import tcgen05_split_twins
from .cutedsl_compat import hinted_cp_async_available

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from ...autotuner.config_spec import ConfigSpec
    from .cute_mma_dag import CuteMmaDagFacts

# The warp-mode tile ladder's seeds and the leading tensor-memory seeds.
SEED_TILES = 8
SEED_TCGEN05 = 4

Tile = tuple["Mapping[int, int]", int, int]
# A ranked ladder entry (``mma_dag_seed_configs``): block sizes by id, warps, stages.
LadderTile = tuple[dict[int, int], int, int]


@dataclasses.dataclass(frozen=True)
class MmaDagSeed:
    """One seed: a tile (block sizes by block id, warps, ring stages) in a
    mode, and the knobs its twins change.  ``epilogue`` / ``lookahead`` are
    None where the seed leaves the knob to the normalizer's default,
    ``persistent`` is the CTAs per SM (0: one tile per CTA)."""

    mode: str
    sizes: Mapping[int, int]
    warps: int
    stages: int
    exchange: str = MMA_DAG_EXCHANGE_AUTO
    pass_warps: str = MMA_DAG_PASS_WARPS_OFF
    derived: str = MMA_DAG_DERIVED_EXCHANGE
    epilogue: str | None = None
    lookahead: str | None = None
    persistent: int = 0
    aux: str = MMA_DAG_AUX_REGISTER
    hinted: bool = False
    occupancy: bool = False
    loads: str = MMA_DAG_LOADS_TMA

    @property
    def tile(self) -> Tile:
        return (self.sizes, self.warps, self.stages)

    def same_seed(self, other: MmaDagSeed) -> bool:
        """Whether ``other`` is this tile in this mode at the same exchange
        and pass-warps setting (the identity the twin steps look up)."""
        return (
            self.mode == other.mode
            and self.tile == other.tile
            and self.exchange == other.exchange
            and self.pass_warps == other.pass_warps
        )


def wide_epilogue_rows(facts: CuteMmaDagFacts, sizes: Mapping[int, object]) -> bool:
    """Whether some epilogue store's fragment row holds more than
    ``MMA_DAG_EPILOGUE_STREAM_COLS`` fp32 columns per thread at ``sizes`` (a
    64-row fragment's lanes hold half rows)."""
    tc = facts.tcgen05
    if tc is None:
        return False
    for slot in tc.root_store_slots:
        rows_axis, cols_axis = facts.frags[slot]
        rows = facts.block_size(rows_axis, sizes) or 0
        cols = facts.block_size(cols_axis, sizes) or 0
        local = cols if rows > 64 else cols // 2
        if local > MMA_DAG_EPILOGUE_STREAM_COLS:
            return True
    return False


def streaming_regions(facts: CuteMmaDagFacts) -> bool:
    """A streaming region program: loops, no root tile axis, and every
    fragment operand of a dot inside a loop a hoisted root exchange (its
    tile written once ahead of the loop) -- the loops' work is a TMA ring
    into tensor-memory accumulators and vectors, nested or not.  A fragment
    exchanged every iteration (a state recurrence's carry, bwd_o_kda's
    per-iteration operand) or a root tile keeps the warp mode first."""
    return (
        bool(facts.loops)
        and all(not edge.in_loop or edge.hoisted for edge in facts.edges)
        and all(axis.kind != AXIS_ROOT for axis in facts.axes)
    )


class MmaDagSeedPlanner:
    """The ranked seeds of one matched program under one config spec.

    ``occupancy_key`` is the launch-bounds knob (``cute_min_blocks_per_mp``)
    whose one-CTA twin follows the first warp seed where the spec searches
    it; the planner takes the name so it depends on no heuristic module.
    """

    def __init__(
        self, facts: CuteMmaDagFacts, spec: ConfigSpec, *, occupancy_key: str
    ) -> None:
        self.facts = facts
        self.spec = spec
        self.occupancy_key = occupancy_key
        # The warp ladder's seeds on its most-CTA tile below one wave
        # (``warp_ladder``); the lane-route twins follow them.
        self.most_cta_ladder: list[LadderTile] = []

    # ----------------------------------------------------------------- ranking

    def ranked(self) -> list[MmaDagSeed]:
        """The seeds in rank order: the tile seeds of both modes, then the
        twins each step inserts behind the seed it varies."""
        ranked, tensor_memory = self.tile_seeds()
        if tensor_memory:
            top = tensor_memory[0]
            self.pass_warps_seed(ranked, top)
            self.split_seeds(ranked, top)
            self.lookahead_twins(ranked, top)
            self.persistent_variants(ranked, top)
        self.derived_twins(ranked)
        self.mma_warp_twins(ranked)
        self.aux_twin(ranked)
        self.hinted_twin(ranked)
        self.occupancy_twin(ranked)
        self.lane_twins(ranked)
        return ranked

    def tile_seeds(self) -> tuple[list[MmaDagSeed], list[MmaDagSeed]]:
        """The tile seeds of both modes, and the tensor-memory ones.

        Tensor-memory seeds lead where the kernel admits them (the promoted
        default is the first seed), then the warp-mode ladder.  Region
        programs (many small CTAs, a few iterations each) lead with tensor
        memory only in the measured classes: wy's tf32 chain (332 vs 513 us
        at s2 at four CTAs per SM) and the streaming programs -- loops,
        every loop dot over loaded tiles or a root fragment exchanged once
        ahead of the loop, no root tile -- whose per-CTA work is a TMA ring
        into TMEM accumulators and vectors (dk at s2 148 vs 165 us, s1 48
        vs 53, s0 21 vs 24; wy_dL 301 vs 404, 87 vs 110, 38 vs 46; gated
        wy_dL 435 vs 474; gated dqkw, whose dattn is hoisted ahead of the
        tile_d loop, 695 vs 730 at s2 and 187 vs 200 at s1 -- plain dqkw
        572 vs 558 and 153 vs 147, the warp seeds behind for the search).
        A fragment operand exchanged every iteration (a state recurrence's
        carry: state_du_kda 278 vs 130; bwd_o_kda's per-iteration operand
        127 vs 95) or a root tile (fwd_o_anch 47 vs 35) keeps the warp mode
        first: the tensor-memory chain of commit waits runs behind it
        there.  The ranked tensor-memory seeds, then (region programs) the
        far corner of their tile / warp / stage box
        (``tcgen05_corner_seeds``).  Measured neighbours of the top warp
        seed (CTA-rich programs; ``mma_dag_seed_variants``) follow the warp
        ladder: the first seed stays.  A streamed K-loop program with a
        pass chain on a grid below 128 CTAs at its top tensor-memory tile
        pays the tensor-memory setup (allocation, barriers, the root TMA
        group) and the chain's passes for a few iterations: the warp mode
        leads (se_block at 256x128: dx warp [16, 16, 32] 2.2 us against
        tcgen05 4.4-5.1 over its tiles, dw 3.3 against 5.6).
        """
        facts = self.facts
        tcgen05_ranked = tcgen05_seed_configs(facts)
        tcgen05_top = tcgen05_ranked[:SEED_TCGEN05]
        tensor_memory = [
            MmaDagSeed(MMA_DAG_TCGEN05, *seed)
            for seed in [
                *tcgen05_top,
                *tcgen05_corner_seeds(facts, tcgen05_ranked, tcgen05_top),
            ]
        ]
        ladder, self.most_cta_ladder = self.warp_ladder()
        warp = [MmaDagSeed(MMA_DAG_WARP, *seed) for seed in ladder]
        warp += [
            MmaDagSeed(MMA_DAG_WARP, sizes, warps, stages, exchange)
            for sizes, warps, stages, exchange in mma_dag_seed_variants(facts, ladder)
        ]
        tensor_memory_first = facts.tcgen05 is not None and (
            not facts.tcgen05.regions or facts.tcgen05.tf32 or streaming_regions(facts)
        )
        if (
            tensor_memory_first
            and tensor_memory
            and facts.pass_chains
            and facts.tcgen05 is not None
            and bool(facts.loops)
            and facts.tcgen05.streamed_loops == len(facts.loops)
        ):
            resolved = self.resolved(tensor_memory[0].sizes)
            if mma_dag_cta_count(facts, resolved) < 128:
                tensor_memory_first = False
        ranked = tensor_memory + warp if tensor_memory_first else warp + tensor_memory
        return ranked, tensor_memory

    def warp_ladder(self) -> tuple[list[LadderTile], list[LadderTile]]:
        """The top ``SEED_TILES`` of the ranked warp ladder and, for a grid
        short of one wave (the device's SM count) at the top seed's tile, up
        to ``SEED_TILES`` more: the ladder's most-CTA tile, in ladder order.
        An idle SM costs more than a smaller tile there, but the ranked
        order (the largest tile first below one wave) is the measured
        default elsewhere, so those seeds join the list instead of
        displacing it (gdn_fwd_h at (1, 4, 256, 64, 32, 32), B200: eight
        CTAs of a 16-column state tile 2.49 us against four of 32 at 3.07;
        the [16] x 4 warps seeds ranked 10-12 of the ladder).  Returns the
        ladder and its seeds on that most-CTA tile (ranked or appended;
        empty at or beyond one wave)."""
        facts = self.facts
        ranked = mma_dag_seed_configs(facts)
        ladder = ranked[:SEED_TILES]
        if not ladder or facts.sm_count <= 0:
            return ladder, []

        def ctas(seed: LadderTile) -> int:
            return mma_dag_cta_count(facts, self.resolved(seed[0]))

        if ctas(ladder[0]) >= facts.sm_count:
            return ladder, []
        most = max(ctas(seed) for seed in ranked)
        extra = [
            seed
            for seed in ranked[SEED_TILES:]
            if ctas(seed) == most and seed not in ladder
        ][:SEED_TILES]
        ladder = ladder + extra
        return ladder, [seed for seed in ladder if ctas(seed) == most]

    def lane_twins(self, ranked: list[MmaDagSeed]) -> None:
        """A grid short of one wave at the first warp seed's tile (fewer CTAs
        than SMs): its rings never reach a steady state, so the TMA ring's
        per-tile issue and barrier round trips stay exposed and the lane
        route (per-thread packets) can lead -- gdn_fwd_h at (1, 4, 256, 64,
        32, 32), B200: [16] x 4 warps x 4 stages 2.49 us against 2.91 with
        the TMA ring pinned, while both cold autotunes of that shape picked
        a TMA-ring config.  Both routes are seeded: the first warp seed and
        every seed on the ladder's most-CTA tile (``warp_ladder``) get a
        lane-route twin behind their other twins; the other seeds keep the
        TMA ring."""
        facts = self.facts
        first_warp = next(
            (i for i, seed in enumerate(ranked) if seed.mode == MMA_DAG_WARP), None
        )
        if first_warp is None or facts.sm_count <= 0:
            return
        if mma_dag_cta_count(facts, self.resolved(ranked[first_warp].sizes)) >= (
            facts.sm_count
        ):
            return
        # The seeds themselves (the first of each ranked identity), not
        # their twins behind them.
        seen: list[MmaDagSeed] = []
        targets: list[int] = []
        for i, seed in enumerate(ranked):
            if seed.mode != MMA_DAG_WARP or any(s.same_seed(seed) for s in seen):
                continue
            seen.append(seed)
            if i == first_warp or seed.tile in self.most_cta_ladder:
                targets.append(i)
        # Behind-to-front: an insertion moves only the later ranks.
        for at in reversed(targets):
            # Behind the seed's own twins (the same identity again: its
            # derived-route, side-load, hint and occupancy twins), ahead of
            # the next seed.  A lane twin of the seed itself, not of the twin
            # ahead of it: it carries the seed's own knobs.
            insert_at = at + 1
            while insert_at < len(ranked) and ranked[insert_at].same_seed(ranked[at]):
                insert_at += 1
            ranked.insert(
                insert_at, dataclasses.replace(ranked[at], loads=MMA_DAG_LOADS_LANE)
            )

    def pass_warps_seed(self, ranked: list[MmaDagSeed], top: MmaDagSeed) -> None:
        """Loop operand passes off the chain: the first tensor-memory seed
        again with warps 4-7 as pass warps and a ring of three slots at
        least (so the next tile lands ahead of its pass).  It leads -- the
        promoted default -- for a one-wave grid whose passes are worth
        MMA_DAG_PASS_WARP_PACKETS per thread (gated fwd_h_delta: 338 -> 270
        us at s2); otherwise it follows the first seed for the autotuner
        (gated dstate's two small passes and the CTA-rich K-loop kernels
        lose with it)."""
        facts = self.facts
        eight = max(MMA_DAG_WARP_CHOICES)
        stages = max(top.stages, MMA_DAG_PASS_WARP_STAGES)
        if (
            mma_dag_pass_warps_reason(facts, eight, stages, MMA_DAG_TCGEN05) is None
            and mma_dag_config_reason(facts, top.sizes, eight, stages, MMA_DAG_TCGEN05)
            is None
        ):
            lead = mma_dag_pass_warps_lead(facts, self.resolved(top.sizes))
            ranked.insert(
                0 if lead else 1,
                MmaDagSeed(
                    MMA_DAG_TCGEN05,
                    top.sizes,
                    eight,
                    stages,
                    pass_warps=MMA_DAG_PASS_WARPS_ON,
                ),
            )

    def split_seeds(self, ranked: list[MmaDagSeed], top: MmaDagSeed) -> None:
        """The split epilogue (``cute_mma_dag_epilogue='split'``, a region
        program without reductions): the largest root tile it admits at
        eight warps, right behind the first tensor-memory seed -- the 128 x
        128 tile the row layout's register budget refuses (the
        sparse-attention indexer's logits: one wave of 128 CTAs, every warp
        over half a row).  Ahead of it, the first tensor-memory seed's
        split twins (``tcgen05_split_twins``: its tile with the rows doubled
        and with the columns doubled at eight warps -- a thread's registers
        of the four-warp row layout over twice the tile; the indexer's 64 x
        128 split at (512, 4096) runs at Triton's 128-column tile's time)."""
        facts = self.facts
        if facts.tcgen05 is None or not facts.tcgen05.regions:
            return
        split_seeds = tcgen05_split_twins(facts, top.tile)
        split_seed = tcgen05_split_seed(facts)
        if split_seed is not None and split_seed not in split_seeds:
            split_seeds.append(split_seed)
        at = self.index(ranked, top) + 1
        for sizes, warps, stages in split_seeds:
            ranked.insert(
                at,
                MmaDagSeed(
                    MMA_DAG_TCGEN05,
                    sizes,
                    warps,
                    stages,
                    epilogue=MMA_DAG_EPILOGUE_SPLIT,
                ),
            )
            at += 1

    def lookahead_twins(self, ranked: list[MmaDagSeed], top: MmaDagSeed) -> None:
        """The lookahead schedule (``cute_mma_dag_lookahead='on'``: the
        loop's dot one iteration ahead of its epilogue, on a doubled slot):
        the first tensor-memory seed and the split seeds again, right behind
        each, at two stages at least (the next tile lands under the
        epilogue) and three where the tile runs one CTA per SM
        (``tcgen05_lookahead_stages``: nothing else overlaps the refill)."""
        facts = self.facts
        if facts.tcgen05 is None or facts.tcgen05.lookahead_loop < 0:
            return
        subjects = {
            self.index(ranked, top),
            *(i for i, seed in enumerate(ranked) if seed.epilogue is not None),
        }
        for at in sorted(subjects, reverse=True):
            seed = ranked[at]
            split = seed.epilogue == MMA_DAG_EPILOGUE_SPLIT
            stages = tcgen05_lookahead_stages(
                facts, seed.sizes, seed.warps, seed.stages, split
            )
            if (
                mma_dag_config_reason(
                    facts,
                    seed.sizes,
                    seed.warps,
                    stages,
                    MMA_DAG_TCGEN05,
                    epilogue=seed.epilogue or MMA_DAG_EPILOGUE_ROW,
                    lookahead=MMA_DAG_LOOKAHEAD_ON,
                )
                is not None
            ):
                continue
            ranked.insert(
                at + 1,
                dataclasses.replace(
                    seed, stages=stages, lookahead=MMA_DAG_LOOKAHEAD_ON
                ),
            )

    def persistent_variants(self, ranked: list[MmaDagSeed], top: MmaDagSeed) -> None:
        """The top tensor-memory seed again as a persistent launch at its
        occupancy (the per-CTA setup paid once and no relaunch between a
        CTA's tiles: b8 h80 mamba2 chunk state 161 -> 154 us), behind it,
        and at one stage, whose occupancy a streaming program may want over
        the ring (that kernel: four CTAs of one 128-deep stage per SM
        against two of two stages).  A region program's tile loop also
        issues the next tile's loads under this tile's epilogue (the
        chunked linear-attention dqkg at (8, 8, 1024, 64, 64): every CTA of
        the flat launch waited ~40 % of its span on its first TMA);
        ``tcgen05_persistent_ctas`` is 0 where its layout does not fit.
        And at one stage over the loop tile doubled (up to the extent)
        where that keeps the CTAs per SM: a persistent CTA walks several
        tiles, so each tile's loop wants the fewest iterations its shared
        memory allows -- one full slot instead of two half slots saves a
        barrier round and a commit wait per tile (dqkg at (8, 8, 1024, 64,
        64): [1, 64, 64] at one stage 19.4 us against [1, 64, 32] at two
        stages 21.7).  (The split / lookahead twins behind the top seed keep
        their places.)"""
        facts = self.facts
        if facts.tcgen05 is None:
            return
        sizes, warps, stages = top.tile
        at = self.index(ranked, top) + 1
        doubled = dict(sizes)
        for axis in facts.axes:
            size = sizes.get(axis.block_id)
            if axis.kind == AXIS_LOOP and type(size) is int and size < axis.extent:
                doubled[axis.block_id] = min(size * 2, axis.extent)
        variants: list[tuple[Mapping[int, int], int]] = [
            (sizes, depth) for depth in dict.fromkeys((stages, 1))
        ]
        if doubled != sizes:
            variants.append((doubled, 1))
        top_ctas = 0
        for variant, depth in variants:
            if (
                mma_dag_config_reason(facts, variant, warps, depth, MMA_DAG_TCGEN05)
                is not None
            ):
                continue
            ctas = tcgen05_persistent_ctas(facts, variant, depth)
            if variant is sizes:
                top_ctas = max(top_ctas, ctas)
            elif ctas < top_ctas:
                continue
            if ctas:
                ranked.insert(
                    at,
                    dataclasses.replace(
                        top, sizes=variant, stages=depth, persistent=ctas
                    ),
                )
                at += 1

    def derived_twins(self, ranked: list[MmaDagSeed]) -> None:
        """Pass chains (a dot operand that is a pointwise chain over loaded
        tiles): the first seed of each mode again with the chain written by
        the derived pass (``cute_mma_dag_derived='pass'``) right behind it,
        so the autotuner compares the two routes at the same tile.  On
        tensor memory the pass route leads where every loop of the program
        streams under it (the K-loop GEMMs with a computed operand: the
        chain's tile is a ring slot and the loop never waits for the
        previous batch; se_block dx at 1024x512 [64, 64, 64] 8w s4: pass
        9.9 us, exchange 12.0; at 4096x1024 [128, 128, 64] 8w s2: 67
        against 72), the exchange route elsewhere."""
        facts = self.facts
        if not facts.pass_chains:
            return
        tc_facts = facts.tcgen05
        pass_leads = (
            tc_facts is not None
            and bool(facts.loops)
            and tc_facts.streamed_loops == len(facts.loops)
        )
        seen_modes: set[str] = set()
        at = 0
        while at < len(ranked):
            seed = ranked[at]
            if seed.mode not in seen_modes:
                seen_modes.add(seed.mode)
                twin = dataclasses.replace(seed, derived=MMA_DAG_DERIVED_PASS)
                if pass_leads and seed.mode == MMA_DAG_TCGEN05:
                    ranked[at] = twin
                    ranked.insert(at + 1, seed)
                else:
                    ranked.insert(at + 1, twin)
                at += 1
            at += 1

    def mma_warp_twins(self, ranked: list[MmaDagSeed]) -> None:
        """The MMA warp (``cute_mma_dag_pass_warps='mma'``): the first
        tensor-memory seed -- and its pass-route twin -- again with the
        streamed K loop warp-specialized (the config's warps run the
        operand passes, one more warp issues the batches and refills; no
        CTA barrier in the loop), right behind each, where the loop's
        passes give the warp a subject at that route (se_block backward at
        4096x1024 on a B200: dx 22.1 -> 21.1 us, dw 39.9 -> 35.9 against
        the exchange-route winners, the cell 69.6 -> 63.4 us graph).  The
        plain seed stays the promoted default; the autotuner measures
        both."""
        facts = self.facts
        if facts.tcgen05 is None or not facts.tcgen05.streamed:
            return
        top_at = next(
            (i for i, seed in enumerate(ranked) if seed.mode == MMA_DAG_TCGEN05),
            None,
        )
        if top_at is None:
            return
        top = ranked[top_at]
        twins = [
            at
            for at in (top_at, top_at + 1)
            if at < len(ranked)
            and ranked[at].mode == MMA_DAG_TCGEN05
            and (
                at == top_at
                or ranked[at].derived == MMA_DAG_DERIVED_PASS
                or top.derived == MMA_DAG_DERIVED_PASS
            )
            and ranked[at].tile == top.tile
        ]
        for at in sorted(twins, reverse=True):
            seed = ranked[at]
            if (
                mma_dag_pass_warps_reason(
                    facts,
                    seed.warps,
                    seed.stages,
                    MMA_DAG_TCGEN05,
                    MMA_DAG_PASS_WARPS_MMA,
                    seed.derived,
                )
                is not None
                or mma_dag_config_reason(
                    facts,
                    seed.sizes,
                    seed.warps,
                    seed.stages,
                    MMA_DAG_TCGEN05,
                    seed.derived,
                    epilogue=seed.epilogue or MMA_DAG_EPILOGUE_ROW,
                    lookahead=seed.lookahead or MMA_DAG_LOOKAHEAD_OFF,
                    persistent=seed.persistent > 0,
                    pass_warps=MMA_DAG_PASS_WARPS_MMA,
                )
                is not None
            ):
                continue
            ranked.insert(
                at + 1, dataclasses.replace(seed, pass_warps=MMA_DAG_PASS_WARPS_MMA)
            )

    def aux_twin(self, ranked: list[MmaDagSeed]) -> None:
        """The first warp seed again with its side loads riding the ring
        (``cute_mma_dag_aux='ring'``), right behind the seed (and its
        derived-route twin, when it has one): a serial chunk loop's decay
        scalar and gate row are first touched from DRAM every iteration,
        which the per-thread loads expose (bwd_dh at (8, 8, 1024, 64, 64):
        19.9 -> 11.1 us on the warp tile; gated fwd_h behind the writer of
        its gates in the mamba2 ssd pipeline).  The plain seed stays the
        promoted default; the autotuner measures both.  No twin without a
        subject (``aux_candidates`` is False when no side load varies with
        a loop whose ring stages a tile for the bound inputs)."""
        if not self.facts.aux_candidates:
            return
        self.insert_warp_twin(
            ranked,
            lambda seed: seed.derived == MMA_DAG_DERIVED_PASS,
            lambda seed: dataclasses.replace(seed, aux=MMA_DAG_AUX_RING),
        )

    def hinted_twin(self, ranked: list[MmaDagSeed]) -> None:
        """The first warp seed again with the staged operand tiles' cp.async
        packets hinted ``L2::evict_last`` (``load_eviction_policies``
        "l2_last" at the staged slots, ``MMA_DAG_L2_HINTS``), right behind
        it and its derived-route / side-load twins: the operand tiles of a
        cold launch stay partly L2-resident across other traffic behind the
        hint (the indexer's decode at s0, B200, campaign protocol: 2.8 ->
        2.5 us against Triton's evict_last kernel's 2.4; s1 7.7 -> 7.5).
        The plain seed stays the promoted default; the autotuner measures
        both.  No twin where the hinted packets cannot be verified
        (``hinted_cp_async_available``: the normalizer pins the slots to ""
        there, so the twin would be the seed again)."""
        facts = self.facts
        if not (
            facts.staged_slots
            and self.spec.load_eviction_policies.length > max(facts.staged_slots)
            and hinted_cp_async_available()
        ):
            return
        self.insert_warp_twin(
            ranked,
            lambda seed: (
                seed.derived == MMA_DAG_DERIVED_PASS or seed.aux == MMA_DAG_AUX_RING
            ),
            lambda seed: dataclasses.replace(seed, hinted=True),
        )

    def occupancy_twin(self, ranked: list[MmaDagSeed]) -> None:
        """The first warp seed again with the one-CTA occupancy hint
        (``cute_min_blocks_per_mp`` 1), behind its other twins: ptxas
        otherwise budgets registers for the CTAs the shared-memory
        footprint admits, and a single-wave kernel gains nothing from that
        occupancy (chunk_scan at s1, 256 CTAs: 4.22 -> 3.95 us with the
        hint on the same tile / warps / stages; the search found it in one
        of three cold draws)."""
        if not self.spec.supports_config_key(self.occupancy_key):
            return
        self.insert_warp_twin(
            ranked,
            lambda seed: (
                seed.derived == MMA_DAG_DERIVED_PASS
                or seed.aux == MMA_DAG_AUX_RING
                or seed.hinted
            ),
            lambda seed: dataclasses.replace(seed, occupancy=True),
        )

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def index(ranked: list[MmaDagSeed], seed: MmaDagSeed) -> int:
        """The rank of the first entry that is ``seed`` (same mode, tile,
        exchange and pass warps; the twins' other knobs do not count)."""
        return next(i for i, entry in enumerate(ranked) if entry.same_seed(seed))

    @staticmethod
    def insert_warp_twin(
        ranked: list[MmaDagSeed],
        behind: Callable[[MmaDagSeed], bool],
        twin: Callable[[MmaDagSeed], MmaDagSeed],
    ) -> None:
        """Insert ``twin`` of the first warp seed behind it and behind the
        twins of it that ``behind`` recognizes."""
        at = next(
            (i for i, seed in enumerate(ranked) if seed.mode == MMA_DAG_WARP), None
        )
        if at is None:
            return
        insert_at = at + 1
        while insert_at < len(ranked) and behind(ranked[insert_at]):
            insert_at += 1
        ranked.insert(insert_at, twin(ranked[at]))

    def resolved(self, sizes: Mapping[int, int]) -> list[int]:
        facts = self.facts
        return [cast("int", facts.block_size(i, sizes)) for i in range(len(facts.axes))]

    # ----------------------------------------------------------------- configs

    def configs(self, ranked: list[MmaDagSeed]) -> list[Config]:
        """The ranked seeds as configs over the spec's default block sizes.

        fp32 snapshots are 8-byte fragment pairs either way: the direct pair
        stores beat the staging round trip (bwd_dh at s2: 255 us against
        263); 16-bit snapshots gain from the 16-byte packets of the staging
        tile (fwd_h: 137 against 180).  Tensor-memory snapshots always go
        through the staging tile: the direct fp32 pair stores from the TMEM
        rows measured 397 us against 176 on bwd_dh at s2.  A thread's row of
        more than two 64-column chunks spills held whole across the
        epilogue: stream it (matmul_layernorm at 2048^3, 256 columns per
        thread: 25.4 -> 23.9 us per call; the 128-column form reads 0.3 us
        better whole).  A seed slicing the cluster-split axis names its
        cluster, and launches as a programmatic dependent: the region
        renderer's prologue (barriers, tensor memory, the cluster sync)
        then overlaps the previous kernel and its epilogue the next one's
        prologue (matmul_layernorm, 16-call per-call us on a B200: 4.10 ->
        3.46 at (128, 256, 256), 7.17 -> 6.67 at (512, 1024, 1024),
        unchanged at 2048^3); both ways of combining the split's rows are
        seeded: the cluster (a portable one) first, the fixup grid behind it
        where the program admits it; a split wider than a cluster is the
        fixup grid alone.
        """
        facts, spec = self.facts, self.spec
        block_ids = spec.block_sizes.valid_block_ids()
        default_sizes = spec.default_config().config.get("block_sizes", [])
        assert isinstance(default_sizes, list)
        store = (
            MMA_DAG_STORE_DIRECT
            if any(elem_bytes == 4 for _rows, _cols, elem_bytes in facts.staging)
            else MMA_DAG_STORE_STAGED
        )
        eviction_length = spec.load_eviction_policies.length
        streamed_epilogue = (
            spec.cute_mma_dag_epilogue is not None
            and MMA_DAG_EPILOGUE_STREAMED in spec.cute_mma_dag_epilogue.choices
        )
        seeds: list[Config] = []
        sleeping: list[Config] = []
        early_retirement: list[Config] = []
        query_tma: list[Config] = []
        value_metadata: list[Config] = []
        value_metadata_policies: list[Config] = []
        value_metadata_probability: list[Config] = []
        probability_seeds: list[Config] = []
        for seed in ranked:
            block_sizes = list(default_sizes)
            for block_id, size in seed.sizes.items():
                if block_id in block_ids:
                    block_sizes[spec.block_sizes.block_id_to_index(block_id)] = size
            config: dict[str, object] = {
                "block_sizes": block_sizes,
                MMA_DAG_KEY: seed.mode,
                MMA_DAG_WARPS_KEY: seed.warps,
                MMA_DAG_STAGES_KEY: seed.stages,
                MMA_DAG_STORE_KEY: (
                    MMA_DAG_STORE_STAGED if seed.mode == MMA_DAG_TCGEN05 else store
                ),
                MMA_DAG_PASS_WARPS_KEY: seed.pass_warps,
                MMA_DAG_LOADS_KEY: seed.loads,
                MMA_DAG_DERIVED_KEY: seed.derived,
            }
            if seed.exchange != MMA_DAG_EXCHANGE_AUTO:
                config[MMA_DAG_EXCHANGE_KEY] = seed.exchange
            if seed.persistent:
                config[MMA_DAG_PERSISTENT_KEY] = seed.persistent
            if seed.epilogue is not None:
                config[MMA_DAG_EPILOGUE_KEY] = seed.epilogue
            if (
                seed.mode == MMA_DAG_TCGEN05
                and streamed_epilogue
                and wide_epilogue_rows(facts, seed.sizes)
            ):
                config[MMA_DAG_EPILOGUE_KEY] = MMA_DAG_EPILOGUE_STREAMED
            if seed.lookahead is not None:
                config[MMA_DAG_LOOKAHEAD_KEY] = seed.lookahead
            if seed.aux != MMA_DAG_AUX_REGISTER:
                config[MMA_DAG_AUX_KEY] = seed.aux
            if seed.occupancy:
                config[self.occupancy_key] = 1
            if seed.hinted:
                policies = [""] * eviction_length
                for slot in facts.staged_slots:
                    policies[slot] = "l2_last"
                config["load_eviction_policies"] = policies
            cluster = facts.cluster_of(seed.sizes)
            if cluster > 1:
                config[MMA_DAG_CLUSTER_KEY] = cluster
                if seed.mode == MMA_DAG_TCGEN05:
                    config["cute_pdl"] = True
                    if cluster <= MMA_DAG_CLUSTER_MAX:
                        seeds.append(
                            Config.from_dict(
                                {**config, MMA_DAG_SPLIT_KEY: MMA_DAG_SPLIT_CLUSTER}
                            )
                        )
                    if fixup_admitted(facts):
                        seeds.append(
                            Config.from_dict(
                                {**config, MMA_DAG_SPLIT_KEY: MMA_DAG_SPLIT_FIXUP}
                            )
                        )
                    continue
            seeds.append(Config.from_dict(config))
        attention = facts.attention
        if attention is not None and spec.supports_config_key(self.occupancy_key):
            from .paged_attention_config import paged_attention_config_reason
            from .paged_attention_config import paged_attention_required_knobs

            # A measured default is not implied by semantic recognition.
            # Append the explicit topology after the ordinary seed ranking.
            sizes = {
                axis.block_id: 1 if axis.kind == AXIS_LEAD else axis.extent
                for axis in facts.axes
            }
            for axis, extent in (
                (attention.query_axis, attention.query_block),
                (attention.key_axis, attention.key_width),
                (attention.dimension_axis, attention.dimension),
            ):
                sizes[facts.axes[axis].block_id] = extent
            block_sizes = list(default_sizes)
            for block_id in block_ids:
                if block_id in sizes:
                    block_sizes[spec.block_sizes.block_id_to_index(block_id)] = sizes[
                        block_id
                    ]
            # The four TMA operands can tune independently. Seed both uniform
            # L2 priorities alongside the unchanged default; no shape selects one.
            priorities = (
                ("", "streaming", "l2_last")
                if any(slot is not None for slot in attention.tma_load_slots)
                else ("",)
            )

            def policy_bundle(policy: str) -> list[str]:
                policies = [""] * eviction_length
                for slot in attention.tma_load_slots:
                    if slot is not None:
                        # The heuristic scatters phase-local seed policies
                        # into the bundle once.
                        policies[slot] = policy
                return policies

            for pass_warps in paged_attention_pass_warp_choices(facts):
                probability_choices = tuple(
                    choice
                    for choice in paged_attention_probability_choices(facts, pass_warps)
                    if choice != MMA_DAG_PROBABILITY_SCHEDULE_SERIAL
                )
                occupancies = attention.min_blocks_per_mp_choices(
                    scoped_registers=pass_warps == MMA_DAG_PASS_WARPS_ATTENTION_SCOPED,
                    paired_items=pass_warps == MMA_DAG_PASS_WARPS_ATTENTION_PAIRED,
                )
                config = {
                    **paged_attention_required_knobs(facts),
                    "block_sizes": block_sizes,
                    MMA_DAG_PASS_WARPS_KEY: pass_warps,
                    self.occupancy_key: occupancies[0],
                    "cute_pdl": False,
                    "pid_type": "flat",
                    "load_eviction_policies": [""] * eviction_length,
                }
                if (
                    paged_attention_config_reason(facts, sizes, config) is None
                    and mma_dag_config_reason(
                        facts,
                        sizes,
                        attention.compute_warps,
                        attention.raw_stages,
                        MMA_DAG_TCGEN05,
                        pass_warps=pass_warps,
                    )
                    is None
                ):
                    for occupancy in occupancies:
                        for v_lag in paged_attention_v_lag_choices(facts, pass_warps):
                            seed = {**config, self.occupancy_key: occupancy}
                            if pass_warps == MMA_DAG_PASS_WARPS_ATTENTION_PAIRED:
                                seed[MMA_DAG_V_LAG_KEY] = v_lag
                            for policy in priorities:
                                seed_config = {
                                    **seed,
                                    "load_eviction_policies": policy_bundle(policy),
                                }
                                seeds.append(Config.from_dict(seed_config))
                                if (
                                    pass_warps == MMA_DAG_PASS_WARPS_ATTENTION_PAIRED
                                    and attention.value_metadata_skip_available
                                ):
                                    metadata_seeds = (
                                        value_metadata
                                        if policy == ""
                                        else value_metadata_policies
                                    )
                                    metadata_seeds.append(
                                        Config.from_dict(
                                            {
                                                **seed_config,
                                                MMA_DAG_VALUE_METADATA_SKIP_KEY: True,
                                            }
                                        )
                                    )
                                if (
                                    pass_warps == MMA_DAG_PASS_WARPS_ATTENTION_PAIRED
                                    and policy == ""
                                ):
                                    if attention.query_tma_available:
                                        query_tma.append(
                                            Config.from_dict(
                                                {
                                                    **seed_config,
                                                    MMA_DAG_QUERY_TMA_KEY: True,
                                                }
                                            )
                                        )
                                    early_retirement.append(
                                        Config.from_dict(
                                            {
                                                **seed_config,
                                                MMA_DAG_EARLY_TMEM_RETIREMENT_KEY: True,
                                            }
                                        )
                                    )
                                    sleeping.append(
                                        Config.from_dict(
                                            {
                                                **seed_config,
                                                MMA_DAG_SLEEP_WAITS_KEY: True,
                                            }
                                        )
                                    )
                                    for probability_schedule in probability_choices:
                                        probability_config = {
                                            **seed_config,
                                            MMA_DAG_PROBABILITY_SCHEDULE_KEY: probability_schedule,
                                        }
                                        if attention.query_tma_available:
                                            probability_config[
                                                MMA_DAG_QUERY_TMA_KEY
                                            ] = True
                                        probability_seeds.append(
                                            Config.from_dict(probability_config)
                                        )
            if attention.value_metadata_skip_available:
                for probability_seed in probability_seeds:
                    for policy in priorities:
                        value_metadata_probability.append(
                            Config.from_dict(
                                {
                                    **probability_seed.config,
                                    MMA_DAG_VALUE_METADATA_SKIP_KEY: True,
                                    "load_eviction_policies": policy_bundle(policy),
                                }
                            )
                        )
        # Keep existing seed order and cache-policy coverage. One sleeping
        # seed per paired V lag explores the scheduling choice without making
        # every topology/cache seed compile twice.
        seeds.extend(sleeping)
        # Probe query transport once per paired V lag, after the old prefix.
        seeds.extend(query_tma)
        # One seed per P schedule and V lag, bundled with available query TMA.
        # Capture overlap requires the caller's immutable-graph opt-in, so no
        # compiler seed enables it. Keep the topology/cache prefix independent.
        seeds.extend(probability_seeds)
        # Keep the original metadata/default-policy prefix. Seed metadata with
        # the existing TMA policy ladder too: a default-policy metadata seed can
        # lose its search path before a beneficial cache-policy neighbor is tried.
        # Keep these serial anchors intact; joint P anchors follow below.
        seeds.extend(value_metadata)
        seeds.extend(value_metadata_policies)
        # Preserve the prefix and offer both shared-slot wait placements for
        # the existing metadata policy/lag anchors, without a schedule product.
        seeds.extend(
            Config.from_dict(
                {**seed.config, MMA_DAG_VALUE_METADATA_READ_OVERLAP_KEY: True}
            )
            for seed in (*value_metadata, *value_metadata_policies)
        )
        # A profitable early-P schedule can require both metadata transport and
        # a different TMA cache policy. Seed that combination directly instead
        # of relying on two independently unprofitable neighbor moves. Preserve
        # the old prefix and leave the other scheduling knobs uncrossed.
        seeds.extend(value_metadata_probability)
        # Keep every old seed in order. One ordinary paired counterpart per
        # V lag exposes retirement independently of PDL and cache policy.
        seeds.extend(early_retirement)
        return seeds
