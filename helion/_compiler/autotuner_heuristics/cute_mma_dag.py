"""Seeds for the ``cute_mma_dag`` warp-MMA state recurrence owner."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
from typing import cast

from ...runtime.config import Config
from ..compile_environment import CompileEnvironment
from ..cute.cute_mma_dag import AXIS_LOOP
from ..cute.cute_mma_dag import AXIS_ROOT
from ..cute.cute_mma_dag import MMA_DAG_AUX_KEY
from ..cute.cute_mma_dag import MMA_DAG_AUX_RING
from ..cute.cute_mma_dag import MMA_DAG_CLUSTER_KEY
from ..cute.cute_mma_dag import MMA_DAG_CLUSTER_MAX
from ..cute.cute_mma_dag import MMA_DAG_DERIVED_EXCHANGE
from ..cute.cute_mma_dag import MMA_DAG_DERIVED_KEY
from ..cute.cute_mma_dag import MMA_DAG_DERIVED_PASS
from ..cute.cute_mma_dag import MMA_DAG_EPILOGUE_KEY
from ..cute.cute_mma_dag import MMA_DAG_EPILOGUE_ROW
from ..cute.cute_mma_dag import MMA_DAG_EPILOGUE_SPLIT
from ..cute.cute_mma_dag import MMA_DAG_EPILOGUE_STREAM_COLS
from ..cute.cute_mma_dag import MMA_DAG_EPILOGUE_STREAMED
from ..cute.cute_mma_dag import MMA_DAG_EXCHANGE_AUTO
from ..cute.cute_mma_dag import MMA_DAG_EXCHANGE_KEY
from ..cute.cute_mma_dag import MMA_DAG_KEY
from ..cute.cute_mma_dag import MMA_DAG_LOADS_KEY
from ..cute.cute_mma_dag import MMA_DAG_LOADS_TMA
from ..cute.cute_mma_dag import MMA_DAG_LOOKAHEAD_KEY
from ..cute.cute_mma_dag import MMA_DAG_LOOKAHEAD_ON
from ..cute.cute_mma_dag import MMA_DAG_PASS_WARP_STAGES
from ..cute.cute_mma_dag import MMA_DAG_PASS_WARPS_KEY
from ..cute.cute_mma_dag import MMA_DAG_PASS_WARPS_OFF
from ..cute.cute_mma_dag import MMA_DAG_PASS_WARPS_ON
from ..cute.cute_mma_dag import MMA_DAG_PERSISTENT_KEY
from ..cute.cute_mma_dag import MMA_DAG_SPLIT_CLUSTER
from ..cute.cute_mma_dag import MMA_DAG_SPLIT_FIXUP
from ..cute.cute_mma_dag import MMA_DAG_SPLIT_KEY
from ..cute.cute_mma_dag import MMA_DAG_STAGES_KEY
from ..cute.cute_mma_dag import MMA_DAG_STORE_DIRECT
from ..cute.cute_mma_dag import MMA_DAG_STORE_KEY
from ..cute.cute_mma_dag import MMA_DAG_STORE_STAGED
from ..cute.cute_mma_dag import MMA_DAG_TCGEN05
from ..cute.cute_mma_dag import MMA_DAG_WARP
from ..cute.cute_mma_dag import MMA_DAG_WARP_CHOICES
from ..cute.cute_mma_dag import MMA_DAG_WARPS_KEY
from ..cute.cute_mma_dag import CuteMmaDagFacts
from ..cute.cute_mma_dag import fixup_admitted
from ..cute.cute_mma_dag import mma_dag_config_reason
from ..cute.cute_mma_dag import mma_dag_cta_count
from ..cute.cute_mma_dag import mma_dag_pass_warps_lead
from ..cute.cute_mma_dag import mma_dag_pass_warps_reason
from ..cute.cute_mma_dag import mma_dag_seed_configs
from ..cute.cute_mma_dag import mma_dag_seed_variants
from ..cute.cutedsl_compat import hinted_cp_async_available
from .cute_launch_bounds import MIN_BLOCKS_KEY
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact


class CuteMmaDagHeuristic(AutotunerHeuristic):
    """Admit and seed the warp-level MMA lowering of serial state recurrences.

    ``register_facts`` runs the config-free matcher (``match_mma_dag``) on
    the device IR; where it matches, the knobs enter the search surface.
    The block-size fragments are left alone (``normalize`` turns tiles the
    mode cannot run into ``InvalidConfig`` / repairs, and the seeds emit
    legal tiles), so admission never changes the search space of the
    ordinary lowering.  The seeds walk the tile
    ladder {16, 32, 64, 128} over the tunable axes, take the most warps that
    tile every fragment within the register budget (and, ranked first, the
    most warps that feed every fragment operand of the loop from registers
    when fewer warps allow it), rank by CTA count (>= 128 first), root tile,
    shared memory for two CTAs per SM, loop tile and stages, and emit the
    top eight with their legal stage depths.  Promotion to the autotune-off
    default is limited to the GPU generations the lowering was validated on.
    """

    name = "cute_mma_dag"
    backend = "cute"
    promote_seed_to_default = True
    PROMOTE_TARGETS = (("cuda", "sm100"), ("cuda", "sm103"))
    CACHE_SPECIALIZATION_FACTS = frozenset({"input_tensor_metadata"})
    SEED_TILES = 8
    SEED_TCGEN05 = 4

    @classmethod
    def _facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> CuteMmaDagFacts | None:
        from ..cute.cute_mma_dag import match_mma_dag
        from ..cute.cute_mma_dag import mma_dag_facts

        host_function = device_ir.host_function
        if host_function is None:
            return None
        # The matcher resolves block ids and symbol origins through the
        # compile environment and the host function; the environment is not
        # current when the seeds are regenerated outside a compile.
        with contextlib.ExitStack() as stack:
            if (
                isinstance(env, CompileEnvironment)
                and not CompileEnvironment.has_current()
            ):
                stack.enter_context(env)
            stack.enter_context(host_function)
            program = match_mma_dag(device_ir.graphs)
            if program is None:
                return None
            # The facts read the bound inputs' alignment (the side tiles of
            # the loops whose cp.async ring stages a tile for them:
            # ``mma_dag_ring_loops``), and seed registration precedes the
            # runtime's storage snapshot: evaluate the registered,
            # metadata-only classifiers here and restore the prior snapshot
            # afterwards (as the tcgen05 facts hook does).  Without a
            # binding (no runtime values) no base is provably aligned and
            # the knob is pinned to the lane loads.
            saved = env.bound_runtime_input_specialization_results
            try:
                if env.runtime_arg_values_by_name:
                    env.snapshot_runtime_input_specialization_results(
                        env.runtime_arg_values_by_name
                    )
                return mma_dag_facts(program)
            finally:
                env.bound_runtime_input_specialization_results = saved

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        facts = cls._facts(env, device_ir)
        if facts is None:
            return frozenset()
        env.config_spec.enable_cute_mma_dag_search(facts)
        return cls.CACHE_SPECIALIZATION_FACTS

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return env.config_spec.cute_mma_dag is not None

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config] | None:
        spec = env.config_spec
        facts = spec.cute_mma_dag_facts
        if facts is None or spec.cute_mma_dag is None:
            return None
        from ..cute.cute_mma_dag_tcgen05 import tcgen05_corner_seeds
        from ..cute.cute_mma_dag_tcgen05 import tcgen05_lookahead_stages
        from ..cute.cute_mma_dag_tcgen05 import tcgen05_persistent_ctas
        from ..cute.cute_mma_dag_tcgen05 import tcgen05_seed_configs
        from ..cute.cute_mma_dag_tcgen05 import tcgen05_split_seed
        from ..cute.cute_mma_dag_tcgen05 import tcgen05_split_twins

        block_ids = spec.block_sizes.valid_block_ids()
        default_sizes = spec.default_config().config.get("block_sizes", [])
        assert isinstance(default_sizes, list)
        seeds: list[Config] = []
        # fp32 snapshots are 8-byte fragment pairs either way: the direct
        # pair stores beat the staging round trip (bwd_dh at s2: 255 us
        # against 263); 16-bit snapshots gain from the 16-byte packets of
        # the staging tile (fwd_h: 137 against 180).
        store = (
            MMA_DAG_STORE_DIRECT
            if any(elem_bytes == 4 for _rows, _cols, elem_bytes in facts.staging)
            else MMA_DAG_STORE_STAGED
        )
        # Tensor-memory seeds lead where the kernel admits them (the promoted
        # default is the first seed), then the warp-mode ladder.  Their
        # snapshots always go through the staging tile: the direct fp32
        # pair stores from the TMEM rows measured 397 us against 176 on
        # bwd_dh at s2.  Region programs (many small CTAs, a few iterations
        # each) lead with tensor memory only in the measured classes: wy's
        # tf32 chain (332 vs 513 us at s2 at four CTAs per SM) and the
        # streaming programs -- loops, every loop dot over loaded tiles or a
        # root fragment exchanged once ahead of the loop, no root tile --
        # whose per-CTA work is a TMA ring into TMEM accumulators and
        # vectors (dk at s2 148 vs 165 us, s1 48 vs 53, s0 21 vs 24; wy_dL
        # 301 vs 404, 87 vs 110, 38 vs 46; gated wy_dL 435 vs 474; gated
        # dqkw, whose dattn is hoisted ahead of the tile_d loop, 695 vs 730
        # at s2 and 187 vs 200 at s1 -- plain dqkw 572 vs 558 and 153 vs
        # 147, the warp seeds behind for the search).  A fragment operand
        # exchanged every iteration (a state recurrence's carry: state_du_kda
        # 278 vs 130; bwd_o_kda's per-iteration operand 127 vs 95) or a root
        # tile (fwd_o_anch 47 vs 35) keeps the warp mode first: the
        # tensor-memory chain of commit waits runs behind it there.
        # The ranked tensor-memory seeds, then (region programs) the far
        # corner of their tile / warp / stage box: see ``tcgen05_corner_seeds``.
        tcgen05_ranked = tcgen05_seed_configs(facts)
        tcgen05_top = tcgen05_ranked[: cls.SEED_TCGEN05]
        tensor_memory = [
            (MMA_DAG_TCGEN05, seed, MMA_DAG_EXCHANGE_AUTO)
            for seed in [
                *tcgen05_top,
                *tcgen05_corner_seeds(facts, tcgen05_ranked, tcgen05_top),
            ]
        ]
        ladder = mma_dag_seed_configs(facts)[: cls.SEED_TILES]
        warp = [(MMA_DAG_WARP, seed, MMA_DAG_EXCHANGE_AUTO) for seed in ladder]
        # Measured neighbours of the top warp seed (CTA-rich programs; see
        # ``mma_dag_seed_variants``) follow the ladder: the first seed stays.
        warp += [
            (MMA_DAG_WARP, (sizes, warps, stages), exchange)
            for sizes, warps, stages, exchange in mma_dag_seed_variants(facts, ladder)
        ]
        tensor_memory_first = facts.tcgen05 is not None and (
            not facts.tcgen05.regions
            or facts.tcgen05.tf32
            or cls.streaming_regions(facts)
        )
        if (
            tensor_memory_first
            and tensor_memory
            and facts.pass_chains
            and facts.tcgen05 is not None
            and bool(facts.loops)
            and facts.tcgen05.streamed_loops == len(facts.loops)
        ):
            # A streamed K-loop program with a pass chain on a grid below 128
            # CTAs at its top tensor-memory tile pays the tensor-memory setup
            # (allocation, barriers, the root TMA group) and the chain's
            # passes for a few iterations: the warp mode leads (se_block at
            # 256x128: dx warp [16, 16, 32] 2.2 us against tcgen05 4.4-5.1
            # over its tiles, dw 3.3 against 5.6).
            sizes = tensor_memory[0][1][0]
            resolved = [
                cast("int", facts.block_size(i, sizes)) for i in range(len(facts.axes))
            ]
            if mma_dag_cta_count(facts, resolved) < 128:
                tensor_memory_first = False
        ranked = [
            (mode, seed, exchange, MMA_DAG_PASS_WARPS_OFF)
            for mode, seed, exchange in (
                tensor_memory + warp if tensor_memory_first else warp + tensor_memory
            )
        ]
        # Loop operand passes off the chain: the first tensor-memory seed
        # again with warps 4-7 as pass warps and a ring of three slots at
        # least (so the next tile lands ahead of its pass).  It leads -- the
        # promoted default -- for a one-wave grid whose passes are worth
        # MMA_DAG_PASS_WARP_PACKETS per thread (gated fwd_h_delta: 338 ->
        # 270 us at s2); otherwise it follows the first seed for the
        # autotuner (gated dstate's two small passes and the CTA-rich K-loop
        # kernels lose with it).
        if tensor_memory:
            sizes, _warps, stages = tensor_memory[0][1]
            eight = max(MMA_DAG_WARP_CHOICES)
            stages = max(stages, MMA_DAG_PASS_WARP_STAGES)
            if (
                mma_dag_pass_warps_reason(facts, eight, stages, MMA_DAG_TCGEN05) is None
                and mma_dag_config_reason(facts, sizes, eight, stages, MMA_DAG_TCGEN05)
                is None
            ):
                resolved = [
                    cast("int", facts.block_size(i, sizes))
                    for i in range(len(facts.axes))
                ]
                lead = mma_dag_pass_warps_lead(facts, resolved)
                ranked.insert(
                    0 if lead else 1,
                    (
                        MMA_DAG_TCGEN05,
                        (sizes, eight, stages),
                        MMA_DAG_EXCHANGE_AUTO,
                        MMA_DAG_PASS_WARPS_ON,
                    ),
                )
        # The split epilogue (``cute_mma_dag_epilogue='split'``, a region
        # program without reductions): the largest root tile it admits at
        # eight warps, right behind the first tensor-memory seed -- the
        # 128 x 128 tile the row layout's register budget refuses (the
        # sparse-attention indexer's logits: one wave of 128 CTAs, every warp
        # over half a row).
        # Ahead of it, the first tensor-memory seed's split twins
        # (``tcgen05_split_twins``: its tile with the rows doubled and with
        # the columns doubled at eight warps -- a thread's registers of the
        # four-warp row layout over twice the tile; the indexer's 64 x 128
        # split at (512, 4096) runs at Triton's 128-column tile's time).
        epilogues: dict[int, str] = {}
        if tensor_memory and facts.tcgen05 is not None and facts.tcgen05.regions:
            top = (
                MMA_DAG_TCGEN05,
                tensor_memory[0][1],
                MMA_DAG_EXCHANGE_AUTO,
                MMA_DAG_PASS_WARPS_OFF,
            )
            split_seeds = tcgen05_split_twins(facts, tensor_memory[0][1])
            split_seed = tcgen05_split_seed(facts)
            if split_seed is not None and split_seed not in split_seeds:
                split_seeds.append(split_seed)
            at = ranked.index(top) + 1
            for seed in split_seeds:
                ranked.insert(
                    at,
                    (
                        MMA_DAG_TCGEN05,
                        seed,
                        MMA_DAG_EXCHANGE_AUTO,
                        MMA_DAG_PASS_WARPS_OFF,
                    ),
                )
                epilogues[at] = MMA_DAG_EPILOGUE_SPLIT
                at += 1
        # The lookahead schedule (``cute_mma_dag_lookahead='on'``: the loop's
        # dot one iteration ahead of its epilogue, on a doubled slot): the
        # first tensor-memory seed and the split seeds again, right behind
        # each, at two stages at least (the next tile lands under the
        # epilogue) and three where the tile runs one CTA per SM
        # (``tcgen05_lookahead_stages``: nothing else overlaps the refill).
        lookaheads: dict[int, str] = {}
        if (
            tensor_memory
            and facts.tcgen05 is not None
            and facts.tcgen05.lookahead_loop >= 0
        ):
            top = (
                MMA_DAG_TCGEN05,
                tensor_memory[0][1],
                MMA_DAG_EXCHANGE_AUTO,
                MMA_DAG_PASS_WARPS_OFF,
            )
            for at in sorted({ranked.index(top), *epilogues}, reverse=True):
                mode, (sizes, warps, stages), exchange, pass_warps = ranked[at]
                stages = tcgen05_lookahead_stages(
                    facts,
                    sizes,
                    warps,
                    stages,
                    epilogues.get(at) == MMA_DAG_EPILOGUE_SPLIT,
                )
                if (
                    mma_dag_config_reason(
                        facts,
                        sizes,
                        warps,
                        stages,
                        MMA_DAG_TCGEN05,
                        epilogue=epilogues.get(at, MMA_DAG_EPILOGUE_ROW),
                        lookahead=MMA_DAG_LOOKAHEAD_ON,
                    )
                    is not None
                ):
                    continue
                ranked.insert(
                    at + 1, (mode, (sizes, warps, stages), exchange, pass_warps)
                )
                epilogues = {
                    (key + 1 if key > at else key): value
                    for key, value in epilogues.items()
                }
                if at in epilogues:
                    epilogues[at + 1] = epilogues[at]
                lookaheads = {
                    (key + 1 if key > at else key): value
                    for key, value in lookaheads.items()
                }
                lookaheads[at + 1] = MMA_DAG_LOOKAHEAD_ON
        # The top tensor-memory seed again as a persistent launch at its
        # occupancy (the per-CTA setup paid once and no relaunch between a
        # CTA's tiles: b8 h80 mamba2 chunk state 161 -> 154 us), behind it,
        # and at one stage, whose occupancy a streaming program may want
        # over the ring (that kernel: four CTAs of one 128-deep stage per SM
        # against two of two stages).  A region program's tile loop also
        # issues the next tile's loads under this tile's epilogue (the
        # chunked linear-attention dqkg at (8, 8, 1024, 64, 64): every CTA
        # of the flat launch waited ~40 % of its span on its first TMA);
        # ``tcgen05_persistent_ctas`` is 0 where its layout does not fit.
        persistent: dict[int, int] = {}
        if tensor_memory and facts.tcgen05 is not None:
            top = (
                MMA_DAG_TCGEN05,
                tensor_memory[0][1],
                MMA_DAG_EXCHANGE_AUTO,
                MMA_DAG_PASS_WARPS_OFF,
            )
            sizes, warps, stages = top[1]
            at = ranked.index(top) + 1
            # ... and at one stage over the loop tile doubled (up to the
            # extent) where that keeps the CTAs per SM: a persistent CTA
            # walks several tiles, so each tile's loop wants the fewest
            # iterations its shared memory allows -- one full slot instead
            # of two half slots saves a barrier round and a commit wait per
            # tile (dqkg at (8, 8, 1024, 64, 64): [1, 64, 64] at one stage
            # 19.4 us against [1, 64, 32] at two stages 21.7).
            doubled = dict(sizes)
            for axis in facts.axes:
                size = sizes.get(axis.block_id)
                if axis.kind == AXIS_LOOP and type(size) is int and size < axis.extent:
                    doubled[axis.block_id] = min(size * 2, axis.extent)
            variants: list[tuple[dict[int, int], int]] = [
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
                    ranked.insert(at, (top[0], (variant, warps, depth), *top[2:]))
                    persistent[at] = ctas
                    # (The split / lookahead twins behind the top seed keep
                    # their entries.)
                    epilogues = {
                        (key + 1 if key >= at else key): value
                        for key, value in epilogues.items()
                    }
                    lookaheads = {
                        (key + 1 if key >= at else key): value
                        for key, value in lookaheads.items()
                    }
                    at += 1
        # Pass chains (a dot operand that is a pointwise chain over loaded
        # tiles): the first seed of each mode again with the chain written
        # by the derived pass (``cute_mma_dag_derived='pass'``) right behind
        # it, so the autotuner compares the two routes at the same tile.
        derived_variants: dict[int, str] = {}
        if facts.pass_chains:
            # On tensor memory the pass route leads where every loop of the
            # program streams under it (the K-loop GEMMs with a computed
            # operand: the chain's tile is a ring slot and the loop never
            # waits for the previous batch; se_block dx at 1024x512 [64, 64,
            # 64] 8w s4: pass 9.9 us, exchange 12.0; at 4096x1024 [128,
            # 128, 64] 8w s2: 67 against 72), the exchange route elsewhere.
            tc_facts = facts.tcgen05
            pass_leads = (
                tc_facts is not None
                and bool(facts.loops)
                and tc_facts.streamed_loops == len(facts.loops)
            )
            seen_modes: set[str] = set()
            at = 0
            while at < len(ranked):
                mode = ranked[at][0]
                if mode not in seen_modes:
                    seen_modes.add(mode)
                    ranked.insert(at + 1, ranked[at])
                    twin = at if pass_leads and mode == MMA_DAG_TCGEN05 else at + 1
                    derived_variants[twin] = MMA_DAG_DERIVED_PASS
                    persistent = {
                        (key + 1 if key > at else key): value
                        for key, value in persistent.items()
                    }
                    epilogues = {
                        (key + 1 if key > at else key): value
                        for key, value in epilogues.items()
                    }
                    if at in epilogues:
                        epilogues[at + 1] = epilogues[at]
                    lookaheads = {
                        (key + 1 if key > at else key): value
                        for key, value in lookaheads.items()
                    }
                    if at in lookaheads:
                        lookaheads[at + 1] = lookaheads[at]
                    at += 1
                at += 1
        # The first warp seed again with its side loads riding the ring
        # (``cute_mma_dag_aux='ring'``), right behind the seed (and its
        # derived-route twin, when it has one): a serial chunk loop's decay
        # scalar and gate row are first touched from DRAM every iteration,
        # which the per-thread loads expose (bwd_dh at (8, 8, 1024, 64, 64):
        # 19.9 -> 11.1 us on the warp tile; gated fwd_h behind the writer of
        # its gates in the mamba2 ssd pipeline).  The plain seed stays the
        # promoted default; the autotuner measures both.  No twin without a
        # subject (``aux_candidates`` is False when no side load varies with
        # a loop whose ring stages a tile for the bound inputs).
        aux_twins: set[int] = set()
        if facts.aux_candidates:
            at = next(
                (i for i, entry in enumerate(ranked) if entry[0] == MMA_DAG_WARP),
                None,
            )
            if at is not None:
                insert_at = at + 1
                while insert_at in derived_variants:
                    insert_at += 1
                ranked.insert(insert_at, ranked[at])
                aux_twins.add(insert_at)
                persistent = {
                    (key + 1 if key >= insert_at else key): value
                    for key, value in persistent.items()
                }
                derived_variants = {
                    (key + 1 if key >= insert_at else key): value
                    for key, value in derived_variants.items()
                }
                epilogues = {
                    (key + 1 if key >= insert_at else key): value
                    for key, value in epilogues.items()
                }
                lookaheads = {
                    (key + 1 if key >= insert_at else key): value
                    for key, value in lookaheads.items()
                }
        # The first warp seed again with the staged operand tiles' cp.async
        # packets hinted ``L2::evict_last`` (``load_eviction_policies``
        # "l2_last" at the staged slots, ``MMA_DAG_L2_HINTS``), right behind
        # it and its derived-route / side-load twins: the operand tiles of a
        # cold launch stay partly L2-resident across other traffic behind
        # the hint (the indexer's decode at s0, B200, campaign protocol: 2.8
        # -> 2.5 us against Triton's evict_last kernel's 2.4; s1 7.7 -> 7.5).
        # The plain seed stays the promoted default; the autotuner measures
        # both.  No twin where the hinted packets cannot
        # be verified (``hinted_cp_async_available``: the normalizer pins
        # the slots to "" there, so the twin would be the seed again).
        # (The occupancy-hint twin below follows it.)
        hinted: set[int] = set()
        eviction_length = spec.load_eviction_policies.length
        if (
            facts.staged_slots
            and eviction_length > max(facts.staged_slots)
            and hinted_cp_async_available()
        ):
            at = next(
                (i for i, entry in enumerate(ranked) if entry[0] == MMA_DAG_WARP),
                None,
            )
            if at is not None:
                insert_at = at + 1
                while insert_at in derived_variants or insert_at in aux_twins:
                    insert_at += 1
                ranked.insert(insert_at, ranked[at])
                hinted.add(insert_at)
                persistent = {
                    (key + 1 if key >= insert_at else key): value
                    for key, value in persistent.items()
                }
                derived_variants = {
                    (key + 1 if key >= insert_at else key): value
                    for key, value in derived_variants.items()
                }
                epilogues = {
                    (key + 1 if key >= insert_at else key): value
                    for key, value in epilogues.items()
                }
                lookaheads = {
                    (key + 1 if key >= insert_at else key): value
                    for key, value in lookaheads.items()
                }
                aux_twins = {
                    (key + 1 if key >= insert_at else key) for key in aux_twins
                }
        # The first warp seed again with the one-CTA occupancy hint
        # (``cute_min_blocks_per_mp`` 1), behind its other twins: ptxas
        # otherwise budgets registers for the CTAs the shared-memory
        # footprint admits, and a single-wave kernel gains nothing from that
        # occupancy (chunk_scan at s1, 256 CTAs: 4.22 -> 3.95 us with the
        # hint on the same tile / warps / stages; the search found it in one
        # of three cold draws).
        occupancy: set[int] = set()
        if spec.supports_config_key(MIN_BLOCKS_KEY):
            at = next(
                (i for i, entry in enumerate(ranked) if entry[0] == MMA_DAG_WARP),
                None,
            )
            if at is not None:
                insert_at = at + 1
                while (
                    insert_at in derived_variants
                    or insert_at in aux_twins
                    or insert_at in hinted
                ):
                    insert_at += 1
                ranked.insert(insert_at, ranked[at])
                occupancy.add(insert_at)
                persistent = {
                    (key + 1 if key >= insert_at else key): value
                    for key, value in persistent.items()
                }
                derived_variants = {
                    (key + 1 if key >= insert_at else key): value
                    for key, value in derived_variants.items()
                }
                epilogues = {
                    (key + 1 if key >= insert_at else key): value
                    for key, value in epilogues.items()
                }
                lookaheads = {
                    (key + 1 if key >= insert_at else key): value
                    for key, value in lookaheads.items()
                }
                aux_twins = {
                    (key + 1 if key >= insert_at else key) for key in aux_twins
                }
                hinted = {(key + 1 if key >= insert_at else key) for key in hinted}
        for index, (mode, (sizes, warps, stages), exchange, pass_warps) in enumerate(
            ranked
        ):
            block_sizes = list(default_sizes)
            for block_id, size in sizes.items():
                if block_id in block_ids:
                    block_sizes[spec.block_sizes.block_id_to_index(block_id)] = size
            config: dict[str, object] = {
                "block_sizes": block_sizes,
                MMA_DAG_KEY: mode,
                MMA_DAG_WARPS_KEY: warps,
                MMA_DAG_STAGES_KEY: stages,
                MMA_DAG_STORE_KEY: (
                    MMA_DAG_STORE_STAGED if mode == MMA_DAG_TCGEN05 else store
                ),
                MMA_DAG_PASS_WARPS_KEY: pass_warps,
                MMA_DAG_LOADS_KEY: MMA_DAG_LOADS_TMA,
                MMA_DAG_DERIVED_KEY: derived_variants.get(
                    index, MMA_DAG_DERIVED_EXCHANGE
                ),
            }
            if exchange != MMA_DAG_EXCHANGE_AUTO:
                config[MMA_DAG_EXCHANGE_KEY] = exchange
            if index in persistent:
                config[MMA_DAG_PERSISTENT_KEY] = persistent[index]
            if index in epilogues:
                config[MMA_DAG_EPILOGUE_KEY] = epilogues[index]
            if (
                mode == MMA_DAG_TCGEN05
                and spec.cute_mma_dag_epilogue is not None
                and MMA_DAG_EPILOGUE_STREAMED in spec.cute_mma_dag_epilogue.choices
                and cls.wide_epilogue_rows(facts, sizes)
            ):
                # A thread's row of more than two 64-column chunks spills
                # held whole across the epilogue: stream it (matmul_layernorm
                # at 2048^3, 256 columns per thread: 25.4 -> 23.9 us per
                # call; the 128-column form reads 0.3 us better whole).
                config[MMA_DAG_EPILOGUE_KEY] = MMA_DAG_EPILOGUE_STREAMED
            if index in lookaheads:
                config[MMA_DAG_LOOKAHEAD_KEY] = lookaheads[index]
            if index in aux_twins:
                config[MMA_DAG_AUX_KEY] = MMA_DAG_AUX_RING
            if index in occupancy:
                config[MIN_BLOCKS_KEY] = 1
            if index in hinted:
                policies = [""] * eviction_length
                for slot in facts.staged_slots:
                    policies[slot] = "l2_last"
                config["load_eviction_policies"] = policies
            # A seed slicing the cluster-split axis names its cluster, and
            # launches as a programmatic dependent: the region renderer's
            # prologue (barriers, tensor memory, the cluster sync) then
            # overlaps the previous kernel and its epilogue the next one's
            # prologue (matmul_layernorm, 16-call per-call us on a B200:
            # 4.10 -> 3.46 at (128, 256, 256), 7.17 -> 6.67 at (512, 1024,
            # 1024), unchanged at 2048^3).
            cluster = facts.cluster_of(sizes)
            if cluster > 1:
                config[MMA_DAG_CLUSTER_KEY] = cluster
                if mode == MMA_DAG_TCGEN05:
                    config["cute_pdl"] = True
                    # Both ways of combining the split's rows: the cluster
                    # (a portable one) first, the fixup grid behind it where
                    # the program admits it; a split wider than a cluster is
                    # the fixup grid alone.
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
        return seeds or None

    @classmethod
    def wide_epilogue_rows(
        cls, facts: CuteMmaDagFacts, sizes: Mapping[int, object]
    ) -> bool:
        """Whether some epilogue store's fragment row holds more than
        ``MMA_DAG_EPILOGUE_STREAM_COLS`` fp32 columns per thread at
        ``sizes`` (a 64-row fragment's lanes hold half rows)."""
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

    @classmethod
    def streaming_regions(cls, facts: CuteMmaDagFacts) -> bool:
        """A streaming region program: loops, no root tile axis, and every
        fragment operand of a dot inside a loop a hoisted root exchange
        (its tile written once ahead of the loop) -- the loops' work is a
        TMA ring into tensor-memory accumulators and vectors, nested or
        not.  A fragment exchanged every iteration (a state recurrence's
        carry, bwd_o_kda's per-iteration operand) or a root tile keeps the
        warp mode first."""
        return (
            bool(facts.loops)
            and all(not edge.in_loop or edge.hoisted for edge in facts.edges)
            and all(axis.kind != AXIS_ROOT for axis in facts.axes)
        )

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        seeds = cls.get_seed_configs(env, device_ir)
        return seeds[0] if seeds else None
