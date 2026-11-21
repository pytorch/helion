"""Check the raw V ring counted in tiles of a persistent partition of the
paired attention schedule (``cute_mma_dag_value_ring_tiles``): the slots
its tiles take, the shared-memory replay that admits it, the config
contract, what it renders and compiles to, and how a search steps onto it
and an override names it."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import helion
from helion import exc
from helion._compiler.cute import _paged_attention_paired_pipeline as template
from helion._compiler.cute.cute_mma_dag import paged_attention_value_ring_tiles_choices
from helion._compiler.cute.paged_attention_config import (
    paged_attention_value_ring_tiles_reason,
)
from helion._compiler.cute.paged_attention_shared_memory import (
    MMA_DAG_VALUE_RING_TILES_CHOICES,
)
from helion._compiler.cute.paged_attention_shared_memory import RAW_TILE_BYTES
from helion._compiler.cute.paged_attention_shared_memory import SCALE_TILE_BYTES
from helion._compiler.cute.paged_attention_shared_memory import ValueRingTiles
from helion._compiler.cute.paged_attention_shared_memory import (
    paired_shared_memory_bytes,
)
from helion._compiler.cute.paged_attention_shared_memory import pipeline_barrier_plan
from helion._compiler.cute.paged_attention_shared_memory import value_ring_reason
from helion._testing import cpu_render_target
from helion._testing.cute_offline import compile_offline
from helion._testing.cute_paged_partition import PAGE
from helion._testing.cute_paged_partition import PAGES
from helion._testing.cute_paged_partition import partition_attention_args
from helion._testing.cute_paged_stream import NO_WINDOW
from helion._testing.cute_paged_stream import PAIRED
from helion._testing.cute_paged_stream import SMS
from helion._testing.cute_paged_stream import body_keywords
from helion._testing.cute_paged_stream import seed_transfers
from helion._testing.cute_paged_stream import stream_bound
from helion._testing.cute_paged_stream import stream_config
from helion._testing.cute_paged_stream import stream_render
from helion._testing.cute_paged_stream import stream_step
from helion._testing.cute_paged_stream import transferred_seeds

pytestmark = pytest.mark.cpu_only
TILES = "cute_mma_dag_value_ring_tiles"
RING = "cute_mma_dag_value_ring"
PAGES_KEY = "cute_mma_dag_boundary_prefetch"
PARTITION = "cute_mma_dag_partition"
ORDER = "cute_mma_dag_page_order"
LEAD = "cute_mma_dag_clipped_page_lead"
SKEW = "cute_mma_dag_partition_skew"
QUERY_TMA = "cute_mma_dag_query_tma"
TEMPLATE = Path(inspect.getfile(template))
# Shared memory one CTA of the modelled B200 may request.
CAPACITY = 232448
# Units on 148 SMs: one wave with idle SMs, and more than two waves, where
# the balanced partition is seeded and the reversed walk runs on it.
ONE_WAVE = 128
WAVES = 300
BALANCED = {PARTITION: "balanced", QUERY_TMA: True}
# The walk a search pairs the ring with.
WALK = {**BALANCED, ORDER: "reversed", LEAD: 8}


def _function(name: str) -> ast.FunctionDef:
    (found,) = [
        node
        for node in ast.walk(ast.parse(TEMPLATE.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    return found


def _paired(moves):
    return [
        (seed, moved)
        for seed, moved in moves
        if seed.config["cute_mma_dag_pass_warps"] == PAIRED
    ]


def test_a_ring_counted_in_tiles_takes_its_slots_in_tile_order():
    """Tile ``2 * step + head`` of the stream takes slot ``tile % tiles``,
    and a slot's barriers alternate phase from use to use, as a ring
    counted in pages has it for whole pages. For an even count the two are
    the same ring."""
    slot, phase = template._ring_slot, template._ring_phase
    for tiles in (5, 7):
        ring = ValueRingTiles(tiles)
        uses: dict[int, int] = {}
        order = []
        for step in range(6 * tiles):
            for head in (0, 1):
                taken = int(slot(step, head, ring))
                assert taken == (2 * step + head) % tiles
                assert int(phase(step, ring, head)) == uses.get(taken, 0) % 2
                uses[taken] = uses.get(taken, 0) + 1
                order.append(taken)
        assert sorted(uses) == list(range(tiles))
        assert len(set(uses.values())) == 1
        # Consecutive tiles take consecutive slots: a tile is requested
        # into the slot the conversion freed longest ago.
        assert order[: 2 * tiles] == list(range(tiles)) * 2
    for depth in (2, 3):
        for step in range(4 * depth):
            for head in (0, 1):
                tiled = ValueRingTiles(2 * depth)
                assert int(slot(step, head, tiled)) == slot(step, head, depth)
                assert int(phase(step, tiled, head)) == phase(step, depth)
    # A page's two heads share the phase of a ring in pages, which reads no
    # head. Those of a ring counted in tiles differ (at step 2 of five
    # tiles: tile 4 closes the ring's first round and tile 5 opens its
    # second), so that ring's phase has no head to fall back on.
    five = ValueRingTiles(5)
    assert (int(phase(2, five, 0)), int(phase(2, five, 1))) == (0, 1)
    with pytest.raises(AssertionError, match="its head's"):
        phase(2, five)
    assert phase(2, 3) == phase(2, 3, 0) == phase(2, 3, 1) == 0
    # The body numbers a tile in Int32 and admits the ring only where that
    # holds twice the pages a CTA streams.
    assert "cutlass.Int32(step) * 2 + cutlass.Int32(head)" in ast.unparse(
        _function("_ring_tile")
    )
    body = ast.unparse(_function("paged_attention_paired_body"))
    for check in (
        "cutlass.const_expr(value_ring_tiles > 2 * PAIRED_DEFAULT_VALUE_RING)",
        "cutlass.const_expr(value_ring == PAIRED_DEFAULT_VALUE_RING)",
        "cutlass.const_expr(partition != 'item' and window is not None)",
        "cutlass.const_expr(0 < stream_page_bound < 2 ** 30)",
    ):
        assert f"assert {check}" in body, check


def test_the_replay_takes_the_tile_count_and_admits_five_tiles_beside_two_stages():
    """Five tiles are the default's four and one more, with its scales and
    one pair of ring barriers inside the block's second round. On a B200 a
    persistent CTA holds them at every panel width with two query stages;
    a third page per head, six tiles, does not fit the widest."""
    assert MMA_DAG_VALUE_RING_TILES_CHOICES == (0, 5)
    for columns in (8, 32, 40):
        for units in (WAVES, 608, 640):
            for balanced in (False, True):
                shape = {
                    "query_stages": 2,
                    "units": units,
                    "balanced": balanced,
                    "scanned_bounds": True,
                }
                default = paired_shared_memory_bytes(columns, **shape)
                five = paired_shared_memory_bytes(
                    columns, value_ring=ValueRingTiles(5), **shape
                )
                assert five - default == RAW_TILE_BYTES + 1024
                assert paired_shared_memory_bytes(
                    columns, value_ring=ValueRingTiles(4), **shape
                ) == (default)
                assert paired_shared_memory_bytes(
                    columns, value_ring=ValueRingTiles(6), **shape
                ) == paired_shared_memory_bytes(columns, value_ring=3, **shape)
    # (One more tile is 16,384 bytes of raw V and 512 of scales; the raw
    # tiles are aligned to 1,024 bytes, which the scales' odd tile pads.)
    assert RAW_TILE_BYTES + SCALE_TILE_BYTES == 16896
    widest = {"query_stages": 2, "units": 640, "balanced": True, "scanned_bounds": True}
    assert paired_shared_memory_bytes(40, **widest) == 205144
    assert (
        paired_shared_memory_bytes(40, value_ring=ValueRingTiles(5), **widest) == 222552
    )
    assert paired_shared_memory_bytes(40, value_ring=3, **widest) == 238936
    shape = {"query_stages": 2, "units": 640, "balanced": True}
    assert value_ring_reason(ValueRingTiles(5), CAPACITY, 40, **shape) is None
    reason = value_ring_reason(ValueRingTiles(5), 222551, 40, **shape)
    assert reason is not None and "5 tiles needs 222552 bytes" in reason
    assert "with 2 query stages" in reason
    # Not a count the option has, and not a ring of one unit per CTA.
    for tiles in (4, 6, 7):
        reason = value_ring_reason(ValueRingTiles(tiles), CAPACITY, 8, **shape)
        assert reason is not None and "is not one of (5,)" in reason
    reason = value_ring_reason(ValueRingTiles(5), CAPACITY, 8)
    assert reason is not None and "persistent partition" in reason
    # An unknown capacity admits the count, as it admits the schedule.
    assert value_ring_reason(ValueRingTiles(5), 0, 40, **shape) is None
    # Ten ring barriers in the place of eight, inside the same block.
    offsets, slots, _rounds = pipeline_barrier_plan(False, 2)
    tiled, tiled_slots, _rounds = pipeline_barrier_plan(
        False, 2, False, ValueRingTiles(5)
    )
    assert slots == tiled_slots == 64
    assert tiled["raw_v_free"] - tiled["raw_v_ready"] == 5
    assert offsets["raw_v_free"] - offsets["raw_v_ready"] == 4


def test_a_tile_count_is_a_windowed_persistent_ctas_beside_the_default_depth():
    bound = stream_bound(WAVES)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    attention = facts.attention
    for partition, choices in (
        ("item", (0,)),
        ("whole", (0, 5)),
        ("balanced", (0, 5)),
    ):
        assert (
            paged_attention_value_ring_tiles_choices(
                attention, PAIRED, False, partition
            )
            == choices
        )
    # Not beside the value-metadata transport or another depth in pages.
    assert paged_attention_value_ring_tiles_choices(
        attention, PAIRED, True, "balanced"
    ) == (0,)
    assert paged_attention_value_ring_tiles_choices(
        attention, PAIRED, False, "balanced", 3
    ) == (0,)
    # A kernel without a window is not offered the ring.
    full = stream_bound(WAVES, window=NO_WINDOW).config_spec.cute_mma_dag_facts
    assert full is not None and full.attention is not None
    assert paged_attention_value_ring_tiles_choices(
        full.attention, PAIRED, False, "balanced"
    ) == (0,)
    assert (
        stream_bound(WAVES, window=NO_WINDOW).config_spec.cute_mma_dag_value_ring_tiles
        is None
    )
    # A device whose CTA has no room for a persistent partition has no count.
    tight = stream_bound(WAVES, smem_capacity_bytes=160000)
    tight_facts = tight.config_spec.cute_mma_dag_facts
    assert tight_facts is not None and tight_facts.attention is not None
    assert tight_facts.attention.partition_choices == ("item",)
    assert paged_attention_value_ring_tiles_choices(
        tight_facts.attention, PAIRED, False, "balanced"
    ) == (0,)
    paired = {"cute_mma_dag_pass_warps": PAIRED}
    # And one that has both persistent partitions and grants less than the
    # count needs beside their query stages, where the shared-memory fit
    # alone decides: 40 query columns with two query stages hold five tiles
    # in 219,992 bytes on the balanced partition, which the B200 grants.
    wide = stream_bound(WAVES, rows=40)
    short = stream_bound(WAVES, rows=40, smem_capacity_bytes=210000)
    for bind, choices in ((wide, (0, 5)), (short, (0,))):
        bind_facts = bind.config_spec.cute_mma_dag_facts
        assert bind_facts is not None and bind_facts.attention is not None
        assert bind_facts.attention.partition_choices == ("item", "whole", "balanced")
        for partition in ("whole", "balanced"):
            assert (
                paged_attention_value_ring_tiles_choices(
                    bind_facts.attention, PAIRED, False, partition
                )
                == choices
            )
        reason = paged_attention_value_ring_tiles_reason(
            bind_facts, {**paired, **BALANCED, TILES: 5}
        )
        if bind is wide:
            assert reason is None
            assert bind.config_spec.cute_mma_dag_value_ring_tiles is not None
        else:
            assert reason is not None
            assert "needs 219992 bytes" in reason and "grants 210000" in reason
            assert bind.config_spec.cute_mma_dag_value_ring_tiles is None
    assert (
        paged_attention_value_ring_tiles_reason(facts, {**paired, **BALANCED, TILES: 5})
        is None
    )
    for config in (
        {**paired, TILES: 5},
        {**paired, **BALANCED, TILES: 6},
        {**paired, **BALANCED, TILES: 5, RING: 3},
        {**paired, **BALANCED, TILES: True},
    ):
        assert paged_attention_value_ring_tiles_reason(facts, config) is not None
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        spec = bound.config_spec
        kept = spec.normalized_config(stream_config(**BALANCED, **{TILES: 5}))
        assert kept.config[TILES] == 5 and RING not in kept.config
        assert (
            TILES
            not in spec.normalized_config(
                stream_config(**BALANCED, **{TILES: 0})
            ).config
        )
        for knobs in ({TILES: 5}, {**BALANCED, TILES: 4}):
            config = stream_config(**knobs)
            with pytest.raises(exc.InvalidConfig, match=TILES):
                spec.normalized_config(config)
            spec.normalize(config, _fix_invalid=True)
            assert TILES not in config.config


# One compile per kernel, about five seconds each.
def test_the_tile_count_renders_and_compiles_on_a_persistent_partition(tmp_path):
    """The walk's balanced kernel without and with the option, rendered and
    compiled the way a launch compiles it: a ring of five tiles is another
    kernel than the ring of four."""
    bound = stream_bound(WAVES)
    args = tuple(partition_attention_args(units=WAVES, rows=8, window=PAGES * PAGE - 1))
    walk = stream_render(bound, stream_config(**WALK))
    keywords = body_keywords(walk)
    assert "value_ring_tiles" not in keywords
    assert keywords["stream_page_bound"] == str(WAVES * PAGES)
    tiled = stream_render(bound, stream_config(**WALK, **{TILES: 5}))
    keywords = body_keywords(tiled)
    assert keywords["value_ring_tiles"] == "5"
    assert keywords["shared_memory_capacity"] == str(CAPACITY)
    assert "value_ring" not in keywords
    compiled = {
        name: compile_offline(bound, args, source, tmp_path / name)
        for name, source in (("walk", walk), ("tiled", tiled))
    }
    assert compiled["walk"].ptx != compiled["tiled"].ptx
    assert all(
        kernel.threads == 640 and kernel.launch_registers == 96
        for kernel in compiled.values()
    )
    # The body refuses a tile count its capacity does not hold, as it
    # refuses a depth in pages.
    capacity = f"shared_memory_capacity={CAPACITY}"
    with pytest.raises(AssertionError):
        compile_offline(
            bound,
            args,
            tiled.replace(capacity, "shared_memory_capacity=100000"),
            tmp_path / "short",
        )
    # And a count beside one unit per CTA.
    plain = stream_render(bound, stream_config())
    with pytest.raises(AssertionError):
        compile_offline(
            bound,
            args,
            plain.replace(
                ", stream_page_bound=",
                f", value_ring_tiles=5, {capacity}, stream_page_bound=",
            ),
            tmp_path / "item",
        )


def test_the_tile_count_is_a_field_stepped_onto_with_a_persistent_partition():
    eager = stream_bound(WAVES)
    fields = eager.config_spec._flat_fields()
    # Sampled in every timing regime, as the depth in pages is.
    assert fields[TILES] is eager.config_spec.cute_mma_dag_value_ring_tiles
    assert fields[TILES].choices == fields[TILES]._active_choices() == (0, 5)
    assert fields[TILES].default() == 0
    # One wave: the tile count stays a persistent CTA's.
    one = stream_bound(ONE_WAVE, input_banks=2).config_spec._flat_fields()
    assert one[TILES].choices == (0, 5)
    banks = stream_bound(WAVES, input_banks=2)
    spec = banks.config_spec
    with cpu_render_target(fake_device_sm_count=SMS), banks.env, banks.host_function:
        # A step onto the tile count from one unit per CTA takes the
        # balanced partition; a persistent config keeps its partition.
        reached, alone = stream_step(spec, stream_config(), **{TILES: 5})
        assert (reached.config[TILES], reached.config[PARTITION]) == (5, "balanced")
        assert reached.config[QUERY_TMA] is True
        assert TILES not in alone.config
        whole = stream_config(**{PARTITION: "whole", QUERY_TMA: True})
        reached, _alone = stream_step(spec, whole, **{TILES: 5})
        assert (reached.config[TILES], reached.config[PARTITION]) == (5, "whole")
        # A step off the persistent partition leaves the count behind.
        parent = spec.normalized_config(stream_config(**WALK, **{TILES: 5}))
        reached, _alone = stream_step(spec, parent, **{PARTITION: "item"})
        assert TILES not in reached.config


def test_a_fixed_tile_count_moves_every_paired_candidate_onto_its_partition():
    """An ``autotune_config_overrides`` entry that names a tile count is
    layered over complete candidates, counted seed by seed: every seed on
    the paired schedule transfers on a persistent partition with the
    count, and what that partition does not run or hold is dropped with
    the move, as a search's step onto the count leaves a row."""
    banks = stream_bound(WAVES, input_banks=2)
    spec = banks.config_spec
    with cpu_render_target(fake_device_sm_count=SMS), banks.env, banks.host_function:
        paired = _paired(seed_transfers(spec, {TILES: 5}))
        # Seeds with one unit per CTA, most of them; a deeper ring in pages,
        # which gives way to the count; pages of one unit per CTA, which
        # are dropped with that partition.
        one_unit = [seed for seed, _moved in paired if PARTITION not in seed.config]
        assert len(one_unit) > len(paired) - len(one_unit) > 0
        assert any(RING in seed.config for seed, _moved in paired)
        assert any(PAGES_KEY in seed.config for seed, _moved in paired)
        for seed, moved in paired:
            assert moved is not None, seed
            assert (moved.config[TILES], moved.config[PARTITION]) == (5, "balanced")
            assert moved.config[QUERY_TMA] is True
            assert RING not in moved.config and PAGES_KEY not in moved.config
            # Past one wave the balanced partition runs the reversed walk.
            assert moved.config.get(ORDER) == seed.config.get(ORDER)
        # The seeds of the other schedules have no ring in tiles: the search
        # rejects them, and only them.
        moves = seed_transfers(spec, {TILES: 5})
        _forced, failures = transferred_seeds(spec, {TILES: 5})
        assert 0 < len(failures) == len(moves) - len(paired)
        assert all(TILES in failure for failure in failures)
        # A candidate's own count goes with what it stands on: under an
        # override that names one unit per CTA, a depth in pages, or pages
        # of one unit per CTA, which take that partition.
        own = spec.normalized_config(stream_config(**WALK, **{TILES: 5}))
        for overrides, kept in (
            ({PARTITION: "item"}, False),
            ({RING: 3}, False),
            ({PAGES_KEY: 12}, False),
            ({SKEW: 1}, True),
            ({PARTITION: "whole"}, True),
        ):
            generation = spec.create_config_generation(overrides=overrides)
            pair = generation.transfer_seed(helion.Config.from_dict(dict(own.config)))
            assert pair is not None, overrides
            assert (pair[1].config.get(TILES) == 5) is kept, overrides
            for key, value in overrides.items():
                # (One unit per CTA is the missing key.)
                assert pair[1].config.get(key, "item") == value
    # Forty query columns: the three seeds with a third page per head move
    # as every other one does.
    wide = stream_bound(WAVES, rows=40, input_banks=2)
    with cpu_render_target(fake_device_sm_count=SMS), wide.env, wide.host_function:
        paired = _paired(seed_transfers(wide.config_spec, {TILES: 5}))
        assert sum(RING in seed.config for seed, _moved in paired) == 3
        assert all(
            moved is not None
            and (moved.config[TILES], moved.config[PARTITION]) == (5, "balanced")
            and RING not in moved.config
            for _seed, moved in paired
        )
    # On a grid of one wave the reversed walk is one unit per CTA's alone
    # (``page_order_needs``): a seed in that order transfers without it, as
    # the step onto the count from it does. 49 paired seeds over two banks
    # and 35 over one, each of them on the balanced partition with the
    # count.
    for input_banks, count, reversed_seeds in ((2, 49, 10), (1, 35, 2)):
        wave = stream_bound(ONE_WAVE, input_banks=input_banks)
        wave_spec = wave.config_spec
        with cpu_render_target(fake_device_sm_count=SMS), wave.env, wave.host_function:
            paired = _paired(seed_transfers(wave_spec, {TILES: 5}))
            assert len(paired) == count
            walked = [
                moved for seed, moved in paired if seed.config.get(ORDER) == "reversed"
            ]
            assert len(walked) == reversed_seeds
            for _seed, moved in paired:
                assert moved is not None
                assert (moved.config[TILES], moved.config[PARTITION]) == (
                    5,
                    "balanced",
                )
                assert ORDER not in moved.config and LEAD not in moved.config
            reached, _alone = stream_step(
                wave_spec, stream_config(**{ORDER: "reversed"}), **{TILES: 5}
            )
            assert (reached.config[TILES], reached.config[PARTITION]) == (5, "balanced")
            assert ORDER not in reached.config
            # Under an entry that names the order as well nothing is
            # dropped: the candidates are rejected by the order they cannot
            # run on that partition.
            forced, failures = transferred_seeds(
                wave_spec, {TILES: 5, ORDER: "reversed"}
            )
            assert not [
                seed
                for seed in forced
                if seed.config["cute_mma_dag_pass_warps"] == PAIRED
            ]
            assert any(f"{ORDER}='reversed'" in failure for failure in failures)
