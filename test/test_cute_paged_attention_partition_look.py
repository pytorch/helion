"""Check the balanced partition's ``cute_mma_dag_partition_look``: the spare
warp's look at the counter of the split unit a CTA runs the lowest part of.
Which partitions, grids and devices have it, the row it adds to the body's
shared-memory table, the config contract, a search's step and a fixed
override, the seeds, the order of the look's barriers and waits in the
device code, that the spare warp leaves for a lowest part the words the
group's own look leaves, that every global write of the compiled kernel
stays behind a dependency wait and every access to the workspace too, that
a lowest part reads its role behind its meeting with the spare warp, and
that a late merge starts behind a barrier of its group: in the compiled
kernel, and in host models, of a group whose warps run the template's
statements with one warp slow in its epilogue, and of a whole CTA, both
groups, the spare warp and the K conversions, on every path of the
exchange. And the other part's panel that the spare warp stages in the
raw K ring for a pair's fold: that it is taken behind the K conversions'
last tile and read behind its copies, in the source, the compiled kernel
and the CTA's model, and that the streamed fold gives the bits of the fold
it replaces."""

from __future__ import annotations

import ast
import contextlib
from fractions import Fraction
import functools
import inspect
from itertools import starmap
import math
import re
import struct
import threading
import time
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from helion import exc
from helion._compiler.cute import _paged_attention_paired_pipeline as template
from helion._compiler.cute import _paged_attention_partition as helpers
from helion._compiler.cute._paged_attention_partition import plan_grid_bound
from helion._compiler.cute.autotune.mma_dag_surface import partition_look_needs
from helion._compiler.cute.cute_mma_dag import paged_attention_partition_look_choices
from helion._compiler.cute.cute_mma_dag import paged_attention_partition_look_fit
from helion._compiler.cute.paged_attention_config import (
    paged_attention_partition_look_reason,
)
from helion._compiler.cute.paged_attention_shared_memory import MAIL_WORDS
from helion._compiler.cute.paged_attention_shared_memory import (
    MMA_DAG_PARTITION_LOOK_CHOICES,
)
from helion._compiler.cute.paged_attention_shared_memory import (
    MMA_DAG_PARTITION_LOOK_KEY,
)
from helion._compiler.cute.paged_attention_shared_memory import (
    PAIRED_DEFAULT_VALUE_RING,
)
from helion._compiler.cute.paged_attention_shared_memory import PANEL_QUAD_BYTES
from helion._compiler.cute.paged_attention_shared_memory import (
    PARTITION_LOOK_KEYS_CONVERTED,
)
from helion._compiler.cute.paged_attention_shared_memory import (
    PARTITION_LOOK_PANEL_STAGED,
)
from helion._compiler.cute.paged_attention_shared_memory import PARTITION_LOOK_STAGE
from helion._compiler.cute.paged_attention_shared_memory import RAW_TILE_BYTES
from helion._compiler.cute.paged_attention_shared_memory import WALK_WORDS
from helion._compiler.cute.paged_attention_shared_memory import ValueRingTiles
from helion._compiler.cute.paged_attention_shared_memory import (
    paired_shared_memory_bytes,
)
from helion._compiler.cute.paged_attention_shared_memory import (
    partition_look_panel_columns,
)
from helion._compiler.cute.paged_attention_shared_memory import partition_look_reason
from helion._compiler.cute.paged_attention_shared_memory import (
    partition_look_stage_words,
)
from helion._compiler.cute.paged_attention_shared_memory import pipeline_barrier_plan
from helion._compiler.cute.paged_attention_shared_memory import pipeline_barriers
from helion._compiler.cute.paged_attention_shared_memory import pipeline_parts
from helion._compiler.cute.paged_attention_shared_memory import pipeline_shared_bytes
from helion._compiler.cute.paged_attention_shared_memory import pipeline_shared_memory
from helion._testing import cpu_render_target
from helion._testing.cute_offline import compile_offline
from helion._testing.cute_paged_partition import PAGE
from helion._testing.cute_paged_partition import PAGES
from helion._testing.cute_paged_partition import partition_attention_args
from helion._testing.cute_paged_partition_plan import active_ctas
from helion._testing.cute_paged_partition_plan import applied_skew
from helion._testing.cute_paged_partition_plan import planner_slot_cta
from helion._testing.cute_paged_partition_plan import reference_partition
from helion._testing.cute_paged_partition_plan import split_units
from helion._testing.cute_paged_partition_plan import unit_headers
from helion._testing.cute_paged_stream import NO_WINDOW
from helion._testing.cute_paged_stream import PAIRED
from helion._testing.cute_paged_stream import SMS
from helion._testing.cute_paged_stream import body_keywords
from helion._testing.cute_paged_stream import jit_host_body
from helion._testing.cute_paged_stream import seed_transfers
from helion._testing.cute_paged_stream import seeds_ahead_of_the_partition_look
from helion._testing.cute_paged_stream import seeds_ahead_of_the_range_ahead
from helion._testing.cute_paged_stream import source_calls
from helion._testing.cute_paged_stream import source_function
from helion._testing.cute_paged_stream import stream_bound
from helion._testing.cute_paged_stream import stream_config
from helion._testing.cute_paged_stream import stream_render
from helion._testing.cute_paged_stream import stream_seeds
from helion._testing.cute_paged_stream import stream_step
from helion._testing.cute_paged_stream import transferred_seeds
from helion._testing.cute_ptx_waits import global_accesses_ahead_of_a_wait
from helion._testing.cute_ptx_waits import instructions_reached_without

pytestmark = pytest.mark.cpu_only
KEY = "cute_mma_dag_partition_look"
PARTITION = "cute_mma_dag_partition"
QUERY_TMA = "cute_mma_dag_query_tma"
PREFETCH = "cute_mma_dag_boundary_prefetch"
AHEAD = "cute_mma_dag_boundary_prefetch_ahead"
TILES = "cute_mma_dag_value_ring_tiles"
RING = "cute_mma_dag_value_ring"
SKEW = "cute_mma_dag_partition_skew"
# Shared memory one CTA of the modelled B200 may request.
CAPACITY = 232448
ONE_WAVE = 128
WAVES = 300
BALANCED = {PARTITION: "balanced", QUERY_TMA: True}
WHOLE = {PARTITION: "whole", QUERY_TMA: True}
SPARE = {KEY: "spare"}
WAIT = "if capture_pdl_wait_mode != 0:\n    cute.arch.griddepcontrol_wait()"


def _range_ahead(seeds):
    """The range-ahead seeds: those that count their raw V ring in tiles
    and leave the look to the compute group."""
    ahead = seeds_ahead_of_the_partition_look(seeds)
    return ahead[len(seeds_ahead_of_the_range_ahead(seeds)) :]


def _attention(bound):
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    return facts.attention


@pytest.fixture(scope="module")
def looks(tmp_path_factory):
    """The partition fixture's balanced kernel with the look of the group
    and with the spare warp's (``spare``; ``wide`` at 40 query columns;
    ``joined`` without the early retirement, whose threads all join at
    the body's end): rendered once, and compiled without a device once in
    a process, when a test first asks for one. The source of ``wide`` and
    of ``joined`` is there once the kernel is compiled."""
    directory = tmp_path_factory.mktemp("looks")

    def arguments(rows: int):
        return partition_attention_args(
            units=WAVES,
            rows=rows,
            window=PAGES * PAGE - 1,
            table_pages=PAGES,
            pool_pages=13,
        )

    bound = stream_bound(WAVES, input_banks=2)
    args = arguments(8)
    sources = {
        "group": stream_render(bound, stream_config(**BALANCED)),
        "spare": stream_render(bound, stream_config(**BALANCED, **SPARE)),
    }

    @functools.cache
    def compiled(name: str):
        if name == "joined":
            # The spare warp's look without the early retirement of the
            # allocation: every thread joins at the body's end, at named
            # barrier 0.
            sources[name] = stream_render(
                bound,
                stream_config(
                    **BALANCED, **SPARE, cute_mma_dag_early_tmem_retirement=False
                ),
            )
        if name == "wide":
            # The spare warp's look at 40 query columns, of which the
            # stage of the other part's panel holds 32.
            wide = stream_bound(WAVES, input_banks=2, rows=40)
            sources[name] = stream_render(wide, stream_config(40, **BALANCED, **SPARE))
            return compile_offline(wide, arguments(40), sources[name], directory / name)
        return compile_offline(bound, args, sources[name], directory / name)

    return SimpleNamespace(bound=bound, args=args, sources=sources, compiled=compiled)


def test_the_look_is_the_balanced_partitions_on_a_grid_of_more_units_than_sms():
    assert KEY == MMA_DAG_PARTITION_LOOK_KEY
    assert MMA_DAG_PARTITION_LOOK_CHOICES == ("group", "spare")
    waves = stream_bound(WAVES)
    attention = _attention(waves)
    for partition, choices in (
        ("item", ("group",)),
        ("whole", ("group",)),
        ("balanced", ("group", "spare")),
    ):
        assert (
            paged_attention_partition_look_choices(attention, PAIRED, False, partition)
            == choices
        )
    for pass_warps, skip in (("attention", False), (PAIRED, True)):
        assert paged_attention_partition_look_choices(
            attention, pass_warps, skip, "balanced"
        ) == ("group",)
    # One wave, and as many units as SMs: no look of the spare warp and no
    # field; one unit more than SMs has both.
    for units, offered in ((ONE_WAVE, False), (SMS, False), (SMS + 1, True)):
        bound = stream_bound(units)
        assert (
            "spare"
            in paged_attention_partition_look_choices(
                _attention(bound), PAIRED, False, "balanced"
            )
        ) is offered
        assert (bound.config_spec.cute_mma_dag_partition_look is not None) is offered
        assert (KEY in bound.config_spec._flat_fields()) is offered
    # A kernel without the balanced partition (no fast math) has none.
    exact = stream_bound(WAVES, fast_math=False)
    assert "balanced" not in _attention(exact).partition_choices
    assert exact.config_spec.cute_mma_dag_partition_look is None
    # With or without a window: the look is the exchange's.
    assert stream_bound(WAVES, window=NO_WINDOW).config_spec.cute_mma_dag_partition_look
    field = waves.config_spec._flat_fields()[KEY]
    assert field is waves.config_spec.cute_mma_dag_partition_look
    assert field.choices == field._active_choices() == ("group", "spare")
    assert field.default() == "group"


def test_the_stage_is_a_row_of_the_shared_memory_table():
    """What the spare warp stages for the two groups is a row of the body's
    table, behind what the groups post to that warp, present only in a
    body with that look, and the config contract admits the look by
    replaying the table with it against the device."""
    for columns in (8, 32, 40):
        assert partition_look_stage_words(columns) == 4 * columns
        arrays = pipeline_shared_memory(columns, False, 2, WAVES, True)
        names = [array.name for array in arrays]
        row = arrays[names.index(PARTITION_LOOK_STAGE)]
        assert names.index(PARTITION_LOOK_STAGE) == names.index("mail_raw") + 1
        assert (row.dtype, row.width, row.count, row.alignment, row.guard) == (
            "Float32",
            4,
            4 * columns,
            16,
            "partition_look",
        )
        # The row and the padding to its alignment.
        plain = paired_shared_memory_bytes(
            columns, query_stages=2, units=WAVES, balanced=True, scanned_bounds=True
        )
        staged = paired_shared_memory_bytes(
            columns,
            query_stages=2,
            units=WAVES,
            balanced=True,
            scanned_bounds=True,
            partition_look=True,
        )
        assert staged - plain == 16 * columns + 8
    assert PARTITION_LOOK_STAGE == "look_stage"
    assert pipeline_parts(True, False, 17, True, True, partition_look=True) == {
        "persistent",
        "scanned_lengths",
        "balanced",
        "partition_look",
    }
    # Only a balanced body has that warp's exchange.
    for balanced_arguments in ((True, False, 17, True, False), (True, False)):
        with pytest.raises(AssertionError):
            pipeline_parts(*balanced_arguments, partition_look=True)
    # The widest table: 40 query columns, two query stages, 640 units.
    shape = {"query_stages": 2, "units": 640}
    assert partition_look_reason(CAPACITY, 40, **shape) is None
    assert partition_look_reason(205791, 40, **shape) == (
        "the spare warp's look stages 640 bytes of row statistics: its CTA "
        "needs 205792 bytes of shared memory with 2 query stages and the "
        "device grants 205791"
    )
    assert partition_look_reason(205792, 40, **shape) is None
    # Beside a ring of five tiles it fits a B200; beside a third page per
    # head, which a balanced CTA of 40 columns does not hold itself, not.
    assert (
        partition_look_reason(CAPACITY, 40, value_ring=ValueRingTiles(5), **shape)
        is None
    )
    refused = partition_look_reason(CAPACITY, 40, value_ring=3, **shape)
    assert refused is not None
    assert "needs 239584 bytes" in refused and f"grants {CAPACITY}" in refused
    # An unknown capacity admits the look, as it admits the schedule.
    assert partition_look_reason(0, 40, value_ring=3, **shape) is None
    # The body takes the row from the table and repeats the check on the
    # table it allocates.
    body = source_function("paged_attention_paired_body", template)
    (parts,) = source_calls(body, "pipeline_parts")
    assert ast.unparse(parts.args[-1]) == "spare_look"
    text = ast.unparse(body)
    assert "spare_look = partition_look == MMA_DAG_PARTITION_LOOK_SPARE" in text
    assert (
        "look_stage = cute.make_tensor(shared.look_stage, "
        "cute.make_layout((partition_look_stage_words(query_columns),)))"
    ) in text
    asserted = {
        ast.unparse(node.test)
        for node in ast.walk(body)
        if isinstance(node, ast.Assert)
    }
    assert {
        "cutlass.const_expr(partition_look in MMA_DAG_PARTITION_LOOK_CHOICES)",
        "cutlass.const_expr(not spare_look or partition == 'balanced')",
    } <= asserted
    # One check, for a deeper raw V ring and the look together: the bytes
    # of the table are computed once.
    (replayed,) = source_calls(body, "pipeline_shared_bytes")
    (checked,) = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.If) and replayed in ast.walk(node)
    ]
    assert ast.unparse(checked) == (
        "if cutlass.const_expr(raw_v_ring != PAIRED_DEFAULT_VALUE_RING or "
        "spare_look):\n"
        "    shared_bytes = pipeline_shared_bytes(shared_table, shared_parts)\n"
        "    assert cutlass.const_expr(shared_memory_capacity == 0 or "
        "shared_bytes <= shared_memory_capacity), "
        "(shared_bytes, shared_memory_capacity)"
    )


def test_the_contract_refuses_the_look_where_the_bind_does_not_hold_it():
    waves = stream_bound(WAVES)
    one = stream_bound(ONE_WAVE)
    facts = waves.config_spec.cute_mma_dag_facts
    paired = {"cute_mma_dag_pass_warps": PAIRED}
    assert (
        paged_attention_partition_look_reason(facts, {**paired, **BALANCED, **SPARE})
        is None
    )
    for config in (
        {**paired, **SPARE},
        {**paired, **WHOLE, **SPARE},
        {"cute_mma_dag_pass_warps": "attention", **BALANCED, **SPARE},
    ):
        reason = paged_attention_partition_look_reason(facts, config)
        assert reason is not None and KEY in reason
        assert "balanced partition's" in reason and "bytes" not in reason
        assert RING not in reason
    # A value that is no look at all, on a config that has both: the
    # choices and nothing else, since neither the partition, the grid nor
    # the fit refuses anything here.
    unknown = f"{KEY} must be one of ('group', 'spare')"
    for value in ("warp", True):
        assert (
            paged_attention_partition_look_reason(
                facts, {**paired, **BALANCED, KEY: value}
            )
            == unknown
        )
    # The whole text of the partition's refusal, and of the grid's on one
    # wave. A value that is no look gets the same text there: it says why
    # the group's look is the only choice.
    grid = (
        f"{KEY} must be one of ('group',) for "
        "cute_mma_dag_pass_warps='attention_paired', "
        "cute_mma_dag_value_metadata_skip=False and "
        "cute_mma_dag_partition={partition!r} on this kernel's grid: the "
        "spare warp's look is the balanced partition's, on a grid known to "
        "hold more units than the device has SMs"
    )
    for look in ("spare", "warp"):
        assert paged_attention_partition_look_reason(
            facts, {**paired, **WHOLE, KEY: look}
        ) == grid.format(partition="whole")
        assert paged_attention_partition_look_reason(
            one.config_spec.cute_mma_dag_facts, {**paired, **BALANCED, KEY: look}
        ) == grid.format(partition="balanced")
    # Where the partition and the grid have the look and the shared-memory
    # fit alone decides, the refusal says so and not whose look it is: it
    # names the ring's two options with their values, and the bytes.
    # Beside the default ring every device that holds the balanced
    # partition holds the stage (the partition's budgets, 188,416 and
    # 208,896 bytes for one and two query stages, exceed its largest
    # layouts with the stage, 185,312 and 205,792). Beside a ring of five
    # tiles a CTA of 40 query columns with two query stages needs 219,992
    # bytes and with the stage 220,640: a device in between has the ring,
    # the look, and not both.
    wide = stream_bound(WAVES, rows=40, input_banks=2)
    short = stream_bound(WAVES, rows=40, input_banks=2, smem_capacity_bytes=220000)
    for bound, both in ((wide, True), (short, False)):
        attention = _attention(bound)
        assert attention.partition_choices == ("item", "whole", "balanced")
        assert attention.partition_q_stages == 2
        assert paged_attention_partition_look_choices(
            attention, PAIRED, False, "balanced"
        ) == ("group", "spare")
        assert bound.config_spec.cute_mma_dag_partition_look is not None
        assert bound.config_spec.cute_mma_dag_value_ring_tiles is not None
        assert (
            "spare"
            in paged_attention_partition_look_choices(
                attention, PAIRED, False, "balanced", 2, 5
            )
        ) is both
        reason = paged_attention_partition_look_reason(
            bound.config_spec.cute_mma_dag_facts,
            {**paired, **BALANCED, **SPARE, TILES: 5},
        )
        if both:
            assert reason is None
        else:
            assert reason == (
                f"{KEY} must be one of ('group',) for {RING}=2 and "
                f"{TILES}=5 on this device: the statistics the spare warp "
                "stages do not fit beside this config's raw V ring: the "
                "spare warp's look stages 640 bytes of row statistics: its "
                "CTA needs 220640 bytes of shared memory with 2 query "
                "stages and the device grants 220000"
            )
        # A value that is no look: where both fit it is told the choices,
        # and where the fit refuses, why the group's look is the only one.
        named = paged_attention_partition_look_reason(
            bound.config_spec.cute_mma_dag_facts,
            {**paired, **BALANCED, KEY: "warp", TILES: 5},
        )
        assert named == (unknown if both else reason)
        # A seed is a legal config: the range-ahead seeds count their ring
        # in tiles, so the device without room for both has those seeds
        # and none with the look.
        seeds = stream_seeds(bound)
        assert len(_range_ahead(seeds)) == 4
        assert sum(KEY in seed.config for seed in seeds) == (4 if both else 0)
    # Beside a ring the CTA does not hold with the stage, the look gives
    # way: 32 query columns hold a third page per head, and the stage
    # beside it, up to a capacity in between.
    ringed = paired_shared_memory_bytes(
        32,
        value_ring=3,
        query_stages=2,
        units=WAVES,
        balanced=True,
        scanned_bounds=True,
        partition_look=True,
    )
    tight = stream_bound(WAVES, rows=32, smem_capacity_bytes=ringed - 1)
    attention = _attention(tight)
    assert paged_attention_partition_look_fit(attention, 3, 0) is not None
    assert paged_attention_partition_look_fit(attention, 2, 0) is None
    assert paged_attention_partition_look_fit(attention, 2, 5) is None
    assert paged_attention_partition_look_choices(
        attention, PAIRED, False, "balanced", 3
    ) == ("group",)
    assert paged_attention_partition_look_choices(
        attention, PAIRED, False, "balanced", 2, 5
    ) == ("group", "spare")
    reason = paged_attention_partition_look_reason(
        tight.config_spec.cute_mma_dag_facts,
        {**paired, **BALANCED, **SPARE, RING: 3},
    )
    assert reason is not None and f"needs {ringed} bytes" in reason
    assert f"for {RING}=3 and {TILES}=0 on this device" in reason
    assert "grid" not in reason and "balanced partition's" not in reason
    # A B200's balanced CTA of 40 query columns beside a third page per
    # head, which the ring's own rule refuses first: asked alone, the look
    # names that ring too.
    reason = paged_attention_partition_look_reason(
        wide.config_spec.cute_mma_dag_facts,
        {**paired, **BALANCED, **SPARE, RING: 3},
    )
    assert reason is not None and "grid" not in reason
    assert f"for {RING}=3 and {TILES}=0 on this device" in reason
    assert "needs 237024 bytes" in reason and f"grants {CAPACITY}" in reason
    with cpu_render_target(fake_device_sm_count=SMS), waves.env, waves.host_function:
        spec = waves.config_spec
        kept = spec.normalized_config(stream_config(**BALANCED, **SPARE))
        assert kept.config[KEY] == "spare"
        # The group's look and the missing key are one identity.
        dropped = spec.normalized_config(stream_config(**BALANCED, **{KEY: "group"}))
        assert KEY not in dropped.config
        for knobs in ({}, WHOLE):
            with pytest.raises(exc.InvalidConfig, match=KEY):
                spec.normalized_config(stream_config(**knobs, **SPARE))
            config = stream_config(**knobs, **SPARE)
            spec.normalize(config, _fix_invalid=True)
            assert KEY not in config.config
            assert config.config.get(PARTITION) == knobs.get(PARTITION)
        # A value that is no look is refused with the choices, and repaired
        # to the group's look on the partition the config names.
        with pytest.raises(exc.InvalidConfig, match=re.escape(unknown)):
            spec.normalized_config(stream_config(**BALANCED, **{KEY: "warp"}))
        config = stream_config(**BALANCED, **{KEY: "warp"})
        spec.normalize(config, _fix_invalid=True)
        assert KEY not in config.config and config.config[PARTITION] == "balanced"
    with (
        cpu_render_target(fake_device_sm_count=SMS),
        one.env,
        one.host_function,
        pytest.raises(exc.InvalidConfig, match=KEY),
    ):
        one.config_spec.normalized_config(stream_config(**BALANCED, **SPARE))


def test_a_step_and_a_fixed_override_take_the_balanced_partition():
    bound = stream_bound(WAVES, input_banks=2)
    spec = bound.config_spec
    assert partition_look_needs() == {
        PARTITION: "balanced",
        QUERY_TMA: True,
        "cute_mma_dag_value_metadata_skip": False,
        "cute_mma_dag_value_metadata_read_overlap": False,
    }
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        # From one unit per CTA and from whole units the requested row
        # alone is the config it started from: the step takes the balanced
        # partition.
        for knobs in ({}, WHOLE):
            taken, asked = stream_step(spec, stream_config(**knobs), **SPARE)
            assert KEY not in asked.config
            assert asked.config.get(PARTITION) == knobs.get(PARTITION)
            assert taken.config[KEY] == "spare"
            assert taken.config[PARTITION] == "balanced" and taken.config[QUERY_TMA]
        taken, asked = stream_step(spec, stream_config(**BALANCED), **SPARE)
        assert taken.config == asked.config and taken.config[KEY] == "spare"
        seeds = spec.compiler_seed_configs
        own = [seed for seed in seeds if KEY in seed.config]
        assert len(own) == 4
        # Seed by seed: every seed on the paired schedule transfers with
        # the look on the balanced partition; the others are refused by
        # the look, and only they.
        moves = seed_transfers(spec, SPARE)
        paired = [
            (seed, moved)
            for seed, moved in moves
            if seed.config["cute_mma_dag_pass_warps"] == PAIRED
        ]
        one_unit = [seed for seed, _moved in paired if PARTITION not in seed.config]
        assert len(one_unit) > 30 and len(paired) > len(one_unit)
        for seed, moved in paired:
            assert moved is not None, seed
            assert (moved.config[KEY], moved.config[PARTITION]) == ("spare", "balanced")
            assert moved.config[QUERY_TMA] is True
            # The balanced partition runs a seed's range-ahead distance
            # and no prefetch of pages.
            assert moved.config.get(AHEAD) == seed.config.get(AHEAD)
            assert PREFETCH not in moved.config
        _forced, failures = transferred_seeds(spec, SPARE)
        assert 0 < len(failures) == len(moves) - len(paired)
        assert all(KEY in failure for failure in failures)
        # An override that takes away what the look stands on takes it
        # along, and no seed is refused for it.
        for overrides in (
            {PARTITION: "item"},
            {PARTITION: "whole"},
            {PREFETCH: 12},
            {KEY: "group"},
        ):
            moved, failures = transferred_seeds(spec, overrides)
            assert moved and not [f for f in failures if KEY in f], overrides
            assert not any(KEY in seed.config for seed in moved), overrides
        for overrides in ({PARTITION: "whole"}, {PREFETCH: 12}):
            moves = seed_transfers(spec, overrides)
            assert all(moved is not None for seed, moved in moves if seed in own)
        # A knob of the balanced partition keeps it, seed by seed.
        for overrides in ({SKEW: 1}, {TILES: 5}, {AHEAD: 96}):
            for seed, moved in seed_transfers(spec, overrides):
                if seed in own:
                    assert moved is not None and moved.config[KEY] == "spare", overrides
    # A fixed depth of the raw V ring that leaves the stage no room: the
    # seeds lose their look and stay; under an override that names the
    # look as well, they are refused by it.
    ringed = paired_shared_memory_bytes(
        32,
        value_ring=3,
        query_stages=2,
        units=WAVES,
        balanced=True,
        scanned_bounds=True,
        partition_look=True,
    )
    tight = stream_bound(WAVES, rows=32, input_banks=2, smem_capacity_bytes=ringed - 1)
    with cpu_render_target(fake_device_sm_count=SMS), tight.env, tight.host_function:
        spec = tight.config_spec
        own = [seed for seed in spec.compiler_seed_configs if KEY in seed.config]
        assert len(own) == 4
        moves = seed_transfers(spec, {RING: 3})
        for seed, moved in moves:
            if seed in own:
                assert moved is not None
                assert KEY not in moved.config and moved.config[RING] == 3
        moves = seed_transfers(spec, {RING: 3, **SPARE})
        assert all(moved is None for _seed, moved in moves)


def test_the_range_ahead_seeds_follow_once_more_with_the_look():
    banks = stream_bound(WAVES, input_banks=2)
    seeds = stream_seeds(banks)
    ahead = seeds_ahead_of_the_partition_look(seeds)
    repeated = _range_ahead(seeds)
    # The reversed walk with a distance and a ring of five tiles, plain and
    # under the store hint, at two distances: the last seeds but the look's.
    assert len(repeated) == 4 and ahead[-4:] == repeated
    assert [seed.config for seed in seeds[len(ahead) :]] == [
        {**seed.config, KEY: "spare"} for seed in repeated
    ]
    assert not any(KEY in seed.config for seed in ahead)
    with cpu_render_target(fake_device_sm_count=SMS), banks.env, banks.host_function:
        for seed in seeds[len(ahead) :]:
            strict = banks.config_spec.normalized_config(seed)
            assert strict.config[KEY] == "spare" and strict.config[TILES] == 5
    keywords = body_keywords(stream_render(banks, seeds[-1]))
    assert keywords["partition_look"] == "'spare'"
    assert keywords["value_ring_tiles"] == "5"
    # The body is told its capacity once, whichever of the ring's two
    # options names it beside the look: counted in the text, since a
    # keyword given twice parses and only fails where the kernel compiles.
    for ring in ({}, {RING: 3}, {TILES: 5}):
        source = stream_render(banks, stream_config(**BALANCED, **SPARE, **ring))
        assert source.count("shared_memory_capacity=") == 1, ring
        keywords = body_keywords(source)
        assert keywords["partition_look"] == "'spare'", ring
        assert keywords["shared_memory_capacity"] == str(CAPACITY), ring
        assert ("value_ring" in keywords, "value_ring_tiles" in keywords) == (
            RING in ring,
            TILES in ring,
        )
    # No seed where there is no range-ahead seed to repeat (one input
    # bank), for a kernel without a window, or on a grid of one wave.
    for bound in (
        stream_bound(WAVES),
        stream_bound(WAVES, input_banks=2, window=NO_WINDOW),
        stream_bound(ONE_WAVE, input_banks=2),
        stream_bound(SMS, input_banks=2),
    ):
        assert not any(KEY in seed.config for seed in stream_seeds(bound))


def test_the_looks_barriers_and_waits_stand_in_program_order():
    """Read from the device code. The spare warp: behind its posts it waits
    for both groups' requests, lanes 0 and 1 wait for the launch's
    dependencies and look once each without arriving, the warp
    synchronizes, stages behind another wait what a complete unit in two
    parts needs, and arrives for the groups that asked. A lane that looks
    leaves the part's role, the unit's lowest CTA and its parts in the
    group's record. A group: it asks at the start of its unit's lowest
    part by an arrival and meets the warp behind that part's pages at the
    first barrier of the part's reduction, where no thread of it has
    decided or waited, and reads the role behind that meeting; at the end
    of its walk a group that did not ask says so and meets the warp at the
    post's reply."""
    look = source_function("look_parts", helpers)
    statements = look.body[1:] if ast.get_docstring(look) else look.body
    order = [
        (
            "requests"
            if source_calls(node, "barrier_cta_sync")
            else "look"
            if source_calls(node, "_count_arrival")
            else "warp"
            if ast.unparse(node) == "cute.arch.sync_warp()"
            else "stage"
            if source_calls(node, "_load_pair")
            else "answer"
            if source_calls(node, "barrier_cta_arrive")
            else None
        )
        for node in statements
    ]
    assert [step for step in order if step] == [
        "requests",
        "look",
        "warp",
        "stage",
        "answer",
        "answer",
    ]
    requests = statements[order.index("requests")]
    assert ast.unparse(requests) == (
        "for head in cutlass.range_constexpr(2):\n"
        "    prims.barrier_cta_sync(LOOK_BARRIER + head, thread_count=POST_THREADS)"
    )
    # Two lanes look, each once, behind the wait and without arriving.
    looking = statements[order.index("look")]
    assert ast.unparse(looking.test) == "lane < 2"
    (asked,) = [
        node
        for node in looking.body
        if isinstance(node, ast.If) and source_calls(node, "_count_arrival")
    ]
    assert ast.unparse(asked.test) == "request != 0"
    assert ast.unparse(asked.body[0]) == WAIT
    (counted,) = source_calls(look, "_count_arrival")
    assert [
        (keyword.arg, ast.unparse(keyword.value)) for keyword in counted.keywords
    ] == [("arrive", "False")]
    # The answer: what one thread of the group leaves in the record where
    # the group looks, decided from the same look. The role is all of it:
    # the warp's lanes store no other word of a record.
    answer = [
        "walks[lane * WALK_WORDS + WALK_ROLE] = role",
        "walks[lane * WALK_WORDS + WALK_LOWEST] = cta",
        "walks[lane * WALK_WORDS + WALK_PARTS] = request + 1",
    ]
    assert [ast.unparse(node) for node in asked.body[1:]] == [
        "cta = cutlass.Int32(cute.arch.block_idx()[0])",
        "role = cutlass.Int32(PART_PUBLISH)",
        (
            "if _count_arrival(workspace, cta, request, lane, record_bytes, "
            "arrive=False):\n"
            "    role = cutlass.Int32(PART_MERGE)\n"
            "    if request == 1:\n"
            "        role = cutlass.Int32(PART_MERGE_PAIR)"
        ),
        *answer,
    ]
    assert [
        ast.unparse(node)
        for node in ast.walk(look)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]).startswith("walks[")
    ] == answer
    # No loop but over the two heads and the rows of a lane; nothing waits
    # for another CTA.
    assert not any(isinstance(node, ast.While) for node in ast.walk(look))
    assert {
        ast.unparse(node.iter) for node in ast.walk(look) if isinstance(node, ast.For)
    } == {
        "cutlass.range_constexpr(2)",
        "cutlass.range_constexpr(0, query_columns, 32)",
    }
    assert not source_calls(look, "barrier") and not source_calls(look, "mbar_wait")
    # What it stages: for a unit in two parts that is complete, behind the
    # wait, the other part's pairs. The role says so, in the record of a
    # group that asked: the lane wrote it there ahead of the warp's
    # synchronization, and another group's record has no answer.
    staging = statements[order.index("stage")]
    answered, noted, staged = staging.body
    assert ast.unparse(answered) == (
        "pair = (cutlass.Int32(walks[head * WALK_WORDS + WALK_ROLE]) == "
        "PART_MERGE_PAIR) & "
        "(cutlass.Int32(walks[head * WALK_WORDS + WALK_LOOK]) != 0)"
    )
    # The warp keeps that answer per head, for the panel it stages behind
    # its arrivals (``stage_panels``): it reads no record again there.
    assert ast.unparse(noted) == "paired.append(pair)"
    assert ast.unparse(statements[-1]) == "return (paired[0], paired[1])"
    assert isinstance(staged, ast.If) and ast.unparse(staged.test) == "pair"
    assert ast.unparse(staged.body[0]) == WAIT
    assert "_partial_slot(workspace, cta + 1, cta, head, record_bytes)" in ast.unparse(
        staged
    )
    # It answers the groups that asked, by arrivals.
    answers = [
        statements[index] for index, step in enumerate(order) if step == "answer"
    ]
    assert [ast.unparse(node) for node in answers] == [
        (
            "if request0 != 0:\n"
            "    prims.barrier_cta_arrive(MEETING_BARRIER, thread_count=POST_THREADS)"
        ),
        (
            "if request1 != 0:\n"
            "    prims.barrier_cta_arrive(MEETING_BARRIER + 1, "
            "thread_count=POST_THREADS)"
        ),
    ]
    # The barriers are their own, of a group and the spare warp.
    assert (helpers.POST_BARRIER, helpers.POSTED_BARRIER) == (8, 10)
    assert (helpers.LOOK_BARRIER, helpers.MEETING_BARRIER) == (12, 14)
    assert helpers.POST_THREADS == 160
    words = (helpers.WALK_LOOK, helpers.WALK_LOOKED)
    assert len(set(words)) == 2 and max(words) < WALK_WORDS
    assert not set(words) & {
        getattr(helpers, name)
        for name in dir(helpers)
        if name.startswith("WALK_")
        and name not in ("WALK_LOOK", "WALK_LOOKED", "WALK_WORDS")
    }
    # In the body the spare warp looks behind its posts.
    body = source_function("paged_attention_paired_body", template)
    (posts,) = source_calls(body, "post_parts")
    (looks,) = source_calls(body, "look_parts")
    assert posts.lineno < looks.lineno
    (guard,) = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.If)
        and looks in ast.walk(node)
        and ast.unparse(node.test) == "cutlass.const_expr(spare_look)"
    ]
    assert not guard.orelse
    # The group's side, in the walk over its units.
    compute = source_function("_compute_units", template)
    (walk,) = [node for node in compute.body if isinstance(node, ast.While)]
    (asking,) = [
        node
        for node in walk.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.const_expr(look_stage is not None)"
    ]
    (lowest,) = [
        node
        for node in asking.body
        if isinstance(node, ast.If) and "WALK_LOOK]" in ast.unparse(node)
    ]
    assert ast.unparse(lowest.test) == "lowest"
    noted, arrived = lowest.body
    # The request is the unit's other parts: from this CTA up to the one
    # that holds the unit's first step, the unit's last page slot.
    assert ast.unparse(noted) == (
        "if thread == 0:\n"
        "    walk[WALK_LOOKED] = cutlass.Int32(1)\n"
        "    walk[WALK_LOOK] = _slot_cta(header + cutlass.Int32(unit_pages), "
        "cutlass.Int32(plan[PLAN_SLOTS]), cutlass.Int32(plan[PLAN_ACTIVE]), "
        "cutlass.Int32(plan[PLAN_SKEW]), slot_bits, skewed) - "
        "cutlass.Int32(cute.arch.block_idx()[0])"
    )
    assert ast.unparse(arrived) == (
        "prims.barrier_cta_arrive(LOOK_BARRIER + pair_head, thread_count=POST_THREADS)"
    )
    pages = next(
        i for i, node in enumerate(walk.body) if source_calls(node, "_compute_pages")
    )
    assert walk.body.index(asking) < pages
    # For the look the group writes its request and its notes, and no word
    # of the answer: the warp's lane writes the role before any thread
    # reads it, so the group clears nothing for it.
    assert {
        ast.unparse(node.targets[0])
        for node in ast.walk(asking)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]).startswith("walk[")
    } == {"walk[WALK_LOOKED]", "walk[WALK_LOOK]", "walk[WALK_POSTED]"}
    # Behind the pages nothing decides for a lowest part, and nothing
    # waits: the group tells whether the spare warp looked for this range,
    # and one thread notes a part to post, which reads no workspace.
    (exchange,) = [
        node
        for node in ast.walk(walk)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.const_expr(look_stage is not None)"
        and source_calls(node, "split_arrival")
    ]
    decision = (
        "split_arrival(workspace, walk, mail, plan, header, planned_pages, "
        "cutlass.Int32(walk[WALK_STEP]), cutlass.Int32(count), item, pair_head, "
        "record_bytes, slot_bits, skewed"
    )
    assert [ast.unparse(node) for node in exchange.body] == [
        "split = _group_uniform(split)",
        "meeting = cutlass.Boolean(False)",
        (
            "if split:\n"
            "    meeting = _group_uniform(cutlass.Int32(walk[WALK_STEP]) + "
            "cutlass.Int32(count) == planned_pages)\n"
            "    if not meeting:\n"
            "        if thread == 0:\n"
            f"            {decision}, posts=True)"
        ),
    ]
    # Where the group looks, the part waits and one thread decides.
    (own,) = exchange.orelse
    assert ast.unparse(own) == (
        f"if split:\n    {WAIT.replace(chr(10), chr(10) + '    ')}\n"
        f"    if thread == 0:\n        {decision})"
    )
    # The meeting is the first barrier of the lowest part's reduction: the
    # barrier that publishes the row sums, of the group and the spare warp
    # for a part the warp looked for and of the group for any other range.
    (reduced,) = source_calls(walk, "_reduce_part")
    assert reduced.lineno > cast("int", exchange.end_lineno)
    assert [ast.unparse(argument) for argument in reduced.args[-5:]] == [
        "look_stage",
        "meeting",
        "capture_pdl_wait_mode",
        "panel_stage",
        "panel_staged",
    ]
    reduction = source_function("_reduce_part", template)
    (summed,) = source_calls(reduction, "_reduce_rows")
    assert ast.unparse(summed.args[-1]) == "meeting"
    rows = source_function("_reduce_rows", template)
    (published,) = [
        node
        for node in rows.body
        if isinstance(node, ast.If) and source_calls(node, "barrier")
    ]
    group_barrier = (
        "cute.arch.barrier(barrier_id=1 + pair_head * 2, number_of_threads=128)"
    )
    assert ast.unparse(published) == (
        f"if cutlass.const_expr(meeting is None):\n    {group_barrier}\n"
        "elif meeting:\n"
        "    prims.barrier_cta_sync(MEETING_BARRIER + pair_head, "
        "thread_count=POST_THREADS)\n"
        f"else:\n    {group_barrier}"
    )
    # Every thread reads the role behind it: the reduction for the pair,
    # then the walk for the stores. Nothing else of the template or of the
    # group's helpers reads the role.
    first = reduction.body.index(
        next(node for node in reduction.body if source_calls(node, "_reduce_rows"))
    )
    role = "cutlass.Int32(walk[WALK_ROLE])"
    reads = [
        index for index, node in enumerate(reduction.body) if role in ast.unparse(node)
    ]
    assert reads and min(reads) > first
    assert ast.unparse(reduction.body[reads[0]]) == (
        f"if split:\n    pair = {role} == PART_MERGE_PAIR"
    )
    walked = [
        node.lineno
        for node in ast.walk(compute)
        if isinstance(node, ast.Assign) and role in ast.unparse(node.value)
    ]
    assert len(walked) == 1 and walked[0] > reduced.lineno
    assert ast.unparse(compute).count("WALK_ROLE") == 1
    assert ast.unparse(reduction).count("WALK_ROLE") == 1
    for name in ("store_part", "merge_parts", "pair_partial"):
        assert "WALK_ROLE" not in ast.unparse(source_function(name, helpers))
    # Behind the walk: a group that did not meet the warp asks for no look,
    # a zero request, which lets the warp finish, and meets it at the
    # post's reply; a group that met it does neither.
    (ending,) = [
        node
        for node in ast.walk(compute)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.const_expr(look_stage is not None)"
        and node.orelse
        and source_calls(node, "barrier_cta_sync")
    ]
    assert cast("int", walk.end_lineno) < ending.lineno
    reply = (
        "prims.barrier_cta_sync(POSTED_BARRIER + pair_head, thread_count=POST_THREADS)"
    )
    assert ast.unparse(ending) == (
        "if cutlass.const_expr(look_stage is not None):\n"
        "    looked = cutlass.Boolean(False)\n"
        "    if not no_units:\n"
        "        looked = cutlass.Int32(walk[WALK_LOOKED]) != 0\n"
        "    looked = _group_uniform(looked)\n"
        "    if not looked:\n"
        "        if thread == 0:\n"
        "            walk[WALK_LOOK] = cutlass.Int32(0)\n"
        "        prims.barrier_cta_arrive(LOOK_BARRIER + pair_head, "
        "thread_count=POST_THREADS)\n"
        f"        {reply}\n"
        "else:\n"
        f"    {reply}"
    )
    # That reply also joined the group's warps ahead of the late merge of
    # a posted unit, and a group that met the warp behind its lowest part
    # comes there without it. The merge's row owners store the unit's
    # weights and row state ahead of the merge's first barrier, rows the
    # lowest part's epilogue reads: so under the look the late merge
    # starts with that barrier of the group, and then waits.
    (late,) = [
        node
        for node in ast.walk(compute)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.Int32(mail[MAIL_KIND]) == MAIL_LAST"
    ]
    assert cast("int", ending.end_lineno) < late.lineno
    joined = late.body[0]
    assert isinstance(joined, ast.If) and not joined.orelse
    assert ast.unparse(joined) == (
        "if cutlass.const_expr(look_stage is not None):\n"
        f"    {group_barrier}\n"
        "    if capture_pdl_wait_mode != 0:\n"
        "        cute.arch.griddepcontrol_wait()"
    )
    (merge,) = source_calls(late, "merge_parts")
    assert merge.lineno > cast("int", joined.end_lineno)
    merged = source_function("merge_parts", helpers)
    (folding,) = [node for node in merged.body if isinstance(node, ast.While)]
    owners, published = folding.body[:2]
    assert isinstance(owners, ast.If)
    assert ast.unparse(owners.test) == "thread < query_columns"
    rewritten = ast.unparse(owners)
    assert "scratch[index, thread] = weight" in rewritten
    assert "state[3, thread] = output_normalization_factor(" in rewritten
    assert ast.unparse(published) == group_barrier
    # The group's thread reads the counter where the group looks, and
    # comes to the same helper for a part to post alone where the spare
    # warp looks: that form states the post and no look.
    arrival = source_function("split_arrival", helpers)
    (forms,) = [
        node
        for node in arrival.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.const_expr(posts)"
    ]
    assert not source_calls(ast.Module(forms.body, []), "_count_arrival")
    assert len(source_calls(ast.Module(forms.orelse, []), "_count_arrival")) == 1
    # What the group branches on around the exchange is one value for its
    # threads, and marked so where the spare warp looks: an arrival or a
    # meeting under a condition the assembler takes for a thread's own
    # would cost every reduction behind it a warp synchronization.
    uniform = source_function("_group_uniform", template)
    assert [ast.unparse(node) for node in uniform.body[1:]] == [
        "word = cutlass.Int32(0)",
        "if condition:\n    word = cutlass.Int32(1)",
        "return cute.arch.make_warp_uniform(word) != 0",
    ]
    marks = sorted(
        source_calls(compute, "_group_uniform"), key=lambda call: call.lineno
    )
    marked = [ast.unparse(call.args[0]) for call in marks]
    assert marked == [
        "step0 + count == unit_pages",
        "closing & (count != unit_pages)",
        "unit == cutlass.Int32(plan[PLAN_FIRST])",
        "split",
        "cutlass.Int32(walk[WALK_STEP]) + cutlass.Int32(count) == planned_pages",
        "looked",
    ]
    # Each of them under the spare warp's look alone: the kernel whose
    # group looks is the kernel it was.
    for call in source_calls(compute, "_group_uniform"):
        assert any(
            isinstance(node, ast.If)
            and ast.unparse(node.test) == "cutlass.const_expr(look_stage is not None)"
            and call in ast.walk(ast.Module(body=node.body, type_ignores=[]))
            for node in ast.walk(compute)
        ), ast.unparse(call)
    # The staged pairs reach the row owners through the reduction, and
    # where they are staged the other part's loads wait for the launch's
    # dependencies there, those of the columns the warp did not stage as
    # well (``test_the_staged_panel_...``): the wait is compiled where one
    # is left.
    (paired,) = source_calls(reduction, "pair_partial")
    assert [ast.unparse(argument) for argument in paired.args[-2:]] == [
        "look_stage",
        "capture_pdl_wait_mode",
    ]
    partial = source_function("pair_partial", helpers)
    assert [argument.arg for argument in partial.args.args[-2:]] == [
        "staged",
        "capture_pdl_wait_mode",
    ]
    (waited,) = [
        node
        for node in partial.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test)
        == "cutlass.const_expr(staged is not None and held < query_columns // 4)"
    ]
    assert not waited.orelse
    assert [ast.unparse(node) for node in waited.body] == [WAIT]
    (source,) = [
        node
        for node in ast.walk(partial)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.const_expr(staged is not None)"
    ]
    assert "staged[pair]" in ast.unparse(source.body)
    assert "_load_pair" in ast.unparse(source.orelse)
    assert "_load_pair" not in ast.unparse(source.body)


def test_the_staged_panel_stands_behind_the_conversions_and_ahead_of_the_fold():
    """Read from the table and the device code. Where the spare warp looks
    it also stages the other part's panel of a complete unit in two parts,
    its leading columns, in the raw K ring: one raw tile a head, which adds
    no byte, behind two barriers that lie in the padding of the barrier
    block.

    - The K conversion warps arrive at the first once each, behind their
      loop over the stream's tiles, and only where the body has the look.
    - The spare warp's lane 0, behind the warp's answer, waits for the
      launch's dependencies and for that barrier's one phase before it
      issues the copies, which it counts from the answer it kept; they
      complete on the second barrier.
    - The lowest part's group waits for that barrier's one phase in front
      of its fold, behind its one load of its own panel, and reads the
      stage nowhere else; the columns the stage does not hold are loaded
      and weighted as where the group looks.
    - Nothing of it is stated for a body without the look."""
    # One raw tile holds eight quads of columns for the group's threads.
    assert (RAW_TILE_BYTES, PANEL_QUAD_BYTES) == (16384, 2048)
    assert [partition_look_panel_columns(columns) for columns in (8, 32, 40)] == [
        8,
        32,
        32,
    ]
    names = (PARTITION_LOOK_KEYS_CONVERTED, PARTITION_LOOK_PANEL_STAGED)
    assert names == ("keys_converted", "panel_staged")
    # The two barriers follow the query stages' in the block: four arrivals
    # for the four K conversion warps, one for the copies' issuer.
    for stages in (1, 2):
        for ring in (2, 3, ValueRingTiles(5)):
            plain = pipeline_barriers(False, stages, False, ring)
            assert pipeline_barriers(False, stages, False, ring, False, True) == (
                *plain,
                ("keys_converted", 1, 4),
                ("panel_staged", 1, 1),
            )
            offsets, count, rounds = pipeline_barrier_plan(
                False, stages, False, ring, False, True
            )
            bare, bare_count, bare_rounds = pipeline_barrier_plan(
                False, stages, False, ring
            )
            # In the padding: the block is as long as the table counts it,
            # and every other barrier lies where it lay.
            assert count == bare_count == 64
            assert {name: offsets[name] for name in bare} == bare
            used = sum(width for _name, width, _arrivals in plain)
            assert [offsets[name] for name in names] == [used, used + 1]
            assert used + 2 <= count
            # The loader's lanes initialize it with the block: the slot of
            # the first of the two starts at four arrivals, the second at
            # the one arrival a slot of the padding had, and no other slot
            # changes.

            def arrivals(block) -> list[int]:
                return [
                    1 + sum(extra for extra, lanes in terms if (lanes >> lane) & 1)
                    for terms in block
                    for lane in range(32)
                ]

            expected = arrivals(bare_rounds)
            assert expected[used : used + 2] == [1, 1]
            expected[used] = 4
            assert arrivals(rounds) == expected
    with pytest.raises(AssertionError):
        pipeline_barriers(False, 0, False, 2, False, True)
    body = source_function("paged_attention_paired_body", template)
    (plan,) = source_calls(body, "pipeline_barrier_plan")
    assert ast.unparse(plan.args[-1]) == "spare_look"
    # The body names the stage's barriers under the look alone.
    stage_names = ("keys_converted", "panel_staged")
    assigned = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) in stage_names
    ]
    assert len(assigned) == 4
    (guard,) = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.const_expr(spare_look)"
        and any(node_ in ast.walk(node) for node_ in assigned)
    ]
    inside = [node for node in assigned if node in ast.walk(guard)]
    assert sorted(ast.unparse(node) for node in inside) == [
        "keys_converted = barriers + barrier_offsets[PARTITION_LOOK_KEYS_CONVERTED]",
        "panel_staged = barriers + barrier_offsets[PARTITION_LOOK_PANEL_STAGED]",
    ]
    assert sorted(ast.unparse(node) for node in assigned if node not in inside) == [
        "keys_converted = None",
        "panel_staged = None",
    ]
    # The K conversions: the arrival is the role's last statement, behind
    # the loop over the stream's tiles, in lane 0 behind the warp's join of
    # the last tile; a body of one unit per CTA passes no barrier.
    convert = source_function("_convert_keys", template)
    loop, signal = convert.body[1:]
    assert isinstance(loop, ast.For) and "converted" not in ast.unparse(loop)
    assert ast.unparse(loop.iter) == "cutlass.range(_tile_steps(count, int32_steps))"
    assert ast.unparse(loop.body[-1].body[-2:]) == (
        "cute.arch.sync_warp()\n"
        "if lane == 0:\n"
        "    prims.mbarrier_arrive(key_ready + head)\n"
        "    prims.mbarrier_arrive(raw_k_free + ring)"
    )
    assert ast.unparse(signal) == (
        "if cutlass.const_expr(converted is not None):\n"
        "    assert converted is not None\n"
        "    if lane == 0:\n"
        "        prims.mbarrier_arrive(converted)"
    )
    assert convert.args.args[-1].arg == "converted"
    assert ast.unparse(convert.args.defaults[-1]) == "None"
    persistent, item = sorted(
        source_calls(body, "_convert_keys"), key=lambda call: call.lineno
    )
    assert ast.unparse(persistent.args[-1]) == "keys_converted"
    assert ast.unparse(item.args[-1]) == "int32_steps"
    # The spare warp: behind its look, whose answer it is handed.
    (looked,) = source_calls(body, "look_parts")
    (staged,) = source_calls(body, "stage_panels")
    (answer,) = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.Assign) and node.value is looked
    ]
    assert ast.unparse(answer.targets[0]) == "paired"
    assert staged.lineno > cast("int", looked.end_lineno)
    (look_guard,) = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.const_expr(spare_look)"
        and looked in ast.walk(node)
    ]
    assert staged in ast.walk(look_guard)
    assert [ast.unparse(argument) for argument in staged.args] == [
        "workspace",
        "raw_k",
        "plan",
        "paired",
        "keys_converted",
        "panel_staged",
        "lane",
        "capture_pdl_wait_mode",
        "partition_record_bytes",
        "query_columns",
    ]
    stage = source_function("stage_panels", helpers)
    statements = stage.body[1:]
    assert [ast.unparse(node) for node in statements[:3]] == [
        "size = partition_look_panel_columns(query_columns) // 4 * PANEL_QUAD_BYTES",
        "assert cutlass.const_expr(0 < size <= RAW_TILE_BYTES and size % 16 == 0)",
        "copies = cutlass.Int32(0)",
    ]
    counting, copying = statements[3:]
    assert ast.unparse(counting) == (
        "for head in cutlass.range_constexpr(2):\n"
        "    if paired[head]:\n"
        "        copies = copies + 1"
    )
    # One lane, and only with a copy to make: the dependency wait, the
    # conversions' end, the proxy fence, the expected bytes, the copies.
    assert ast.unparse(copying.test) == "lane == 0" and not copying.orelse
    (issuing,) = copying.body
    assert ast.unparse(issuing.test) == "copies != 0" and not issuing.orelse
    copy = (
        "cp_bulk_global_to_shared(slots + cute.assume(tile * RAW_TILE_BYTES, "
        "divby=RAW_TILE_BYTES), "
        "_partial_slot(workspace, cta + 1, cta, head, record_bytes), size, staged)"
    )
    assert [ast.unparse(node) for node in issuing.body] == [
        WAIT,
        "mbar_wait_sleep(converted, 0)",
        "cta = cutlass.Int32(cute.arch.block_idx()[0])",
        "cute.arch.fence_proxy('async.global')",
        "prims.mbarrier_arrive_expect_tx(staged, copies * size)",
        (
            "for head in cutlass.range_constexpr(2):\n"
            "    if paired[head]:\n"
            "        tile = stage_tile(plan, head)\n"
            f"        {copy}"
        ),
    ]
    # The tile is the head's of the ring's position behind the stream's
    # last step, which the plan's pages name: the slot the stream's next
    # step would take, not the one its last step took. The spare warp and
    # the group's view take it from the one rule.
    tile = source_function("stage_tile", helpers)
    assert ast.unparse(tile.body[-1]) == (
        "return 2 * (cutlass.Int32(plan[PLAN_PAGES]) & 1) + head"
    )
    for pages in range(1, 40):
        for head in range(2):
            rule = 2 * (pages & 1) + head
            assert rule == template._ring_slot(pages, head, PAIRED_DEFAULT_VALUE_RING)
            assert rule != template._ring_slot(
                pages - 1, head, PAIRED_DEFAULT_VALUE_RING
            )
    assert len(source_calls(stage, "stage_tile")) == 1
    assert len(source_calls(body, "stage_tile")) == 1
    # It reads no record and meets nobody at a named barrier; its one wait
    # on a barrier is for this CTA's own K conversions.
    text = ast.unparse(stage)
    assert "walks" not in text and "WALK_" not in text
    assert not any(isinstance(node, ast.While) for node in ast.walk(stage))
    for name in ("barrier_cta_sync", "barrier_cta_arrive", "barrier", "mbar_wait"):
        assert not source_calls(stage, name), name
    assert len(source_calls(stage, "mbar_wait_sleep")) == 1
    # The group: the stage and its barrier reach the reduction through the
    # walk, which hands each on once.

    def uses(function: ast.FunctionDef, name: str) -> int:
        return sum(
            isinstance(node, ast.Name) and node.id == name
            for node in ast.walk(function)
        )

    # A group is handed its head's tile of the ring and the barrier of the
    # copies.
    (walked,) = source_calls(body, "_compute_units")
    assert [ast.unparse(argument) for argument in walked.args[-4:]] == [
        "look_stage",
        (
            "cute.make_tensor(cute.recast_ptr(raw_k, dtype=cutlass.Float32) + "
            "stage_tile(plan, head) * (RAW_TILE_BYTES // 4), "
            "cute.make_layout((RAW_TILE_BYTES // 4,))) if spare_look else None"
        ),
        "panel_staged",
        "store_policy",
    ]
    compute = source_function("_compute_units", template)
    for name in ("panel_stage", "panel_staged"):
        assert uses(compute, name) == 1, name
    reduction = source_function("_reduce_part", template)
    asserted = {
        ast.unparse(node.test)
        for node in reduction.body
        if isinstance(node, ast.Assert)
    }
    assert {
        "cutlass.const_expr((look_stage is None) == (panel_stage is None))",
        "cutlass.const_expr((panel_stage is None) == (panel_staged is None))",
    } <= asserted
    # Ahead of the drain a pair weights what it loaded: every column where
    # the group looks, the columns behind the stage's where the warp does.
    drain = next(
        index
        for index, node in enumerate(reduction.body)
        if source_calls(node, "_drain_last_pv")
    )
    first_term = reduction.body[drain - 1]
    assert ast.unparse(first_term) == (
        "if pair:\n"
        "    if cutlass.const_expr(panel_stage is None):\n"
        "        merge_pair_other(scratch, panel, query_columns)\n"
        "    else:\n"
        "        merge_pair_loaded(scratch, panel, query_columns)"
    )
    # Behind it, with the stage: one load of the range's own panel, and for
    # a pair the wait for the copies right in front of the fold that reads
    # them. Without it: the fold of the group's look, which the kernel
    # whose group looks keeps.
    fold = reduction.body[drain + 1]
    assert isinstance(fold, ast.If)
    assert ast.unparse(fold.test) == "cutlass.const_expr(panel_stage is not None)"
    assert [ast.unparse(node) for node in fold.body] == [
        "assert panel_stage is not None and panel_staged is not None",
        "own = load_query_panel(output_address, query_columns)",
        "prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)",
        (
            "if pair:\n"
            "    _wait_non_mma(panel_staged, 0, sleep_waits)\n"
            "    merge_pair_staged(scratch, panel, own, panel_stage, thread, "
            "query_columns)\n"
            "else:\n"
            "    for query in cutlass.range_constexpr(query_columns):\n"
            "        panel[query] = cutlass.Float32(own[query])"
        ),
    ]
    (chunked,) = fold.orelse
    assert isinstance(chunked, ast.If) and ast.unparse(chunked.test) == "pair"
    assert source_calls(chunked, "merge_pair_own") and len(chunked.orelse) == 3
    assert "panel_stage" not in ast.unparse(chunked)
    # The barrier is named in the asserts and at the wait, the stage in
    # the asserts, the two conditions and at the fold.
    assert (uses(reduction, "panel_staged"), uses(reduction, "panel_stage")) == (3, 6)
    # The fold reads the stage where it weights a value, two words a step,
    # at the thread's quad of the column pair: the workspace's layout.
    staged_fold = source_function("merge_pair_staged", helpers)
    reads = [
        ast.unparse(node)
        for node in ast.walk(staged_fold)
        if isinstance(node, ast.Subscript) and ast.unparse(node.value) == "stage"
    ]
    assert reads == ["stage[word]", "stage[word + 1]"]
    assert "word = (query // 4 * 128 + thread) * 4 + query % 4" in ast.unparse(
        staged_fold
    )
    assert "if cutlass.const_expr(query < staged):" in ast.unparse(staged_fold)
    # The loaded columns' first term covers what the stage does not hold.
    loaded = source_function("merge_pair_loaded", helpers)
    (columns,) = [node for node in loaded.body if isinstance(node, ast.For)]
    assert ast.unparse(columns.iter) == (
        "cutlass.range_constexpr(first // 2, query_columns // 2)"
    )
    assert "first = partition_look_panel_columns(query_columns)" in ast.unparse(loaded)


def test_the_spare_warp_leaves_the_words_of_the_groups_own_look():
    """Run on the host. What a group notes when its unit's lowest part
    starts, the template's own statement, is the count the group's own
    look hands the counter's reader (``split_arrival``'s ``highest -
    cta``): the unit's parts above this CTA by the partition's definition.
    And what the spare warp's lane leaves in the group's record behind its
    look (``look_parts``) is what the deciding thread leaves there where
    the group looks, for the same answer of the same reader: the part's
    role (to publish where the look finds a part missing; to merge, as a
    pair for a unit in two parts), this CTA as the unit's lowest, and the
    unit's parts. The role is the warp's one answer: its lanes stage the
    other part's rows where it is a pair's merge in the record of the
    group that asked, and nothing for the other group, whose record holds
    such a role left over. What it returns says the same, per head: a
    complete unit in two parts in the record of the group that asked,
    which is all it stages another part's panel for (``stage_panels``).
    Over every lowest part of balanced plans with and without a range
    skew: units in two parts whose last page slot ends a CTA's range, the
    eval's 320 requests of a 17-page window, a unit in more than a hundred
    parts, long units whose skew is applied whole, and units in three and
    four parts. Each of them is its CTA's last range: the unit goes on in
    the next CTA, so the CTA's slots end inside it, and the raw K ring
    the panel is staged in has no later range to serve."""
    typed = SimpleNamespace(
        Int32=int,
        Uint32=int,
        Int64=int,
        const_expr=lambda value: value,
        range_constexpr=range,
    )
    compute = source_function("_compute_units", template)
    (request,) = [
        node
        for node in ast.walk(compute)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "walk[WALK_LOOK]"
        and source_calls(node, "_slot_cta")
    ]
    noted = compile(ast.Module([request], []), "<request>", "exec")
    slot_cta = jit_host_body(helpers._slot_cta)
    arrival = jit_host_body(helpers.split_arrival)
    look = jit_host_body(helpers.look_parts)
    counted: list[tuple[int, int, int]] = []
    complete = [False]

    def count(_workspace, lowest, others, head, _bytes, arrive):
        assert not arrive
        counted.append((lowest, others, head))
        return complete[0]

    meetings: list[tuple[str, int, int]] = []
    words = (helpers.WALK_ROLE, helpers.WALK_LOWEST, helpers.WALK_PARTS)
    compared = 0
    above_zero = 0
    range_ends = 0
    most = 0
    skews = set()
    roles: dict[int, int] = {}
    counts: set[int] = set()
    with (
        patch.object(helpers, "cutlass", typed),
        patch.object(helpers, "_slot_boundary", jit_host_body(helpers._slot_boundary)),
        patch.object(helpers, "_slot_cta", slot_cta),
        patch.object(helpers, "_count_arrival", count),
        patch.object(helpers, "_partial_slot", jit_host_body(helpers._partial_slot)),
        patch.object(helpers, "_load_pair", lambda address: (0.0, 1.0)),
        patch.object(
            helpers,
            "prims",
            SimpleNamespace(
                barrier_cta_sync=lambda barrier, thread_count: meetings.append(
                    ("sync", barrier, thread_count)
                ),
                barrier_cta_arrive=lambda barrier, thread_count: meetings.append(
                    ("arrive", barrier, thread_count)
                ),
            ),
        ),
    ):
        for pages, grid in (
            ([35] * 8, 16),
            ([17] * 320, 148),
            ([8192, *[17] * 127], 148),
            ([211] * 320, 148),
            ([100] * 8, 24),
        ):
            for skew in (0, 8):
                headers = unit_headers(pages)
                slots = headers[-1]
                bound = plan_grid_bound(len(pages), max(pages))
                active = active_ctas(pages, grid, grid_bound=bound, workspace_ctas=grid)
                applied = applied_skew(slots, active, skew)
                skews.add(applied)
                plan = {
                    helpers.PLAN_SLOTS: slots,
                    helpers.PLAN_ACTIVE: active,
                    helpers.PLAN_SKEW: applied,
                }
                ctas = reference_partition(
                    pages,
                    grid,
                    "balanced",
                    grid_bound=bound,
                    workspace_ctas=grid,
                    skew=skew,
                )
                parts = split_units(ctas)
                counts.update(len(where) for where in parts.values())
                for cta, segments in enumerate(ctas):
                    device = SimpleNamespace(
                        arch=SimpleNamespace(
                            block_idx=lambda cta=cta: (cta, 0, 0),
                            griddepcontrol_wait=lambda: None,
                            sync_warp=lambda: None,
                        )
                    )
                    for segment in segments:
                        if segment.whole or not segment.tail:
                            continue
                        # A unit's lowest part is the last range its CTA
                        # runs.
                        assert segment is segments[-1], (pages[:2], grid, skew, cta)
                        header = headers[segment.unit]
                        walk: dict[int, int] = {}
                        exec(
                            noted,
                            {
                                "walk": walk,
                                "plan": plan,
                                "header": header,
                                "unit_pages": segment.pages,
                                "slot_bits": 32,
                                "skewed": bool(skew),
                                "cutlass": typed,
                                "cute": device,
                                "_slot_cta": slot_cta,
                                "WALK_LOOK": helpers.WALK_LOOK,
                                "PLAN_SLOTS": helpers.PLAN_SLOTS,
                                "PLAN_ACTIVE": helpers.PLAN_ACTIVE,
                                "PLAN_SKEW": helpers.PLAN_SKEW,
                            },
                        )
                        asked = walk[helpers.WALK_LOOK]
                        what = (pages[:2], grid, skew, cta, segment)
                        where = parts[segment.unit]
                        assert where[0] == cta and asked == len(where) - 1 > 0, what
                        assert asked == where[-1] - cta, what
                        # The two heads' groups ask alike: each lowest
                        # part takes one of them here, the other asks for
                        # nothing. The warp left no role in that group's
                        # record: it holds a pair's merge from elsewhere.
                        head = compared & 1
                        base = head * WALK_WORDS
                        idle = {
                            WALK_WORDS - base + helpers.WALK_LOOK: 0,
                            WALK_WORDS - base + helpers.WALK_ROLE: (
                                helpers.PART_MERGE_PAIR
                            ),
                        }
                        for answer in (False, True):
                            complete[0] = answer
                            counted.clear()
                            meetings.clear()
                            own: dict[int, int] = {}
                            walks = {base + helpers.WALK_LOOK: asked, **idle}
                            stage: dict[int, float] = {}
                            with patch.object(helpers, "cute", device):
                                arrival(
                                    0,
                                    own,
                                    {},
                                    plan,
                                    header,
                                    segment.pages,
                                    segment.step,
                                    segment.count,
                                    0,
                                    head,
                                    16,
                                    32,
                                    bool(skew),
                                )
                                # The lane that looks for this head, and
                                # one that looks for neither.
                                paired = look(0, walks, stage, head, 1, 16, 8)
                                before = dict(walks)
                                others = look(0, walks, stage, 2 + head, 1, 16, 8)
                            assert counted == [(cta, asked, head)] * 2, what
                            assert walks == before, what
                            # Every lane tells the same heads' pairs: the
                            # head that asked, if its unit is complete and
                            # in two parts, and never the other.
                            assert paired == others, what
                            assert paired == tuple(
                                each == head and answer and len(where) == 2
                                for each in range(2)
                            ), what
                            assert [walks[base + word] for word in words] == [
                                own[word] for word in words
                            ], what
                            assert own[helpers.WALK_LOWEST] == cta, what
                            assert own[helpers.WALK_PARTS] == len(where), what
                            role = own[helpers.WALK_ROLE]
                            assert role == (
                                helpers.PART_PUBLISH
                                if not answer
                                else helpers.PART_MERGE_PAIR
                                if len(where) == 2
                                else helpers.PART_MERGE
                            ), what
                            roles[role] = roles.get(role, 0) + 1
                            # Nothing is written or staged for the head
                            # that did not ask, and only a complete pair
                            # is staged: two lanes' rows of the head's 8.
                            assert set(walks) == {
                                *before,
                                *(base + word for word in words),
                            }
                            assert idle.items() <= walks.items(), what
                            assert bool(stage) == (answer and len(where) == 2)
                            assert {index // 16 for index in stage} <= {head}
                            assert len(stage) in (0, 4), what
                            # The warp answers the head that asked, behind
                            # both requests, and only that head.
                            assert (
                                meetings
                                == [
                                    *(
                                        ("sync", helpers.LOOK_BARRIER + each, 160)
                                        for each in range(2)
                                    ),
                                    ("arrive", helpers.MEETING_BARRIER + head, 160),
                                ]
                                * 2
                            ), what
                        compared += 1
                        above_zero += cta > 0
                        most = max(most, asked)
                        # The unit's last page slot ends its CTA's range:
                        # the next slot is another CTA's.
                        last = header + segment.pages
                        range_ends += planner_slot_cta(
                            last + 1, slots, active, applied, 32
                        ) != planner_slot_cta(last, slots, active, applied, 32)
    # The cases tell the request from its neighbours: the highest CTA
    # itself where the lowest is not CTA 0, the owner of the next slot
    # where the unit ends a range (each unit of the first plan), and a
    # count above one; with every applied skew from none to the whole.
    assert compared > 600 and above_zero > 600 and range_ends >= 16 and most > 100
    assert skews == {0, 1, 2, 8}
    # Every role the look can leave was left: a publish for every part, a
    # pair's merge for the units in two parts and a merge for those in more.
    assert roles[helpers.PART_PUBLISH] == compared
    assert roles[helpers.PART_MERGE_PAIR] > 400 and roles[helpers.PART_MERGE] >= 10
    assert {2, 3, 4} <= counts and max(counts) > 100
    assert set(roles) == {
        helpers.PART_PUBLISH,
        helpers.PART_MERGE,
        helpers.PART_MERGE_PAIR,
    }


def test_every_global_write_of_the_compiled_kernel_is_behind_a_wait(tmp_path, looks):
    """Compiled without a device, with the look of the group and of the
    spare warp: the spare warp's kernel has the look's named barriers, the
    relaxed load of the counter in the spare warp's code, and no path from
    the kernel's entry reaches a write to global memory without a
    dependency wait, in either wait mode. The late merge of a posted unit,
    which a group that met the spare warp behind its lowest part reaches
    without the post's reply, waits itself. Behind its answers the spare
    warp stages the other part's panel: a wait for the dependencies, a
    suspended wait for the K conversions' end, the proxy fence, the bytes
    it tells the copies' barrier to expect and one bulk copy from global
    memory per head; the kernel whose group looks has none of these."""
    bound, args, sources = looks.bound, looks.args, looks.sources
    assert "partition_look" not in body_keywords(sources["group"])
    keywords = body_keywords(sources["spare"])
    assert keywords["partition_look"] == "'spare'"
    assert keywords["shared_memory_capacity"] == str(CAPACITY)
    kernels = ("group", "spare", "wide", "joined")
    compiled = {name: looks.compiled(name) for name in kernels}

    def named(ptx: str) -> dict[str, int]:
        """The barrier instructions of a group and the spare warp, 160
        threads each, by kind and barrier; a group's barrier is its head's
        and so a register."""
        found: dict[str, int] = {}
        for kind, barrier in re.findall(
            r"barrier\.(arrive|sync)\s+(%r\d+|\d+), 160;", ptx
        ):
            name = f"{kind} {'head' if barrier.startswith('%') else barrier}"
            found[name] = found.get(name, 0) + 1
        return found

    group, spare = compiled["group"].ptx, compiled["spare"].ptx
    # Where the group looks: the post and its reply. A group arrives for
    # a part it posts or for none, and waits for the reply.
    assert named(group) == {
        "sync 8": 1,
        "sync 9": 1,
        "arrive 10": 1,
        "arrive 11": 1,
        "arrive head": 2,
        "sync head": 1,
    }
    # Where the spare warp looks: a request is one more arrival of a group
    # (at its lowest part's start or its walk's end; and its first range's
    # early post) and a wait of the spare warp; the answer is an arrival of
    # the warp and one more wait of the group.
    assert named(spare) == {
        **named(group),
        "sync 12": 1,
        "sync 13": 1,
        "arrive 14": 1,
        "arrive 15": 1,
        "arrive head": 5,
        "sync head": 2,
    }
    # The spare warp's code in the order the assembler was given it: each
    # post behind its barrier with the wait ahead of the arrival, then both
    # requests, the wait, the one relaxed load with its fence and reset,
    # the warp's synchronization, a wait ahead of each head's staged
    # loads, and the answers; then the panels' stage: the wait, the
    # suspended wait for the conversions' end, the proxy fence, the bytes
    # the copies' barrier expects and the two copies.
    suspended = (
        r"mbarrier\.try_wait\.parity\.shared::cta\.b64 P1, \[%r\d+\], %r\d+, 10000000"
    )
    bulk = "cp.async.bulk.shared::cta.global"
    steps = re.findall(
        r"barrier\.(?:arrive|sync)\s+\d+, 160|griddepcontrol\.wait"
        r"|ld\.relaxed\.gpu\.global|atom\.add\.acq_rel\.gpu\.global"
        r"|fence\.acq_rel\.gpu|st\.global\.L2::cache_hint\.u32"
        r"|bar\.warp\.sync|ld\.global\.cg\.v2"
        rf"|{suspended}|fence\.proxy\.async\.global"
        rf"|mbarrier\.arrive\.expect_tx|{re.escape(bulk)}",
        spare[spare.rindex("barrier.sync \t8, 160") :],
    )
    post = [
        "griddepcontrol.wait",
        "fence.acq_rel.gpu",
        "atom.add.acq_rel.gpu.global",
        "st.global.L2::cache_hint.u32",
    ]
    assert [re.sub(r"%r\d+", "%r", re.sub(r"\s+", " ", step)) for step in steps] == [
        "barrier.sync 8, 160",
        *post,
        "barrier.arrive 10, 160",
        "barrier.sync 9, 160",
        *post,
        "barrier.arrive 11, 160",
        "barrier.sync 12, 160",
        "barrier.sync 13, 160",
        "griddepcontrol.wait",
        "ld.relaxed.gpu.global",
        "fence.acq_rel.gpu",
        "st.global.L2::cache_hint.u32",
        "bar.warp.sync",
        "griddepcontrol.wait",
        "ld.global.cg.v2",
        "griddepcontrol.wait",
        "ld.global.cg.v2",
        "barrier.arrive 14, 160",
        "barrier.arrive 15, 160",
        "griddepcontrol.wait",
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%r], %r, 10000000",
        "fence.proxy.async.global",
        "mbarrier.arrive.expect_tx",
        bulk,
        bulk,
    ]
    # The stage is the look's: the kernel whose group looks copies nothing
    # from global memory, has no fence for such a copy and no suspended
    # wait, and its K conversions arrive nowhere behind their tiles (the
    # look adds one arrival, one expected transfer and two waits, the
    # warp's for the conversions and the group's for the copies).
    assert (group.count(bulk), spare.count(bulk)) == (0, 2)
    assert "fence.proxy.async.global" not in group
    assert not re.search(suspended, group) and len(re.findall(suspended, spare)) == 1
    for instruction, added in (
        ("mbarrier.arrive.shared", 1),
        ("mbarrier.arrive.expect_tx", 1),
        ("mbarrier.try_wait.parity", 2),
        ("mbarrier.init", 0),
    ):
        assert spare.count(instruction) == group.count(instruction) + added, instruction
    for name, kernel in compiled.items():
        for mode_nonzero in (False, True):
            writes, _reads = global_accesses_ahead_of_a_wait(
                kernel.ptx, mode_nonzero=mode_nonzero
            )
            assert not writes, (name, mode_nonzero, writes[:3])
    # The exchange's conditions are marked warp-uniform where the spare
    # warp looks: the kernel has no more warp synchronizations than the
    # one whose group looks (with the conditions left to the assembler,
    # under two more meetings, it has several times as many).
    if all(kernel.sass is not None for kernel in compiled.values()):
        synchronized = {
            name: len(re.findall(r"\bWARPSYNC\b", cast("str", kernel.sass)))
            for name, kernel in compiled.items()
        }
        assert synchronized["spare"] <= synchronized["group"], synchronized
    # The check has teeth: without its waits the kernel writes ahead.
    unguarded = spare.replace("griddepcontrol.wait;", "")
    writes, _reads = global_accesses_ahead_of_a_wait(unguarded, mode_nonzero=True)
    assert writes
    # The body refuses the spare warp's look beside another partition and
    # on a table its capacity does not hold.
    with pytest.raises(AssertionError):
        compile_offline(
            bound,
            args,
            sources["spare"].replace(
                f"shared_memory_capacity={CAPACITY}", "shared_memory_capacity=100000"
            ),
            tmp_path / "short",
        )
    whole = stream_render(bound, stream_config(**WHOLE))
    assert ", stream_page_bound=" in whole
    with pytest.raises(AssertionError):
        compile_offline(
            bound,
            args,
            whole.replace(
                ", stream_page_bound=", ", partition_look='spare', stream_page_bound="
            ),
            tmp_path / "whole",
        )
    # A kernel without the look names no capacity on its account.
    assert "shared_memory_capacity" not in body_keywords(sources["group"])


def _shared_offsets(columns: int, units: int, query_stages: int) -> dict[str, int]:
    """Where the arrays of a balanced body's shared memory lie, with the
    spare warp's look: the body's table, each array at the next offset of
    its alignment, as the DSL's allocator places it."""
    arrays = pipeline_shared_memory(columns, False, query_stages, units, True)
    parts = pipeline_parts(True, False, units, True, True, partition_look=True)
    where: dict[str, int] = {}
    offset = 0
    for array in arrays:
        if array.guard is not None and array.guard not in parts:
            continue
        alignment = max(array.alignment, array.width)
        offset = -(-offset // alignment) * alignment
        where[array.name] = offset
        offset += array.width * array.count
    assert offset == pipeline_shared_bytes(arrays, parts)
    return where


def test_the_late_merge_starts_behind_a_barrier_of_its_group(looks):
    """Compiled without a device. A group reads the verdict of its post
    behind its walk and, if the posted part arrived last, merges that unit
    there: the merge's row owners store the unit's weights into the group's
    scratch rows, and its maximum, denominator and normalization into the
    row state, ahead of the merge's first barrier. Where the group looks,
    every group meets the spare warp ahead of the verdict, which joins its
    warps. Where the spare warp looks, a group that met the warp behind its
    lowest part does not, and other warps of it may still be in that part's
    epilogue, reading those rows: so in that kernel no path leads from a
    load of the verdict to one of those stores without a barrier of the
    group's 128 threads. With that barrier taken out of the PTX every one
    of the stores is reached."""
    spare, group = looks.compiled("spare").ptx, looks.compiled("group").ptx
    keywords = body_keywords(looks.sources["spare"])
    columns = int(keywords["query_columns"])
    where = _shared_offsets(columns, int(keywords["units"]), int(keywords["q_stages"]))
    # A row of the scratch rows or of the row state: its first column's
    # offset, to which a store adds its head and its thread in a register.
    rows = {
        where[array] + row * 4 * columns: f"{name} {row}"
        for array, name in (("scratch_raw", "scratch"), ("state_raw", "state"))
        for row in range(4)
    }
    # The verdict is the kind of a group's post, the first word of its
    # four: a group loads its own, the spare warp both.
    kind = where["mail_raw"] + 4 * helpers.MAIL_KIND
    heard = re.compile(
        rf"^ld\.shared\.b32\s+%r\d+, \[%r\d+\+({kind}|{kind + 4 * helpers.MAIL_WORDS})\];"
    )
    joined = re.compile(r"^bar\.sync\s+%r\d+, 128;")

    def stored(ptx: str) -> list[str]:
        found = []
        for text in instructions_reached_without(ptx, start=heard, stop=joined):
            store = re.match(r"st\.shared\.b32\s+\[%r\d+\+(\d+)\]", text)
            if store is not None and int(store.group(1)) in rows:
                found.append(rows[int(store.group(1))])
        return found

    late = [
        "scratch 0",
        "scratch 1",
        "scratch 2",
        "scratch 3",
        "state 0",
        "state 1",
        "state 3",
    ]
    assert stored(spare) == []
    # The group's own load of the verdict is the one compared with the
    # last arrival's, and the barrier is the first of the group behind it.
    lines = spare.splitlines()
    (verdict,) = [
        index
        for index, line in enumerate(lines)
        if heard.match(line.strip())
        and any(
            re.match(
                rf"setp\.(?:eq|ne)\.b32\s+%p\d+, "
                rf"{re.escape(line.split()[1].rstrip(','))}, {helpers.MAIL_LAST};",
                following.strip(),
            )
            for following in lines[index + 1 : index + 4]
        )
    ]
    barrier = next(
        index
        for index in range(verdict, len(lines))
        if joined.match(lines[index].strip())
    )
    without = "\n".join([*lines[:barrier], *lines[barrier + 1 :]])
    assert stored(without) == late
    # That barrier is the one instruction of its kind the look adds, and
    # the kernel whose group looks has none there: its paths to the same
    # stores come through the meeting ahead of the verdict.
    aligned = r"bar\.sync\s+%r\d+, 128;"
    assert len(re.findall(aligned, spare)) == len(re.findall(aligned, group)) + 1
    assert stored(group) == late
    met = re.compile(r"^barrier\.sync\s+%r\d+, 160;")
    walked = [
        text
        for text in (line.strip() for line in group.splitlines())
        if met.match(text) or heard.match(text)
    ]
    assert met.match(walked[0]) and heard.match(walked[1])


def _record_words(source: str) -> SimpleNamespace:
    """Where the words of a compute group's record lie in the shared
    memory of the balanced kernel ``source`` renders: the offset a group
    adds to its head's base, by word, and those of the words without a
    name (``unnamed``)."""
    keywords = body_keywords(source)
    where = _shared_offsets(
        int(keywords["query_columns"]),
        int(keywords["units"]),
        int(keywords["q_stages"]),
    )
    named = {
        getattr(helpers, name)
        for name in dir(helpers)
        if name.startswith("WALK_") and name != "WALK_WORDS"
    }
    return SimpleNamespace(
        **{
            name: where["walk_raw"] + 4 * getattr(helpers, f"WALK_{name.upper()}")
            for name in ("unit", "role", "lowest", "parts")
        },
        unnamed=[
            where["walk_raw"] + 4 * index
            for index in range(WALK_WORDS)
            if index not in named
        ],
    )


def test_a_lowest_part_reads_its_role_behind_the_meeting_and_waits_at_its_loads(
    looks,
):
    """Compiled without a device, where the spare warp looks. Behind a
    range's pages a group reloads its record and reduces the range; the
    first barrier of that reduction is the meeting with the spare warp for
    a unit's lowest part and the group's own for any other range, and no
    path leads from the reload to a load of the part's role without one of
    the two. With the meeting taken out of the PTX the role is read
    without a barrier: ahead of what the spare warp wrote. The group reads
    no counter, since no thread of it decides for a lowest part; the spare
    warp's lanes store the role, the lowest CTA and the parts behind their
    look and ahead of the warp's synchronization and its arrivals at the
    meetings, and the kernel touches no word of a record beside the named
    ones: the role is the warp's whole answer.

    And no path from the kernel's entry reaches an access to the
    workspace without a dependency wait: the spare warp's copies of
    another part's panel wait themselves, and so do the other part's loads
    of a pair, which are those of the columns the copies do not hold: none
    at eight query columns, the last eight of forty. The kernel whose
    group looks keeps the pair's loads reachable without a wait, on a path
    no launch takes (a range that is split where the reduction asks and
    not where the part's wait stands); without the wait in front of them
    the spare warp's kernel does too, and without the wait in front of
    its copies the copies."""
    spare, group = looks.compiled("spare").ptx, looks.compiled("group").ptx
    words = _record_words(looks.sources["spare"])
    assert words == _record_words(looks.sources["group"])

    def word(offset: int, kind: str) -> re.Pattern[str]:
        return re.compile(
            rf"^{kind}\.shared\.(?:v[24]\.)?b32\s.*\[%r\d+\+({offset}|{offset + 4 * WALK_WORDS})\]"
        )

    reloaded, role = word(words.unit, "ld"), word(words.role, "ld")
    barriers = re.compile(r"^(?:bar\.sync\s+%r\d+, 128|barrier\.sync\s+%r\d+, 160);")

    def unpublished(ptx: str) -> list[str]:
        return [
            text
            for text in instructions_reached_without(ptx, start=reloaded, stop=barriers)
            if role.match(text)
        ]

    lines = [line.strip() for line in spare.splitlines()]
    # The group loads the role in its reduction, behind its stores' wait
    # and behind its own arrival: nowhere ahead of a barrier.
    assert sum(bool(role.match(line)) for line in lines) >= 3
    assert unpublished(spare) == [] == unpublished(group)
    # The meeting is the first barrier with the spare warp behind the
    # reload, one of the two alternatives of a branch: without it a lowest
    # part reads its role unpublished.
    reload = next(index for index, line in enumerate(lines) if reloaded.match(line))
    met = re.compile(r"^barrier\.sync\s+%r\d+, 160;")
    meeting = next(
        index for index in range(reload, len(lines)) if met.match(lines[index])
    )
    without = "\n".join([*lines[:meeting], *lines[meeting + 1 :]])
    assert len(unpublished(without)) == 1
    # The other alternative is the group's barrier, for a range that is no
    # lowest part: without it such a range reads the role one thread of
    # the group wrote for a part to post, unpublished as well.
    joined = next(
        index
        for index in range(meeting, len(lines))
        if lines[index].startswith("bar.sync")
    )
    assert not any(role.match(line) for line in lines[meeting:joined])
    without = "\n".join([*lines[:joined], *lines[joined + 1 :]])
    assert len(unpublished(without)) == 1
    # No thread of the group decides for a lowest part: the kernel reads
    # the counter once, in the spare warp, where the kernel whose group
    # looks reads it in the group. And the warp leaves the group no word
    # beside the named ones: neither kernel loads or stores another.
    assert len(words.unnamed) == 4
    for ptx in (lines, [line.strip() for line in group.splitlines()]):
        assert not any(
            word(offset, kind).match(line)
            for offset in words.unnamed
            for kind in ("ld", "st")
            for line in ptx
        )
    counter = "ld.relaxed.gpu.global"
    warp = spare.rindex("barrier.sync \t8, 160")
    assert spare.count(counter) == 1 == group.count(counter)
    assert spare.index(counter) > warp > group.index(counter)
    # The spare warp's lane leaves the three words of its answer between
    # its look and the warp's synchronization, ahead of its arrivals.
    steps = [
        (index, line)
        for index, line in enumerate(spare[warp:].splitlines())
        if counter in line
        or "bar.warp.sync" in line
        or re.search(r"barrier\.arrive\s+1[45], 160", line)
    ]
    assert [line.split()[0] for _index, line in steps] == [
        "ld.relaxed.gpu.global.u32",
        "bar.warp.sync",
        "barrier.arrive",
        "barrier.arrive",
    ]
    looked, synchronized = steps[0][0], steps[1][0]
    stored = {
        name: [
            index
            for index, line in enumerate(spare[warp:].splitlines())
            if word(getattr(words, name), "st").match(line.strip())
        ]
        for name in ("role", "parts")
    }
    for name, where in stored.items():
        assert len(where) == 1 and looked < where[0] < synchronized, (name, where)
    # The role and the lowest CTA are neighbours, stored as one pair.
    assert "st.shared.v2.b32" in spare[warp:].splitlines()[stored["role"][0]]
    # The workspace: its loads from L2, the counter's, and the spare
    # warp's bulk copies from it.
    bulk = "cp.async.bulk.shared::cta.global"
    workspace = re.compile(
        rf"ld\.global\.cg\.|ld\.relaxed\.gpu\.global|{re.escape(bulk)}"
    )

    def touched(ptx: str) -> list[str]:
        _writes, reads = global_accesses_ahead_of_a_wait(ptx, mode_nonzero=True)
        return [text for text in reads if workspace.search(text)]

    assert touched(spare) == []
    assert {text.split()[0] for text in touched(group)} == {
        "ld.global.cg.v4.f32",
        "ld.global.cg.v2.f32",
    }
    # The copies stand behind a wait of their own: the last one in front
    # of them, with nothing of the workspace between.
    copies = [index for index, line in enumerate(lines) if line.startswith(bulk)]
    assert len(copies) == 2
    waited = max(
        index
        for index in range(copies[0])
        if lines[index].startswith("griddepcontrol.wait")
    )
    assert not any(workspace.search(line) for line in lines[waited : copies[0]])
    unguarded = "\n".join([*lines[:waited], *lines[waited + 1 :]])
    assert {text.split()[0] for text in touched(unguarded)} == {
        f"{bulk}.mbarrier::complete_tx::bytes"
    }
    # At eight query columns the stage holds the whole panel: a pair's
    # two quads, which the kernel whose group looks loads, are not loaded.
    quad = "ld.global.cg.v4.f32"
    assert spare.count(quad) == group.count(quad) - 2
    # At forty it holds thirty-two: the pair's loads of the last eight
    # columns are the first of their kind behind the meeting, two quads,
    # with the wait right in front of them.
    wide = [line.strip() for line in looks.compiled("wide").ptx.splitlines()]
    assert body_keywords(looks.sources["wide"])["query_columns"] == "40"
    assert touched("\n".join(wide)) == []
    reloaded = word(_record_words(looks.sources["wide"]).unit, "ld")
    reload = next(index for index, line in enumerate(wide) if reloaded.match(line))
    meeting = next(
        index for index in range(reload, len(wide)) if met.match(wide[index])
    )
    loads = [
        index for index in range(meeting, len(wide)) if wide[index].startswith(quad)
    ]
    waited = max(
        index
        for index in range(meeting, loads[0])
        if wide[index].startswith("griddepcontrol.wait")
    )
    assert not any(
        barriers.match(line) or line.startswith("ld.global")
        for line in wide[waited : loads[0]]
    )
    joined = next(
        index
        for index in range(loads[0], len(wide))
        if wide[index].startswith("bar.sync")
    )
    assert sum(line.startswith(quad) for line in wide[loads[0] : joined]) == 2
    unguarded = "\n".join([*wide[:waited], *wide[waited + 1 :]])
    assert {text.split()[0] for text in touched(unguarded)} == {quad}


def test_the_compiled_kernel_takes_and_reads_the_stage_behind_its_barriers(looks):
    """Compiled without a device, where the spare warp looks, at 8 and at
    40 query columns. The two barriers of the staged panel lie where the
    barrier plan puts them, in the padding of the block. A K conversion
    warp arrives at the first in one instruction, from behind which no
    path leads back to a tile's conversion. The spare warp's wait for that
    barrier is the kernel's one suspended wait, and no path from the
    kernel's entry reaches a copy from global memory without it; with it
    taken out of the PTX both copies are reached. Each head's copy goes to
    the ring position after the stream's last step, raw K tile
    ``2 * (pages & 1) + head``, and the copies complete on the second
    barrier, which their issuer told to expect them. A compute group waits
    for that barrier's phase zero once, behind its panel load with no load
    from shared memory between the two, and behind that wait each thread
    loads its quads of the stage, 2,048 bytes apart: two at 8 columns,
    eight at 40. The same holds in a kernel whose threads all join at the
    body's end, at named barrier 0, which a config without the early
    retirement of the allocation has: the stage needs no named barrier."""
    bulk = "cp.async.bulk.shared::cta.global"
    never = re.compile(r"(?!)")
    entry = re.compile(r"^ld\.param")
    suspended = re.compile(
        r"^mbarrier\.try_wait\.parity\.shared::cta\.b64 P1, \[(\S+)\], %r\d+, 10000000;"
    )
    join = re.compile(r"^\s*bar\.sync\s+0;", re.MULTILINE)
    for name, columns in (("spare", 8), ("wide", 40), ("joined", 8)):
        ptx = looks.compiled(name).ptx
        keywords = body_keywords(looks.sources[name])
        assert int(keywords["query_columns"]) == columns
        # The join of all threads is the kernel's one use of named
        # barrier 0, where the allocation is not retired early.
        assert ("early_tmem_retirement" not in keywords) is (name == "joined")
        assert len(join.findall(ptx)) == (name == "joined")
        stages = int(keywords["q_stages"])
        block = _shared_offsets(columns, int(keywords["units"]), stages)["barriers"]
        offsets, _count, _rounds = pipeline_barrier_plan(
            False, stages, False, 2, False, True
        )
        converted = block + 8 * offsets[PARTITION_LOOK_KEYS_CONVERTED]
        staged = block + 8 * offsets[PARTITION_LOOK_PANEL_STAGED]
        pages = (
            _shared_offsets(columns, int(keywords["units"]), stages)["plan"]
            + 4 * helpers.PLAN_PAGES
        )
        lines = [line.strip() for line in ptx.splitlines()]
        copies = [index for index, line in enumerate(lines) if line.startswith(bulk)]
        assert len(copies) == 2
        operands = [re.findall(r"\[([^\]]+)\]", lines[index]) for index in copies]
        # The base of the CTA's shared memory: what the copies' barrier is
        # an offset of.
        (barrier,) = {barrier for _target, _source, barrier in operands}
        base, _plus, offset = barrier.partition("+")
        if not offset:
            ((base, offset),) = re.findall(
                rf"add\.s32\s+{re.escape(barrier)}, (%r\d+), (\d+);", ptx
            )
        assert int(offset) == staged

        def where(
            operand: str, stream: int = 0, ptx: str = ptx, base: str = base
        ) -> int | None:
            """The offset in shared memory an address operand names: the
            base plus a constant, or a register the kernel made of them
            and of the pages of the stream the plan states, ``stream``.
            Another register counts as zero: a thread's index in its
            group, or its group's head, of thread 0 of head 0."""
            register, _plus, offset = operand.partition("+")
            return made(register, stream, ptx, base) + int(offset or 0)

        def made(
            name: str, stream: int, ptx: str, base: str, pages: int = pages
        ) -> int:
            if name == base:
                return 0
            if re.fullmatch(r"-?\d+", name):
                return int(name)
            defined = re.findall(
                rf"^\s*(\S+)\s+{re.escape(name)}, ([^;]+);", ptx, re.MULTILINE
            )
            if len(defined) != 1:
                return 0
            ((operation, text),) = defined
            if operation == "ld.shared.b32":
                return stream if text == f"[{base}+{pages}]" else 0
            values = [made(operand, stream, ptx, base) for operand in text.split(", ")]
            if operation == "add.s32":
                return sum(values)
            if operation == "shl.b32":
                return values[0] << values[1]
            if operation == "and.b32":
                return values[0] & values[1]
            if operation == "or.b32":
                return values[0] | values[1]
            return 0

        # Each head's copy lands in its tile of the ring's position behind
        # the stream's last step, whatever the stream's pages.
        for stream in range(1, 6):
            assert [
                where(target, stream) for target, _source, _barrier in operands
            ] == [RAW_TILE_BYTES * (2 * (stream & 1) + head) for head in range(2)]
        # The K conversions' arrival, one lane's (a branch of the others
        # stands right in front of it), and nothing of a tile behind it.
        (arrival,) = [
            line
            for line in lines
            if (
                found := re.match(
                    r"mbarrier\.arrive\.shared\.b64\s+%rd\d+, \[(\S+)\];", line
                )
            )
            and where(found.group(1)) == converted
        ]
        assert re.match(r"@!?%p\d+ bra\s", lines[lines.index(arrival) - 1])
        behind = instructions_reached_without(
            ptx, start=re.compile(re.escape(arrival)), stop=never
        )
        assert not any(
            text.startswith(("tcgen05.st", "ld.shared", "mbarrier")) for text in behind
        )
        # The spare warp's wait for it, ahead of every copy.
        (waited,) = [index for index, line in enumerate(lines) if suspended.match(line)]
        found = suspended.match(lines[waited])
        assert found is not None and where(found.group(1)) == converted
        reached = instructions_reached_without(ptx, start=entry, stop=suspended)
        assert not any(text.startswith(bulk) for text in reached)
        without = "\n".join(
            line for line in ptx.splitlines() if not suspended.match(line.strip())
        )
        reached = instructions_reached_without(without, start=entry, stop=suspended)
        assert sum(text.startswith(bulk) for text in reached) == 2
        # The bytes the copies' barrier expects: told once, between the
        # wait and the copies.
        (told,) = [
            index
            for index, line in enumerate(lines)
            if (
                found := re.match(
                    r"mbarrier\.arrive\.expect_tx\S*\s+%rd\d+, \[(\S+)\], ", line
                )
            )
            and where(found.group(1)) == staged
        ]
        assert waited < told < copies[0]
        # The group's wait for the copies: one phase, zero.
        (wait,) = [
            index
            for index, line in enumerate(lines)
            if (
                found := re.match(
                    r"mbarrier\.try_wait\.parity\S*\s+%p\d+, \[(\S+)\], 0;", line
                )
            )
            and where(found.group(1)) == staged
        ]
        loaded = max(
            index for index in range(wait) if lines[index].startswith("tcgen05.ld")
        )
        assert not any(line.startswith("ld.shared") for line in lines[loaded:wait])
        # A loop on the wait, and the fold in the block behind it.
        assert lines[wait - 1].endswith(":") and lines[wait + 1].startswith("not.pred")
        assert lines[wait + 2].endswith(f"bra \t{lines[wait - 1][:-1]};")
        end = next(
            index for index in range(wait + 3, len(lines)) if lines[index].endswith(":")
        )
        quads: dict[str, list[int]] = {}
        for line in lines[wait + 3 : end]:
            load = re.match(
                r"ld\.shared\.v2\.b64\s+\{[^}]+\}, \[(%r\d+)(?:\+(\d+))?\];", line
            )
            if load and int(load.group(2) or 0) < RAW_TILE_BYTES:
                quads.setdefault(load.group(1), []).append(int(load.group(2) or 0))
        assert list(quads.values()) == [
            [
                quad * PANEL_QUAD_BYTES
                for quad in range(partition_look_panel_columns(columns) // 4)
            ]
        ]
        # They start at the tile the copy of the thread's head took: head
        # 0's here, of its thread 0.
        (view,) = quads
        for stream in range(1, 6):
            assert where(view, stream) == RAW_TILE_BYTES * 2 * (stream & 1)
    # The kernel whose group looks has neither barrier's instructions.
    group = looks.compiled("group").ptx
    assert bulk not in group and not any(
        suspended.match(line.strip()) for line in group.splitlines()
    )
    assert not re.search(r"mbarrier\.try_wait\.parity\S*\s+%p\d+, \[\S+\], 0;", group)


def _walk_statements() -> SimpleNamespace:
    """What a thread of a compute group runs from the second barrier of
    its last range's reduction to its end, as the template states it: for
    a unit in two parts the other part's first term and, behind the drain,
    the fold of its own part onto it (``_reduce_part``: both in the form
    of the group's look and in the streamed form of the spare warp's,
    which reads the other part's staged panel), the wait and the stores of
    the walk, and the block behind the walk with the late merge
    (``_compute_units``); and that block once more without the barrier the
    late merge starts with."""
    reduce = source_function("_reduce_part", template)
    drain = next(
        index
        for index, node in enumerate(reduce.body)
        if source_calls(node, "_drain_last_pv")
    )
    first, fold = reduce.body[drain - 1], reduce.body[drain + 1]
    assert isinstance(first, ast.If) and ast.unparse(first.test) == "pair"
    assert source_calls(first, "merge_pair_other")
    assert source_calls(first, "merge_pair_loaded")
    assert isinstance(fold, ast.If)
    assert ast.unparse(fold.test) == "cutlass.const_expr(panel_stage is not None)"
    assert source_calls(fold, "merge_pair_staged")
    assert source_calls(ast.Module(fold.orelse, []), "merge_pair_own")
    compute = source_function("_compute_units", template)
    (walk,) = [node for node in compute.body if isinstance(node, ast.While)]
    (waited,) = [node for node in walk.body if ast.unparse(node) == WAIT]
    (stores,) = [
        node
        for node in walk.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "stores"
    ]
    assert walk.body.index(waited) < walk.body.index(stores)
    (tail,) = [
        node
        for node in compute.body[compute.body.index(walk) + 1 :]
        if isinstance(node, ast.If) and source_calls(node, "merge_parts")
    ]
    (unjoined,) = ast.parse(ast.unparse(tail)).body
    (late,) = [
        node
        for node in ast.walk(unjoined)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.Int32(mail[MAIL_KIND]) == MAIL_LAST"
    ]
    looked = late.body[0]
    assert isinstance(looked, ast.If)
    (barrier,) = [
        node
        for node in looked.body
        if isinstance(node, ast.Expr) and source_calls(node, "barrier")
    ]
    looked.body.remove(barrier)

    def code(node: ast.stmt, name: str):
        return compile(ast.Module([node], []), f"<{name}>", "exec")

    return SimpleNamespace(
        first=code(first, "first"),
        fold=code(fold, "fold"),
        waited=code(waited, "wait"),
        stores=code(stores, "stores"),
        tail=code(tail, "tail"),
        unjoined=code(ast.fix_missing_locations(unjoined), "unjoined"),
    )


def _part(seed: int, columns: int, below=lambda row: 0) -> SimpleNamespace:
    """One CTA's part of a unit, as its pages leave it: the panel by row
    and thread, the rows' maxima (``below`` lowers a row's), and the rows'
    denominators by warp. Exact values: maxima are integers and everything
    else a small dyadic number, so every product and sum of a fold is
    exact."""

    def number(index: int) -> float:
        return 0.25 * (1 + ((seed * 31 + index) * 7919 + 13) % 23)

    sums = [[8.0 * number(900 + 5 * w + r) for r in range(columns)] for w in range(4)]
    return SimpleNamespace(
        panel=[[number(3 * r + 5 * t) for t in range(128)] for r in range(columns)],
        maximum=[float((seed + r) % 3 - below(r)) for r in range(columns)],
        sums=sums,
        denominator=[sum(sums[w][r] for w in range(4)) for r in range(columns)],
    )


def _fold(parts: list[SimpleNamespace], row: int) -> tuple[list[float], float, float]:
    """A unit's fold of ``parts`` for ``row``: every part's weight, two to
    its maximum less the unit's, the row's normalization and its LSE."""
    top = max(part.maximum[row] for part in parts)
    weights = [2.0 ** (part.maximum[row] - top) for part in parts]
    total = sum(
        part.denominator[row] * weight
        for part, weight in zip(parts, weights, strict=True)
    )
    return weights, 1.0 / total, top * math.log(2) + math.log(total)


def _folded(parts: list[SimpleNamespace], row: int, thread: int) -> tuple[float, float]:
    """What a unit of ``parts`` stores for a row and a thread, and the
    row's LSE."""
    weights, factor, log = _fold(parts, row)
    panel = sum(
        part.panel[row][thread] * weight
        for part, weight in zip(parts, weights, strict=True)
    )
    return panel * factor, log


def _host(
    cta: int,
    columns: int,
    barrier,
    waited,
    touch=lambda: None,
    fused=lambda value, weight, total: value * weight + total,
    number=float,
    **arch,
) -> SimpleNamespace:
    """What both host models run the template's and the device helpers'
    own bodies on, for CTA ``cta`` of a kernel of ``columns`` query
    columns: a ``cutlass`` of the host's types; a ``cute`` with the host's
    arithmetic, the model's ``barrier`` of a compute group, its dependency
    wait (``waited``), its fused multiply-add (``fused``: the host's
    double arithmetic unless a model rounds as the device does, on values
    of the type ``number``) and whatever else of ``cute.arch`` it models
    (``arch``); the bodies of the merge helpers, of the stores and of the
    group's conditions (``bodies``); a workspace by address, which a
    thread reads behind the model's ``touch``, with the slot of a part in
    it (``slot``) and a part as its CTA published it (``publish``); and
    what a group's stores fill, by tile, row and thread (``output``,
    ``lse``). ``replaced`` lists these as what the two modules get in
    place of their own (``_replaced``)."""
    panel_bytes, slot_bytes = helpers.partial_slot_bytes(columns)
    record_bytes = helpers.RECORD_HEADER_BYTES + 4 * slot_bytes
    workspace: dict[int, tuple[float, ...]] = {}
    output: dict[tuple[int, int, int], float] = {}
    lse: dict[tuple[int, int], float] = {}
    typed = SimpleNamespace(
        Int32=int,
        Int64=int,
        Float32=number,
        BFloat16=float,
        Boolean=bool,
        const_expr=lambda value: value,
        range_constexpr=range,
        range=lambda *bounds, **_unroll: range(*bounds),
        inttoptr=lambda address, _space, _dtype: address,
        Array=lambda _dtype, count, alignment: [0.0] * count,
    )
    device = SimpleNamespace(
        arch=SimpleNamespace(
            barrier=lambda barrier_id, number_of_threads: barrier(
                barrier_id, number_of_threads
            ),
            fma_packed_f32x2=lambda values, weights, totals: tuple(
                starmap(fused, zip(values, weights, totals, strict=True))
            ),
            fmax=lambda first, second, nan: max(first, second),
            make_warp_uniform=lambda value: value,
            block_idx=lambda: (cta, 0, 0),
            griddepcontrol_wait=waited,
            **arch,
        ),
        math=SimpleNamespace(
            exp2=lambda value, fastmath: 2.0**value,
            log=lambda value, fastmath: math.log(value),
        ),
        make_rmem_tensor=lambda shape, _dtype: [0.0] * shape[0],
        make_tensor=lambda pointer, _layout: pointer,
        assume=lambda value, divby: value,
    )
    slot = jit_host_body(helpers._partial_slot)
    merges = (
        "merge_pair_rows",
        "merge_pair_other",
        "merge_pair_own",
        "merge_pair_loaded",
        "merge_pair_staged",
        "merge_parts",
    )
    merge = {
        name: jit_host_body(getattr(helpers, name))
        for name in ("_fold_part", "_fused_sum", "_part_weight", *merges)
    }
    walked = {
        name: jit_host_body(getattr(template, name))
        for name in ("_compute_store", "_group_uniform")
    }

    def load(address: int) -> tuple[float, ...]:
        touch()
        return workspace[address]

    def publish(part: SimpleNamespace, where: int, lowest: int, head: int) -> None:
        # What ``store_part`` of CTA ``where`` leaves of its part of the
        # unit whose lowest CTA is ``lowest``.
        base = slot(0, where, lowest, head, record_bytes)
        for thread in range(128):
            for quad in range(columns // 4):
                workspace[base + (quad * 128 + thread) * 16] = tuple(
                    part.panel[quad * 4 + offset][thread] for offset in range(4)
                )
        for row in range(columns):
            workspace[base + panel_bytes + row * 8] = (
                part.maximum[row],
                part.denominator[row],
            )

    def stored_element(_output, tile, query, thread, value, _heads, _group, _layouts):
        output[tile, query, thread] = value

    def stored_row(_rows, tile, thread, value, _heads, _group, _layout):
        lse[tile, thread] = value

    both = (helpers, template)
    replaced = [
        *((module, "cutlass", typed) for module in both),
        *((module, "cute", device) for module in both),
        *(
            (
                module,
                "output_normalization_factor",
                lambda total, _reciprocal, output_scale_log2: 1.0 / total,
            )
            for module in both
        ),
        (helpers, "_partial_slot", slot),
        (helpers, "_load_pair", load),
        (helpers, "_load_quad", load),
        *((helpers, name, body) for name, body in merge.items()),
        *((template, name, merge[name]) for name in merges),
        *((template, name, body) for name, body in walked.items()),
        (template, "normalize_output", lambda value, factor, _r: value * factor),
        (template, "store_output_element", stored_element),
        (template, "store_row_element", stored_row),
    ]
    return SimpleNamespace(
        slot=slot,
        bodies={**merge, **walked},
        publish=publish,
        workspace=workspace,
        output=output,
        lse=lse,
        record_bytes=record_bytes,
        panel_bytes=panel_bytes,
        replaced=replaced,
    )


@contextlib.contextmanager
def _replaced(replaced: list[tuple[object, str, object]]):
    """The modules' attributes ``replaced`` names, for the block."""
    with contextlib.ExitStack() as patches:
        for module, name, value in replaced:
            patches.enter_context(patch.object(module, name, value))
        yield


def _group_run(
    statements: SimpleNamespace,
    *,
    look: str,
    looked: bool = True,
    joined: bool = True,
    slow: int | None = None,
    site: str = "fold",
    chunk: int = 0,
    columns: int = 40,
) -> SimpleNamespace:
    """One compute group of a balanced CTA from the second barrier of its
    last range's reduction to its end, on the host: 128 threads that each
    run the template's statements (``_walk_statements``) and the device
    helpers' own bodies (``_host``). The last range is the lowest part of
    a unit in two parts that merges the pair; the CTA's first range was a
    part it posted, which arrived last, so the group merges that unit
    behind its walk. A barrier of the group is a rendezvous of the 128
    threads.

    Where the spare warp looks the fold is the streamed one: the other
    part's leading columns lie in the head's stage, each thread's words of
    it as the workspace holds them, which a thread may read behind its wait
    for the copies alone; the columns behind them are in registers, as the
    part's loads left them.

    ``slow`` is a warp that stops at ``site`` (in its fold: behind the
    panel load of ``chunk`` where the group looks and at the wait for the
    copies, behind its one panel load, where the spare warp does; or at
    the wait ahead of its stores) until every other warp of the group
    stands at a barrier: the furthest they get without it. The lanes of a
    warp are in step, and so are the warps that are not slow, as the
    model's schedule has it: each of them has stored the pair's unit
    before one goes on. Returns what the group stored, by unit, row and
    thread, and the two units' parts in source order.
    """
    threads = 128
    cta, posted_item, last_item = 5, 10, 20
    # The pair of the last range (unit ``last_item``): the other CTA's part
    # and this CTA's, which weighs a half or a quarter. The posted unit
    # (``posted_item``): this CTA's part, its first steps, and the lowest
    # CTA's, which weighs one. No row's weights are those of the other unit.
    pair = [_part(3, columns), _part(6, columns, below=lambda row: 1 + row % 2)]
    posted = [_part(4, columns, below=lambda row: 2 + row % 2), _part(7, columns)]
    other, own = pair
    lowest = cta - 1
    barriers: dict[int, threading.Barrier] = {}
    me = threading.local()
    state: dict[tuple[int, int], float] = {}
    scratch: dict[tuple[int, int], float] = {}
    stalled = threading.Semaphore(0)
    released = threading.Event()
    in_step = {True: threading.Barrier(32), False: threading.Barrier(threads - 32)}
    errors: list[tuple[int, BaseException]] = []

    def meet(barrier: int, parties: int) -> None:
        # A barrier releases the threads it counts: one that counted
        # fewer than the group would let it go on without its last warp.
        barriers.setdefault(barrier, threading.Barrier(parties)).wait(timeout=60)

    def stall() -> None:
        if slow is not None and me.thread // 32 == slow:
            stalled.release()
            assert released.wait(timeout=60)

    spare = look == "spare"
    staged = partition_look_panel_columns(columns) if spare else 0

    class Stage:
        """The head's raw K tile behind the copies: a quad of the other
        part's leading columns per thread."""

        def __getitem__(self, word: int) -> float:
            quad, rest = divmod(word, 4 * threads)
            thread, offset = divmod(rest, 4)
            assert thread == me.thread and quad * 4 + offset < staged
            assert me.copies == 1, "the stage ahead of the wait for its copies"
            return other.panel[quad * 4 + offset][thread]

    def load(_form: str, address: int, num: int) -> list[float]:
        me.loads += 1
        return [own.panel[address + row][me.thread] for row in range(num)]

    def load_panel(_address: int, count: int) -> list[float]:
        me.loads += 1
        return [own.panel[row][me.thread] for row in range(count)]

    def loaded(kind: str) -> None:
        if site == "fold" and me.loads - 1 == chunk and not spare:
            stall()

    def copied(barrier: str, phase: int, _sleep: bool) -> None:
        # The copies landed pages ago: the wait returns at once.
        assert (barrier, phase, me.loads) == ("staged", 0, 1)
        me.copies += 1
        if site == "fold":
            stall()

    def waited() -> None:
        me.waits += 1
        if site == "stores" and me.waits == 1:
            stall()

    host = _host(cta, columns, meet, waited)
    # What the statements name beside the template's own names.
    names = {
        "prims": SimpleNamespace(
            tcgen05_ld=load,
            tcgen05_wait=loaded,
            Tcgen05Wait=SimpleNamespace(LOAD="load"),
            barrier_cta_arrive=lambda barrier, thread_count: None,
            # The spare warp's 32 lanes arrived at its reply long ago: the
            # meeting is the group's.
            barrier_cta_sync=lambda barrier, thread_count: meet(
                barrier, thread_count - 32
            ),
        ),
        "pair": True,
        "stores": True,
        "partition": "balanced",
        "no_units": False,
        "look_stage": [] if spare else None,
        "panel_stage": Stage() if spare else None,
        "panel_staged": "staged" if spare else None,
        "_wait_non_mma": copied,
        "load_query_panel": load_panel,
        "sleep_waits": False,
        "walk": {
            helpers.WALK_POSTED: 1,
            helpers.WALK_LOOKED: int(looked),
            helpers.WALK_LOOK: 1,
        },
        "mail": {
            helpers.MAIL_KIND: helpers.MAIL_LAST,
            helpers.MAIL_LOWEST: lowest,
            helpers.MAIL_PARTS: 2,
            helpers.MAIL_ITEM: posted_item,
        },
        "capture_pdl_wait_mode": 1,
        "pair_head": 0,
        "item": last_item,
        "state": state,
        "scratch": scratch,
        "output": host.output,
        "lse": host.lse,
        "workspace": 0,
        "output_address": 0,
        "record_bytes": host.record_bytes,
        "query_columns": columns,
        "query_rows": columns,
        "heads": 2,
        "group": 8,
        "fast_math": True,
        "base2_scores": True,
        "output_reciprocal": True,
        "output_scale_log2": 0,
        "io_layouts": None,
        "maximum_output": None,
        "denominator_output": None,
        "statistic_layouts": None,
        "store_policy": "default",
    }

    def run(thread: int) -> None:
        me.thread, me.loads, me.waits, me.copies = thread, 0, 0, 0
        # What the other part's loads left in registers: every column, or
        # the columns behind the stage's.
        panel = [0.0] * staged + [
            other.panel[row][thread] for row in range(staged, columns)
        ]
        local = {**vars(template), **names}
        local.update(thread=thread, panel=panel, result=panel)
        try:
            # Behind the reduction's second barrier: the other part's
            # term of what is in registers, then the fold: this part's
            # eight columns at a time onto it, or both terms streamed.
            exec(statements.first, local)
            exec(statements.fold, local)
            assert me.copies == int(spare) and me.loads == (
                1 if spare else columns // 8
            )
            exec(statements.waited, local)
            exec(statements.stores, local)
            if slow is not None:
                in_step[thread // 32 == slow].wait(timeout=60)
            exec(statements.tail if joined else statements.unjoined, local)
        except Exception as error:  # reported by the caller's assertion
            errors.append((thread, error))
            released.set()
            for barrier in (*in_step.values(), *barriers.values()):
                barrier.abort()

    with _replaced(host.replaced):
        # The workspace holds both parts of the posted unit: the CTA's own
        # in its first pair of slots, the lowest CTA's in that CTA's second.
        for part, source in enumerate(posted):
            host.publish(source, cta - part, lowest, 0)
        # Ahead of the reduction's second barrier the row owners left the
        # pair's weights and the unit's row state.
        for row in range(columns):
            state[0, row] = own.maximum[row]
            host.bodies["merge_pair_rows"](
                state,
                scratch,
                row,
                own.denominator[row],
                other.maximum[row],
                other.denominator[row],
                True,
                True,
                True,
                0,
            )
        group = [
            threading.Thread(target=run, args=(thread,)) for thread in range(threads)
        ]
        deadline = time.monotonic() + 60
        if slow is None:
            for member in group:
                member.start()
        else:
            late = [member for index, member in enumerate(group) if index // 32 == slow]
            for member in late:
                member.start()
            for _member in late:
                assert stalled.acquire(timeout=60), errors
            for index, member in enumerate(group):
                if index // 32 != slow:
                    member.start()
            while (
                sum(barrier.n_waiting for barrier in barriers.values()) < threads - 32
            ):
                assert time.monotonic() < deadline and not errors, errors
                time.sleep(0.0005)
            released.set()
        for member in group:
            member.join(timeout=60)
        assert not errors and not any(member.is_alive() for member in group), errors
    return SimpleNamespace(
        output=host.output,
        lse=host.lse,
        pair=pair,
        posted=posted,
        last_item=last_item,
        posted_item=posted_item,
        meetings=sorted(barriers),
    )


def test_a_warp_slow_in_its_epilogue_reads_the_rows_of_its_own_unit():
    """Run on the host: a compute group as 128 threads that run the
    template's own statements from the second barrier of the last range's
    reduction to the group's end, a pair's fold and stores and then the
    late merge of a posted unit, with one warp stopped in its epilogue
    until every other warp stands at a barrier.

    Where the spare warp looks and the group met it behind its lowest
    part, the others stand at the barrier the late merge starts with: the
    slow warp folds with its own unit's weights and normalizes with its
    rows, whichever warp it is and wherever it stopped. Without that
    barrier, the same statements less one, the others stand at the merge's
    first barrier, behind its row owners' stores, and the slow warp's 32
    columns are wrong in every row that another warp owns: normalized with
    the posted unit's rows and, if it stopped in its fold, folded with
    that unit's weights. The fold there is the streamed one, which weights
    each value of the other part where it reads it from the stage, behind
    the wait for the copies: both weights are the posted unit's in the
    rows the stage holds, and this part's weight in the rows behind them,
    whose other term the warp formed ahead of its last PV. A group that
    did not meet the warp, and every group where the group looks, meets it
    at the walk's end, as ever."""
    statements = _walk_statements()
    columns, threads = 40, 128
    free = _group_run(statements, look="spare")
    assert free.meetings == [1]
    other, own = free.pair
    staged = partition_look_panel_columns(columns)
    assert staged == 32

    def expected(row: int, thread: int, first=(), second=(), rows=()):
        """What the group stores for the pair's unit: the other part's
        term, then this part's. A row of ``rows`` is normalized with the
        posted unit's row; in a row of ``first`` the other part's value is
        weighted with that unit's first weight, and in one of ``second``
        this part's with its second."""
        weights, factor, _log = _fold(free.pair, row)
        late, late_factor, _log = _fold(free.posted, row)
        return (
            other.panel[row][thread] * (late if row in first else weights)[0]
            + own.panel[row][thread] * (late if row in second else weights)[1]
        ) * (late_factor if row in rows else factor)

    def check(run, wrong_warp=None, **wrong):
        assert len(run.output) == 2 * columns * threads
        for row in range(columns):
            for thread in range(threads):
                late = wrong if thread // 32 == wrong_warp else {}
                assert run.output[run.last_item, row, thread] == expected(
                    row, thread, **late
                ), (row, thread)
                posted, _log = _folded(free.posted, row, thread)
                assert run.output[run.posted_item, row, thread] == posted
        assert run.lse == free.lse and len(run.lse) == 2 * columns

    check(free)
    # The model tells the two units' rows apart in every row: both weights
    # and the normalizations.
    for row in range(columns):
        pair, late_unit = _fold(free.pair, row), _fold(free.posted, row)
        assert pair[0][0] != late_unit[0][0] and pair[0][1] != late_unit[0][1], row
        assert pair[1] != late_unit[1], row
    # With the barrier: any warp, stopped in its fold or ahead of its
    # stores, stores what the free group stores.
    for slow, site in ((3, "fold"), (1, "fold"), (2, "stores")):
        run = _group_run(statements, look="spare", slow=slow, site=site)
        assert run.meetings == [1]
        check(run)
    # Without it. A warp that owns no row: every row is another warp's.
    everything = range(columns)
    run = _group_run(statements, look="spare", joined=False, slow=3)
    check(run, wrong_warp=3, first=range(staged), second=everything, rows=everything)
    # Stopped ahead of its stores it folded rightly and normalizes wrongly.
    run = _group_run(statements, look="spare", joined=False, slow=2, site="stores")
    check(run, wrong_warp=2, rows=everything)
    # The second warp owns the last eight rows of forty, which it has not
    # rewritten itself: the other thirty-two are wrong.
    run = _group_run(statements, look="spare", joined=False, slow=1)
    check(run, wrong_warp=1, first=range(32), second=range(32), rows=range(32))
    # A group that did not meet the spare warp behind a lowest part, and
    # any group where the group looks, meets it at the post's reply: the
    # others stand there, with or without the late merge's barrier. Where
    # the group looks the fold is the one of eight columns at a time.
    reply = helpers.POSTED_BARRIER
    for look, joined in (("spare", True), ("spare", False), ("group", True)):
        run = _group_run(
            statements, look=look, looked=False, joined=joined, slow=3, chunk=2
        )
        assert run.meetings == [1, reply]
        check(run)


def _round32(exact: Fraction) -> float:
    """``exact``, which is not zero, rounded to the nearest float32, a tie
    to the even one."""
    magnitude = abs(exact)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    # A subnormal has the least normal's unit in the last place.
    unit = Fraction(2) ** (max(exponent, -126) - 23)
    steps, rest = divmod(magnitude, unit)
    if 2 * rest > unit or (2 * rest == unit and steps % 2):
        steps += 1
    rounded = steps * unit
    result = math.inf if rounded >= Fraction(2) ** 128 else float(rounded)
    return -result if exact < 0 else result


def _fma32(value: float, weight: float, total: float) -> float:
    """``fma.rn.f32`` of three float32 values: ``value * weight + total``,
    rounded once."""
    if math.isnan(value) or math.isnan(weight) or math.isnan(total):
        return math.nan
    if math.isinf(value) or math.isinf(weight):
        if value == 0 or weight == 0:
            return math.nan
        product = math.copysign(math.inf, value) * math.copysign(1.0, weight)
        return math.nan if math.isinf(total) and total != product else product
    if math.isinf(total):
        return total
    product = Fraction(value) * Fraction(weight)
    exact = product + Fraction(total)
    if exact != 0:
        return _round32(exact)
    # A sum of two zeros keeps the sign both have; a sum that cancels is
    # the positive zero.
    sign = math.copysign(1.0, value) * math.copysign(1.0, weight)
    if product == 0 and sign == math.copysign(1.0, total):
        return math.copysign(0.0, sign)
    return 0.0


def test_the_streamed_fold_gives_the_bits_of_the_fold_it_replaces():
    """Run on the host: one thread's fold of a unit in two parts by the
    template's own statements (``_reduce_part``), in the form of the
    group's look, the other part's panel in registers and this part's
    folded onto it eight columns at a time, and in the streamed form of
    the spare warp's look, the other part's leading columns read from the
    stage.

    With the fused operation left symbolic, every element of the unit's
    panel is one expression in both forms, this part's value times its
    weight onto the other part's value times its weight onto zero, at 8,
    32 and 40 query columns: the same operations on the same operands in
    the same order, whatever the values. And rounded as the device rounds
    (``fma.rn.f32``, exactly) both forms give the same bits over values
    that tell a wrong form: zeros of both signs, subnormals, the least
    normal and the largest value, plus and minus 2**120, infinities, a
    NaN, values one unit in the last place around a power of two whose
    products are ties, and weights from zero to one and around it. Folding
    the two terms in the other order, or without fusing, gives other bits
    on these."""
    statements = _walk_statements()
    threads = 128

    def folds(columns: int, fused, number):
        """The fold of one thread's panel in both forms: a function of the
        thread, the two parts' values by row and the two weight rows."""
        host = _host(0, columns, None, None, fused=fused, number=number)
        staged = partition_look_panel_columns(columns)

        def fold(thread: int, other, own, weights, streamed: bool):
            stage = {
                ((row // 4) * threads + thread) * 4 + row % 4: other[row]
                for row in range(staged)
            }
            # What the other part's loads left in registers.
            panel = [*([number(0)] * staged), *other[staged:]] if streamed else [*other]
            waits: list[tuple[object, int]] = []
            local = {
                **vars(template),
                "pair": True,
                "thread": thread,
                "panel": panel,
                "scratch": {
                    (part, row): weights[part][row]
                    for part in range(2)
                    for row in range(columns)
                },
                "query_columns": columns,
                "output_address": 0,
                "sleep_waits": False,
                "panel_stage": stage if streamed else None,
                "panel_staged": "staged" if streamed else None,
                "_wait_non_mma": lambda barrier, phase, _sleep: waits.append(
                    (barrier, phase)
                ),
                "load_query_panel": lambda _address, count: own[:count],
                "prims": SimpleNamespace(
                    tcgen05_ld=lambda _form, address, num: own[address : address + num],
                    tcgen05_wait=lambda kind: None,
                    Tcgen05Wait=SimpleNamespace(LOAD="load"),
                ),
            }
            exec(statements.first, local)
            exec(statements.fold, local)
            assert waits == ([("staged", 0)] if streamed else [])
            return panel

        return host, fold

    # Symbolically: one expression per element, in both forms.
    for columns in (8, 32, 40):
        host, fold = folds(
            columns,
            lambda value, weight, total: ("fma", value, weight, total),
            lambda value: value,
        )
        other = [("other", row) for row in range(columns)]
        own = [("own", row) for row in range(columns)]
        weights = [[("weight", part, row) for row in range(columns)] for part in (0, 1)]
        with _replaced(host.replaced):
            for thread in (0, 37, threads - 1):
                expected = [
                    (
                        "fma",
                        own[row],
                        weights[1][row],
                        ("fma", other[row], weights[0][row], 0),
                    )
                    for row in range(columns)
                ]
                assert fold(thread, other, own, weights, False) == expected
                assert fold(thread, other, own, weights, True) == expected
    # In float32, over every combination of these values and weights.
    one, last = 1.0, 2.0**-23
    largest = (2.0 - last) * 2.0**127
    values = [
        0.0,
        -0.0,
        one,
        -one,
        one + last,
        one - last / 2,
        1.5 + last,
        2.0**-149,
        -3 * 2.0**-149,
        2.0**-126,
        largest,
        2.0**120,
        -(2.0**120),
        math.inf,
        -math.inf,
        math.nan,
    ]
    factors = [0.0, one, 0.5, one - last / 2, one + last, 0.75, 2.0**-126, 2.0**-10]
    cases = [
        (other, own, first, second)
        for other in values
        for own in values
        for first in factors
        for second in factors
    ]
    columns = 40
    host, fold = folds(columns, _fma32, float)

    def bits(value: float) -> bytes:
        return struct.pack("<f", value)

    reordered = unfused = 0
    with _replaced(host.replaced):
        for start in range(0, len(cases), columns):
            rows = cases[start : start + columns]
            rows += cases[: columns - len(rows)]
            other, own, first, second = (
                list(column) for column in zip(*rows, strict=True)
            )
            thread = (start // columns) % threads
            weights = [first, second]
            plain = fold(thread, other, own, weights, False)
            streamed = fold(thread, other, own, weights, True)
            assert [bits(value) for value in streamed] == [
                bits(value) for value in plain
            ], start
            for row in range(columns):
                reference = _fma32(
                    own[row], second[row], _fma32(other[row], first[row], 0.0)
                )
                assert bits(plain[row]) == bits(reference)
                swapped = _fma32(
                    other[row], first[row], _fma32(own[row], second[row], 0.0)
                )
                reordered += bits(swapped) != bits(reference)
                separate = _fma32(
                    _fma32(own[row], second[row], 0.0),
                    1.0,
                    _fma32(other[row], first[row], 0.0),
                )
                unfused += bits(separate) != bits(reference)
    # The values tell the forms apart.
    assert len(cases) == 16384 and reordered > 300 and unfused > 200


class _NamedBarriers:
    """A CTA's named barriers on the host. A barrier completes when as many
    threads have arrived at it as its participants name, and then releases
    those that wait at it; an arrival does not wait. Everybody at one
    barrier names the same participants, or the meeting is none."""

    def __init__(self, timeout: float = 60.0) -> None:
        self.condition = threading.Condition()
        self.timeout = timeout
        self.named: dict[int, int] = {}
        self.arrived: dict[int, int] = {}
        self.waiting: dict[int, int] = {}
        self.rounds: dict[int, int] = {}
        self.watched = 0
        self.broken = False

    def _arrive(self, barrier: int, threads: int) -> None:
        named = self.named.setdefault(barrier, threads)
        assert named == threads, f"barrier {barrier}: {threads} among {named}"
        self.arrived[barrier] = self.arrived.get(barrier, 0) + 1
        if self.arrived[barrier] == threads:
            self.arrived[barrier] = 0
            self.rounds[barrier] = self.rounds.get(barrier, 0) + 1
            self.condition.notify_all()
        elif self.watched:
            self.condition.notify_all()

    def arrive(self, barrier: int, threads: int) -> None:
        with self.condition:
            self._arrive(barrier, threads)

    def sync(self, barrier: int, threads: int) -> None:
        with self.condition:
            before = self.rounds.get(barrier, 0)
            self.waiting[barrier] = self.waiting.get(barrier, 0) + 1
            self._arrive(barrier, threads)
            met = self.condition.wait_for(
                lambda: self.broken or self.rounds.get(barrier, 0) > before,
                self.timeout,
            )
            self.waiting[barrier] -= 1
            assert met and not self.broken, f"no meeting at barrier {barrier}"

    def until(self, barrier: int, threads: int) -> None:
        """Hold the caller until ``threads`` threads wait at ``barrier``."""
        with self.condition:
            self.watched += 1
            held = self.condition.wait_for(
                lambda: self.broken or self.waiting.get(barrier, 0) >= threads,
                self.timeout,
            )
            self.watched -= 1
            assert held and not self.broken, f"nobody came to barrier {barrier}"

    def abort(self) -> None:
        with self.condition:
            self.broken = True
            self.condition.notify_all()


class _Phases:
    """A CTA's barriers of shared memory that complete one phase in its
    life, on the host: the phase completes when the arrivals it was
    initialized with have come and the bytes those arrivals told it to
    expect have landed, and only then releases a thread that waits for
    it."""

    def __init__(self, arrivals: dict[str, int], timeout: float = 60.0) -> None:
        self.condition = threading.Condition()
        self.timeout = timeout
        self.left = dict(arrivals)
        self.expected = dict.fromkeys(arrivals, 0)
        self.waiting = dict.fromkeys(arrivals, 0)
        self.waited = dict.fromkeys(arrivals, 0)
        self.broken = False

    def complete(self, name: str) -> bool:
        return self.left[name] == 0 and self.expected[name] == 0

    def arrive(self, name: str, expect: int = 0) -> None:
        with self.condition:
            assert self.left[name] > 0, f"one arrival too many at {name}"
            self.expected[name] += expect
            self.left[name] -= 1
            self.condition.notify_all()

    def land(self, name: str, count: int) -> None:
        with self.condition:
            assert 0 < count <= self.expected[name], f"{count} bytes at {name}"
            self.expected[name] -= count
            self.condition.notify_all()

    def wait(self, name: str, phase: int) -> None:
        assert phase == 0, f"{name} has one phase"
        with self.condition:
            self.waiting[name] += 1
            self.waited[name] += 1
            self.condition.notify_all()
            met = self.condition.wait_for(
                lambda: self.broken or self.complete(name), self.timeout
            )
            self.waiting[name] -= 1
            assert met and not self.broken, f"the phase of {name} did not complete"

    def until(self, held) -> None:
        """Hold the caller until ``held`` says so of this object."""
        with self.condition:
            met = self.condition.wait_for(
                lambda: self.broken or held(self), self.timeout
            )
            assert met and not self.broken, "held for nothing"

    def abort(self) -> None:
        with self.condition:
            self.broken = True
            self.condition.notify_all()


class _Pointer:
    """A pointer into a CTA's shared memory on the host: an array's name
    and an offset in it."""

    def __init__(self, name: str, offset: int = 0) -> None:
        self.name = name
        self.offset = offset

    def __add__(self, offset: int) -> _Pointer:
        return _Pointer(self.name, self.offset + offset)

    def align(self, _bytes: int) -> _Pointer:
        return self


class _Words:
    """One head's words of an array both heads share."""

    def __init__(self, words: dict[int, int], base: int) -> None:
        self.words = words
        self.base = base

    def __getitem__(self, index: int) -> int:
        return self.words[self.base + index]

    def __setitem__(self, index: int, value: int) -> None:
        self.words[self.base + index] = value


class _Answer:
    """What the spare warp's lanes store for the two compute groups in an
    array of ``width`` words a head: the groups' records, or the rows the
    warp stages. A group reads them behind its meeting with the warp, which
    completes with the last arrival of its 160 threads: a lane's store for
    a head is ahead of the release of that head's group if it is ahead of
    the lane's own arrival at the head's meeting. One behind that arrival
    fails here, since the barriers would not show it: a thread that arrives
    goes on at once, as a rule ahead of the threads its arrival releases.
    ``me`` holds the barriers the calling thread has arrived at."""

    def __init__(self, words: dict, width: int, me: threading.local) -> None:
        self.words = words
        self.width = width
        self.me = me

    def __getitem__(self, index: int):
        return self.words[index]

    def __setitem__(self, index: int, value) -> None:
        head = index // self.width
        assert helpers.MEETING_BARRIER + head not in self.me.arrived, (
            f"a store for head {head} behind the lane's arrival at its meeting"
        )
        self.words[index] = value


def _cta_run(
    ranges: tuple[str, ...],
    *,
    parts: int = 2,
    posted_parts: int = 2,
    at_look: int | tuple[int, int] = 0,
    after_look: int = 0,
    posted_last: bool = False,
    slow: str | None = None,
    columns: int = 40,
    conversions: str = "late",
    stream: int = 2,
) -> SimpleNamespace:
    """One balanced CTA where the spare warp looks, on the host: its two
    compute groups, 128 threads each, run the template's walk over the
    CTA's ranges (``_compute_units``), its spare warp, 32 threads, runs
    ``post_parts``, ``look_parts`` and ``stage_panels``, and lane 0 of
    each of its four K conversion warps runs ``_convert_keys`` over a
    stream of ``stream`` pages, with the device helpers' own bodies below
    them.
    Only what the pages leave behind (a range's panel, maxima and
    denominators), the plan, the tiles' conversion and the memory
    operations are the model's. The barriers are rendezvous that count
    their participants (``_NamedBarriers``); a thread that touches the
    workspace must have waited for the launch's dependencies, and a lane
    of the spare warp that stores a word of a group's record or a staged
    row of its head must not have arrived at that group's meeting yet
    (``_Answer``).

    The raw K ring is the stream's until the K conversions have read every
    tile of it: a copy of another part's panel into a slot while a tile is
    unread fails, as do a tile read from a slot a copy took and a K
    conversion warp's arrival at the conversions' barrier with a tile of
    its own unread. The two barriers of that stage complete one phase each
    (``_Phases``): the conversions' with the fourth warp's arrival, the
    copies' when the bytes the issuer told it to expect have landed. A
    thread of a group that reads a word of the stage must have waited for
    the copies, and reads its own words of the tile the template's rule
    names for its head (``stage_tile``), as the body's view of it does.

    ``ranges`` are the CTA's ranges in order: ``post``, the part of a unit
    in ``posted_parts`` parts that began in a lower CTA (its highest part,
    or the middle one of three, which is all a CTA runs); ``whole`` units;
    and ``lowest``, the lowest part of a unit in ``parts`` parts. The other
    CTAs are the workspace: their partials lie there, ``at_look`` other
    parts of the lowest part's unit have arrived when the spare warp looks
    (one count for both heads, or head 0's and head 1's), and
    ``after_look`` more arrive right behind that look; the posted unit's
    other parts have all arrived ahead of this CTA's (``posted_last``) or
    none has.

    ``slow``: ``look`` holds the spare warp's look until every thread of
    both groups waits at the meeting; ``stores`` holds the last warp of
    each group ahead of its stores of the lowest part's unit until the
    group's other warps wait at a barrier of the group.

    ``conversions``: ``late`` holds each K conversion warp at the stream's
    last tile until the spare warp's lane 0 waits for the conversions' end
    or has nothing to copy, so a copy that did not wait finds the tile
    unread; ``early`` lets them finish ahead of the look; ``held`` is
    ``late`` with the last warp's arrival held until every thread of both
    groups waits for the copies, as far as they get without it."""
    cta, heads, threads = 5, 2, 128
    assert conversions in ("late", "early", "held")
    assert ranges.count("post") <= 1 and ranges.count("lowest") <= 1
    assert "post" not in ranges[1:] and "lowest" not in ranges[:-1]
    found = (at_look, at_look) if isinstance(at_look, int) else at_look
    # A middle part's slots are its CTA's whole range.
    assert posted_parts == 2 or ranges == ("post",)
    # The plan: every range's unit with its pages and this CTA's steps, the
    # unit's header slot, and the owners of the two slots the exchange
    # asks for (the one behind a unit's header, its lowest CTA's, and its
    # last, its highest CTA's).
    plan_ranges = []
    owners: dict[int, int] = {}
    header = 100
    units: dict[tuple[int, int], list[tuple[int, SimpleNamespace]]] = {}
    for index, kind in enumerate(ranges):
        pages = 6
        step, count = {"post": (0, 2), "whole": (0, pages), "lowest": (4, 2)}[kind]
        lowest = highest = cta
        if kind == "post":
            lowest = cta - 1
            if posted_parts == 3:
                step, highest = 2, cta + 1
        if kind == "lowest":
            highest = cta + parts - 1
        owners[header + 1], owners[header + pages] = lowest, highest
        plan_ranges.append(
            SimpleNamespace(
                kind=kind,
                item=10 * index,
                pages=pages,
                step=step,
                count=count,
                header=header,
                lowest=lowest,
                highest=highest,
            )
        )
        for head in range(heads):
            # The unit's parts in source order: its highest CTA's first.
            units[head, index] = [
                (other, _part(1000 * head + 100 * index + other, columns))
                for other in range(highest, lowest - 1, -1)
            ]
        header += pages + 1
    plan = {
        helpers.PLAN_FIRST: 0,
        helpers.PLAN_LAST: len(ranges),
        helpers.PLAN_PAGES: stream,
        helpers.PLAN_HEADER: 100,
        helpers.PLAN_SLOTS: 0,
        helpers.PLAN_ACTIVE: 0,
        helpers.PLAN_SKEW: 0,
    }
    barriers = _NamedBarriers()
    me = threading.local()
    lock = threading.Lock()
    counters: dict[int, int] = {}
    # A CTA starts on what its shared memory held: here each group's record
    # holds a role from elsewhere, a pair's merge, which is no answer of
    # this CTA's spare warp.
    walks = {
        head * WALK_WORDS + helpers.WALK_ROLE: helpers.PART_MERGE_PAIR
        for head in range(heads)
    }
    mail: dict[int, int] = {}
    stage: dict[int, float] = {}
    state: list[dict[tuple[int, int], float]] = [{} for _head in range(heads)]
    scratch: list[dict[tuple[int, int], float]] = [{} for _head in range(heads)]
    errors: list[tuple[str, BaseException]] = []
    looks: list[tuple[int, int, int]] = []
    # The stage of the other part's panel: its two barriers, and the raw K
    # ring, with the tiles of the stream's pages that the four K
    # conversion warps have yet to read, the slots the spare warp took,
    # what its copies left there by slot and word, and whether its lane 0
    # is done.
    phases = _Phases(
        {
            name: arrivals
            for name, _width, arrivals in pipeline_barriers(
                False, 2, False, 2, False, True
            )[-2:]
        }
    )
    assert set(phases.left) == {"keys_converted", "panel_staged"}
    ring = SimpleNamespace(
        unread=4 * 2 * stream, taken=set(), words={}, copies=[], finished=False
    )

    def touch() -> None:
        assert me.waited, "the workspace ahead of the dependency wait"

    def waited() -> None:
        me.waited = True

    def joined() -> None:
        # The lanes of the spare warp are threads here: its
        # synchronization is a meeting of the 32. A K conversion warp is
        # its lane 0.
        if me.role == "spare":
            barriers.sync(100, 32)

    def fenced(kind: str) -> None:
        assert kind == "async.global"
        me.fenced = True

    host = _host(
        cta, columns, barriers.sync, waited, touch, sync_warp=joined, fence_proxy=fenced
    )
    workspace, record_bytes = host.workspace, host.record_bytes

    def own(head: int, index: int) -> SimpleNamespace:
        return dict(units[head, index])[cta]

    def counter(lowest: int, head: int) -> int:
        return lowest * record_bytes + head * 4

    def store(address: int, *values: object) -> None:
        touch()
        workspace[address] = tuple(cast("float", value) for value in values[:-1])

    def look(address: int) -> int:
        touch()
        head = (address % record_bytes) // 4
        if slow == "look":
            barriers.until(helpers.MEETING_BARRIER + head, threads)
        with lock:
            seen = counters.get(address, 0)
            looks.append((address, seen, head))
            # The parts that arrive right behind the look.
            counters[address] = seen + after_look
        return seen

    def add(address: int, amount: int) -> int:
        touch()
        with lock:
            seen = counters.get(address, 0)
            counters[address] = seen + amount
        return seen

    def reset(address: int, value: int, _policy: object) -> None:
        touch()
        with lock:
            counters[address] = value

    def arrive(barrier: int, thread_count: int) -> None:
        me.arrived.add(barrier)
        barriers.arrive(barrier, thread_count)

    class Stage:
        """One head's raw K tile as its group reads it, the one the body
        hands the group (``stage_tile``): behind the wait for the copies,
        each thread its own words."""

        def __init__(self, head: int) -> None:
            self.tile = exchange["stage_tile"](plan, head)

        def __getitem__(self, word: int) -> float:
            assert me.copied, "the stage ahead of the wait for its copies"
            assert phases.complete("panel_staged")
            assert (word % (4 * threads)) // 4 == me.thread
            return ring.words[self.tile, word]

    def held(pointer: _Pointer, phase: int, _sleep: bool) -> None:
        # ``_wait_non_mma``: a group's wait for the copies, and a K
        # conversion warp's for its next raw tile.
        if pointer.name == "panel_staged":
            phases.wait("panel_staged", phase)
            me.copied = True
        elif pointer.name == "raw_k_ready":
            me.tiles += 1
            if me.tiles == 2 * stream and conversions != "early":
                phases.until(
                    lambda the: the.waiting["keys_converted"] > 0 or ring.finished
                )

    def read_tile(tile: _Pointer, *_rows: object, **_forms: object) -> None:
        slot, rest = divmod(tile.offset, RAW_TILE_BYTES)
        with lock:
            assert (tile.name, rest) == ("raw_k", 0) and slot not in ring.taken, (
                "a K tile read from a slot the spare warp took"
            )
            ring.unread -= 1
        me.tiles_read += 1

    def arrived(pointer: _Pointer) -> None:
        if pointer.name == "keys_converted":
            assert me.tiles_read == 2 * stream, (
                "a K conversion warp at the conversions' barrier with a tile unread"
            )
            if conversions == "held" and me.warp == 3:
                phases.until(lambda the: the.waiting["panel_staged"] == heads * threads)
            phases.arrive("keys_converted")

    def expected(pointer: _Pointer, count: int) -> None:
        assert pointer.name == "panel_staged"
        phases.arrive("panel_staged", count)

    def slept(pointer: _Pointer, phase: int) -> None:
        assert pointer.name == "keys_converted"
        phases.wait("keys_converted", phase)

    def copy(target: _Pointer, source: int, size: int, barrier: _Pointer) -> None:
        touch()
        assert me.fenced, "a copy through the async proxy without its fence"
        slot, rest = divmod(target.offset, RAW_TILE_BYTES)
        assert (target.name, rest, barrier.name) == ("raw_k", 0, "panel_staged")
        with lock:
            assert ring.unread == 0, (
                "a copy into the raw K ring ahead of the conversions' end"
            )
            ring.taken.add(slot)
            ring.copies.append(slot)
            for offset in range(0, size, 16):
                for index, value in enumerate(workspace[source + offset]):
                    ring.words[slot, offset // 4 + index] = value
        phases.land("panel_staged", size)

    device_prims = SimpleNamespace(
        barrier_cta_sync=lambda barrier, thread_count: barriers.sync(
            barrier, thread_count
        ),
        barrier_cta_arrive=arrive,
        mbarrier_arrive=arrived,
        mbarrier_arrive_expect_tx=expected,
        tcgen05_ld=lambda _form, address, num: [
            own(me.head, me.unit).panel[address + row][me.thread] for row in range(num)
        ],
        tcgen05_wait=lambda kind: None,
        tcgen05_fence=lambda kind: None,
        Tcgen05Wait=SimpleNamespace(LOAD="load", STORE="store"),
        Tcgen05Fence=SimpleNamespace(
            BEFORE_THREAD_SYNC="before", AFTER_THREAD_SYNC="after"
        ),
    )

    def unit_range(unit, _header, *_rest):
        me.unit = unit
        this = plan_ranges[unit]
        return this.item, 0, 0, 0, this.pages, this.step, this.count

    def pages(*arguments):
        # What a range's pages leave: this thread's share of the rows'
        # denominators, its row's maximum, and its panel for the loads.
        partials, part = arguments[13], own(me.head, me.unit)
        for row in range(columns):
            partials[row] = part.sums[me.thread // 32][row] / 32.0
        if me.thread < columns:
            state[me.head][0, me.thread] = part.maximum[me.thread]

    exchange = {
        name: jit_host_body(getattr(helpers, name))
        for name in (
            "_count_arrival",
            "_note_post",
            "arrive_part",
            "look_parts",
            "pair_partial",
            "post_parts",
            "split_arrival",
            "stage_panels",
            "stage_tile",
            "store_part",
        )
    }
    walked = {
        name: jit_host_body(getattr(template, name))
        for name in (
            "_compute_begin",
            "_compute_units",
            "_convert_keys",
            "_drain_last_pv",
            "_reduce_part",
            "_reduce_rows",
        )
    }
    last = len(ranges) - 1

    def stores(*arguments):
        tile = arguments[4]
        if (
            slow == "stores"
            and me.thread // 32 == 3
            and tile == plan_ranges[last].item + me.head
        ):
            # The furthest the group's other warps get without this one.
            barriers.until(1 + 2 * me.head, threads - 32)
        return host.bodies["_compute_store"](*arguments)

    def run(name: str, body, **local) -> None:
        for key, value in local.items():
            setattr(me, key, value)
        me.waited = me.copied = me.fenced = False
        me.arrived = set()
        me.tiles = me.tiles_read = 0
        try:
            body()
        except BaseException as error:  # reported by the caller's assertion
            errors.append((name, error))
            barriers.abort()
            phases.abort()

    arguments = dict.fromkeys(inspect.signature(walked["_compute_units"]).parameters)
    arguments.update(
        output=host.output,
        lse=host.lse,
        plan=plan,
        workspace=0,
        first_unit=0,
        last_unit=0,
        partition="balanced",
        record_bytes=record_bytes,
        slot_bits=32,
        skewed=False,
        group=8,
        query_rows=columns,
        heads=heads,
        fast_math=True,
        query_columns=columns,
        base2_scores=True,
        output_reciprocal=True,
        output_scale_log2=0,
        capture_pdl_wait_mode=1,
        probability_schedule="serial",
        early_tmem_retirement=False,
        look_stage=stage,
        panel_staged=_Pointer("panel_staged"),
        store_policy="default",
        sleep_waits=False,
        base=0,
    )

    def group_thread(head: int, thread: int):
        return lambda: walked["_compute_units"](
            **{
                **arguments,
                "state": state[head],
                "scratch": scratch[head],
                "pair_head": head,
                "walk": _Words(walks, head * WALK_WORDS),
                "mail": _Words(mail, head * MAIL_WORDS),
                "panel_stage": Stage(head),
            }
        )

    def warp_thread(lane: int):
        def body() -> None:
            exchange["post_parts"](0, mail, lane, 1, record_bytes)
            # The look's answer: no store of it behind the lane's arrival.
            paired = exchange["look_parts"](
                0,
                _Answer(walks, WALK_WORDS, me),
                _Answer(stage, 2 * columns, me),
                lane,
                1,
                record_bytes,
                columns,
            )
            exchange["stage_panels"](
                0,
                _Pointer("raw_k"),
                plan,
                paired,
                _Pointer("keys_converted"),
                _Pointer("panel_staged"),
                lane,
                1,
                record_bytes,
                columns,
            )
            if lane == 0:
                with phases.condition:
                    ring.finished = True
                    phases.condition.notify_all()

        return body

    def keys_thread(warp: int):
        # Lane 0 of a K conversion warp: its rows of every tile of the
        # stream, then the arrival that says so.
        return lambda: walked["_convert_keys"](
            _Pointer("raw_k"),
            _Pointer("scale_k"),
            _Pointer("raw_k_ready"),
            _Pointer("raw_k_free"),
            _Pointer("key_ready"),
            _Pointer("key_free"),
            None,
            None,
            0,
            warp * 32,
            0,
            stream,
            0,
            False,
            True,
            _Pointer("keys_converted"),
        )

    def owner(where, *_plan):
        return owners[where]

    replaced = [
        *host.replaced,
        *((module, "prims", device_prims) for module in (helpers, template)),
        *((module, "_slot_cta", owner) for module in (helpers, template)),
        (helpers, "stg_v4_f32_l2", store),
        (helpers, "stg_v2_f32_l2", store),
        (helpers, "stg_u32_l2", reset),
        (helpers, "ld_relaxed_gpu_u32", look),
        (helpers, "atom_add_acq_rel_gpu_u32", add),
        (helpers, "fence_acq_rel_gpu", lambda: None),
        (helpers, "l2_policy_evict_first", lambda: None),
        (helpers, "l2_policy_evict_last", lambda: None),
        (helpers, "cp_bulk_global_to_shared", copy),
        (helpers, "mbar_wait_sleep", slept),
        *(
            (helpers, name, body)
            for name, body in exchange.items()
            if name not in ("post_parts", "look_parts", "stage_panels")
        ),
        (template, "_wait_non_mma", held),
        (template, "stage_scaled_s8_rows_to_tmem", read_tile),
        (template, "_packed_scale_layout_supported", lambda scales: True),
        (
            template,
            "_compute_group",
            lambda *_quota: (me.thread, me.thread // 32, me.thread % 32, 0, None),
        ),
        (template, "_unit_range", unit_range),
        (template, "_compute_pages", pages),
        (template, "store_query_panel", lambda *_panel: None),
        (
            template,
            "load_query_panel",
            lambda _address, count: [
                own(me.head, me.unit).panel[row][me.thread] for row in range(count)
            ],
        ),
        # A warp's sum of a row: its lanes hold equal shares here.
        (
            template,
            "warp_row_sum",
            lambda values, lane: (
                lane % len(values),
                values[lane % len(values)] * 32.0,
            ),
        ),
        *(
            (template, name, exchange[name])
            for name in ("arrive_part", "pair_partial", "split_arrival", "store_part")
        ),
        *(
            (template, name, body)
            for name, body in walked.items()
            if name != "_compute_units"
        ),
        (template, "_compute_store", stores),
    ]
    with _replaced(replaced):
        # The other CTAs' partials of the split units, each in its slot,
        # and their arrivals so far.
        published = {}
        for index, this in enumerate(plan_ranges):
            if this.kind == "whole":
                continue
            for head in range(heads):
                # Where this CTA's own part goes: its row statistics.
                published[index, head] = (
                    host.slot(0, cta, this.lowest, head, record_bytes)
                    + host.panel_bytes
                )
                for other, part in units[head, index]:
                    if other != cta:
                        host.publish(part, other, this.lowest, head)
                ahead = this.highest - this.lowest if posted_last else 0
                counters[counter(this.lowest, head)] = (
                    found[head] if this.kind == "lowest" else ahead
                )
        members = [
            threading.Thread(
                target=run,
                args=(f"head {head} thread {thread}", group_thread(head, thread)),
                kwargs={"head": head, "thread": thread, "unit": None, "role": "group"},
            )
            for head in range(heads)
            for thread in range(threads)
        ]
        members += [
            threading.Thread(
                target=run,
                args=(f"spare lane {lane}", warp_thread(lane)),
                kwargs={"head": None, "thread": lane, "unit": None, "role": "spare"},
            )
            for lane in range(32)
        ]
        members += [
            threading.Thread(
                target=run,
                args=(f"keys warp {warp}", keys_thread(warp)),
                kwargs={"head": None, "thread": 0, "role": "keys", "warp": warp},
            )
            for warp in range(4)
        ]
        for member in members:
            member.start()
        for member in members:
            member.join(timeout=90)
        assert not errors and not any(member.is_alive() for member in members), errors[
            :3
        ]
    return SimpleNamespace(
        output=host.output,
        lse=host.lse,
        units=units,
        ranges=plan_ranges,
        barriers=barriers,
        counters={
            (index, head): counters[counter(this.lowest, head)]
            for index, this in enumerate(plan_ranges)
            if this.kind != "whole"
            for head in range(heads)
        },
        looks=looks,
        walks=walks,
        stage=stage,
        phases=phases,
        ring=ring,
        stream=stream,
        mail=mail,
        workspace=workspace,
        published={
            (index, head): published[index, head] in workspace
            for index, head in published
        },
    )


def test_a_cta_exchanges_its_parts_on_every_path_without_a_deadlock():
    """Run on the host: a balanced CTA where the spare warp looks, as 292
    threads that run the template's walk, the spare warp's posts, look and
    stage, the K conversions' loop and the device helpers' own bodies,
    with barriers that count their participants. On every path of the
    exchange the CTA ends, each barrier's participants name the same
    count, and the CTA stores what the fold of the parts gives, for both
    heads:

    - a lowest part whose look finds the other part merges the pair in its
      reduction, with or without a part the CTA posted first, whose unit
      it merges behind its walk if that part arrived last; the spare warp
      copies the other part's panel into the raw K ring for it, behind
      the K conversions' last tile, and the group folds from there behind
      its wait for the copies: at 40 query columns with the last eight
      loaded, at 32 and at 8 with nothing loaded; with the conversions
      done ahead of the look, and with one conversion warp's arrival held
      until both groups wait for the copies;
    - a look that finds a part missing publishes: the part merges if its
      own arrival is the last, and stores nothing otherwise;
    - where head 0's look finds the other part and head 1's does not,
      head 0's group merges the pair and head 1's publishes: each does
      what the warp left in its own record;
    - a unit in three parts is merged from the workspace, by the look or
      by the last arrival, and a middle part's unit behind its CTA's walk:
      nothing is copied for it;
    - a CTA without a lowest part asks for no look and meets the spare
      warp at its post's reply; the warp stages nothing for it, though
      each group's record starts with a pair's role left over.

    The warp copies for the heads whose group merges a pair and for no
    other, one tile a head, the ring's position behind the stream's last
    step, for a stream of two pages and of three; every thread of such a
    group waits for the copies once and reads its own words of the stage
    behind that wait; the K conversions have read every tile of the stream
    ahead of a copy and arrive once each. A copy ahead of the dependency
    wait or of a tile's conversion, and a read of the stage ahead of the
    copies, fail the run.

    A group that ran a lowest part met the spare warp once, at the first
    barrier of that part's reduction, with the warp's 32 lanes among the
    160; the warp's arrivals at the reply then stay unmatched. In every
    run a lane of the warp stores what it leaves for a group, the words
    of the group's record and its staged rows, ahead of the lane's own
    arrival at that meeting, which completes with the last of the 160: a
    store behind the arrival fails the run, since the barriers here would
    not show whether the group read ahead of it. With the look held until
    every thread of both groups waits at that meeting the CTA stores the
    same; with a warp of each group held ahead of its stores until the
    group's other warps wait at a barrier of the group, as far as they get
    without it, it stores its unit's rows."""
    columns, threads, heads = 40, 128, 2
    post, meeting = helpers.POST_BARRIER, helpers.MEETING_BARRIER
    look, reply = helpers.LOOK_BARRIER, helpers.POSTED_BARRIER

    def check(
        run,
        stored: set[int] | tuple[set[int], set[int]],
        counters: dict[int, int] | tuple[dict[int, int], dict[int, int]],
        met: bool,
        copied: tuple[int, ...] = (),
        columns: int = columns,
    ) -> None:
        # ``stored``: the ranges whose units the CTA stores, and
        # ``counters``: what it leaves in the counters of its split units,
        # by range. One of each for both heads, or head 0's and head 1's.
        # ``copied``: the heads whose other part's panel the warp staged.
        by_head = stored if isinstance(stored, tuple) else (stored, stored)
        left = counters if isinstance(counters, tuple) else (counters, counters)
        expected = {}
        logs = {}
        for head in range(heads):
            for index in by_head[head]:
                parts = [part for _cta, part in run.units[head, index]]
                tile = run.ranges[index].item + head
                for row in range(columns):
                    for thread in range(threads):
                        value, log = _folded(parts, row, thread)
                        expected[tile, row, thread] = value
                    logs[tile, row] = log
        assert run.output == expected
        assert run.lse == logs
        assert run.counters == {
            (index, head): count
            for head in range(heads)
            for index, count in left[head].items()
        }
        # Every barrier's participants: a group's own are its 128 threads,
        # one with the spare warp counts the 160; and what completed.
        named = run.barriers.named
        assert {threads for barrier, threads in named.items() if barrier < 5} == {128}
        assert {threads for barrier, threads in named.items() if 8 <= barrier < 16} == {
            160
        }
        rounds = {
            barrier: count
            for barrier, count in run.barriers.rounds.items()
            if 8 <= barrier < 16
        }
        both = range(heads)
        assert rounds == {
            **{post + head: 1 for head in both},
            **{look + head: 1 for head in both},
            **{(meeting if met else reply) + head: 1 for head in both},
        }
        # Nobody is left waiting, and only the spare warp's arrivals at a
        # reply nobody waits for are left over.
        assert not any(run.barriers.waiting.values())
        assert {b: n for b, n in run.barriers.arrived.items() if n} == (
            {reply + head: 32 for head in both} if met else {}
        )
        # Each head's lane looked once where a lowest part ran, and the
        # warp staged nothing for a group that did not ask.
        assert sorted(head for _address, _seen, head in run.looks) == (
            [0, 1] if met else []
        )
        assert met or not run.stage
        # The staged panels: one copy for each head whose group merges a
        # pair, the head's tile of the ring's position behind the stream's
        # last step, with the stage's columns of every thread; each thread
        # of such a group waited for the copies once, and the warp's lane
        # 0 for the conversions if it copied. The conversions read the
        # whole stream and arrived once each.
        tiles = [2 * (run.stream & 1) + head for head in copied]
        assert sorted(run.ring.copies) == tiles == sorted(run.ring.taken)
        assert len(run.ring.words) == (
            len(copied) * partition_look_panel_columns(columns) * threads
        )
        assert run.ring.unread == 0
        assert run.phases.left == {
            "keys_converted": 0,
            "panel_staged": 0 if copied else 1,
        }
        assert run.phases.waited == {
            "keys_converted": 1 if copied else 0,
            "panel_staged": threads * len(copied),
        }
        assert not any(run.phases.waiting.values())
        assert not any(run.phases.expected.values())

    role = helpers.WALK_ROLE
    both_heads = (0, 1)
    # A pair merged at the look, behind a posted part that arrived last.
    run = _cta_run(("post", "whole", "lowest"), at_look=1, posted_last=True)
    check(run, {0, 1, 2}, {0: 0, 2: 0}, met=True, copied=both_heads)
    for head in range(heads):
        assert run.walks[head * WALK_WORDS + role] == helpers.PART_MERGE_PAIR
        assert run.mail[head * MAIL_WORDS + helpers.MAIL_KIND] == helpers.MAIL_LAST
    # The same with the look held until both groups wait at the meeting,
    # and with a warp of each group held ahead of its stores.
    for slow in ("look", "stores"):
        held = _cta_run(
            ("post", "whole", "lowest"), at_look=1, posted_last=True, slow=slow
        )
        check(held, {0, 1, 2}, {0: 0, 2: 0}, met=True, copied=both_heads)
        assert held.output == run.output
    # The same with the K conversions done ahead of the look, and with the
    # last conversion warp's arrival held until every thread of both groups
    # waits for the copies.
    for conversions in ("early", "held"):
        held = _cta_run(
            ("post", "whole", "lowest"),
            at_look=1,
            posted_last=True,
            conversions=conversions,
        )
        check(held, {0, 1, 2}, {0: 0, 2: 0}, met=True, copied=both_heads)
        assert held.output == run.output
    # A lowest part alone in its CTA: its first access to the workspace
    # is the other part's loads of the columns the stage does not hold.
    run = _cta_run(("lowest",), at_look=1)
    check(run, {0}, {0: 0}, met=True, copied=both_heads)
    assert sorted(run.ring.taken) == [0, 1]
    # A stream of three pages: its last step took the ring's first two
    # tiles, and the stage is the other two.
    odd = _cta_run(("lowest",), at_look=1, stream=3)
    check(odd, {0}, {0: 0}, met=True, copied=both_heads)
    assert sorted(odd.ring.taken) == [2, 3] and odd.output == run.output
    # At 32 and at 8 query columns the stage holds the whole panel: the
    # group loads nothing of the other part, and its first access to the
    # workspace is behind the stores' wait.
    for narrow in (32, 8):
        run = _cta_run(("lowest",), at_look=1, columns=narrow)
        check(run, {0}, {0: 0}, met=True, copied=both_heads, columns=narrow)
    # A look that finds nothing. The other part arrives behind it: this
    # part's own arrival is the last, and it merges.
    run = _cta_run(("whole", "lowest"), at_look=0, after_look=1)
    check(run, {0, 1}, {1: 0}, met=True)
    for head in range(heads):
        assert run.walks[head * WALK_WORDS + role] == helpers.PART_MERGE
    # It does not arrive: the part publishes its partial and is done, as
    # is the part the CTA posted, which did not arrive last either.
    run = _cta_run(("post", "lowest"), at_look=0)
    check(run, set(), {0: 1, 1: 1}, met=True)
    for head in range(heads):
        assert run.walks[head * WALK_WORDS + role] == helpers.PART_DONE
        assert run.mail[head * MAIL_WORDS + helpers.MAIL_KIND] == helpers.MAIL_ARRIVED
    # Both parts lie in the CTA's slots for them.
    assert (
        run.published == dict.fromkeys(run.published, True) and len(run.published) == 4
    )
    # The two heads' looks with different answers: head 0's finds the other
    # part, and its group merges the pair; head 1's finds it missing, and
    # its group publishes its part and stores nothing. A word of the
    # answer left in the other head's record gives neither.
    run = _cta_run(("whole", "lowest"), at_look=(1, 0))
    check(run, ({0, 1}, {0}), ({1: 0}, {1: 1}), met=True, copied=(0,))
    assert sorted((head, seen) for _address, seen, head in run.looks) == [
        (0, 1),
        (1, 0),
    ]
    assert [run.walks[head * WALK_WORDS + role] for head in range(heads)] == [
        helpers.PART_MERGE_PAIR,
        helpers.PART_DONE,
    ]
    # The warp staged the other part's rows for head 0 alone.
    assert {index // (2 * columns) for index in run.stage} == {0}
    assert len(run.stage) == 2 * columns
    assert run.published == {(1, 0): False, (1, 1): True}
    # A unit in three parts: merged by the look, behind a posted part
    # that arrived last and is merged behind the walk.
    run = _cta_run(("post", "lowest"), parts=3, at_look=2, posted_last=True)
    check(run, {0, 1}, {0: 0, 1: 0}, met=True)
    for head in range(heads):
        assert run.walks[head * WALK_WORDS + role] == helpers.PART_MERGE
        assert run.walks[head * WALK_WORDS + helpers.WALK_PARTS] == 3
    # And by the last arrival, where the look saw one of the two others.
    run = _cta_run(("lowest",), parts=3, at_look=1, after_look=1)
    check(run, {0}, {0: 0}, met=True)
    # The middle part of a unit in three parts is all its CTA runs: it
    # arrived last, and the CTA merges the three behind its walk.
    run = _cta_run(("post",), posted_parts=3, posted_last=True)
    check(run, {0}, {0: 0}, met=False)
    for head in range(heads):
        assert run.mail[head * MAIL_WORDS + helpers.MAIL_PARTS] == 3
    # A CTA without a lowest part: no look, and the post's reply.
    run = _cta_run(("post", "whole"), posted_last=True)
    check(run, {0, 1}, {0: 0}, met=False)
    run = _cta_run(("whole", "whole"))
    check(run, {0, 1}, {}, met=False)
