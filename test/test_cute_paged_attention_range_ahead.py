"""Check the range-ahead prefetch of the balanced partition of the paired
attention schedule (``cute_mma_dag_boundary_prefetch_ahead``): the spare
warp of a CTA requests the first pages of the CTA a distance ahead. Which
grids have it, the config contract, what it renders and compiles to, the
structure of the device role, and where a search samples and seeds it,
together with the raw V ring counted in tiles and the reversed walk."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import helion
from helion import exc
from helion._compiler.cute import _paged_attention_paired_pipeline as template
from helion._compiler.cute import paged_attention_codegen
from helion._compiler.cute.autotune.mma_dag_surface import boundary_prefetch_ahead_needs
from helion._compiler.cute.cute_mma_dag import MMA_DAG_BOUNDARY_PREFETCH_AHEAD_CHOICES
from helion._compiler.cute.cute_mma_dag import MMA_DAG_BOUNDARY_PREFETCH_AHEAD_KEY
from helion._compiler.cute.cute_mma_dag import MMA_DAG_BOUNDARY_PREFETCH_CHOICES
from helion._compiler.cute.cute_mma_dag import (
    paged_attention_boundary_prefetch_ahead_choices,
)
from helion._compiler.cute.cute_mma_dag import paged_attention_boundary_prefetch_choices
from helion._compiler.cute.cute_mma_dag_seeds import VALUE_RING_TILES_SEED
from helion._compiler.cute.paged_attention_config import (
    paged_attention_boundary_prefetch_ahead_reason,
)
from helion._compiler.cute.paged_attention_config import (
    paged_attention_boundary_prefetch_reason,
)
from helion._compiler.cute.paged_attention_prefetch import (
    BOUNDARY_PREFETCH_RANGE_AHEAD_DISTANCES,
)
from helion._compiler.cute.paged_attention_prefetch import RANGE_AHEAD_KEY_PAGES
from helion._compiler.cute.paged_attention_prefetch import RANGE_AHEAD_UNIT_LIMIT
from helion._compiler.cute.paged_attention_prefetch import RANGE_AHEAD_VALUE_PAGES
from helion._compiler.cute.paged_attention_prefetch import range_ahead_distances
from helion._compiler.cute.paged_attention_shared_memory import plan_geometry
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
from helion._testing.cute_paged_stream import seeds_ahead_of_the_range_ahead
from helion._testing.cute_paged_stream import stream_bound
from helion._testing.cute_paged_stream import stream_config
from helion._testing.cute_paged_stream import stream_render
from helion._testing.cute_paged_stream import stream_seeds
from helion._testing.cute_paged_stream import stream_step
from helion._testing.cute_paged_stream import transferred_seeds
from helion.language import memory_ops

pytestmark = pytest.mark.cpu_only
KEY = "cute_mma_dag_boundary_prefetch_ahead"
# The boundary prefetch of one unit per CTA, whose values are pages.
PAGES_KEY = "cute_mma_dag_boundary_prefetch"
COLD = "cute_mma_dag_boundary_prefetch_cold"
TILES = "cute_mma_dag_value_ring_tiles"
RING = "cute_mma_dag_value_ring"
PARTITION = "cute_mma_dag_partition"
ORDER = "cute_mma_dag_page_order"
LEAD = "cute_mma_dag_clipped_page_lead"
SKEW = "cute_mma_dag_partition_skew"
QUERY_TMA = "cute_mma_dag_query_tma"
PREFETCH_INSTRUCTION = "cp.async.bulk.prefetch.L2.global"
TEMPLATE = Path(inspect.getfile(template))
# Units on 148 SMs: one wave with idle SMs, and more than two waves, where
# the balanced partition is seeded and the reversed walk runs on it.
ONE_WAVE = 128
WAVES = 300
BALANCED = {PARTITION: "balanced", QUERY_TMA: True}
# The walk the range-ahead seeds stand on.
WALK = {**BALANCED, ORDER: "reversed", LEAD: 8}


def _function(name: str) -> ast.FunctionDef:
    (found,) = [
        node
        for node in ast.walk(ast.parse(TEMPLATE.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    return found


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and ast.unparse(call.func).rpartition(".")[2] == name
    ]


def _paired(moves):
    return [
        (seed, moved)
        for seed, moved in moves
        if seed.config["cute_mma_dag_pass_warps"] == PAIRED
    ]


def test_a_grid_of_more_units_than_sms_has_the_range_ahead_distances():
    assert MMA_DAG_BOUNDARY_PREFETCH_AHEAD_KEY == KEY
    assert BOUNDARY_PREFETCH_RANGE_AHEAD_DISTANCES == (48, 96)
    # Zero, off, is the default; no distance is a page count of the
    # prefetch of one unit per CTA, and nothing tells them apart by value:
    # each has its own option.
    assert MMA_DAG_BOUNDARY_PREFETCH_AHEAD_CHOICES == (0, 48, 96)
    assert MMA_DAG_BOUNDARY_PREFETCH_CHOICES == (0, 8, 12, 4)
    assert (RANGE_AHEAD_KEY_PAGES, RANGE_AHEAD_VALUE_PAGES) == (3, 5)
    # The spare warp of a persistent CTA scans no length up to this many
    # units: the nineteen warps below it take one unit a thread.
    assert RANGE_AHEAD_UNIT_LIMIT == 608
    assert plan_geometry(RANGE_AHEAD_UNIT_LIMIT) == (1, 19)
    assert plan_geometry(RANGE_AHEAD_UNIT_LIMIT + 1) == (1, 20)
    for units, capacity, distances in (
        # Every unit has an SM: one unit per CTA, whatever the partition.
        (128, 148, ()),
        (148, 148, ()),
        (149, 148, (48, 96)),
        (320, 148, (48, 96)),
        (608, 148, (48, 96)),
        # The spare warp scans.
        (609, 148, ()),
        # A distance names a CTA of the grid.
        (200, 97, (48, 96)),
        (200, 96, (48,)),
        (200, 49, (48,)),
        (200, 48, ()),
        (200, 1, ()),
        (0, 148, ()),
    ):
        assert range_ahead_distances(units, capacity) == distances, (units, capacity)
    waves = stream_bound(WAVES).config_spec.cute_mma_dag_facts
    assert waves is not None and waves.attention is not None
    assert waves.attention.boundary_prefetch_range_ahead == (48, 96)
    # The grid's form of one unit per CTA is the page prefetch's alone.
    assert waves.attention.boundary_prefetch_form == "wave_ahead"
    one = stream_bound(ONE_WAVE).config_spec.cute_mma_dag_facts
    assert one is not None and one.attention is not None
    assert one.attention.boundary_prefetch_range_ahead == ()
    # A device of fewer SMs offers the distances that name one of its CTAs.
    small = stream_bound(WAVES, sms=96)
    small_facts = small.config_spec.cute_mma_dag_facts
    assert small_facts is not None and small_facts.attention is not None
    assert small_facts.attention.boundary_prefetch_range_ahead == (48,)
    assert small.config_spec._flat_fields()[KEY].choices == (0, 48)
    # Without fast math there is no balanced partition, hence no distance.
    exact = stream_bound(WAVES, fast_math=False)
    exact_facts = exact.config_spec.cute_mma_dag_facts
    assert exact_facts is not None and exact_facts.attention is not None
    assert exact_facts.attention.boundary_prefetch_range_ahead == ()
    assert exact.config_spec.cute_mma_dag_boundary_prefetch_ahead is None


def test_a_distance_is_the_balanced_partitions_and_pages_are_one_unit_per_ctas():
    bound = stream_bound(WAVES)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    attention = facts.attention
    # Pages on one unit per CTA and nowhere else, as without this option.
    assert paged_attention_boundary_prefetch_choices(attention, PAIRED) == (0, 8, 12, 4)
    for partition in ("whole", "balanced"):
        assert paged_attention_boundary_prefetch_choices(
            attention, PAIRED, False, partition
        ) == (0,)
    # A distance on the balanced partition and nowhere else.
    for partition, choices in (
        ("item", (0,)),
        ("whole", (0,)),
        ("balanced", (0, 48, 96)),
    ):
        assert (
            paged_attention_boundary_prefetch_ahead_choices(
                attention, PAIRED, False, partition
            )
            == choices
        )
    # The value-metadata actor is the warp a prefetch takes.
    assert paged_attention_boundary_prefetch_ahead_choices(
        attention, PAIRED, True, "balanced"
    ) == (0,)
    assert paged_attention_boundary_prefetch_ahead_choices(
        attention, "attention", False, "balanced"
    ) == (0,)
    paired = {"cute_mma_dag_pass_warps": PAIRED}
    for config, legal in (
        ({**paired, **BALANCED, KEY: 48}, True),
        ({**paired, **BALANCED, KEY: 96}, True),
        ({**paired, **BALANCED, KEY: 0}, True),
        ({**paired, KEY: 48}, False),
        ({**paired, PARTITION: "whole", QUERY_TMA: True, KEY: 48}, False),
        # No other distance, no page count, and an int.
        ({**paired, **BALANCED, KEY: 1}, False),
        ({**paired, **BALANCED, KEY: 49}, False),
        ({**paired, **BALANCED, KEY: 8}, False),
        ({**paired, **BALANCED, KEY: True}, False),
        ({**paired, **BALANCED, KEY: "48"}, False),
    ):
        reason = paged_attention_boundary_prefetch_ahead_reason(facts, config)
        assert (reason is None) is legal, config
        if not legal:
            assert reason is not None and reason.startswith(f"{KEY} must be one of")
    # The page prefetch reads its own option: a distance beside it changes
    # nothing of its rule, and pages have no role on the balanced partition
    # whatever their value.
    for config, legal in (
        ({**paired, PAGES_KEY: 8}, True),
        ({**paired, **BALANCED, PAGES_KEY: 8}, False),
        ({**paired, **BALANCED, PAGES_KEY: 48}, False),
        ({**paired, PAGES_KEY: 48}, False),
        ({**paired, **BALANCED, PAGES_KEY: 0, KEY: 48}, True),
    ):
        reason = paged_attention_boundary_prefetch_reason(facts, config)
        assert (reason is None) is legal, config
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        spec = bound.config_spec
        for distance in (48, 96):
            kept = spec.normalized_config(stream_config(**BALANCED, **{KEY: distance}))
            assert kept.config[KEY] == distance
            assert kept.config[PARTITION] == "balanced"
            assert PAGES_KEY not in kept.config
        assert (
            KEY
            not in spec.normalized_config(stream_config(**BALANCED, **{KEY: 0})).config
        )
        for knobs, refused in (
            ({KEY: 48}, KEY),
            ({PARTITION: "whole", QUERY_TMA: True, KEY: 96}, KEY),
            ({**BALANCED, KEY: 8}, KEY),
            # A config has at most one of the two: pages take one unit per
            # CTA, a distance the balanced partition.
            ({**BALANCED, KEY: 48, PAGES_KEY: 8}, PAGES_KEY),
            ({KEY: 48, PAGES_KEY: 8}, KEY),
            ({**BALANCED, PAGES_KEY: 48}, PAGES_KEY),
        ):
            config = stream_config(**knobs)
            with pytest.raises(exc.InvalidConfig, match=f"{refused} must be one of"):
                spec.normalized_config(config)
            # Repaired to the same config without the option its partition
            # does not run: the partition is not a prefetch's to change.
            spec.normalize(config, _fix_invalid=True)
            assert refused not in config.config
            assert config.config.get(PARTITION) == knobs.get(PARTITION)
            other = PAGES_KEY if refused == KEY else KEY
            kept = knobs.get(other) if knobs.get(PARTITION) != "whole" else None
            assert config.config.get(other) == kept
    # What a distance needs of a config, for a step and for an override.
    assert boundary_prefetch_ahead_needs() == {
        PARTITION: "balanced",
        QUERY_TMA: True,
        "cute_mma_dag_value_metadata_skip": False,
        "cute_mma_dag_value_metadata_read_overlap": False,
    }


def test_the_spare_warp_plans_the_served_range_behind_the_scan_alone():
    """The role's place in the body: the spare warp waits at the scan's
    barrier beside the loader, finds the launch's slots from the tables,
    arrives at the setup rendezvous without waiting there and requests. It
    reads nothing the rendezvous publishes: not the plan, which the other
    control warps read behind it."""
    body = _function("paged_attention_paired_body")
    source = ast.unparse(body)
    # The distance is the body's own parameter, beside the page count of
    # the roles of one unit per CTA: a body has at most one of the two.
    assert "range_ahead = boundary_prefetch_ahead > 0" in source
    for check in (
        "cutlass.const_expr(partition == 'balanced')",
        "cutlass.const_expr(boundary_prefetch == 0)",
        "cutlass.const_expr(boundary_prefetch_form is None)",
    ):
        assert f"assert {check}" in source, check
    # The barrier counts the warp where it waits there.
    (counted,) = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.const_expr(range_ahead)"
        and "plan_threads = plan_threads + 32" in ast.unparse(node)
    ]
    assert "assert cutlass.const_expr(plan_warps <= 19)" in ast.unparse(counted)
    (rendezvous,) = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "warp == 19"
        and _calls(node, "_prefetch_range_ahead")
    ]
    steps = [
        ast.unparse(node.value.func if isinstance(node.value, ast.Call) else node.value)
        for node in rendezvous.body
        if isinstance(node, (ast.Expr, ast.Assign))
    ]
    assert steps == [
        "prims.barrier_cta_sync",
        "partition_plan",
        "prims.barrier_cta_arrive",
        "_prefetch_range_ahead",
    ]
    wait, _plan, arrival, _role = (
        node for node in rendezvous.body if isinstance(node, (ast.Expr, ast.Assign))
    )
    assert ast.unparse(wait.value) == (
        "prims.barrier_cta_sync(7, thread_count=plan_threads)"
    )
    assert ast.unparse(arrival.value) == (
        "prims.barrier_cta_arrive(6, thread_count=640)"
    )
    # Every other warp of that branch waits at the rendezvous.
    assert ast.unparse(rendezvous.orelse[0]) == (
        "prims.barrier_cta_sync(6, thread_count=640)"
    )
    # The plan is read by the control warps that waited for it.
    (planned,) = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.const_expr(range_ahead)"
        and _calls(node, "_planned_units")
    ]
    (guarded,) = [node for node in planned.body if isinstance(node, ast.If)]
    assert ast.unparse(guarded.test) == "warp != 19"
    assert _calls(guarded, "_planned_units")


def test_the_range_ahead_role_reads_the_page_table_alone_and_requests_in_the_pool():
    role = _function("_prefetch_range_ahead")
    source = ast.unparse(role)
    # The served CTA's range and its first unit as that CTA's loader finds
    # them, its part of the unit from the one function every walking role
    # takes it from, and a step's tile from the body's one walk.
    assert "serving = cta + distance < active" in source
    for call in (
        "_slot_boundary(served, slots, active, skew, slot_bits, skewed)",
        "_slot_boundary(served + 1, slots, active, skew, slot_bits, skewed)",
    ):
        assert call in source
    (located,) = _calls(role, "_locate_slot")
    assert ast.unparse(located.args[3]) == "begin"
    (ranged,) = _calls(role, "_unit_range")
    assert [ast.unparse(argument) for argument in ranged.args[:6]] == [
        "unit",
        "header",
        "begin",
        "end",
        "None",
        "plan_lengths",
    ]
    assert ast.unparse(ranged.args[-1]) == "'balanced'"
    (mapped,) = _calls(role, "attention_page_coordinates")
    assert [ast.unparse(argument) for argument in mapped.args] == [
        "first_page",
        "last_page",
        "step0 + cutlass.Int64(lane)",
        "page_walk",
    ]
    # Global memory: one page id a lane, inside the table's page extent.
    # The K and V tensors are only addresses of requests.
    loads = [
        ast.unparse(node)
        for node in ast.walk(role)
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Load)
        and ast.unparse(node.value)
        in ("pages", "key", "key_scales", "value", "value_scales")
    ]
    assert loads == ["pages[request, logical_page]"]
    guards = [
        ast.unparse(node.test)
        for node in ast.walk(role)
        if isinstance(node, ast.If)
        and "pages[request, logical_page]" in ast.unparse(node)
    ]
    assert guards == [
        "serving & (lane < depth) & (cutlass.Int64(lane) < count)",
        "(logical_page >= 0) & (logical_page < table_pages)",
    ]
    assert "table_pages = cute.size(pages, mode=[1])" in source
    # A request is formed only for an id inside the operand's pool: three
    # lanes request K pages, five V pages.
    requests = _calls(role, "_request_page")
    assert [ast.unparse(call.args[0]) for call in requests] == ["key", "value"]
    for call, operand, lanes in zip(
        requests,
        ("key", "value"),
        ("RANGE_AHEAD_KEY_PAGES", "RANGE_AHEAD_VALUE_PAGES"),
        strict=True,
    ):
        bounded, counted = (
            node
            for node in reversed(list(ast.walk(role)))
            if isinstance(node, ast.If)
            and any(part is call for part in ast.walk(node))
            and (
                "physical" in ast.unparse(node.test)
                or "RANGE_AHEAD" in ast.unparse(node.test)
            )
        )
        tests = {ast.unparse(bounded.test), ast.unparse(counted.test)}
        assert tests == {
            f"(physical >= 0) & (physical < {operand}_layout[2])",
            f"lane < {lanes}",
        }
    # The lengths come from the scan's words in shared memory: the role
    # has the twin that takes a scanned length, and one unit a thread.
    assert "assert cutlass.const_expr(chunk == 1)" in source
    body = ast.unparse(_function("paged_attention_paired_body"))
    assert "assert cutlass.const_expr(unit_bounds_scanned is not None)" in body
    # Nothing is stored, and no barrier is reached.
    assert not any(
        isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
        for node in ast.walk(role)
    )
    called = {
        ast.unparse(call.func).rpartition(".")[2]
        for call in ast.walk(role)
        if isinstance(call, ast.Call)
    }
    assert not [name for name in called if "barrier" in name or "mbar" in name]


# One compile per kernel, about five seconds each.
def test_the_role_renders_and_compiles_on_the_balanced_partition(tmp_path):
    """The walk's balanced kernel without the option, with it and with it
    beside a ring of five tiles, rendered and compiled the way a launch
    compiles it: the role's four requests (a page's rows and scales, of K
    and of V)."""
    bound = stream_bound(WAVES)
    args = tuple(partition_attention_args(units=WAVES, rows=8, window=PAGES * PAGE - 1))
    walk = stream_render(bound, stream_config(**WALK))
    assert not [
        name for name in body_keywords(walk) if name.startswith("boundary_prefetch")
    ]
    ahead = stream_render(bound, stream_config(**WALK, **{KEY: 96}))
    keywords = body_keywords(ahead)
    # The distance is the body's own keyword: the page count and its form
    # are those of one unit per CTA, which this kernel does not have.
    assert keywords["boundary_prefetch_ahead"] == "96"
    assert "boundary_prefetch" not in keywords
    assert "boundary_prefetch_form" not in keywords
    assert (keywords["boundary_prefetch_key"], keywords["boundary_prefetch_value"]) == (
        "k",
        "v",
    )
    assert keywords["boundary_prefetch_key_scales"] == "k_scale"
    assert keywords["boundary_prefetch_value_scales"] == "v_scale"
    # A page's byte strides in the raw and the scale tensor, and the pool.
    assert keywords["boundary_prefetch_layout"] == "(33792, 33792, 13)"
    assert keywords["boundary_prefetch_value_layout"] == "(33792, 33792, 13)"
    assert "unit_bounds_scanned" in keywords
    both = stream_render(bound, stream_config(**WALK, **{KEY: 48, TILES: 5}))
    keywords = body_keywords(both)
    assert keywords["boundary_prefetch_ahead"] == "48"
    assert keywords["value_ring_tiles"] == "5"
    compiled = {
        name: compile_offline(bound, args, source, tmp_path / name)
        for name, source in (("walk", walk), ("ahead", ahead), ("both", both))
    }
    requests = {
        name: kernel.ptx.count(PREFETCH_INSTRUCTION)
        for name, kernel in compiled.items()
    }
    assert requests == {"walk": 0, "ahead": 4, "both": 4}
    assert len({kernel.ptx for kernel in compiled.values()}) == 3
    assert all(
        kernel.threads == 640 and kernel.launch_registers == 96
        for kernel in compiled.values()
    )
    # The body refuses a distance beside a page count, and on another
    # partition than the balanced one.
    with pytest.raises(AssertionError):
        compile_offline(
            bound,
            args,
            ahead.replace(
                "boundary_prefetch_ahead=96",
                "boundary_prefetch_ahead=96, boundary_prefetch=8",
            ),
            tmp_path / "pages",
        )
    with pytest.raises(AssertionError):
        compile_offline(
            bound,
            args,
            ahead.replace("partition='balanced'", "partition='whole'"),
            tmp_path / "whole",
        )


def test_the_render_leaves_the_role_out_where_the_program_stores_a_tensor_it_loads(
    monkeypatch,
):
    """The role reads the page table where a launch may read its inputs,
    ahead of a captured launch's dependency wait, whose proof covers the
    tensors the program loads. The matcher declines a program that loads a
    tensor it stores, so no kernel reaches the render with one, and the
    config contract does not look: the render repeats the test against the
    program it emits. With every store counted as a load of its tensor as
    well, a config that names a distance renders the kernel without the
    role."""
    bound = stream_bound(WAVES)
    walk = stream_config(**WALK)
    ahead = stream_config(**WALK, **{KEY: 96})
    # The control: the program the matcher admitted renders the role.
    assert body_keywords(stream_render(bound, ahead))["boundary_prefetch_ahead"] == "96"
    original = paged_attention_codegen._is_call

    def loads_what_it_stores(node, target):
        return original(node, target) or (
            target is memory_ops.load and original(node, memory_ops.store)
        )

    monkeypatch.setattr(paged_attention_codegen, "_is_call", loads_what_it_stores)
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        assert bound.config_spec.normalized_config(ahead).config[KEY] == 96
    source = stream_render(bound, ahead)
    assert not [
        name for name in body_keywords(source) if name.startswith("boundary_prefetch")
    ]
    assert source == stream_render(bound, walk)


def test_the_distance_is_a_field_and_seeded_together_with_the_ring_and_the_walk():
    eager = stream_bound(WAVES)
    fields = eager.config_spec._flat_fields()
    # A field of its own beside the page counts of one unit per CTA, which
    # hold no distance, sampled where those are: over bank cycles.
    assert fields[KEY] is eager.config_spec.cute_mma_dag_boundary_prefetch_ahead
    assert fields[KEY].choices == (0, 48, 96) and fields[KEY].default() == 0
    assert fields[KEY]._active_choices() == (0,)
    assert fields[PAGES_KEY].choices == (0, 8, 12, 4)
    assert fields[PAGES_KEY]._active_choices() == (0,)
    # One wave: no distance.
    one = stream_bound(ONE_WAVE, input_banks=2)
    assert KEY not in one.config_spec._flat_fields()
    assert one.config_spec.cute_mma_dag_boundary_prefetch_ahead is None
    # Three waves of one unit per CTA have no form for pages; the balanced
    # partition of that grid has its distances.
    three = stream_bound(444, input_banks=2).config_spec._flat_fields()
    assert PAGES_KEY not in three and three[KEY].choices == (0, 48, 96)
    # No seed names a distance or a tile count where the search does not
    # sample the prefetches.
    assert not any(
        TILES in seed.config or KEY in seed.config for seed in stream_seeds(eager)
    )
    banks = stream_bound(WAVES, input_banks=2)
    searched = banks.config_spec._flat_fields()
    assert searched[KEY]._active_choices() == (0, 48, 96)
    assert searched[PAGES_KEY]._active_choices() == (0, 8, 12, 4)
    assert VALUE_RING_TILES_SEED == 5
    seeds = stream_seeds(banks)
    ahead = seeds_ahead_of_the_range_ahead(seeds)
    # The balanced seed in reversed order, with the clipped page eight
    # pages from a unit's end, and its form under the store hint.
    (walked,) = [
        seed
        for seed in ahead
        if seed.config.get(PARTITION) == "balanced"
        and seed.config.get(ORDER) == "reversed"
        and "cute_store_policy" not in seed.config
    ]
    assert walked.config[LEAD] == 8
    triples = [{**walked.config, KEY: distance, TILES: 5} for distance in (48, 96)]
    assert [seed.config for seed in seeds[len(ahead) :]] == [
        *triples,
        *({**triple, "cute_store_policy": "l2_evict_last"} for triple in triples),
    ]
    # No seed has one of the three without the others, and none names a
    # distance as a page count.
    assert not any(TILES in seed.config or KEY in seed.config for seed in ahead)
    assert all(seed.config.get(PAGES_KEY, 0) in (0, 8, 12, 4) for seed in seeds)
    # One wave has no such seed, nor has a kernel without a window, which
    # has the field.
    assert not any(
        TILES in seed.config or KEY in seed.config
        for seed in stream_seeds(stream_bound(ONE_WAVE, input_banks=2))
    )
    full = stream_bound(WAVES, input_banks=2, window=NO_WINDOW)
    assert full.config_spec._flat_fields()[KEY].choices == (0, 48, 96)
    assert not any(
        TILES in seed.config or KEY in seed.config for seed in stream_seeds(full)
    )
    spec = banks.config_spec
    with cpu_render_target(fake_device_sm_count=SMS), banks.env, banks.host_function:
        generation = spec.create_config_generation()
        for seed in seeds[len(ahead) :]:
            # A complete legal config, and a search's row is that config.
            normalized = spec.normalized_config(seed)
            assert all(
                normalized.config[key] == value for key, value in seed.config.items()
            )
            _flat, row = generation.canonicalize_flat(generation.flatten(normalized))
            assert all(
                row.config.get(key) == seed.config.get(key)
                for key in (KEY, PAGES_KEY, TILES, PARTITION, ORDER, LEAD)
            )
        # A step onto a distance from one unit per CTA takes the balanced
        # partition and its transport, and leaves the pages of one unit per
        # CTA behind; onto pages from the balanced partition, one unit per
        # CTA, which leaves the distance behind.
        reached, alone = stream_step(spec, stream_config(), **{KEY: 48})
        assert (reached.config[KEY], reached.config[PARTITION]) == (48, "balanced")
        assert reached.config[QUERY_TMA] is True
        assert KEY not in alone.config
        reached, _alone = stream_step(
            spec, stream_config(**{PAGES_KEY: 8}), **{KEY: 48}
        )
        assert (reached.config[KEY], reached.config[PARTITION]) == (48, "balanced")
        assert PAGES_KEY not in reached.config
        reached, _alone = stream_step(spec, stream_config(**BALANCED), **{KEY: 96})
        assert (reached.config[KEY], reached.config[PARTITION]) == (96, "balanced")
        parent = spec.normalized_config(stream_config(**WALK, **{KEY: 48, TILES: 5}))
        reached, _alone = stream_step(spec, parent, **{PAGES_KEY: 8})
        assert reached.config[PAGES_KEY] == 8 and PARTITION not in reached.config
        assert KEY not in reached.config and TILES not in reached.config
        # A step between the two distances changes nothing else.
        reached, _alone = stream_step(spec, parent, **{KEY: 96})
        assert all(
            reached.config[key] == value
            for key, value in {**parent.config, KEY: 96}.items()
        )
        # A step off the balanced partition leaves the distance behind.
        reached, _alone = stream_step(spec, parent, **{PARTITION: "item"})
        assert KEY not in reached.config and TILES not in reached.config
        reached, _alone = stream_step(spec, parent, **{PARTITION: "whole"})
        assert KEY not in reached.config and reached.config[TILES] == 5


def test_a_fixed_distance_moves_every_paired_candidate_onto_the_balanced_partition():
    """An ``autotune_config_overrides`` entry that names a distance is
    layered over complete candidates, counted seed by seed: every seed on
    the paired schedule transfers on the balanced partition with it, and
    what that partition does not run or hold is dropped with the move, as
    a search's step onto the distance leaves a row."""
    banks = stream_bound(WAVES, input_banks=2)
    spec = banks.config_spec
    with cpu_render_target(fake_device_sm_count=SMS), banks.env, banks.host_function:
        moves = seed_transfers(spec, {KEY: 96})
        paired = _paired(moves)
        # 51 seeds on the paired schedule: 43 with one unit per CTA, which
        # take the balanced partition and its query TMA; 6 that prefetch
        # pages, which are dropped with that partition; 3 with a third page
        # per head, which eight query columns hold on a persistent CTA.
        assert len(paired) == 51
        assert sum(PARTITION not in seed.config for seed, _moved in paired) == 43
        assert sum(PAGES_KEY in seed.config for seed, _moved in paired) == 6
        assert sum(RING in seed.config for seed, _moved in paired) == 3
        for seed, moved in paired:
            assert moved is not None, seed
            assert (moved.config[KEY], moved.config[PARTITION]) == (96, "balanced")
            assert moved.config[QUERY_TMA] is True
            assert PAGES_KEY not in moved.config and COLD not in moved.config
            assert moved.config.get(RING) == seed.config.get(RING)
            assert moved.config.get(TILES) == seed.config.get(TILES)
            assert moved.config.get(ORDER) == seed.config.get(ORDER)
        # The seeds of the other schedules have no such prefetch: the search
        # rejects them, and only them.
        forced, failures = transferred_seeds(spec, {KEY: 96})
        assert len(failures) == len(moves) - len(paired) == 12
        assert all(KEY in failure for failure in failures)
        assert forced[0] == paired[0][1]
        # A fixed page count moves every paired seed to one unit per CTA,
        # the four range-ahead seeds among them: their distance and their
        # tile count name nothing there and go.
        paged = _paired(seed_transfers(spec, {PAGES_KEY: 12}))
        assert len(paged) == 51
        assert sum(KEY in seed.config for seed, _moved in paged) == 4
        for seed, moved in paged:
            assert moved is not None, seed
            assert moved.config[PAGES_KEY] == 12
            assert not any(key in moved.config for key in (PARTITION, TILES, KEY))
        # Under a fixed tile count the four seeds with a distance keep it.
        tiled = _paired(seed_transfers(spec, {TILES: 5}))
        assert len(tiled) == 51 and all(moved is not None for _seed, moved in tiled)
        for seed, moved in tiled:
            assert moved is not None
            assert moved.config.get(KEY) == seed.config.get(KEY)
        assert sum(KEY in seed.config for seed, _moved in tiled) == 4
        # An override that takes away what a distance stands on takes it
        # along; a skew is the balanced partition's own.
        for overrides, kept in (
            ({PARTITION: "item"}, 0),
            ({PARTITION: "whole"}, 0),
            ({SKEW: 1}, 4),
            ({RING: 3}, 4),
        ):
            moved, failures = transferred_seeds(spec, overrides)
            assert moved and not [failure for failure in failures if KEY in failure], (
                overrides
            )
            assert sum(KEY in seed.config for seed in moved) == kept, overrides
    # Forty query columns, where a persistent CTA does not hold a third
    # page per head: the three seeds that name it transfer at the default
    # depth, and all 51 transfer.
    wide = stream_bound(WAVES, rows=40, input_banks=2)
    with cpu_render_target(fake_device_sm_count=SMS), wide.env, wide.host_function:
        paired = _paired(seed_transfers(wide.config_spec, {KEY: 96}))
        assert len(paired) == 51
        assert sum(RING in seed.config for seed, _moved in paired) == 3
        for seed, moved in paired:
            assert moved is not None, seed
            assert (moved.config[KEY], moved.config[PARTITION]) == (96, "balanced")
            assert RING not in moved.config
        # An entry that names the depth as well drops nothing: the
        # candidates are rejected by the ring their partition cannot hold.
        forced, failures = transferred_seeds(wide.config_spec, {KEY: 96, RING: 3})
        assert not [
            seed for seed in forced if seed.config["cute_mma_dag_pass_warps"] == PAIRED
        ]
        assert any(f"{RING} must be one of (2,)" in failure for failure in failures)
    # Without a window, over a page table long enough for it, the bucketed
    # order is the only other order, and one unit per CTA's: a candidate in
    # that order transfers in the source's, as the step onto a distance
    # from it does, and every paired seed transfers.
    full = stream_bound(
        WAVES, input_banks=2, window=NO_WINDOW, table_pages=64, pool_pages=80
    )
    spec = full.config_spec
    with cpu_render_target(fake_device_sm_count=SMS), full.env, full.host_function:
        assert spec._flat_fields()[ORDER].choices == ("source", "bucketed")
        own = spec.normalized_config(stream_config(**{ORDER: "bucketed"}))
        assert own.config[ORDER] == "bucketed" and PARTITION not in own.config
        generation = spec.create_config_generation(overrides={KEY: 48})
        pair = generation.transfer_seed(helion.Config.from_dict(dict(own.config)))
        assert pair is not None
        assert (pair[1].config[KEY], pair[1].config[PARTITION]) == (48, "balanced")
        assert ORDER not in pair[1].config
        reached, _alone = stream_step(spec, own, **{KEY: 48})
        assert (reached.config[KEY], reached.config[PARTITION]) == (48, "balanced")
        assert ORDER not in reached.config
        paired = _paired(seed_transfers(spec, {KEY: 48}))
        assert paired and all(
            moved is not None
            and (moved.config[KEY], moved.config[PARTITION]) == (48, "balanced")
            for _seed, moved in paired
        )
        # An entry that names the order as well drops nothing.
        generation = spec.create_config_generation(
            overrides={KEY: 48, ORDER: "bucketed"}
        )
        assert (
            generation.transfer_seed(helion.Config.from_dict(dict(own.config))) is None
        )
