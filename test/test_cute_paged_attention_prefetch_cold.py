"""Check the depth the sibling boundary prefetch takes in a cold launch
(``cute_mma_dag_boundary_prefetch_cold``): the device rule it adds to the
role, the config contract that ties it to the sibling prefetch, what it
renders and compiles to, and where a search samples and seeds it."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from helion import exc
from helion._compiler.cute import _paged_attention_prefetch as device
from helion._compiler.cute._paged_attention_prefetch import SIBLING_LANES
from helion._compiler.cute.autotune import mma_dag_surface
from helion._compiler.cute.autotune.mma_dag_surface import PAIRED_NEED_DEFAULTS
from helion._compiler.cute.cute_mma_dag import MMA_DAG_BOUNDARY_PREFETCH_COLD_CHOICES
from helion._compiler.cute.cute_mma_dag import (
    paged_attention_boundary_prefetch_cold_choices,
)
from helion._compiler.cute.cute_mma_dag_seeds import BOUNDARY_PREFETCH_COLD_SEED_PAGES
from helion._compiler.cute.paged_attention_config import (
    paged_attention_boundary_prefetch_cold_reason,
)
from helion._compiler.cute.pdl_release import CUTE_PDL_CAPTURE_DEFAULT
from helion._testing import cpu_render_target
from helion._testing.cute_offline import compile_offline
from helion._testing.cute_paged_partition import partition_attention_args
from helion._testing.cute_paged_stream import CAPTURE_POLICY
from helion._testing.cute_paged_stream import NO_WINDOW
from helion._testing.cute_paged_stream import PAIRED
from helion._testing.cute_paged_stream import SMS
from helion._testing.cute_paged_stream import body_keywords
from helion._testing.cute_paged_stream import seed_transfers
from helion._testing.cute_paged_stream import seeds_ahead_of_the_cold_prefetch
from helion._testing.cute_paged_stream import stream_bound
from helion._testing.cute_paged_stream import stream_config
from helion._testing.cute_paged_stream import stream_render
from helion._testing.cute_paged_stream import stream_seeds
from helion._testing.cute_paged_stream import stream_step
from helion._testing.cute_paged_stream import transferred_seeds

pytestmark = pytest.mark.cpu_only
KEY = "cute_mma_dag_boundary_prefetch_cold"
PAGES = "cute_mma_dag_boundary_prefetch"
PARTITION = "cute_mma_dag_partition"
PREFETCH_INSTRUCTION = "cp.async.bulk.prefetch.L2.global"
# Units on 148 SMs: a grid of one wave that leaves a few SMs idle, whose
# prefetch has the sibling form, and one past a wave, whose prefetch serves
# the unit a capacity ahead and has no cold launch to speak of.
ONE_WAVE = 128
WAVES = 300


def _role() -> ast.FunctionDef:
    (role,) = [
        node
        for node in ast.walk(ast.parse(Path(inspect.getfile(device)).read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "prefetch_boundary_units"
    ]
    return role


def test_a_cold_launch_serves_as_an_overlapped_one_would_to_its_own_depth():
    """The rule the option adds to the sibling form, read from the role and
    replayed on the host: a launch whose head start is zero takes the head
    start an overlapped launch of the grid would have, the SMs it leaves
    idle, and requests to the cold depth; a launch with a head start is
    untouched."""
    role = _role()
    (sibling,) = [
        node
        for node in role.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.const_expr(form == 'sibling')"
    ]
    (cold,) = [
        node
        for node in sibling.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.const_expr(cold_depth > 0)"
    ]
    assert [ast.unparse(node) for node in cold.body] == [
        "cold = cutlass.Int32(head == 0)",
        "head = head + cutlass.Int32(capacity - units) * cold",
        "limit = limit + cutlass.Int32(cold_depth - depth) * cold",
    ]
    # The rule sits between the head start the host installed and the
    # serving rule that reads it.
    order = [ast.unparse(node).split(" = ")[0] for node in sibling.body]
    assert order[0] == "head" and order[2:] == ["target", "serving"]
    source = ast.unparse(role)
    assert "resolved = max(depth, cold_depth)" in source
    assert "limit = cutlass.Int32(depth)" in source
    # Only the sibling form of a grid of fewer units than SMs has it.
    assert (
        "cutlass.const_expr(cold_depth == 0 or (form == 'sibling' and units < capacity))"
        in source
    )
    # A page beyond the launch's depth keeps no id, so no request is formed
    # for it: checked only for the pages one of the two depths leaves out.
    (dropped,) = [
        node
        for node in ast.walk(role)
        if isinstance(node, ast.If)
        and ast.unparse(node.test)
        == "cutlass.const_expr(cold_depth > 0 and index >= min(depth, cold_depth))"
    ]
    assert ast.unparse(dropped.body[0]) == (
        "if index >= limit:\n    page = cutlass.Int32(-1)"
    )

    def launch(units: int, installed: int, depth: int, cold_depth: int):
        """Served unit -> its server, and the pages requested for it."""
        head, limit = installed, depth
        if cold_depth:
            cold = int(head == 0)
            head = head + (SMS - units) * cold
            limit = limit + (cold_depth - depth) * cold
        served = {}
        for unit in range(units):
            for lane in range(32):
                target = unit + head * (lane + 1)
                if unit < head and lane < SIBLING_LANES and target < units:
                    assert target not in served
                    served[target] = unit
        return served, limit

    for units in (120, 128, 140, 147):
        idle = SMS - units
        overlapped, depth = launch(units, idle, 8, 0)
        assert depth == 8 and overlapped
        # Without the option a launch without a head start serves nobody.
        assert launch(units, 0, 8, 0) == ({}, 8)
        # With it the same units have the same servers, to two pages.
        assert launch(units, 0, 8, 2) == (overlapped, 2)
        assert set(overlapped.values()) <= set(range(idle))
        # An overlapped launch keeps its own depth, whatever head start
        # the host installed.
        assert launch(units, idle, 8, 2) == (overlapped, 8)
        assert launch(units, 3, 12, 2) == (launch(units, 3, 12, 0)[0], 12)


def test_a_cold_depth_needs_the_sibling_prefetch_it_extends():
    bound = stream_bound(ONE_WAVE)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    assert facts.attention.boundary_prefetch_form == "sibling"
    assert MMA_DAG_BOUNDARY_PREFETCH_COLD_CHOICES == (0, 2)
    legal = {"cute_mma_dag_pass_warps": PAIRED, PAGES: 8, KEY: 2}
    assert paged_attention_boundary_prefetch_cold_reason(facts, legal) is None
    assert paged_attention_boundary_prefetch_cold_reason(facts, {}) is None
    assert paged_attention_boundary_prefetch_cold_reason(facts, {KEY: 0}) is None
    for pages in (4, 8, 12):
        assert paged_attention_boundary_prefetch_cold_choices(
            facts.attention, PAIRED, False, "item", pages
        ) == (0, 2)
    # No pages, pages the option does not have, another schedule, the
    # value-metadata transport, a persistent partition: no sibling role.
    for change in (
        {PAGES: 0},
        {PAGES: 2},
        {"cute_mma_dag_pass_warps": "attention"},
        {"cute_mma_dag_value_metadata_skip": True},
        {PARTITION: "balanced"},
    ):
        reason = paged_attention_boundary_prefetch_cold_reason(
            facts, {**legal, **change}
        )
        assert reason is not None and reason.startswith(f"{KEY} must be one of (0,)")
    # Another depth than the option's, or another type.
    for value in (1, 3, 8, -2, True, "2", 2.0):
        reason = paged_attention_boundary_prefetch_cold_reason(
            facts, {**legal, KEY: value}
        )
        assert reason is not None and "(0, 2)" in reason
    # Past one wave the prefetch has the other form, which has no launch
    # without servers.
    waves = stream_bound(WAVES).config_spec.cute_mma_dag_facts
    assert waves is not None and waves.attention is not None
    assert waves.attention.boundary_prefetch_form == "wave_ahead"
    assert paged_attention_boundary_prefetch_cold_reason(waves, legal) is not None
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        spec = bound.config_spec
        kept = spec.normalized_config(stream_config(**{PAGES: 8, KEY: 2}))
        assert (kept.config[PAGES], kept.config[KEY]) == (8, 2)
        # The default is the missing key.
        assert (
            KEY
            not in spec.normalized_config(stream_config(**{PAGES: 8, KEY: 0})).config
        )
        for knobs in (
            {KEY: 2},
            {PAGES: 8, KEY: 3},
            {PAGES: 8, KEY: 2, PARTITION: "balanced", "cute_mma_dag_query_tma": True},
            # Without the capture policy the pages have no role on this
            # grid, and the depth goes with them.
            {PAGES: 8, KEY: 2, "cute_pdl": False},
        ):
            config = stream_config(**knobs)
            with pytest.raises(exc.InvalidConfig):
                spec.normalized_config(config)
            spec.normalize(config, _fix_invalid=True)
            assert KEY not in config.config, knobs
        # A config the kernel selects without a search loses the depth with
        # the pages when overrides decline the policy both need.
        compiled = spec.unsearched_config(kept, {"cute_pdl": False})
        assert KEY not in compiled.config and PAGES not in compiled.config


# One compile per kernel: about four seconds each.
def test_the_cold_depth_renders_and_compiles_with_the_role(tmp_path):
    """The render states the depth beside the sibling role, and the role
    compiles with it the way a launch compiles it: the same requests in the
    PTX (a serving lane resolves the deeper of the two depths and requests
    up to the launch's), in another kernel than the role without it."""
    bound = stream_bound(ONE_WAVE)
    args = tuple(partition_attention_args(units=ONE_WAVE, rows=8, heads=2))
    plain = stream_render(bound, stream_config(**{PAGES: 8}))
    cold = stream_render(bound, stream_config(**{PAGES: 8, KEY: 2}))
    assert KEY.removeprefix("cute_mma_dag_") not in body_keywords(plain)
    keywords = body_keywords(cold)
    assert keywords["boundary_prefetch_cold"] == "2"
    assert keywords["boundary_prefetch"] == "8"
    assert keywords["boundary_prefetch_form"] == "'sibling'"
    assert cold.replace(", boundary_prefetch_cold=2", "") == plain
    with_depth = compile_offline(bound, args, cold, tmp_path / "cold")
    without = compile_offline(bound, args, plain, tmp_path / "plain")
    assert with_depth.ptx != without.ptx
    assert with_depth.ptx.count(PREFETCH_INSTRUCTION) == 2 * 8
    assert without.ptx.count(PREFETCH_INSTRUCTION) == 2 * 8
    assert with_depth.threads == 640 and with_depth.launch_registers == 96
    # No barrier or warp synchronization of its own.
    for instruction in ("bar.warp.sync", "bar.sync", "barrier.sync"):
        assert with_depth.ptx.count(instruction) == without.ptx.count(instruction)
    # The body refuses a depth beside another form.
    wave = stream_bound(WAVES)
    wave_args = tuple(partition_attention_args(units=WAVES, rows=8, heads=2))
    wave_source = stream_render(wave, stream_config(**{PAGES: 8}))
    form = "boundary_prefetch_form='wave_ahead'"
    assert wave_source.count(form) == 1
    with pytest.raises(AssertionError):
        compile_offline(
            wave,
            wave_args,
            wave_source.replace(form, f"{form}, boundary_prefetch_cold=2"),
            tmp_path / "wave",
        )


def test_the_cold_depth_is_sampled_and_seeded_with_the_sibling_prefetch():
    eager = stream_bound(ONE_WAVE)
    fragment = eager.config_spec._flat_fields()[KEY]
    assert fragment is eager.config_spec.cute_mma_dag_boundary_prefetch_cold
    assert fragment.choices == (0, 2) and fragment.default() == 0
    # Off is all a search without bank cycles draws, as for the pages.
    assert fragment._active_choices() == (0,)
    banks = stream_bound(ONE_WAVE, input_banks=2)
    assert banks.config_spec._flat_fields()[KEY]._active_choices() == (0, 2)
    # Past one wave, where the prefetch has its other form, no field.
    for units in (WAVES,):
        waves = stream_bound(units, input_banks=2)
        assert waves.config_spec.cute_mma_dag_boundary_prefetch is not None
        assert KEY not in waves.config_spec._flat_fields()
        assert not any(KEY in seed.config for seed in stream_seeds(waves))
    # No seed names it where the search does not sample the prefetch.
    assert not any(KEY in seed.config for seed in stream_seeds(eager))
    # Where it does, on a windowed kernel: every seed that prefetches
    # follows once more with two pages in a cold launch, behind every other
    # seed.
    assert BOUNDARY_PREFETCH_COLD_SEED_PAGES == 2
    seeds = stream_seeds(banks)
    ahead = seeds_ahead_of_the_cold_prefetch(seeds)
    prefetching = [seed for seed in ahead if PAGES in seed.config]
    assert len(prefetching) == 6
    assert [seed.config for seed in seeds[len(ahead) :]] == [
        {**seed.config, KEY: 2} for seed in prefetching
    ]
    # A kernel without a window has no such seed: the depth was measured on
    # a window's few pages, which a later launch finds again.
    full = stream_bound(ONE_WAVE, input_banks=2, window=NO_WINDOW)
    assert KEY in full.config_spec._flat_fields()
    assert not any(KEY in seed.config for seed in stream_seeds(full))
    spec = banks.config_spec
    with cpu_render_target(fake_device_sm_count=SMS), banks.env, banks.host_function:
        generation = spec.create_config_generation()
        for seed in seeds[len(ahead) :]:
            # A complete legal config, and a search's row is that config.
            normalized = spec.normalized_config(seed)
            assert normalized.config[KEY] == 2
            _flat, row = generation.canonicalize_flat(generation.flatten(normalized))
            assert (row.config[KEY], row.config[PAGES]) == (2, seed.config[PAGES])
        # A step onto the depth from a config without pages takes the pages
        # and what they need; alone it would be an alias of its parent.
        reached, alone = stream_step(spec, stream_config(cute_pdl=False), **{KEY: 2})
        assert (reached.config[KEY], reached.config[PAGES]) == (2, 8)
        assert all(
            reached.config[name] == value for name, value in CAPTURE_POLICY.items()
        )
        assert KEY not in alone.config and PAGES not in alone.config
        # From a config that prefetches it keeps that config's pages.
        reached, _alone = stream_step(spec, stream_config(**{PAGES: 12}), **{KEY: 2})
        assert (reached.config[KEY], reached.config[PAGES]) == (2, 12)
        # And a step off the pages leaves the depth nothing to extend.
        parent = spec.normalized_config(stream_config(**{PAGES: 12, KEY: 2}))
        reached, _alone = stream_step(spec, parent, **{PAGES: 0})
        assert KEY not in reached.config and PAGES not in reached.config
        # A fixed depth is layered over complete candidates, and every seed
        # on the paired schedule transfers with it, counted seed by seed: one
        # that prefetches (the six and their forms with a depth) keeps its
        # pages, and one without pages takes the first page count with what
        # the pages need.
        moves = seed_transfers(spec, {KEY: 2})
        paired = [
            (seed, moved)
            for seed, moved in moves
            if seed.config["cute_mma_dag_pass_warps"] == PAIRED
        ]
        without = [seed for seed, _moved in paired if PAGES not in seed.config]
        assert without and len(paired) - len(without) == 2 * len(prefetching)
        for seed, moved in paired:
            assert moved is not None, seed
            assert moved.config[KEY] == 2
            assert moved.config[PAGES] == seed.config.get(PAGES, 8)
            assert all(
                moved.config[name] == value for name, value in CAPTURE_POLICY.items()
            )
        # The seeds of the other schedules have no prefetch to extend: the
        # search rejects them, and only them.
        forced, failures = transferred_seeds(spec, {KEY: 2})
        assert 0 < len(failures) == len(moves) - len(paired)
        assert all(KEY in failure for failure in failures)
        assert forced[0] == paired[0][1]
        assert {seed.config[PAGES] for seed in forced} == {8, 12}
        # An override that takes the pages away takes a candidate's own
        # depth with them.
        for overrides in ({PAGES: 0}, {PARTITION: "balanced"}, {"cute_pdl": False}):
            moved, failures = transferred_seeds(spec, overrides)
            assert moved and not [failure for failure in failures if KEY in failure]
            assert not any(KEY in seed.config for seed in moved), overrides
        # One that keeps them keeps it.
        kept, _failures = transferred_seeds(spec, {PAGES: 12})
        assert any(seed.config.get(KEY) == 2 for seed in kept)
        assert all(seed.config[PAGES] == 12 for seed in kept if KEY in seed.config)
    # A fixed override is no sample: a search that does not draw the prefetch
    # gives every paired candidate the pages its fixed depth needs as well.
    with cpu_render_target(fake_device_sm_count=SMS), eager.env, eager.host_function:
        moves = seed_transfers(eager.config_spec, {KEY: 2})
        paired = [
            moved
            for seed, moved in moves
            if seed.config["cute_mma_dag_pass_warps"] == PAIRED
        ]
        assert paired and len(paired) < len(moves)
        assert all(
            moved is not None and (moved.config[KEY], moved.config[PAGES]) == (2, 8)
            for moved in paired
        )


def test_every_option_a_need_names_has_its_default_in_the_table():
    """A fixed override gives each paired candidate what its knob needs,
    wherever the candidate's own value, or the option's default for an
    option it does not name, differs. The default is looked up in
    ``PAIRED_NEED_DEFAULTS`` (the capture policy's three options are taken
    by name), so every option a need can name must be in that table: an
    option outside it would be a ``KeyError`` on the override path alone.
    Every function of the surface that states a need (``*_needs``) is
    called here over its cases; one this test has no cases for fails it."""
    binds = [
        stream_bound(ONE_WAVE, input_banks=2),
        stream_bound(WAVES, input_banks=2),
        # Forty query columns: a persistent CTA does not hold a deeper ring.
        stream_bound(WAVES, rows=40, input_banks=2),
        stream_bound(WAVES, input_banks=2, window=NO_WINDOW),
        # A device without room for a persistent partition.
        stream_bound(WAVES, smem_capacity_bytes=160000),
    ]
    attentions = [None]
    for bound in binds:
        facts = bound.config_spec.cute_mma_dag_facts
        assert facts is not None and facts.attention is not None
        attentions.append(facts.attention)
    forms = (None, "", "wave_ahead", "sibling")
    partitions = ("item", "whole", "balanced")
    cases: dict[str, list[tuple[object, ...]]] = {
        "boundary_prefetch_needs": [(form,) for form in forms],
        "boundary_prefetch_cold_needs": [
            (form, pages) for form in forms for pages in (0, 8, 12, 4)
        ],
        "boundary_prefetch_ahead_needs": [()],
        "_paired_loader_needs": [(True,), (False,)],
        "page_order_needs": [
            (attention, order)
            for attention in attentions
            for order in ("source", "bucketed", "reversed")
        ],
        "value_ring_needs": [
            (attention, ring, partition)
            for attention in attentions
            for ring in (2, 3)
            for partition in partitions
        ],
        "value_ring_tiles_needs": [
            (attention, tiles, partition)
            for attention in attentions
            for tiles in (0, 5)
            for partition in partitions
        ],
    }
    needs = {
        name: value
        for name, value in vars(mma_dag_surface).items()
        if name.endswith("_needs")
        and inspect.isfunction(value)
        and value.__module__ == mma_dag_surface.__name__
    }
    assert sorted(needs) == sorted(cases)
    named: set[str] = set()
    for name, function in needs.items():
        for arguments in cases[name]:
            needed = function(*arguments)
            missing = (
                set(needed) - set(CUTE_PDL_CAPTURE_DEFAULT) - set(PAIRED_NEED_DEFAULTS)
            )
            assert not missing, (name, arguments, missing)
            named.update(needed)
    # The table holds those defaults and no other entry.
    assert named - set(CUTE_PDL_CAPTURE_DEFAULT) == set(PAIRED_NEED_DEFAULTS)
    # And each is the option's default: what the search's field of the
    # option has for a config that leaves it out, on every bind that has
    # the field.
    compared = set()
    for bound in binds:
        fields = bound.config_spec._flat_fields()
        for key, default in PAIRED_NEED_DEFAULTS.items():
            if key in fields:
                assert fields[key].default() == default, key
                compared.add(key)
    assert compared == set(PAIRED_NEED_DEFAULTS) - {
        "cute_mma_dag_value_metadata_skip",
        "cute_mma_dag_value_metadata_read_overlap",
    }
