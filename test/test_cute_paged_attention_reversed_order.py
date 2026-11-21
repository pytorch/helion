"""Check the paired schedule's reversed page order: the step-to-page mapping
of the template against the option's definition for every step, which
kernels are offered the walk, what a config may name and what it renders,
that every role takes its tiles from the one mapping, and where a search
samples and seeds the walk and the lead of its clipped page."""

from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
from pathlib import Path
import random
import re
import tempfile
from typing import cast
from unittest.mock import patch

import pytest

import helion
from helion import exc
from helion._compiler.cute import _paged_attention_page_order as bucketed_role
from helion._compiler.cute import _paged_attention_paired_pipeline as template
from helion._compiler.cute import _paged_attention_partial_pipeline as partial
from helion._compiler.cute import _paged_attention_prefetch as prefetch
from helion._compiler.cute import paged_attention_transport as transport
from helion._compiler.cute.autotune.mma_dag_surface import page_order_needs
from helion._compiler.cute.cute_mma_dag import paged_attention_clipped_page_lead_choices
from helion._compiler.cute.cute_mma_dag import paged_attention_page_order_choices
from helion._compiler.cute.cute_mma_dag_seeds import BOUNDARY_PREFETCH_SEED_PAGES
from helion._compiler.cute.cute_mma_dag_seeds import CLIPPED_PAGE_LEAD_SEED
from helion._compiler.cute.cute_mma_dag_seeds import VALUE_RING_SEED_PAGES
from helion._compiler.cute.paged_attention_config import (
    paged_attention_clipped_page_lead_reason,
)
from helion._compiler.cute.paged_attention_config import (
    paged_attention_page_order_reason,
)
from helion._compiler.cute.paged_attention_page_order import (
    MMA_DAG_CLIPPED_PAGE_LEAD_CHOICES,
)
from helion._compiler.cute.paged_attention_page_order import (
    MMA_DAG_CLIPPED_PAGE_LEAD_KEY,
)
from helion._compiler.cute.paged_attention_page_order import MMA_DAG_PAGE_ORDER_KEY
from helion._compiler.cute.paged_attention_page_order import PAGE_KEYS
from helion._compiler.cute.paged_attention_page_order import REVERSED_ORDER_NEEDS_PAIR
from helion._compiler.cute.paged_attention_page_order import ReversedWalk
from helion._compiler.cute.paged_attention_page_order import reversed_order_missing
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_offline import compile_offline
from helion._testing.cute_paged_partition import PAGE
from helion._testing.cute_paged_partition import SHORT_WINDOW
from helion._testing.cute_paged_partition import partition_attention_args
from helion._testing.cute_paged_partition import partition_attention_kernel
from helion._testing.cute_paged_partition_plan import reference_partition
from helion._testing.cute_paged_positive_zero import positive_zero_attention_fixture
from helion._testing.cute_paged_reversed_order import reversed_walk
from helion._testing.cute_paged_reversed_order import template_walk
from helion._testing.cute_paged_stream import NO_WINDOW
from helion._testing.cute_paged_stream import PAIRED
from helion._testing.cute_paged_stream import SMS
from helion._testing.cute_paged_stream import body_keywords
from helion._testing.cute_paged_stream import seeds_ahead_of_the_reversed_order
from helion._testing.cute_paged_stream import seeds_ahead_of_the_store_hint
from helion._testing.cute_paged_stream import stream_bound
from helion._testing.cute_paged_stream import stream_config
from helion._testing.cute_paged_stream import stream_render
from helion._testing.cute_paged_stream import stream_seeds
from helion._testing.cute_paged_stream import stream_step
from helion._testing.cute_paged_stream import transferred_seeds
from helion._testing.cute_template_source import source_function

pytestmark = pytest.mark.cpu_only
KEY = MMA_DAG_PAGE_ORDER_KEY
LEAD = MMA_DAG_CLIPPED_PAGE_LEAD_KEY
RING = "cute_mma_dag_value_ring"
PREFETCH = "cute_mma_dag_boundary_prefetch"
# The sibling prefetch's depth in a cold launch: the prefetching seeds in
# reversed order follow once more with it, behind every other seed
# (``test_cute_paged_attention_prefetch_cold`` pins those).
COLD = "cute_mma_dag_boundary_prefetch_cold"
PARTITION = "cute_mma_dag_partition"
SKIP = "cute_mma_dag_value_metadata_skip"
READ_OVERLAP = "cute_mma_dag_value_metadata_read_overlap"
V_LAG = "cute_mma_dag_v_lag"
# Units on 148 SMs: a grid of one wave, and one of more than two.
ONE_WAVE = 128
WAVES = 300
# A page table of 32 pages a request and a window of 1500 keys: a request
# longer than the window streams twelve or thirteen pages.
TABLE = 32
POOL = 48
WINDOW = 1500


def _bound(units: int = ONE_WAVE, **options: object):
    options.setdefault("table_pages", TABLE)
    options.setdefault("pool_pages", POOL)
    options.setdefault("window", WINDOW)
    return stream_bound(units, **options)  # type: ignore[arg-type]


def _reflected_bound():
    """The fixture whose source iterates its pages in reverse, under the
    same window."""
    args = partition_attention_args(
        units=ONE_WAVE,
        rows=8,
        reverse_pages=True,
        window=WINDOW,
        table_pages=TABLE,
        pool_pages=POOL,
    )
    with cpu_render_target(fake_device_sm_count=SMS, num_sm=SMS):
        return cpu_bind(
            cast("object", partition_attention_kernel(reverse_pages=True)),  # type: ignore[arg-type]
            args,
            num_sm=SMS,
        )


def _attention(bound):
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    return facts.attention


# A template function's definition (``source_function``), by the name this
# module's tests call it.
_function = source_function


def test_every_step_of_every_walk_takes_the_page_the_option_names():
    """The template's mapping, run from its original body, against the
    definition: every page of the range once, newest first, the oldest
    where the lead says."""
    assert KEY == "cute_mma_dag_page_order"
    assert LEAD == "cute_mma_dag_clipped_page_lead"
    assert MMA_DAG_CLIPPED_PAGE_LEAD_CHOICES == (0, 8)
    # Ranges of none to forty pages from several first pages, and every
    # range the fixture's source derives from a length: its windows from a
    # few keys to a 2047-key window's seventeen pages, one to five query
    # positions a request.
    shapes = {(first, first + count) for first in (0, 1, 7, 100) for count in range(41)}
    for window in (0, 100, SHORT_WINDOW, WINDOW, 2047):
        for query_len in (1, 4, 5):
            for length in range(0, 40 * PAGE, 13):
                begin = max(length - query_len - window, 0)
                shapes.add((begin // PAGE, (length + PAGE - 1) // PAGE))
    assert {last - first for first, last in shapes} >= set(range(41))
    walks = 0
    for first, last in sorted(shapes):
        count = last - first
        source = list(range(first, last))
        # The source's loop, and a source that reflects it.
        assert template_walk(first, last, None) == [
            (page, page * PAGE) for page in source
        ]
        reflected = template_walk(first, last, int)
        assert reflected == [(page, page * PAGE) for page in reversed(source)]
        for lead in (0, 1, 2, 3, 5, 8, 12, 17, 64):
            walk = template_walk(first, last, ReversedWalk(lead))
            pages = [page for page, _key_base in walk]
            # A step's key base is its own page's: the tile's mask and key
            # clip are those of the page it streams.
            assert all(key_base == page * PAGE for page, key_base in walk)
            # Every page of the source's range, once.
            assert sorted(pages) == source
            assert pages == reversed_walk(first, last, lead)
            if lead <= 1:
                # The page last: the reversal itself.
                assert walk == reflected
            if count:
                # The oldest page when the walk has the lead's pages left,
                # at its first step where it has fewer, and every other
                # page newest first around it.
                step = max(count - max(lead, 1), 0)
                assert pages[step] == first
                assert pages[:step] + pages[step + 1 :] == source[:0:-1]
            walks += 1
            # A step outside the range has no page; the mapping stays
            # defined for it (a prefetch evaluates a fixed depth of steps).
            beyond = template_walk(
                first, last, ReversedWalk(lead), range(count, count + 12)
            )
            assert all(key_base == page * PAGE for page, key_base in beyond)
    assert walks == 9 * len(shapes)
    # The eval's sliding window: seventeen pages, the clipped one at step
    # nine with a lead of eight, seven newer pages behind it.
    assert [page for page, _ in template_walk(3, 20, ReversedWalk(8))] == [
        *range(19, 10, -1),
        3,
        *range(10, 3, -1),
    ]


def test_a_kernel_is_offered_the_walk_under_fast_math_with_a_window():
    # What the kernel must have: the pair proof, fast math, a window in
    # its mask, and a source loop that is not itself reflected.
    wanted = {"paired": True, "fast_math": True, "windowed": True, "reflected": False}
    assert reversed_order_missing(**wanted) == ""
    for change, word in (
        ({"paired": False}, "pair proof"),
        ({"fast_math": False}, "fast_math"),
        ({"windowed": False}, "sliding window"),
        ({"reflected": True}, "oldest first"),
    ):
        missing = reversed_order_missing(**{**wanted, **change})
        assert word in missing
    assert reversed_order_missing(**{**wanted, "paired": False}) == (
        REVERSED_ORDER_NEEDS_PAIR
    )
    bound = _bound()
    attention = _attention(bound)
    assert attention.windowed and attention.reversed_order_missing == ""
    # A 1500-key window has no pages to sort: no bucketed order.
    assert not attention.page_order_available
    assert paged_attention_page_order_choices(attention, PAIRED) == (
        "source",
        "reversed",
    )
    # A window with pages to sort offers both.
    long = _attention(_bound(table_pages=64, pool_pages=80, window=50 * PAGE))
    assert paged_attention_page_order_choices(long, PAIRED) == (
        "source",
        "bucketed",
        "reversed",
    )
    # The eval's kernels are offered the walk past one wave as on one.
    assert _attention(_bound(WAVES)).reversed_order_missing == ""
    for other, word in (
        (_bound(window=NO_WINDOW), "sliding window"),
        (_bound(fast_math=False), "fast_math"),
        (_reflected_bound(), "oldest first"),
    ):
        attention = _attention(other)
        assert word in attention.reversed_order_missing
        assert "reversed" not in paged_attention_page_order_choices(attention, PAIRED)
        assert other.config_spec.cute_mma_dag_page_order is None
        assert other.config_spec.cute_mma_dag_clipped_page_lead is None
        fields = other.config_spec._flat_fields()
        assert KEY not in fields and LEAD not in fields
        assert not any(
            KEY in seed.config or LEAD in seed.config for seed in stream_seeds(other)
        )
    # The reflected fixture is windowed and runs under fast math: only its
    # loop keeps it from the walk.
    reflected = _attention(_reflected_bound())
    assert reflected.windowed and reflected.paired_items
    # One unit per CTA of the paired schedule, without the value-metadata
    # transport.
    attention = _attention(bound)
    for arguments in (
        ("attention",),
        (PAIRED, True),
        (PAIRED, False, "whole"),
        (PAIRED, False, "balanced"),
    ):
        assert paged_attention_page_order_choices(attention, *arguments) == ("source",)
        assert paged_attention_clipped_page_lead_choices(
            attention, *arguments, *(False, "item")[len(arguments) - 1 :], "reversed"
        ) == (0,)
    assert paged_attention_page_order_choices(None, PAIRED) == ("source",)
    assert paged_attention_page_order_choices(
        dataclasses.replace(attention, paired_items=False), PAIRED
    ) == ("source",)
    # The lead is the reversed walk's alone.
    assert paged_attention_clipped_page_lead_choices(
        attention, PAIRED, page_order="reversed"
    ) == (0, 8)
    for order in ("source", "bucketed"):
        assert paged_attention_clipped_page_lead_choices(
            long, PAIRED, page_order=order
        ) == (0,)


def test_the_source_order_renders_the_kernel_without_the_options():
    body = _function("paged_attention_paired_body")
    defaults = dict(
        zip(
            [argument.arg for argument in body.args.args][-len(body.args.defaults) :],
            [ast.unparse(default) for default in body.args.defaults],
            strict=True,
        )
    )
    assert defaults["page_order"] == "MMA_DAG_PAGE_ORDER_SOURCE"
    assert defaults["clipped_page_lead"] == "0"
    bound = _bound()
    plain = stream_render(bound, stream_config())
    assert "page_order" not in plain and "clipped_page_lead" not in plain
    for named in ({KEY: "source"}, {LEAD: 0}, {KEY: "source", LEAD: 8}, {LEAD: 8}):
        assert stream_render(bound, stream_config(**named)) == plain, named
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        spec = bound.config_spec
        default = spec.normalized_config(stream_config())
        assert KEY not in default.config and LEAD not in default.config
        assert KEY not in spec.default_config().config
        assert LEAD not in spec.default_config().config
    # The reversed walk names itself, and its lead where it has one, and
    # changes nothing else of the kernel's source: the walk is the body's.
    source = stream_render(bound, stream_config(**{KEY: "reversed"}))
    keywords = body_keywords(source)
    assert keywords["page_order"] == "'reversed'"
    assert "clipped_page_lead" not in keywords and "page_order_shift" not in keywords
    assert source.replace(", page_order='reversed'", "") == plain
    assert stream_render(bound, stream_config(**{KEY: "reversed", LEAD: 0})) == source
    source = stream_render(bound, stream_config(**{KEY: "reversed", LEAD: 8}))
    assert body_keywords(source)["clipped_page_lead"] == "8"
    assert source.replace(", page_order='reversed', clipped_page_lead=8", "") == plain


def test_a_lead_names_nothing_without_the_reversed_walk():
    """Normalization drops a lead beside every order but the reversed one,
    without an error, as it drops the default; a value that is none of the
    option's is an error anywhere."""
    assert paged_attention_clipped_page_lead_reason({}) is None
    for lead in MMA_DAG_CLIPPED_PAGE_LEAD_CHOICES:
        assert paged_attention_clipped_page_lead_reason({LEAD: lead}) is None
    for value in (5, 1, -8, 16, True, "8", 8.0, None):
        reason = paged_attention_clipped_page_lead_reason({LEAD: value})
        assert reason is not None and "(0, 8)" in reason and LEAD in reason
    bound = _bound(table_pages=64, pool_pages=80, window=50 * PAGE)
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        spec = bound.config_spec
        plain = spec.normalized_config(stream_config())
        for order in (None, "source", "bucketed"):
            named = {} if order is None else {KEY: order}
            for lead in (0, 8):
                config = spec.normalized_config(stream_config(**named, **{LEAD: lead}))
                assert LEAD not in config.config
                assert config == spec.normalized_config(stream_config(**named))
        kept = spec.normalized_config(stream_config(**{KEY: "reversed", LEAD: 8}))
        assert (kept.config[KEY], kept.config[LEAD]) == ("reversed", 8)
        last = spec.normalized_config(stream_config(**{KEY: "reversed", LEAD: 0}))
        assert last.config[KEY] == "reversed" and LEAD not in last.config
        assert last == spec.normalized_config(stream_config(**{KEY: "reversed"}))
        for order in (None, "reversed"):
            named = {} if order is None else {KEY: order}
            for value in (5, True, "8"):
                config = stream_config(**named, **{LEAD: value})
                with pytest.raises(exc.InvalidConfig, match=re.escape(LEAD)):
                    spec.normalized_config(config)
                spec.normalize(config, _fix_invalid=True)
                assert LEAD not in config.config
                assert config == spec.normalized_config(stream_config(**named))
        # A reversed order the config cannot run goes, and its lead with it.
        config = stream_config(**{KEY: "reversed", LEAD: 8, PARTITION: "balanced"})
        with pytest.raises(exc.InvalidConfig, match="one unit per CTA"):
            spec.normalized_config(config)
        # Another body pins both.
        off = spec.normalized_config(
            stream_config(**{KEY: "reversed", LEAD: 8}, cute_mma_dag="off")
        )
        assert KEY not in off.config and LEAD not in off.config
        assert off == spec.normalized_config(stream_config(cute_mma_dag="off"))
        assert plain == spec.normalized_config(stream_config(**{LEAD: 8}))


def test_the_walk_needs_fast_math_a_window_and_a_partition_that_runs_it():
    bound = _bound()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    assert not facts.attention.past_one_wave_units
    legal = {KEY: "reversed", "cute_mma_dag_pass_warps": PAIRED}
    assert paged_attention_page_order_reason(facts, legal) is None
    assert paged_attention_page_order_reason(facts, {**legal, LEAD: 8}) is None
    # Each config-side requirement by name. A grid of no more units than
    # SMs runs the walk with one unit per CTA alone.
    more_units = "more units than the device has SMs"
    for change, words in (
        ({"cute_mma_dag_pass_warps": "attention"}, ("paired attention schedule",)),
        ({PARTITION: "whole"}, ("one unit per CTA", more_units, "'whole'")),
        ({PARTITION: "balanced"}, ("one unit per CTA", more_units, "'balanced'")),
        ({SKIP: True}, (f"{SKIP}=False",)),
    ):
        reason = paged_attention_page_order_reason(facts, {**legal, **change})
        assert reason is not None and reason.startswith(f"{KEY}='reversed' ")
        assert all(word in reason for word in words), reason
    # A grid of more units than SMs, whose persistent CTAs stream several
    # units, runs it on every partition; the transport and the schedule
    # stay requirements there.
    waves = _bound(WAVES)
    wave_facts = waves.config_spec.cute_mma_dag_facts
    assert wave_facts is not None and wave_facts.attention is not None
    assert wave_facts.attention.past_one_wave_units
    for partition in ("item", "whole", "balanced"):
        named = {**legal, PARTITION: partition}
        assert paged_attention_page_order_reason(wave_facts, named) is None
        assert paged_attention_page_order_reason(wave_facts, {**named, LEAD: 8}) is None
        assert paged_attention_page_order_choices(
            wave_facts.attention, PAIRED, False, partition
        ) == ("source", "reversed")
        assert paged_attention_page_order_choices(
            facts.attention, PAIRED, False, partition
        ) == (("source", "reversed") if partition == "item" else ("source",))
        reason = paged_attention_page_order_reason(wave_facts, {**named, SKIP: True})
        assert reason is not None and f"{SKIP}=False" in reason
    with cpu_render_target(fake_device_sm_count=SMS), waves.env, waves.host_function:
        for partition in ("whole", "balanced"):
            for lead in (0, 8):
                config = stream_config(
                    **{KEY: "reversed", LEAD: lead, PARTITION: partition}
                )
                kept = waves.config_spec.normalized_config(config)
                assert kept.config[KEY] == "reversed"
                assert kept.config[PARTITION] == partition
                assert kept.config.get(LEAD, 0) == lead
    # And each kernel-side one, ahead of the config's.
    for missing, word in (
        (REVERSED_ORDER_NEEDS_PAIR, "pair proof"),
        (
            reversed_order_missing(
                paired=True, fast_math=False, windowed=True, reflected=False
            ),
            "fast_math",
        ),
    ):
        without = dataclasses.replace(
            facts,
            attention=dataclasses.replace(
                facts.attention, reversed_order_missing=missing
            ),
        )
        for config in (legal, {**legal, PARTITION: "balanced"}):
            reason = paged_attention_page_order_reason(without, config)
            assert reason is not None and word in reason and "newest first" in reason
    unpaired = dataclasses.replace(
        facts, attention=dataclasses.replace(facts.attention, paired_items=False)
    )
    assert "pair proof" in cast(
        "str", paged_attention_page_order_reason(unpaired, legal)
    )
    for value in ("reverse", "Reversed", True, 1, None):
        reason = paged_attention_page_order_reason(facts, {**legal, KEY: value})
        assert reason is not None and "('source', 'reversed')" in reason
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        spec = bound.config_spec
        kept = spec.normalized_config(stream_config(**{KEY: "reversed"}))
        assert kept.config[KEY] == "reversed"
        # Each is rejected, and repaired to the same config in source
        # order: no other option changes.
        for change, word in (
            ({PARTITION: "whole"}, "one unit per CTA"),
            ({PARTITION: "balanced"}, "one unit per CTA"),
            (
                {
                    "cute_mma_dag_pass_warps": "attention",
                    "cute_mma_dag_query_tma": False,
                },
                "paired attention schedule",
            ),
        ):
            for lead in (0, 8):
                config = stream_config(**{KEY: "reversed", LEAD: lead}, **change)
                with pytest.raises(exc.InvalidConfig, match=word):
                    spec.normalized_config(config)
                spec.normalize(config, _fix_invalid=True)
                assert KEY not in config.config and LEAD not in config.config
                assert config == spec.normalized_config(stream_config(**change))
    # A kernel without fast math, one without a window and one whose source
    # reflects its loop reject the walk by the requirement they lack.
    for other, word in (
        (_bound(fast_math=False), "fast_math"),
        (_bound(window=NO_WINDOW), "sliding window"),
        (_reflected_bound(), "oldest first"),
    ):
        with (
            cpu_render_target(fake_device_sm_count=SMS),
            other.env,
            other.host_function,
        ):
            config = stream_config(**{KEY: "reversed", LEAD: 8})
            with pytest.raises(exc.InvalidConfig, match=word):
                other.config_spec.normalized_config(config)
            other.config_spec.normalize(config, _fix_invalid=True)
            assert KEY not in config.config and LEAD not in config.config


def test_every_role_takes_its_tiles_from_the_one_walk():
    """The loader, the compute groups, a boundary prefetch and the
    value-metadata actor map a step through ``attention_page_coordinates``
    with the walk the body hands them, and nothing else looks at it."""
    body = _function("paged_attention_paired_body")
    # The body binds the walk once, ahead of every role.
    (bind,) = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.Assign)
        and [ast.unparse(target) for target in node.targets] == ["page_walk"]
    ]
    assert ast.unparse(bind.value) == (
        "ReversedWalk(clipped_page_lead) if page_order == "
        "MMA_DAG_PAGE_ORDER_REVERSED else reflected_dtype"
    )
    roles = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.If) and ast.unparse(node.test).startswith("warp")
    ]
    assert roles and bind.lineno < min(node.lineno for node in roles)
    # Behind the binding the body names the walk alone: the source's
    # reflection, which the binding read, reaches no role, so none can be
    # handed the source's walk where the config names another.
    reflection = [
        node.lineno
        for node in ast.walk(body)
        if isinstance(node, ast.Name) and node.id == "reflected_dtype"
    ]
    assert reflection and max(reflection) <= bind.end_lineno
    # It asserts what the walk was admitted on.
    asserted = {
        ast.unparse(node.test)
        for node in ast.walk(body)
        if isinstance(node, ast.Assert) and node.lineno < bind.lineno
    }
    assert {
        "cutlass.const_expr(fast_math)",
        "cutlass.const_expr(window is not None and reflected_dtype is None)",
        "cutlass.const_expr(value_metadata_scales is None)",
    } <= asserted
    # Everywhere else the walk is only handed on: an argument of a call or
    # a parameter. The one function that reads it is the mapping.
    mapped = 0
    narrow = []
    for module in (template, prefetch, bucketed_role):
        tree = ast.parse(Path(inspect.getfile(module)).read_text())
        handed = {
            id(argument)
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            for argument in (*call.args, *(keyword.value for keyword in call.keywords))
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "page_walk":
                assert (
                    id(node) in handed
                    or isinstance(node.ctx, ast.Store)
                    or bind.lineno - 12 <= node.lineno <= bind.end_lineno
                ), (module.__name__, node.lineno)
        for call in ast.walk(tree):
            if (
                isinstance(call, ast.Call)
                and ast.unparse(call.func) == "attention_page_coordinates"
            ):
                assert len(call.args) == 4
                assert ast.unparse(call.args[3]) == "page_walk"
                # Only the integer type of a walk in Int32 is named as well.
                narrow += [ast.unparse(keyword) for keyword in call.keywords]
                mapped += 1
    # The loader's step, which both of its V-lag forms begin with, the
    # compute groups, the value-metadata actor, the prefetch of one unit
    # per CTA, the balanced partition's range-ahead prefetch and the
    # bucketed order's role; and the compute groups once more, in Int32,
    # for a bind that bounds its table's keys.
    assert mapped == 7
    assert narrow == ["integer=cutlass.Int32"]
    mapping = _function("attention_page_coordinates", partial)
    assert "ReversedWalk" in ast.unparse(mapping)


def test_the_parts_of_a_unit_take_each_page_of_its_walk_once():
    """The walk on a balanced partition: a CTA runs the steps of a unit
    whose slots it owns, with the unit's own page range, so a part takes
    the pages the unit's walk has at its steps. Over random page counts,
    grids and skews the parts of every unit take each of its pages once,
    newest first, and the oldest where the lead says, in whichever part
    holds that step: also a part of fewer steps than the lead, and one that
    is the unit's last step alone."""
    generator = random.Random(20261120)
    split = short = alone = 0
    for _trial in range(60):
        units = generator.randint(1, 24)
        pages = [generator.choice((0, 1, 2, 3, 7, 8, 9, 13, 17)) for _ in range(units)]
        firsts = [generator.randint(0, 6) for _ in range(units)]
        grid = generator.randint(1, 40)
        skew = generator.choice((0, 0, 1, 2))
        partition = reference_partition(pages, grid, "balanced", skew=skew)
        for lead in MMA_DAG_CLIPPED_PAGE_LEAD_CHOICES:
            taken: dict[int, list[int]] = {unit: [] for unit in range(units)}
            for segments in partition:
                for segment in segments:
                    first = firsts[segment.unit]
                    last = first + segment.pages
                    steps = range(segment.step, segment.step + segment.count)
                    mapped = template_walk(first, last, ReversedWalk(lead), steps)
                    walk = reversed_walk(first, last, lead)
                    assert [page for page, _base in mapped] == walk[
                        segment.step : segment.step + segment.count
                    ]
                    # Each step's tile is its page's own.
                    assert all(base == page * PAGE_KEYS for page, base in mapped)
                    taken[segment.unit].extend(page for page, _base in mapped)
                    if not segment.whole:
                        split += 1
                        short += 0 < segment.count < lead
                        alone += segment.count == 1 and segment.tail
            for unit, count in enumerate(pages):
                assert sorted(taken[unit]) == list(
                    range(firsts[unit], firsts[unit] + count)
                )
    assert split and short and alone


def _seeds(bound, spec=None):
    """The compiler seeds ahead of the store hint's, which follow the
    walk's (``test_cute_paged_attention_store_policy`` pins those)."""
    return seeds_ahead_of_the_store_hint(stream_seeds(bound, spec))


def test_the_walk_is_searched_everywhere_and_seeded_in_two_forms():
    eager = _bound()
    fields = eager.config_spec._flat_fields()
    order = fields[KEY]
    assert order is eager.config_spec.cute_mma_dag_page_order
    # The walk changes which tiles of the launch's own stream rescale,
    # which every timing of the kernel shows.
    assert order.choices == order._active_choices() == ("source", "reversed")
    assert order.default() == "source"
    lead = fields[LEAD]
    assert lead is eager.config_spec.cute_mma_dag_clipped_page_lead
    assert lead.choices == lead._active_choices() == (0, 8) and lead.default() == 0
    assert (CLIPPED_PAGE_LEAD_SEED, VALUE_RING_SEED_PAGES) == (8, 3)
    seeds = _seeds(eager)
    head = seeds[0]
    assert head.get("cute_mma_dag_pass_warps") == PAIRED
    assert KEY not in head.config and head.config[V_LAG] == 2
    # The leading paired family twice in reversed order, behind every other
    # seed: the clipped page eight pages ahead of the walk's end beside
    # three pages of raw V per head, then the page last beside the default
    # ring. Both keep the family's V lag.
    pair = {KEY: "reversed", LEAD: 8, RING: 3}
    last = {KEY: "reversed"}
    named = [seed for seed in seeds if KEY in seed.config]
    assert named == seeds[-2:]
    assert [seed.config for seed in named] == [
        {**head.config, **pair},
        {**head.config, **last},
    ]
    # Every other seed keeps its place: they are the seeds of the same
    # kernel without the walk.
    without = copy.copy(eager.config_spec)
    without.cute_mma_dag_page_order = None
    assert _seeds(eager, without) == seeds[:-2]
    assert seeds_ahead_of_the_reversed_order(seeds) == seeds[:-2]
    # Where a search samples a boundary prefetch and seeds it, each of
    # those seeds follows in both forms as well.
    banks = _bound(input_banks=2)
    seeds = _seeds(banks)
    head = seeds[0]
    named = [seed for seed in seeds if KEY in seed.config]
    assert named == seeds[-6:]
    families = [
        head.config,
        *({**head.config, PREFETCH: pages} for pages in BOUNDARY_PREFETCH_SEED_PAGES),
    ]
    assert BOUNDARY_PREFETCH_SEED_PAGES == (8, 12)
    assert [seed.config for seed in named] == [
        {**family, **form} for family in families for form in (pair, last)
    ]
    ahead = seeds_ahead_of_the_reversed_order(seeds)
    assert [seed.config for seed in ahead if PREFETCH in seed.config] == families[1:]
    # No seed in reversed order has V issued with its K, whose form with
    # the deeper ring lost the walk's gains, and none names the ring beside
    # another order: a window's stream gains nothing from the depth alone.
    assert all(seed.config[V_LAG] == 2 for seed in named)
    assert not any(RING in seed.config for seed in ahead)
    # The guards: a device without room for the deeper ring keeps the
    # second form alone, and a search that does not sample the prefetch
    # seeds the leading family alone.
    spec = banks.config_spec
    shallow = copy.copy(spec)
    shallow.cute_mma_dag_value_ring = None
    assert [seed.config for seed in _seeds(banks, shallow) if KEY in seed.config] == [
        {**family, **last} for family in families
    ]
    small = _bound(input_banks=2, smem_capacity_bytes=200_000, rows=40)
    assert small.config_spec.cute_mma_dag_value_ring is None
    assert [seed.config.get(LEAD) for seed in _seeds(small) if KEY in seed.config] == [
        None
    ] * 3
    unsearched = copy.copy(spec)
    unsearched.cute_mma_dag_boundary_prefetch_searched = False
    assert [
        seed.config for seed in _seeds(banks, unsearched) if KEY in seed.config
    ] == [{**head.config, **form} for form in (pair, last)]
    # Past one wave the walk is seeded as on one wave, and once more on the
    # balanced partition, which runs it there: that partition's seed with
    # the clipped page eight pages ahead of a unit's end and the default
    # ring, the last of the walk's seeds.
    waves = _bound(WAVES, input_banks=2)
    wave_seeds = _seeds(waves)
    walked = [seed for seed in wave_seeds if KEY in seed.config]
    assert len(walked) == 7 and walked == wave_seeds[-7:]
    (balanced,) = [
        seed
        for seed in wave_seeds
        if seed.config.get(PARTITION) == "balanced" and KEY not in seed.config
    ]
    assert [PARTITION in seed.config for seed in walked] == [False] * 6 + [True]
    assert walked[-1].config == {**balanced.config, KEY: "reversed", LEAD: 8}
    assert RING not in walked[-1].config
    keywords = body_keywords(stream_render(waves, walked[-1]))
    assert keywords["partition"] == "'balanced'"
    assert (keywords["page_order"], keywords["clipped_page_lead"]) == (
        "'reversed'",
        "8",
    )
    # A search that has no lead to name seeds no walk on that partition.
    leadless = copy.copy(waves.config_spec)
    leadless.cute_mma_dag_clipped_page_lead = None
    assert not any(
        PARTITION in seed.config and KEY in seed.config
        for seed in _seeds(waves, leadless)
    )
    # On one wave no seed names a persistent partition, with the walk or
    # without it.
    assert not any(PARTITION in seed.config for seed in seeds)
    for bound in (eager, banks):
        spec = bound.config_spec
        walked = [seed for seed in _seeds(bound) if KEY in seed.config]
        with (
            cpu_render_target(fake_device_sm_count=SMS),
            bound.env,
            bound.host_function,
        ):
            assert KEY not in spec.default_config().config
            generation = spec.create_config_generation()
            for seed in walked:
                # A complete legal config: strict normalization keeps every
                # knob it names, and a search's row is that config.
                normalized = spec.normalized_config(seed)
                assert all(
                    normalized.config[key] == value
                    for key, value in seed.config.items()
                )
                _flat, transferred = generation.canonicalize_flat(
                    generation.flatten(normalized)
                )
                assert all(
                    transferred.config.get(key) == seed.config.get(key)
                    for key in (KEY, LEAD, RING, PREFETCH, V_LAG)
                )
            pairs = generation.seed_flat_config_pairs()
            assert [
                config.config.get(LEAD)
                for _flat, config in pairs
                if KEY in config.config and COLD not in config.config
            ] == [8, None] * (len(walked) // 2)
    keywords = body_keywords(stream_render(banks, named[2]))
    assert keywords["page_order"] == "'reversed'"
    assert (keywords["clipped_page_lead"], keywords["value_ring"]) == ("8", "3")
    assert keywords["boundary_prefetch"] == "8" and keywords["v_lag"] == "2"
    keywords = body_keywords(stream_render(banks, named[1]))
    assert keywords["page_order"] == "'reversed'"
    assert not {"clipped_page_lead", "value_ring", "boundary_prefetch"} & set(keywords)


@pytest.mark.parametrize("units", [ONE_WAVE, WAVES])
def test_a_step_onto_the_walk_takes_a_partition_that_runs_it(units):
    """And a step onto a lead takes the walk. With no more units than SMs
    the walk has one unit per CTA alone, so a step onto it from the
    balanced partition takes that; with more units the balanced partition
    runs the walk and the step keeps it."""
    bound = _bound(units)
    spec = bound.config_spec
    persistent = units == WAVES
    # What the walk takes of a row is the bind's to say, whatever partition
    # the row is on: no value-metadata transport, and one unit per CTA
    # unless the bind runs the walk on every partition. The bucketed order,
    # which no persistent partition runs, always takes one unit per CTA.
    no_transport = {SKIP: False, READ_OVERLAP: False}
    one_unit = {PARTITION: "item", "cute_mma_dag_partition_skew": 0}
    assert page_order_needs(_attention(bound), "reversed") == (
        no_transport if persistent else {**no_transport, **one_unit}
    )
    for order in ("bucketed", "sorted"):
        assert page_order_needs(_attention(bound), order) == {
            **no_transport,
            **one_unit,
        }
    assert page_order_needs(None, "reversed") == {**no_transport, **one_unit}
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        plain = stream_config()
        balanced = stream_config(**{PARTITION: "balanced"})
        for parent in (plain, balanced):
            kept = "balanced" if parent is balanced and persistent else None
            # The search's row of the parent itself.
            row, _row = stream_step(spec, parent)
            reached, alone = stream_step(spec, parent, **{KEY: "reversed"})
            assert reached.config[KEY] == "reversed" and LEAD not in reached.config
            assert spec.normalized_config(reached).config.get(PARTITION) == kept
            if parent is plain or persistent:
                assert alone == reached
            else:
                # Alone the step is an alias of where it started: this
                # grid's walk belongs to one unit per CTA, and normalization
                # repairs it to the source's order and changes nothing else.
                assert KEY not in alone.config
                assert alone.config[PARTITION] == "balanced"
            # A lead names a place in the reversed walk: the step takes
            # that order, and with it a partition that runs it. Alone it is
            # an alias of where it started.
            reached, alone = stream_step(spec, parent, **{LEAD: 8})
            assert (reached.config[KEY], reached.config[LEAD]) == ("reversed", 8)
            assert spec.normalized_config(reached).config.get(PARTITION) == kept
            assert alone == row
        on = spec.normalized_config(stream_config(**{KEY: "reversed", LEAD: 8}))
        # Back to the source's order: the lead goes with the walk.
        reached, alone = stream_step(spec, on, **{KEY: "source"})
        assert reached == alone == stream_step(spec, plain)[0]
        # The page last again keeps the walk.
        reached, alone = stream_step(spec, on, **{LEAD: 0})
        assert reached == alone
        assert reached.config[KEY] == "reversed" and LEAD not in reached.config
        # A step onto a partition from the walk is the partition's: it
        # drops a deeper ring, which a search samples with one unit per CTA
        # alone, and the walk and its lead where the partition does not
        # run them on this grid.
        deep = spec.normalized_config(
            stream_config(**{KEY: "reversed", LEAD: 8, RING: 3})
        )
        assert deep.config[RING] == 3
        for parent in (on, deep):
            reached, _alone = stream_step(spec, parent, **{PARTITION: "balanced"})
            moved = spec.normalized_config(reached)
            assert moved.config[PARTITION] == "balanced"
            assert RING not in moved.config
            if persistent:
                assert (moved.config[KEY], moved.config[LEAD]) == ("reversed", 8)
            else:
                assert not {KEY, LEAD} & set(moved.config)
        # The walk pins no V lag or ring depth and takes none.
        for move in ({V_LAG: 0}, {RING: 3}):
            reached, alone = stream_step(spec, on, **move)
            assert reached == alone
            config = spec.normalized_config(reached).config
            assert (config[KEY], config[LEAD]) == ("reversed", 8)


def test_a_step_onto_the_bucketed_order_past_one_wave_takes_one_unit_per_cta():
    """A bind past one wave that offers both orders: the walk on every
    partition, the bucketed order with one unit per CTA alone. A step takes
    what the order it names needs, so from a balanced row the step onto the
    bucketed order leaves the partition and the step onto the walk keeps
    it."""
    bound = _bound(WAVES, window=50 * 128, table_pages=64, pool_pages=96)
    spec = bound.config_spec
    attention = _attention(bound)
    assert attention.past_one_wave_units and attention.page_order_available
    for partition, orders in (
        ("item", ("source", "bucketed", "reversed")),
        ("whole", ("source", "reversed")),
        ("balanced", ("source", "reversed")),
    ):
        assert (
            paged_attention_page_order_choices(attention, PAIRED, False, partition)
            == orders
        )
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        balanced = stream_config(**{PARTITION: "balanced"})
        reached, alone = stream_step(spec, balanced, **{KEY: "bucketed"})
        assert reached.config[KEY] == "bucketed"
        assert PARTITION not in spec.normalized_config(reached).config
        # Alone the step is an alias of where it started: the balanced
        # partition does not run the bucketed order.
        assert KEY not in alone.config and alone.config[PARTITION] == "balanced"
        reached, alone = stream_step(spec, balanced, **{KEY: "reversed"})
        assert reached == alone
        assert spec.normalized_config(reached).config[PARTITION] == "balanced"


def test_a_step_onto_the_walk_drops_the_value_metadata_transport():
    """A kernel that has the value-metadata transport (the positive-zero
    fixture, with a window, under fast math): the walk was not examined
    beside that transport's actor, so a step onto it from a config with the
    transport takes the transport off with it (``page_order_needs``), as the
    bucketed order's step does. Alone the step is an alias of where it
    started."""
    kernel, args = positive_zero_attention_fixture(
        5, -22, form="mapped", reference_scores=False
    )
    fast = helion.kernel(kernel.fn, backend="cute", static_shapes=True, fast_math=True)
    with cpu_render_target(fake_device_sm_count=SMS, num_sm=SMS):
        bound = cpu_bind(fast, args, num_sm=SMS)
    spec = bound.config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    assert facts.attention.value_metadata_skip_available
    assert not facts.attention.reversed_order_missing
    head = stream_seeds(bound)[0]
    assert head.get("cute_mma_dag_pass_warps") == PAIRED and SKIP not in head.config
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        for overlap in (False, True):
            parent = spec.normalized_config(
                helion.Config(**{**head.config, SKIP: True, READ_OVERLAP: overlap})
            )
            assert parent.config[SKIP] is True
            assert parent.config[READ_OVERLAP] is overlap
            row, _row = stream_step(spec, parent)
            for move in ({KEY: "reversed"}, {LEAD: 8}):
                reached, alone = stream_step(spec, parent, **move)
                assert reached.config[KEY] == "reversed"
                assert reached.config.get(LEAD) == move.get(LEAD)
                assert reached.config[SKIP] is False
                assert reached.config[READ_OVERLAP] is False
                # Nothing else of the parent moved.
                assert {
                    key: value
                    for key, value in reached.config.items()
                    if key not in (KEY, LEAD, SKIP, READ_OVERLAP)
                } == {
                    key: value
                    for key, value in row.config.items()
                    if key not in (SKIP, READ_OVERLAP)
                }
                # The step alone keeps the transport and the source's order.
                assert alone == row
                assert KEY not in alone.config and alone.config[SKIP] is True


def test_the_planner_gives_no_schedule_to_a_lead_beside_another_order():
    """Normalization drops a lead beside every order but the reversed one,
    so no config reaches the planner with one. A config that does, because
    it was not normalized, gets no schedule, as with every other option the
    planner checks: the kernel is not rendered in the source's order as if
    the lead had not been named."""
    bound = _bound()
    planned = {}
    plan = transport.plan_paged_attention_schedule

    def planning(env, semantic, selected, config):
        named = dict(config)
        assert (named[KEY], named[LEAD]) == ("reversed", 8)
        without = {key: value for key, value in named.items() if key != KEY}
        for name, changed in {
            "the walk": named,
            "the page last": {**named, LEAD: 0},
            "a lead that is none of the option's": {**named, LEAD: 5},
            "a lead beside the source's order": {**named, KEY: "source"},
            "a lead beside no order": without,
            "the source's order": {**without, LEAD: 0},
        }.items():
            planned[name] = plan(env, semantic, selected, helion.Config(**changed))
        return planned["the walk"]

    with patch.object(transport, "plan_paged_attention_schedule", planning):
        source = stream_render(bound, stream_config(**{KEY: "reversed", LEAD: 8}))
    assert body_keywords(source)["clipped_page_lead"] == "8"
    assert len(planned) == 6
    walk, last, plain = (
        planned[name] for name in ("the walk", "the page last", "the source's order")
    )
    assert (walk.page_order, walk.clipped_page_lead) == ("reversed", 8)
    assert (last.page_order, last.clipped_page_lead) == ("reversed", 0)
    assert (plain.page_order, plain.clipped_page_lead) == ("source", 0)
    for name in (
        "a lead that is none of the option's",
        "a lead beside the source's order",
        "a lead beside no order",
    ):
        assert planned[name] is None, name


def test_overrides_keep_the_paired_seeds_with_or_without_their_walk():
    bound = _bound(input_banks=2)
    spec = bound.config_spec
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        seeded, failures = transferred_seeds(spec, None)
        assert not failures
        paired = [
            seed for seed in seeded if seed.get("cute_mma_dag_pass_warps") == PAIRED
        ]
        assert [
            seed.config.get(LEAD)
            for seed in seeded
            if KEY in seed.config and COLD not in seed.config
        ] == [8, None] * 3
        # A fixed order is layered over complete candidates: every paired
        # one takes it and keeps its place; a candidate on another schedule
        # is left to that body's rule, which has no walk.
        forced, failures = transferred_seeds(spec, {KEY: "reversed"})
        kept = [
            seed for seed in forced if seed.get("cute_mma_dag_pass_warps") == PAIRED
        ]
        assert kept and forced[0] == kept[0] and len(forced) == len(kept)
        assert all(seed.config[KEY] == "reversed" for seed in kept)
        assert len(failures) == len(seeded) - len(paired)
        assert all(KEY in failure for failure in failures)
        # With the lead as well, each of them has both.
        both, _failures = transferred_seeds(spec, {KEY: "reversed", LEAD: 8})
        assert both and all(
            (seed.config[KEY], seed.config[LEAD]) == ("reversed", 8) for seed in both
        )
        # A fixed lead alone is the walk's: the candidates in reversed
        # order have it, and for every other one it names nothing.
        led, failures = transferred_seeds(spec, {LEAD: 8})
        assert not failures
        walked = [seed for seed in led if KEY in seed.config]
        assert walked and all(seed.config[LEAD] == 8 for seed in walked)
        assert not any(LEAD in seed.config for seed in led if KEY not in seed.config)
        for overrides, drops in (
            ({PARTITION: "balanced"}, True),
            ({PARTITION: "whole"}, True),
            ({"cute_mma_dag_pass_warps": "attention"}, True),
            ({KEY: "source"}, True),
            ({PARTITION: "item"}, False),
            ({V_LAG: 3}, False),
            ({"cute_pdl": False}, False),
        ):
            moved, failures = transferred_seeds(spec, overrides)
            assert not [failure for failure in failures if KEY in failure], overrides
            orders = [seed.config[KEY] for seed in moved if KEY in seed.config]
            assert bool(orders) is not drops, overrides
            # A candidate that loses its walk loses its lead with it.
            assert not any(
                LEAD in seed.config for seed in moved if KEY not in seed.config
            )
        moved, failures = transferred_seeds(
            spec, {KEY: "reversed", PARTITION: "balanced"}
        )
        assert not moved and len(failures) == len(seeded)
    # Past one wave the persistent partitions run the walk, so a fixed order
    # takes no partition from a candidate: the balanced seeds stay balanced,
    # and the order holds beside a fixed partition.
    waves = _bound(WAVES, input_banks=2)
    spec = waves.config_spec
    with cpu_render_target(fake_device_sm_count=SMS), waves.env, waves.host_function:
        seeded, failures = transferred_seeds(spec, None)
        assert not failures
        balanced = [seed for seed in seeded if seed.config.get(PARTITION) == "balanced"]
        assert {seed.config.get(KEY) for seed in balanced} == {None, "reversed"}
        forced, _failures = transferred_seeds(spec, {KEY: "reversed"})
        assert all(seed.config[KEY] == "reversed" for seed in forced)
        kept = [seed for seed in forced if seed.config.get(PARTITION) == "balanced"]
        assert len(kept) == len(balanced)
        assert sorted(seed.config.get(LEAD, 0) for seed in kept) == sorted(
            seed.config.get(LEAD, 0) for seed in balanced
        )
        moved, _failures = transferred_seeds(
            spec, {KEY: "reversed", PARTITION: "balanced"}
        )
        assert moved and all(
            (seed.config[KEY], seed.config[PARTITION]) == ("reversed", "balanced")
            for seed in moved
        )


def test_the_reversed_kernel_compiles_with_no_role_of_its_own():
    bound = _bound(17, rows=40)
    arguments = partition_attention_args(
        units=17, rows=40, window=WINDOW, table_pages=TABLE, pool_pages=POOL
    )
    kernels = {}
    with tempfile.TemporaryDirectory(prefix="helion-cute-reversed-") as dump_dir:
        for name, knobs in (
            ("plain", {}),
            ("last", {KEY: "reversed"}),
            ("lead", {KEY: "reversed", LEAD: 8}),
        ):
            with cpu_render_target(fake_device_sm_count=SMS, num_sm=SMS):
                source = bound.to_code(stream_config(40, **knobs))
            kernels[name] = compile_offline(
                bound, arguments, source, Path(dump_dir) / name
            )
    plain, last, lead = (kernels[name] for name in ("plain", "last", "lead"))
    # Three kernels: the walk and its lead each change the device code.
    assert len({plain.ptx, last.ptx, lead.ptx}) == 3
    # The walk has no warp of its own, no barrier and no register site: the
    # same threads, quotas and synchronizations as the source order.
    for walked in (last, lead):
        assert walked.threads == plain.threads == 640
        assert walked.ptx_quotas() == plain.ptx_quotas()
        for instruction in ("bar.sync", "bar.warp.sync", "barrier.cta", "mbarrier"):
            assert walked.ptx.count(instruction) == plain.ptx.count(instruction)
