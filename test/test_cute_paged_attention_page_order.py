"""Check the paired schedule's bucketed page order: which kernels are offered
it, what a config may name and what it renders, the structure of the device
role and of the loader's handoff, the sort itself against a stable sort, and
where a search samples and seeds it."""

from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
from pathlib import Path
import random
import re
import tempfile

import pytest
import torch

from helion import exc
from helion._compiler.cute import _paged_attention_page_order as device
from helion._compiler.cute.cute_mma_dag import paged_attention_page_order_choices
from helion._compiler.cute.cute_mma_dag_seeds import PAGE_ORDER_PREFETCH_SEED_PAGES
from helion._compiler.cute.paged_attention_config import (
    paged_attention_page_order_reason,
)
from helion._compiler.cute.paged_attention_page_order import MMA_DAG_PAGE_ORDER_CHOICES
from helion._compiler.cute.paged_attention_page_order import MMA_DAG_PAGE_ORDER_KEY
from helion._compiler.cute.paged_attention_page_order import PAGE_ORDER_BUCKETS
from helion._compiler.cute.paged_attention_page_order import PAGE_ORDER_CAPACITY
from helion._compiler.cute.paged_attention_page_order import PAGE_ORDER_HEAD
from helion._compiler.cute.paged_attention_page_order import PAGE_ORDER_MIN_SORTED
from helion._compiler.cute.paged_attention_page_order import PAGE_ORDER_TAIL
from helion._compiler.cute.paged_attention_page_order import page_order_bucket
from helion._compiler.cute.paged_attention_page_order import page_order_offered
from helion._compiler.cute.paged_attention_page_order import page_order_shift
from helion._compiler.cute.paged_attention_page_order import page_order_sortable_bound
from helion._compiler.cute.paged_attention_shared_memory import (
    paired_shared_memory_bytes,
)
from helion._compiler.cute.paged_attention_shared_memory import pipeline_barrier_plan
from helion._compiler.cute.paged_attention_shared_memory import pipeline_parts
from helion._compiler.cute.paged_attention_shared_memory import pipeline_shared_memory
from helion._testing import cpu_render_target
from helion._testing.cute_offline import compile_offline
from helion._testing.cute_paged_page_order import bucketed_page_table
from helion._testing.cute_paged_page_order import mirror_sort
from helion._testing.cute_paged_page_order import ordered_run
from helion._testing.cute_paged_page_order import run_device_sort
from helion._testing.cute_paged_page_order import stream_page
from helion._testing.cute_paged_page_order import unit_page_range
from helion._testing.cute_paged_partition import PAGE
from helion._testing.cute_paged_partition import SHORT_WINDOW
from helion._testing.cute_paged_partition import partition_attention_args
from helion._testing.cute_paged_partition import randomize_partition_inputs
from helion._testing.cute_paged_partition import set_request_lengths
from helion._testing.cute_paged_stream import NO_WINDOW
from helion._testing.cute_paged_stream import PAIRED
from helion._testing.cute_paged_stream import SMS
from helion._testing.cute_paged_stream import body_keywords
from helion._testing.cute_paged_stream import pair_seed
from helion._testing.cute_paged_stream import stream_bound
from helion._testing.cute_paged_stream import stream_config
from helion._testing.cute_paged_stream import stream_render
from helion._testing.cute_paged_stream import stream_seeds
from helion._testing.cute_paged_stream import stream_step
from helion._testing.cute_paged_stream import transferred_seeds
from helion._testing.cute_template_source import source_calls
from helion._testing.cute_template_source import source_function
from helion.autotuner.config_fragment import EnumFragment

pytestmark = pytest.mark.cpu_only
KEY = MMA_DAG_PAGE_ORDER_KEY
LEAD = "cute_mma_dag_release_lead"
RING = "cute_mma_dag_value_ring"
PREFETCH = "cute_mma_dag_boundary_prefetch"
PARTITION = "cute_mma_dag_partition"
SKIP = "cute_mma_dag_value_metadata_skip"
V_LAG = "cute_mma_dag_v_lag"
DEVICE_ROLE = Path(inspect.getfile(device))
# Units on 148 SMs: a grid of one wave, and one of more than two.
ONE_WAVE = 128
WAVES = 300
# A page table of 64 pages a request over a pool of 80 pages: a request can
# have more interior pages behind the head and ahead of the tail than the
# fewest a sort orders, and a bucket is four pages of the pool.
TABLE = 64
POOL = 80
# Pages of a request that keep their places, and the fewest pages of a
# request that is reordered.
KEPT = PAGE_ORDER_HEAD + PAGE_ORDER_TAIL
FEWEST = KEPT + PAGE_ORDER_MIN_SORTED
assert FEWEST + 2 <= TABLE


def _bound(units: int = ONE_WAVE, **options: object):
    options.setdefault("table_pages", TABLE)
    options.setdefault("pool_pages", POOL)
    options.setdefault("window", NO_WINDOW)
    return stream_bound(units, **options)  # type: ignore[arg-type]


def test_a_kernel_is_offered_the_order_where_some_unit_has_pages_to_sort():
    assert KEY == "cute_mma_dag_page_order"
    assert MMA_DAG_PAGE_ORDER_CHOICES == ("source", "bucketed", "reversed")
    assert (PAGE_ORDER_HEAD, PAGE_ORDER_TAIL) == (4, 12)
    assert (PAGE_ORDER_CAPACITY, PAGE_ORDER_MIN_SORTED) == (416, 32)
    # The most pages of one unit the order can permute: the page table's
    # extent less the head and the tail, the whole pages of a window's keys,
    # and one sort's capacity.
    for page_limit, window, sortable in (
        (17, None, 1),
        (47, None, 31),
        (48, None, 32),
        (64, None, 48),
        (211, None, 195),
        (391, None, 375),
        (500, None, 416),
        (10, None, 0),
        # A 2047-key window sees at most sixteen whole pages, whatever the
        # table: the SDG eval's sliding window is not offered the order.
        (17, 2047, 1),
        (391, 2047, 16),
        (391, 4094, 31),
        (391, 4095, 32),
        (64, 6400, 48),
    ):
        assert page_order_sortable_bound(page_limit, window) == sortable
        assert page_order_offered(page_limit, window) is (sortable >= 32)
    # The bucket is the page id without the bits below a 32nd of the pool.
    for extent, shift in (
        (1, 0),
        (13, 0),
        (32, 0),
        (33, 1),
        (80, 2),
        (4484, 8),
        (7132, 8),
        (91577, 12),
        (102702, 12),
        (131072, 12),
        (131073, 13),
    ):
        assert page_order_shift(extent) == shift
        assert page_order_bucket(extent - 1, shift) < PAGE_ORDER_BUCKETS
        assert {page_order_bucket(page, shift) for page in range(extent)} == set(
            range(page_order_bucket(extent - 1, shift) + 1)
        )
    # The bind's facts: fast math, the page table and the window.
    facts = _bound().config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    assert facts.attention.page_order_available
    assert facts.attention.page_order_shift == 2
    assert paged_attention_page_order_choices(facts.attention, PAIRED) == (
        "source",
        "bucketed",
    )
    for other in (
        # The fixture's seventeen-page table and a kernel without fast math.
        _bound(table_pages=17),
        _bound(fast_math=False),
    ):
        attention = other.config_spec.cute_mma_dag_facts.attention
        assert not attention.page_order_available
        assert paged_attention_page_order_choices(attention, PAIRED) == ("source",)
        assert other.config_spec.cute_mma_dag_page_order is None
        assert KEY not in other.config_spec._flat_fields()
        assert not any(KEY in seed.config for seed in stream_seeds(other))
    # A window of four pages over the long table has no pages to sort. The
    # option stays a field there for the reversed walk, which a windowed
    # kernel is offered (test_cute_paged_attention_reversed_order).
    short = _bound(window=SHORT_WINDOW)
    attention = short.config_spec.cute_mma_dag_facts.attention
    assert not attention.page_order_available
    assert paged_attention_page_order_choices(attention, PAIRED) == (
        "source",
        "reversed",
    )
    assert not any(seed.config.get(KEY) == "bucketed" for seed in stream_seeds(short))
    # A window of fifty pages has room.
    windowed = _bound(window=50 * 128).config_spec.cute_mma_dag_facts.attention
    assert windowed.windowed and windowed.page_order_available
    assert paged_attention_page_order_choices(windowed, PAIRED) == (
        "source",
        "bucketed",
        "reversed",
    )
    # One unit per CTA of the paired schedule, without the value-metadata
    # transport.
    for arguments in (
        ("attention",),
        (PAIRED, True),
        (PAIRED, False, "whole"),
        (PAIRED, False, "balanced"),
    ):
        assert paged_attention_page_order_choices(facts.attention, *arguments) == (
            "source",
        )
    assert paged_attention_page_order_choices(None, PAIRED) == ("source",)
    assert paged_attention_page_order_choices(
        dataclasses.replace(facts.attention, paired_items=False), PAIRED
    ) == ("source",)


def test_the_source_order_renders_the_kernel_without_the_option():
    body = source_function("paged_attention_paired_body")
    defaults = dict(
        zip(
            [argument.arg for argument in body.args.args][-len(body.args.defaults) :],
            [ast.unparse(default) for default in body.args.defaults],
            strict=True,
        )
    )
    assert defaults["page_order"] == "MMA_DAG_PAGE_ORDER_SOURCE"
    assert defaults["page_order_shift"] == "0"
    bound = _bound()
    plain = stream_render(bound, stream_config())
    assert stream_render(bound, stream_config(**{KEY: "source"})) == plain
    assert "page_order" not in plain
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        spec = bound.config_spec
        named = spec.normalized_config(stream_config(**{KEY: "source"}))
        assert KEY not in named.config
        assert named == spec.normalized_config(stream_config())
        assert KEY not in spec.default_config().config
    # The bucketed order names itself and the bits below a bucket, from the
    # K tensor's page extent, and changes nothing else of the kernel.
    source = stream_render(bound, stream_config(**{KEY: "bucketed"}))
    keywords = body_keywords(source)
    assert keywords["page_order"] == "'bucketed'"
    assert keywords["page_order_shift"] == "2"
    assert source.replace(", page_order='bucketed', page_order_shift=2", "") == plain
    small = _bound(pool_pages=13)
    assert (
        body_keywords(stream_render(small, stream_config(**{KEY: "bucketed"})))[
            "page_order_shift"
        ]
        == "0"
    )


def test_the_order_needs_fast_math_one_unit_per_cta_and_no_value_metadata():
    bound = _bound(WAVES)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    legal = {KEY: "bucketed", "cute_mma_dag_pass_warps": PAIRED}
    assert paged_attention_page_order_reason(facts, legal) is None
    assert paged_attention_page_order_reason(facts, {}) is None
    assert paged_attention_page_order_reason(facts, {KEY: "source"}) is None
    for change in (
        {"cute_mma_dag_pass_warps": "attention"},
        {SKIP: True},
        {PARTITION: "whole"},
        {PARTITION: "balanced"},
    ):
        reason = paged_attention_page_order_reason(facts, {**legal, **change})
        assert reason is not None and reason.startswith(
            f"{KEY} must be one of ('source',) for"
        )
    for value in ("sorted", "Bucketed", True, 1, None):
        reason = paged_attention_page_order_reason(facts, {**legal, KEY: value})
        assert reason is not None and "('source', 'bucketed')" in reason
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        spec = bound.config_spec
        kept = spec.normalized_config(stream_config(**{KEY: "bucketed"}))
        assert kept.config[KEY] == "bucketed"
        # Each is rejected, and repaired to the same config without the
        # order: no other option changes.
        for order, change in (
            ("bucketed", {PARTITION: "whole"}),
            ("bucketed", {PARTITION: "balanced"}),
            ("sorted", {}),
            (
                "bucketed",
                {
                    "cute_mma_dag_pass_warps": "attention",
                    "cute_mma_dag_query_tma": False,
                },
            ),
        ):
            config = stream_config(**{KEY: order}, **change)
            with pytest.raises(exc.InvalidConfig, match=re.escape(KEY)):
                spec.normalized_config(config)
            spec.normalize(config, _fix_invalid=True)
            assert KEY not in config.config
            assert config == spec.normalized_config(stream_config(**change))
        off = spec.normalized_config(
            stream_config(**{KEY: "bucketed"}, cute_mma_dag="off")
        )
        assert KEY not in off.config
        assert off == spec.normalized_config(stream_config(cute_mma_dag="off"))
    # A kernel without fast math, and one whose units cannot have enough
    # interior pages, reject the order by name, and say what the kernel
    # lacks: the options the config names are the ones that run the order.
    lacks = (
        f"{KEY}='bucketed' needs a kernel bound with fast_math=True, some of "
        "whose units can have 32 or more interior pages to order: this kernel "
        "does not offer it"
    )
    for other in (_bound(fast_math=False), _bound(table_pages=17)):
        other_facts = other.config_spec.cute_mma_dag_facts
        assert other_facts is not None
        assert paged_attention_page_order_reason(other_facts, legal) == lacks
        # An option that excludes the order is named there as well, and an
        # order that does not exist is one of none.
        for change in ({PARTITION: "whole"}, {KEY: "sorted"}):
            reason = paged_attention_page_order_reason(other_facts, {**legal, **change})
            assert reason is not None and reason.startswith(
                f"{KEY} must be one of ('source',) for"
            )
        with (
            cpu_render_target(fake_device_sm_count=SMS),
            other.env,
            other.host_function,
        ):
            config = stream_config(**{KEY: "bucketed"})
            with pytest.raises(exc.InvalidConfig, match=re.escape(lacks)):
                other.config_spec.normalized_config(config)
            other.config_spec.normalize(config, _fix_invalid=True)
            assert KEY not in config.config


def test_the_orders_tables_and_barrier_exist_only_with_the_order():
    plain = pipeline_shared_memory(40, False)
    ordered = pipeline_shared_memory(40, False, page_order=True)
    added = [array for array in ordered if array.guard == "page_order"]
    assert [(array.name, array.count) for array in added] == [
        ("page_order", PAGE_ORDER_CAPACITY),
        ("page_order_buckets", PAGE_ORDER_BUCKETS),
        ("page_order_members", PAGE_ORDER_BUCKETS),
        ("page_order_span", device.SPAN_WORDS),
    ]
    assert all((array.dtype, array.width) == ("Int32", 4) for array in added)
    # Every other array is the one of a body without the order, apart from
    # the barrier block's second round.
    assert [a for a in ordered if a not in added and a.name != "barriers"] == [
        a for a in plain if a.name != "barriers" and a.guard != "page_order"
    ]
    assert pipeline_parts(True, False, page_order=True) == {"query_map", "page_order"}
    assert "page_order" not in pipeline_parts(True, False)
    offsets, count, rounds = pipeline_barrier_plan(False, 0, False, 2, True)
    assert count == 64 and offsets["page_order_ready"] == 32
    # Its 32 arrivals: one per lane of the sorting warp.
    assert rounds[1] == ((31, 1),)
    assert "page_order_ready" not in pipeline_barrier_plan(False)[0]
    words = PAGE_ORDER_CAPACITY + 2 * PAGE_ORDER_BUCKETS + device.SPAN_WORDS
    assert paired_shared_memory_bytes(40, page_order=True) == (
        paired_shared_memory_bytes(40) + 4 * words + 32 * 8
    )
    # Beside a deeper V ring the block has its second round already.
    assert paired_shared_memory_bytes(40, value_ring=3, page_order=True) == (
        paired_shared_memory_bytes(40, value_ring=3) + 4 * words
    )
    # The role is the value-metadata actor's warp and a persistent
    # partition's spare one.
    for arguments in ((True, 0), (False, 2)):
        with pytest.raises(AssertionError):
            pipeline_barrier_plan(*arguments, False, 2, True)


def test_the_loader_takes_the_order_behind_its_head_pages():
    body = source_function("paged_attention_paired_body")
    text = ast.unparse(body)
    # Only the bucketed body asks for it, and only under fast math, with one
    # unit per CTA and without the value-metadata actor.
    assert "bucketed = page_order == MMA_DAG_PAGE_ORDER_BUCKETED" in text
    assert "assert cutlass.const_expr(fast_math and partition == 'item')" in text
    (ordered,) = source_calls(body, "_load_ordered_pages")
    (plain,) = source_calls(body, "_load_pages")
    (guard,) = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.const_expr(bucketed)"
        and source_calls(node, "_load_ordered_pages")
    ]
    assert ordered in list(ast.walk(guard.body[-1]))
    assert plain in list(ast.walk(guard.orelse[0]))
    # The source-order call is the one the body always made.
    passed = dict(
        zip(
            [argument.arg for argument in source_function("_load_pages").args.args],
            [ast.unparse(argument) for argument in plain.args],
            strict=False,
        )
    )
    assert (passed["g0"], passed["step0"]) == ("cutlass.Int64(0)", "cutlass.Int64(0)")
    assert (passed["count"], passed["total"]) == ("count", "count")
    assert "page_order" not in passed
    # The ordered loader: the head in source order, the wait for the
    # sorting warp's 32 arrivals, then the rest of the same stream with the
    # run the warp published.
    loader = source_function("_load_ordered_pages")
    head, rest = source_calls(loader, "_load_pages")
    names = [argument.arg for argument in source_function("_load_pages").args.args]
    first = dict(zip(names, map(ast.unparse, head.args), strict=False))
    second = dict(zip(names, map(ast.unparse, rest.args), strict=False))
    # Both halves are the one stream's: each releases from the loader's
    # tail with the config's lead (a stream whose release step lies in the
    # head releases there) and walks the raw V ring at the config's depth,
    # as the body hands them to the ordered loader.
    handed = dict(
        zip(
            [argument.arg for argument in loader.args.args],
            [ast.unparse(argument) for argument in ordered.args],
            strict=True,
        )
    )
    for name in ("capture_release_tail", "capture_release_lead", "value_ring"):
        # The ring is the one the body binds: its depth in pages, or the
        # ring counted in tiles in its place.
        passed = "raw_v_ring" if name == "value_ring" else name
        assert first[name] == second[name] == name and handed[name] == passed
        # The ordered loader has no default to fall back on.
        assert name in [argument.arg for argument in loader.args.args]
    assert not loader.args.defaults
    assert (first["g0"], first["step0"], first["count"], first["total"]) == (
        "cutlass.Int64(0)",
        "cutlass.Int64(0)",
        "head",
        "count",
    )
    assert "page_order" not in first
    assert (second["g0"], second["step0"], second["count"], second["total"]) == (
        "head",
        "head",
        "count - head",
        "count",
    )
    assert (second["page_order"], second["order_first"], second["order_stop"]) == (
        "page_order",
        "order_first",
        "order_stop",
    )
    lag = (
        "previous_page",
        "older_page",
        "previous_shift",
        "older_shift",
        "previous_head",
        "older_head",
    )
    assert all(second[name] == name for name in lag)
    # The names the two calls pass hold what the body handed in: the loader
    # binds only its head, the lagging tiles' state and the run.
    assert {
        node.id
        for node in ast.walk(loader)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    } == {"head", *lag, "order_first", "order_stop"}
    statements = [ast.unparse(statement) for statement in loader.body]
    assert "head = cutlass.Int64(PAGE_ORDER_HEAD)" in statements
    # A unit of fewer pages than the head streams the pages it has.
    clamp = statements.index("head = cutlass.Int64(PAGE_ORDER_HEAD)") + 1
    assert statements[clamp] == "if count < head:\n    head = count"
    # Behind the head the run is the empty one, and only a unit with a step
    # behind the head waits for the sorting warp and reads the run it left:
    # a shorter unit's loader never depends on that warp.
    empty = statements.index("order_first = cutlass.Int64(0)")
    assert statements[empty + 1] == "order_stop = cutlass.Int64(0)"
    assert any("= _load_pages(" in statement for statement in statements[:empty])
    longer = loader.body[empty + 2]
    assert isinstance(longer, ast.If) and not longer.orelse
    assert ast.unparse(longer.test) == "count > head"
    waited = [ast.unparse(statement) for statement in longer.body]
    assert waited[0] == "_wait_non_mma(page_order_ready, cutlass.Int32(0), sleep_waits)"
    assert waited[1].startswith("order_first = ")
    assert "page_order_span[SPAN_FIRST]" in waited[1]
    assert waited[2].startswith("order_stop = ")
    assert "page_order_span[SPAN_STOP]" in waited[2]
    assert len(waited) == 3
    assert statements[empty + 3].startswith("return _load_pages(")
    text = ast.unparse(loader)
    assert text.count("_wait_non_mma(") == 1 and text.count("page_order_span[") == 2
    # A step of the run streams the order's page; every other step the page
    # table's, and source order asks for no order at all. Both V-lag forms
    # of the loader begin a step with the one helper, which takes the page
    # from the one lookup.
    pages = ast.unparse(source_function("_load_pages"))
    assert pages.count("physical, key_shift = _loader_step(pages, page_order, ") == 2
    assert "_ordered_page(" not in pages and "pages[request" not in pages
    step = ast.unparse(source_function("_loader_step"))
    assert (
        step.count(
            "physical = _ordered_page(pages, page_order, request, logical_page, "
            "step0 + step, order_first, order_stop)"
        )
        == 1
    )
    assert "pages[request, logical_page]" not in step
    lookup = source_function("_ordered_page")
    (source_order,) = [
        node
        for node in lookup.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.const_expr(page_order is None)"
    ]
    assert [ast.unparse(node) for node in source_order.body] == [
        "physical = cutlass.Int32(pages[request, logical_page])"
    ]
    ordered = "\n".join(ast.unparse(node) for node in source_order.orelse)
    assert "if (step >= order_first) & (step < order_stop):" in ordered
    assert "page_order[cutlass.Int32(step - order_first)]" in ordered
    assert "physical = cutlass.Int32(pages[request, logical_page])" in ordered


def test_the_sorting_warp_orders_interior_tiles_alone_and_owns_its_barrier():
    role = source_function("order_unit_pages", device)
    text = ast.unparse(role)
    # The candidates: behind the head, ahead of the tail, one sort at most.
    assert "if count > PAGE_ORDER_HEAD + PAGE_ORDER_TAIL:" in text
    assert "count - (PAGE_ORDER_HEAD + PAGE_ORDER_TAIL)" in text
    assert "cutlass.Int64(PAGE_ORDER_CAPACITY)" in text
    # Each candidate is tested by the template's own interior test, with the
    # coordinates of its own stream step, and a page id is read only for a
    # tile that passes.
    (coordinates,) = source_calls(role, "attention_page_coordinates")
    assert [ast.unparse(argument) for argument in coordinates.args] == [
        "first_page",
        "last_page",
        "cutlass.Int64(PAGE_ORDER_HEAD + index)",
        "page_walk",
    ]
    (test,) = source_calls(role, "logical_attention_tile_is_interior")
    assert [ast.unparse(argument) for argument in test.args] == [
        "logical_key_base",
        "PAGE_KEYS",
        "length",
        "query_columns",
        "query_len",
        "group",
        "window",
        "coordinate_dtype",
    ]
    (passing,) = [
        node
        for node in ast.walk(role)
        if isinstance(node, ast.If) and test is node.test
    ]
    assert [ast.unparse(statement) for statement in passing.body] == [
        "page = cutlass.Int32(pages[request, logical_page])",
        "inside = cutlass.Int32(inside | cutlass.Int32(1 << round_index))",
        "own = cutlass.Int32(own + 1)",
        "lowest = cutlass.Int32(cutlass.min(lowest, index))",
        "highest = index",
    ]
    assert text.count("pages[") == 1
    # The interior candidates must be one run of at least the minimum: the
    # warp reduces their number, the first and the last over its lanes, and
    # the span from the first to the last must hold that number.
    shuffled = "cutlass.Int32(cute.arch.shuffle_sync_bfly({0}, 1 << bit))"
    for reduced in (
        f"total = cutlass.Int32(total + {shuffled.format('total')})",
        f"first = cutlass.Int32(cutlass.min(first, {shuffled.format('first')}))",
        f"last = cutlass.Int32(cutlass.max(last, {shuffled.format('last')}))",
    ):
        assert reduced in text
    assert (
        "ordered = (last - first + 1 == total) & (total >= PAGE_ORDER_MIN_SORTED)"
    ) in text
    # Only interior candidates are counted and placed.
    for atomic, test in (
        ("atomic_add", "inside & 1 << round_index != 0"),
        ("atomic_or", "placed"),
    ):
        (call,) = source_calls(role, f"cute.arch.{atomic}")
        (guard,) = [
            node
            for node in ast.walk(role)
            if isinstance(node, ast.If)
            and call in list(ast.walk(node))
            and ast.unparse(node.test) == test
        ]
        assert guard is not None
    assert "placed = inside & 1 << round_index != 0" in text
    stores = [
        node
        for node in ast.walk(role)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]).startswith("order[")
    ]
    assert len(stores) == 1 and ast.unparse(stores[0].value) == "ids[round_index]"
    # Every lane arrives at the role's own pipeline barrier, behind its
    # stores; no named barrier of another role is used.
    assert ast.unparse(role.body[-1]) == "prims.mbarrier_arrive(ready)"
    source = DEVICE_ROLE.read_text()
    assert "barrier_cta" not in source and "cute.arch.barrier(" not in source
    assert "setmaxregister" not in source
    # The body: the role on the idle control warp, ahead of a boundary
    # prefetch, with nothing of its own between the role and what follows.
    body = source_function("paged_attention_paired_body")
    (call,) = source_calls(body, "order_unit_pages")
    passed = dict(
        zip(
            [argument.arg for argument in role.args.args],
            [ast.unparse(argument) for argument in call.args],
            strict=True,
        )
    )
    assert passed["ready"] == "page_order_ready"
    assert passed["shift"] == "page_order_shift"
    assert (passed["count"], passed["length"]) == ("count", "length")
    (spare,) = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "warp == 19"
        and call in list(ast.walk(node))
    ]
    roles = [ast.unparse(statement.test) for statement in spare.body]
    assert roles == [
        "cutlass.const_expr(value_metadata_scales is not None)",
        "cutlass.const_expr(bucketed)",
        "cutlass.const_expr(boundary_prefetch > 0)",
    ]
    # The role's last statement on the warp is its own arrival: the body
    # adds no warp synchronization behind it. The barrier instruction of
    # the body's final join makes each thread wait for its warp's other
    # threads itself, so the sort's lanes need not be joined ahead of it.
    ordered = spare.body[1]
    assert isinstance(ordered.body[-1], ast.Expr)
    assert ordered.body[-1].value is call
    assert "sync_warp" not in ast.unparse(ordered)
    # The sort's own synchronizations stay: one behind each step in which
    # lanes store or update words that other lanes of the warp then use.
    assert text.count("cute.arch.sync_warp()") == 6


def test_the_device_sort_is_a_stable_sort_by_bucket():
    generator = random.Random(7)
    sorted_runs = 0
    for trial in range(300):
        shift = generator.choice((0, 2, 5, 12))
        pool = generator.choice((13, 80, 5000, 131072))
        candidates = generator.choice(
            (0, 1, 31, 32, 33, 50, 64, 197, 377, 415, PAGE_ORDER_CAPACITY)
        )
        ids = [generator.randrange(pool) for _ in range(candidates)]
        # One run of interior tiles, with boundary tiles ahead of it and
        # behind it, and now and then a stray tile that breaks the run.
        lead = generator.choice((0, 0, 1, 3))
        trail = generator.choice((0, 0, 2, 40))
        inside = [lead <= index < candidates - trail for index in range(candidates)]
        broken = trial % 7 == 0 and candidates > 40
        if broken:
            inside[generator.randrange(lead + 1, candidates - trail - 1)] = False
        first, total, order = mirror_sort(ids, inside, shift)
        run = [page for page, keep in zip(ids, inside, strict=True) if keep]
        if broken or len(run) < PAGE_ORDER_MIN_SORTED:
            assert (first, total, order) == (0, 0, [])
            continue
        sorted_runs += 1
        assert (first, total) == (lead, len(run))
        # Python's sort is stable: equal buckets keep their source order.
        assert order == sorted(run, key=lambda page: page_order_bucket(page, shift))
    assert sorted_runs > 100


# What the device role is run on, on the host: the window, whether the pages
# are iterated in reverse, the page table's extent, each request's page range
# when a tensor gives it, and the requests' lengths in pages (a few keys
# short of whole pages for every third request).
DEVICE_CASES = {
    # No window: requests of every length, on either side of the fewest
    # pages a request is reordered for, and shorter than the tail.
    "no window": (
        NO_WINDOW,
        False,
        TABLE,
        None,
        [64, FEWEST, FEWEST - 1, FEWEST + 1, FEWEST + 2, 10, 0, 33, 60, 1],
    ),
    # A window of fifty pages: a long request's first tile is a clipped
    # window edge, and its interior run starts behind it.
    "window": (50 * PAGE, False, TABLE, None, [64, 63, 60, 52, 40, 20]),
    # Page ranges from a tensor, longer than the keys a request has: the
    # tiles past its length are fully masked and lie among the candidates.
    "masked tail": (
        NO_WINDOW,
        False,
        TABLE,
        [(0, TABLE), (3, TABLE - 2), (0, 40)],
        [40, 50, 36, 64, 38, 20],
    ),
    # The same in reverse over 96 pages: the masked tiles come first, so
    # the run starts in mid-stream, at the warp's last lane and around it
    # (the second and third requests, a few keys short of whole pages: the
    # masked tiles, the boundary tile, and the run from candidate 31 and 32).
    "masked head": (
        NO_WINDOW,
        True,
        96,
        [(0, 96)],
        [
            63,
            96 + 1 - PAGE_ORDER_HEAD - 31,
            96 + 1 - PAGE_ORDER_HEAD - 32,
            62,
            53,
            68,
            66,
            67,
            90,
        ],
    ),
    # More interior pages than one sort orders: the first 416.
    "capacity": (
        NO_WINDOW,
        False,
        500,
        None,
        [
            500,
            PAGE_ORDER_CAPACITY + KEPT + 1,
            PAGE_ORDER_CAPACITY + KEPT,
            PAGE_ORDER_CAPACITY + KEPT - 1,
            300,
        ],
    ),
}


def _device_args(case: str) -> tuple[object, ...]:
    window, reverse, table, bounds, pages = DEVICE_CASES[case]
    values = list(
        partition_attention_args(
            units=len(pages),
            rows=8,
            reverse_pages=reverse,
            window=window,
            table_pages=table,
            pool_pages=POOL,
        )
    )
    if bounds is not None:
        values[10] = torch.tensor(
            [bounds[index % len(bounds)] for index in range(len(pages))],
            dtype=torch.int32,
        )
    args = tuple(values)
    randomize_partition_inputs(args, sorted(DEVICE_CASES).index(case))
    set_request_lengths(
        args, [max(count * PAGE - index % 3, 0) for index, count in enumerate(pages)]
    )
    return args


@pytest.mark.parametrize("case", list(DEVICE_CASES))
def test_the_device_role_orders_what_the_definitions_name(case):
    """The role's own source, 32 lanes in lockstep on the host, against the
    model: the same run of stream steps and the same order, with a page id
    read for every interior candidate and for no other tile, and no word of
    shared memory loaded or stored by one lane while another stores it."""
    args = _device_args(case)
    table = args[5]
    reverse = bool(args[14])
    shift = 2
    reordered = 0
    for request in range(table.size(0)):
        first_step, stop_step, order, reads = run_device_sort(args, request, shift)
        model_first, model_order = ordered_run(args, request, shift)
        assert order == model_order, (case, request)
        if not model_order:
            assert (first_step, stop_step) == (0, 0), (case, request)
            continue
        reordered += 1
        assert (first_step, stop_step) == (
            model_first,
            model_first + len(model_order),
        ), (case, request)
        # The order is a permutation of the run's own pages, and the run
        # lies behind the head and ahead of the tail.
        first, last = unit_page_range(args, request)
        own = [
            int(table[request, stream_page(first, last, step, reverse)])
            for step in range(first_step, stop_step)
        ]
        assert sorted(order) == sorted(own), (case, request)
        assert first_step >= PAGE_ORDER_HEAD
        assert stop_step <= last - first - PAGE_ORDER_TAIL
        assert stop_step - first_step <= PAGE_ORDER_CAPACITY
        # A page id was read for every interior candidate: at least the
        # run's, and no more than one sort's candidates.
        assert stop_step - first_step <= reads <= PAGE_ORDER_CAPACITY
    assert reordered > 0
    if case == "masked head":
        firsts = {
            ordered_run(args, request, shift)[0] - PAGE_ORDER_HEAD
            for request in range(table.size(0))
        }
        # The run's first candidate: the warp's last lane, the next round's
        # first and others.
        assert {31, 32} <= firsts
    if case == "capacity":
        lengths = [
            stop - first
            for first, stop, _order, _reads in (
                run_device_sort(args, request, shift) for request in range(3)
            )
        ]
        assert lengths == [
            PAGE_ORDER_CAPACITY,
            PAGE_ORDER_CAPACITY,
            PAGE_ORDER_CAPACITY,
        ]
    # The model's page table is the original one outside the runs.
    modelled, requests = bucketed_page_table(args, shift)
    assert len(requests) == reordered
    assert sorted(modelled.flatten().tolist()) == sorted(table.flatten().tolist())


def test_the_order_is_searched_everywhere_and_seeded_for_kernels_without_a_window():
    eager = _bound()
    fragment = eager.config_spec._flat_fields()[KEY]
    assert fragment is eager.config_spec.cute_mma_dag_page_order
    # The order shortens the launch's own page period, which every timing
    # of the kernel shows.
    assert fragment.choices == fragment._active_choices() == ("source", "bucketed")
    assert fragment.default() == "source"
    seeds = stream_seeds(eager)
    head = seeds[0]
    assert head.get("cute_mma_dag_pass_warps") == PAIRED and KEY not in head.config
    # The leading paired family once more in bucketed order, behind every
    # other seed.
    named = [seed for seed in seeds if KEY in seed.config]
    assert named == seeds[-1:]
    assert named[0].config == {**head.config, KEY: "bucketed"}
    # Where a search samples the lead of the tail release and seeds it with
    # the deeper ring, the last of those seeds follows in bucketed order,
    # and where it also samples and seeds a boundary prefetch, that seed
    # once more with a prefetch of the order's head.
    banks = _bound(input_banks=2)
    seeds = stream_seeds(banks)
    head = seeds[0]
    named = [seed for seed in seeds if KEY in seed.config]
    assert named == seeds[-3:]
    assert named[0].config == {**head.config, KEY: "bucketed"}
    combined = {**head.config, LEAD: 16, RING: 3, V_LAG: 0, KEY: "bucketed"}
    assert named[1].config == combined
    assert PAGE_ORDER_PREFETCH_SEED_PAGES == PAGE_ORDER_HEAD == 4
    assert named[2].config == {**combined, PREFETCH: 4}
    assert seeds[-4].config == {**head.config, LEAD: 16, RING: 3, V_LAG: 0}
    # It joins the prefetch family, the leading family with 8 and with 12
    # pages, and is the only seed that names a prefetch beside another of
    # the three knobs or a register split.
    prefetching = [seed for seed in seeds if PREFETCH in seed.config]
    assert [seed.config[PREFETCH] for seed in prefetching] == [8, 12, 4]
    assert [seed.config for seed in prefetching[:2]] == [
        {**head.config, PREFETCH: pages} for pages in (8, 12)
    ]
    # The guards are the existing combination's and the prefetch family's:
    # no such seed where the search does not sample the prefetch, where the
    # prefetch has no four pages, or where the device has no room for the
    # deeper ring (the combination is then the lead in bucketed order).
    spec = banks.config_spec
    assert spec.cute_mma_dag_boundary_prefetch is not None
    assert spec.cute_mma_dag_boundary_prefetch_searched
    unsearched = copy.copy(spec)
    unsearched.cute_mma_dag_boundary_prefetch_searched = False
    shorter = copy.copy(spec)
    shorter.cute_mma_dag_boundary_prefetch = EnumFragment(choices=(0, 8, 12))
    absent = copy.copy(spec)
    absent.cute_mma_dag_boundary_prefetch = None
    for other in (unsearched, shorter, absent):
        ordered = [seed for seed in stream_seeds(banks, other) if KEY in seed.config]
        assert [seed.config for seed in ordered] == [named[0].config, combined]
    shallow = copy.copy(spec)
    shallow.cute_mma_dag_value_ring = None
    ordered = [seed for seed in stream_seeds(banks, shallow) if KEY in seed.config]
    assert [seed.config for seed in ordered] == [
        named[0].config,
        {**head.config, LEAD: 16, KEY: "bucketed"},
    ]
    assert not any(
        PREFETCH in seed.config for seed in stream_seeds(eager) if KEY in seed.config
    )
    # No seed in bucketed order for a windowed kernel or past one wave; the
    # field stays, for a windowed kernel with the reversed walk as well.
    for other, choices in (
        (_bound(window=50 * 128), ("source", "bucketed", "reversed")),
        (_bound(WAVES), ("source", "bucketed")),
    ):
        assert other.config_spec._flat_fields()[KEY]._active_choices() == choices
        assert not any(
            seed.config.get(KEY) == "bucketed" for seed in stream_seeds(other)
        )
    for bound in (eager, banks):
        spec = bound.config_spec
        with (
            cpu_render_target(fake_device_sm_count=SMS),
            bound.env,
            bound.host_function,
        ):
            assert KEY not in spec.default_config().config
            generation = spec.create_config_generation()
            for seed in named:
                normalized = spec.normalized_config(seed)
                assert normalized.config[KEY] == "bucketed"
                _flat, transferred = generation.canonicalize_flat(
                    generation.flatten(normalized)
                )
                assert transferred.config[KEY] == "bucketed"
    keywords = body_keywords(stream_render(banks, named[1]))
    assert keywords["page_order"] == "'bucketed'"
    assert (keywords["value_ring"], keywords["capture_release_lead"]) == ("3", "16")
    assert "boundary_prefetch" not in keywords
    # The seed on the prefetch family is a complete legal config: strict
    # normalization keeps every knob it names, a search's row is that
    # config, and it renders all four.
    wanted = {LEAD: 16, RING: 3, V_LAG: 0, KEY: "bucketed", PREFETCH: 4}
    with cpu_render_target(fake_device_sm_count=SMS), banks.env, banks.host_function:
        normalized = spec.normalized_config(named[2])
        assert {key: normalized.config[key] for key in wanted} == wanted
        assert all(
            normalized.config[key] == value for key, value in named[2].config.items()
        )
        generation = spec.create_config_generation()
        _flat, transferred = generation.canonicalize_flat(generation.flatten(named[2]))
        assert {key: transferred.config[key] for key in wanted} == wanted
        # The search starts from it: the last of the seeds it transfers.
        pairs = generation.seed_flat_config_pairs()
        assert pairs[-1][1] == transferred
        assert sum(PREFETCH in config.config for _flat, config in pairs) == 3
    keywords = body_keywords(stream_render(banks, named[2]))
    assert keywords["page_order"] == "'bucketed'"
    assert (keywords["value_ring"], keywords["capture_release_lead"]) == ("3", "16")
    assert keywords["boundary_prefetch"] == "4" and "v_lag" not in keywords


def test_a_step_onto_the_order_takes_one_unit_per_cta():
    bound = _bound(WAVES)
    spec = bound.config_spec
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        plain = stream_config()
        balanced = stream_config(**{PARTITION: "balanced"})
        for parent in (plain, balanced):
            reached, alone = stream_step(spec, parent, **{KEY: "bucketed"})
            assert reached.config[KEY] == "bucketed"
            assert PARTITION not in spec.normalized_config(reached).config
            if parent is plain:
                assert alone == reached
            else:
                # Alone the step is an alias of where it started: the order
                # belongs to one unit per CTA, and normalization repairs it
                # to the source's and changes nothing else.
                assert KEY not in alone.config
                assert alone.config[PARTITION] == "balanced"
        on = spec.normalized_config(stream_config(**{KEY: "bucketed"}))
        reached, alone = stream_step(spec, on, **{KEY: "source"})
        assert reached == alone and KEY not in spec.normalized_config(reached).config
        # A step onto a partition from the bucketed order is the
        # partition's: it drops the order.
        reached, _alone = stream_step(spec, on, **{PARTITION: "balanced"})
        moved = spec.normalized_config(reached)
        assert moved.config[PARTITION] == "balanced" and KEY not in moved.config
        # The order pins no V lag, ring depth or lead and takes none.
        for move in ({V_LAG: 0}, {RING: 3}):
            reached, alone = stream_step(spec, on, **move)
            assert reached == alone
            assert spec.normalized_config(reached).config[KEY] == "bucketed"


def test_overrides_keep_the_paired_seeds_with_or_without_their_order():
    bound = _bound(input_banks=2)
    spec = bound.config_spec
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        seeded, failures = transferred_seeds(spec, None)
        assert not failures
        paired = [
            seed for seed in seeded if seed.get("cute_mma_dag_pass_warps") == PAIRED
        ]
        assert [seed.config[KEY] for seed in seeded if KEY in seed.config] == [
            "bucketed"
        ] * 3
        # A fixed order is layered over complete candidates: every paired
        # one takes it and keeps its place; a candidate on another schedule
        # is left to that body's rule, which has no such role.
        forced, failures = transferred_seeds(spec, {KEY: "bucketed"})
        kept = [
            seed for seed in forced if seed.get("cute_mma_dag_pass_warps") == PAIRED
        ]
        assert kept and forced[0] == kept[0] and len(forced) == len(kept)
        assert all(seed.config[KEY] == "bucketed" for seed in kept)
        assert len(failures) == len(seeded) - len(paired)
        assert all(KEY in failure for failure in failures)
        for overrides, drops in (
            ({PARTITION: "balanced"}, True),
            ({PARTITION: "whole"}, True),
            ({"cute_mma_dag_pass_warps": "attention"}, True),
            ({PARTITION: "item"}, False),
            ({V_LAG: 3}, False),
            ({"cute_pdl": False}, False),
        ):
            moved, failures = transferred_seeds(spec, overrides)
            assert not [failure for failure in failures if KEY in failure], overrides
            orders = [seed.config[KEY] for seed in moved if KEY in seed.config]
            assert bool(orders) is not drops, overrides
        moved, failures = transferred_seeds(
            spec, {KEY: "bucketed", PARTITION: "balanced"}
        )
        assert not moved and len(failures) == len(seeded)


def test_the_bucketed_kernel_compiles_with_its_sort_and_no_new_register_site():
    bound = _bound(17, rows=40)
    config = stream_config(40, **{KEY: "bucketed", V_LAG: 0})
    plain_config = stream_config(40, **{V_LAG: 0})
    with cpu_render_target(fake_device_sm_count=SMS, num_sm=SMS):
        source = bound.to_code(config)
        plain_source = bound.to_code(plain_config)
    arguments = partition_attention_args(
        units=17, rows=40, window=NO_WINDOW, table_pages=TABLE, pool_pages=POOL
    )
    with tempfile.TemporaryDirectory(prefix="helion-cute-order-") as dump_dir:
        ordered = compile_offline(bound, arguments, source, Path(dump_dir) / "ordered")
        plain = compile_offline(
            bound, arguments, plain_source, Path(dump_dir) / "plain"
        )
    # The sort's shared-memory atomics and warp-level votes exist only in
    # the bucketed kernel, and the role adds no register-quota site: the
    # sorting warp keeps the control warpgroup's.
    assert not re.findall(r"atom\S*\.(?:add|or)", plain.ptx)
    # Three reductions over the warp's 32 lanes, five butterfly steps each.
    assert ordered.ptx.count("shfl.sync.bfly") - plain.ptx.count("shfl.sync.bfly") == 15
    assert len(re.findall(r"atom\S*\.add", ordered.ptx)) >= 1
    assert len(re.findall(r"atom\S*\.or", ordered.ptx)) >= 1
    assert ordered.ptx_quotas() == plain.ptx_quotas()
    assert ordered.threads == plain.threads == 640


def test_a_gpu_pairs_inputs_are_seeded_by_the_pairs_own_arguments():
    """The GPU tests bound the page orders' rounding on the random inputs of
    a compiled pair (``stream_pair``). Their seed is a function of the
    arguments the pair is cached under, the same in every process and
    whatever pairs a worker compiled before, so a bound holds or fails in
    every order the tests run in."""
    # The bucketed order's first pair: the candidate, no schedule knobs, no
    # window, the fixture's page order, rows, reserved SMs, table and pool.
    key = (((KEY, "bucketed"),), (), NO_WINDOW, False, 8, 0, TABLE, POOL, False, False)
    # A checksum of the key's text, which no process salts as it does a hash.
    assert pair_seed(key) == 1142789880
    assert pair_seed((*key[:-1], True)) != pair_seed(key)
