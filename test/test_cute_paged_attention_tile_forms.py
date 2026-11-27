"""Check the forms the paired attention body gives its per-tile work on the
packed scale tile: a V tile converted ahead of the vote on its scales, and
the per-tile loops counted in Int32 below the bound the bind's page table
states. The conversions themselves are compiled standalone in
``test_cute_paged_attention_decode``; here the paired kernel is rendered and
compiled offline, and its PTX read. And the first tile of a work item, whose
event has no output panel to rescale."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import re
import tempfile

import pytest

from helion._compiler.cute import _paged_attention_paired_pipeline as paired
from helion._compiler.cute import _paged_attention_partial_pipeline as partial
from helion._compiler.cute import _paged_attention_pipeline as single
from helion._compiler.cute.paged_attention_partition import paired_stream_page_bound
from helion._testing.cute_offline import compile_offline
from helion._testing.cute_offline import partition_bound
from helion._testing.cute_paged_partition import PAGES
from helion._testing.cute_paged_stream import SMS
from helion._testing.cute_paged_stream import body_keywords
from helion._testing.cute_paged_stream import stream_bound
from helion._testing.cute_paged_stream import stream_config
from helion._testing.cute_paged_stream import stream_render

pytestmark = pytest.mark.cpu_only
BOUND = "stream_page_bound"
PARTITION = "cute_mma_dag_partition"
# One ``16x256b`` store stages a 16-key block of a V warp's two dimension
# halves; a tile of 128 keys is sixteen of them.
_V_STORE = "tcgen05.st.sync.aligned.16x256b"
_VOTE = re.compile(r"vote\.sync\.all\.pred\s+(%p\d+),")
# A loop counter's increment: ``add.s64 %rdN, %rdN, 1``.
_COUNT64 = re.compile(r"add\.s64\s+(%rd\d+), \1, 1;")
_COUNT32 = re.compile(r"add\.s32\s+(%r\d+), \1, 1;")


def _keyword_values(module: object, callee: str, keyword: str) -> list[str]:
    """The values a module's calls of ``callee`` pass for ``keyword``."""
    tree = ast.parse(Path(inspect.getfile(module)).read_text())  # type: ignore[arg-type]
    return [
        ast.unparse(passed.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == callee
        for passed in node.keywords
        if passed.arg == keyword
    ]


def test_only_the_paired_body_asks_for_the_forms_of_the_packed_tile():
    """The paired body's K and V conversions ask for the forms; the two
    single-head pipelines call the same conversions without them and keep
    the guard at the head of every tile."""
    for callee in ("stage_scaled_s8_rows_to_tmem", "stage_scaled_s8_transpose_to_tmem"):
        assert _keyword_values(paired, callee, "packed_tile_forms") == ["True"]
        for module in (single, partial):
            assert _keyword_values(module, callee, "packed_tile_forms") == []


def test_a_bound_is_the_page_table_s_static_extent_over_the_cta_s_units():
    assert paired_stream_page_bound(17, 1) == 17
    assert paired_stream_page_bound(391, 320) == 391 * 320
    # No static extent, no unit, or a product an Int32 does not hold.
    assert paired_stream_page_bound(object(), 1) == 0
    assert paired_stream_page_bound(0, 1) == 0
    assert paired_stream_page_bound(17, 0) == 0
    assert paired_stream_page_bound(2**31 - 1, 1) == 2**31 - 1
    assert paired_stream_page_bound(2**31, 1) == 0
    assert paired_stream_page_bound(2**16, 2**15) == 0


@pytest.mark.parametrize("partition", ["item", "whole", "balanced"])
def test_a_paired_kernel_renders_the_bound_of_its_partition(partition):
    """One unit per CTA streams at most the page table's pages, a persistent
    CTA at most all units' (the product its plan already sums in Int32)."""
    units = 300
    bound = stream_bound(units)
    overrides = {} if partition == "item" else {PARTITION: partition}
    if partition != "item":
        overrides["cute_mma_dag_query_tma"] = True
    keywords = body_keywords(stream_render(bound, stream_config(**overrides)))
    expected = PAGES if partition == "item" else units * PAGES
    assert keywords[BOUND] == str(expected)
    if partition != "item":
        assert keywords["units"] == str(units)
        assert keywords["page_limit"] == str(PAGES)


def test_the_tile_loops_count_in_int32_below_the_bound_and_a_v_tile_votes_once():
    """The paired kernel, compiled offline: its V conversion stages a tile's
    sixteen blocks ahead of the vote on the tile's scales, and with the
    rendered bound the per-tile loops (the compute groups', both
    conversions', both issuers') count in Int32, where the same kernel
    without the bound counts them in the source's Int64."""
    bound, args = partition_bound()
    source = stream_render(bound, stream_config(block_sizes=[64]), sms=SMS)
    stated = f", {BOUND}={PAGES}"
    assert source.count(stated) == 1
    with tempfile.TemporaryDirectory(prefix="helion-cute-tile-") as dump_dir:
        counted = compile_offline(bound, args, source, Path(dump_dir) / "bound")
        unbounded = compile_offline(
            bound, args, source.replace(stated, ""), Path(dump_dir) / "unbounded"
        )
        with pytest.raises(AssertionError):
            # A bound that is no page count.
            compile_offline(
                bound,
                args,
                source.replace(stated, f", {BOUND}=-1"),
                Path(dump_dir) / "negative",
            )
    lines = counted.ptx.splitlines()
    stores = [index for index, line in enumerate(lines) if _V_STORE in line]
    votes = [index for index, line in enumerate(lines) if _VOTE.search(line)]
    assert len(stores) == 32
    # Between a tile's first store and the first store of the arm that
    # replaces it there is one vote, behind the sixteenth store.
    (vote,) = [line for line in votes if stores[0] < line < stores[16]]
    assert stores[15] < vote
    # Five loops over tiles: the two compute groups share one, as do the
    # warps of each conversion, and each issuer has its own.
    wide = len(_COUNT64.findall(unbounded.ptx)) - len(_COUNT64.findall(counted.ptx))
    narrow = len(_COUNT32.findall(counted.ptx)) - len(_COUNT32.findall(unbounded.ptx))
    assert wide == narrow == 5


def _template_function(name: str) -> ast.FunctionDef:
    (found,) = [
        node
        for node in ast.walk(ast.parse(Path(inspect.getfile(paired)).read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    return found


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and ast.unparse(call.func) == name
    ]


def test_a_work_item_s_first_tile_does_not_rescale_the_panel_it_starts_with():
    """An event multiplies the O panel by the rows' alpha. A work item's
    first tile is always an event, on the panel of zeros the item starts
    with, so its rescale would load, scale and store those zeros: it is
    left out, and every later event keeps it. Both drivers zero the panel
    ahead of the range they hand the recurrence, a part of a split unit
    included."""
    pages = _template_function("_compute_pages")
    (loop,) = [node for node in pages.body if isinstance(node, ast.For)]
    assert ast.unparse(loop.target) == "step"
    assert ast.unparse(loop.iter) == "cutlass.range(_tile_steps(count, int32_steps))"
    rescales = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.If)
        and _calls(node, "load_query_panel")
        and _calls(node, "store_query_panel")
    ]
    (rescale,) = rescales
    assert ast.unparse(rescale.test) == "maximum_changed & (step > 0)"
    # The panel is loaded and stored nowhere else in the loop.
    assert len(_calls(loop, "load_query_panel")) == 1
    assert len(_calls(loop, "store_query_panel")) == 1
    # The range starts on the zeros ``_compute_begin`` stored.
    begin = _template_function("_compute_begin")
    (zeroed, *_denominators) = _calls(begin, "store_query_panel")
    assert ast.unparse(zeroed.args[0]) == "output_address"
    assert ast.unparse(zeroed.args[1]) == "zero"
    for driver in ("_compute_head", "_compute_units"):
        function = _template_function(driver)
        (begun,) = _calls(function, "_compute_begin")
        (ran,) = _calls(function, "_compute_pages")
        assert begun.lineno < ran.lineno
        # In the persistent driver both sit in the loop over the units.
        blocks = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.While) and _calls(node, "_compute_pages")
        ]
        assert bool(blocks) == (driver == "_compute_units")
        assert all(_calls(block, "_compute_begin") for block in blocks)
