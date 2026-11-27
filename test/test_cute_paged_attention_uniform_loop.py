"""Check the warp-uniform form of the paired attention schedule's softmax
tile loop, which every paired kernel has where its bind bounds the loop's
trip count: what the template marks at a unit's head, that no tile reloads
the unit's first step, the one page mapping walked in Int32 with the bound
that walk needs, and what the compiled loop then lacks beside the same kernel
without the bound."""

from __future__ import annotations

import ast
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest

from helion._compiler.cute import _paged_attention_paired_pipeline as template
from helion._compiler.cute import _paged_attention_partial_compute as per_key
from helion._compiler.cute import _paged_attention_partial_pipeline as partial
from helion._compiler.cute import _paged_attention_reference_compute as reference
from helion._compiler.cute.paged_attention_page_order import ReversedWalk
from helion._testing.cute_offline import compile_offline
from helion._testing.cute_paged_partition import PAGE
from helion._testing.cute_paged_partition import PAGES
from helion._testing.cute_paged_partition import partition_attention_args
from helion._testing.cute_paged_reversed_order import reversed_walk
from helion._testing.cute_paged_stream import LOOP_WAITS_IN_REGIONS
from helion._testing.cute_paged_stream import body_keywords
from helion._testing.cute_paged_stream import jit_host_body
from helion._testing.cute_paged_stream import softmax_loop_forms
from helion._testing.cute_paged_stream import source_calls
from helion._testing.cute_paged_stream import source_function
from helion._testing.cute_paged_stream import stream_bound
from helion._testing.cute_paged_stream import stream_config
from helion._testing.cute_paged_stream import stream_render

if TYPE_CHECKING:
    from pathlib import Path
    from typing_extensions import Self

    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only
PARTITION = "cute_mma_dag_partition"
QUERY_TMA = "cute_mma_dag_query_tma"
# Units on 148 SMs: more than two waves, where a persistent CTA streams
# several units and the balanced partition cuts units between CTAs.
WAVES = 300
BALANCED = {PARTITION: "balanced", QUERY_TMA: True}
WHOLE = {PARTITION: "whole", QUERY_TMA: True}
EARLY_TMEM = {"cute_mma_dag_probability_schedule": "early_tmem"}


def _wrapping(bits: int) -> type[int]:
    """A host integer of ``bits`` bits for the mapping's body: two's
    complement arithmetic that wraps as the device's does, between values
    of its own width and plain constants only, so a walk that mixed the
    widths raises."""

    class Wrapped(int):
        def __new__(cls, value: int = 0) -> Self:
            low = int(value) & ((1 << bits) - 1)
            return super().__new__(cls, low - ((low >> (bits - 1)) << bits))

        def _operand(self, other: int) -> int:
            assert type(other) in (int, type(self)), (bits, type(other))
            return int(other)

        def __add__(self, other: int) -> Self:
            return type(self)(int(self) + self._operand(other))

        def __sub__(self, other: int) -> Self:
            return type(self)(int(self) - self._operand(other))

        def __mul__(self, other: int) -> Self:
            return type(self)(int(self) * self._operand(other))

    return Wrapped


def _walk_pages(first: int, last: int, walk: object) -> list[int]:
    """The logical page of each step of a walk over ``[first, last)``,
    apart from the mapping: the source's loop, a ``ReversedWalk``, or a
    source reflection (a dtype)."""
    if walk is None:
        return list(range(first, last))
    if isinstance(walk, ReversedWalk):
        return reversed_walk(first, last, walk.clipped_page_lead)
    return list(range(last - 1, first - 1, -1))


def test_the_one_mapping_walks_in_int32_what_it_walks_in_int64():
    """``attention_page_coordinates`` is one mapping in two integer types:
    told to walk in Int32 it gives every step of every walk the page and
    the Int64 key base it gives in Int64, with integers that wrap at their
    width and never mix widths: over the source's loop, the reversed walk
    with each lead of the clipped page (ranges shorter than the lead among
    them) and a source reflection, which keeps its own dtype and is
    widened first. That holds up to the last page whose keys an Int32
    holds; one page further the Int32 key base wraps."""
    narrow, wide = _wrapping(32), _wrapping(64)
    typed = SimpleNamespace(Int32=narrow, Int64=wide, const_expr=lambda value: value)
    mapping = jit_host_body(partial.attention_page_coordinates)
    # Pages whose key base, 128 times the index, an Int32 holds.
    held = 2**31 // 128
    walks = (None, ReversedWalk(0), ReversedWalk(1), ReversedWalk(8), narrow, wide)
    compared = top = 0
    with patch.object(partial, "cutlass", typed):
        for walk in walks:
            reflected = walk in (narrow, wide)
            for first in (0, 3, 1000, held - 40):
                for pages in (0, 1, 2, 7, 8, 9, 17, 40):
                    last = first + pages
                    expected = _walk_pages(first, last, walk)
                    for step in range(pages):
                        page, key_base = mapping(
                            wide(first), wide(last), wide(step), walk
                        )
                        assert (page, key_base) == (
                            expected[step],
                            expected[step] * 128,
                        )
                        assert type(page) is type(key_base) is wide
                        walked = mapping(
                            narrow(first), narrow(last), narrow(step), walk, narrow
                        )
                        assert walked == (page, key_base), (walk, first, last, step)
                        # The page in the walk's type (a reflection's
                        # walk is the Int64 one), the key base widened.
                        assert type(walked[0]) is (wide if reflected else narrow)
                        assert type(walked[1]) is wide
                        compared += 1
                        top = max(top, int(page))
        # The last page an Int32 holds the keys of was among them.
        assert compared == len(walks) * 4 * 84 and top == held - 1
        for walk in walks[:4]:
            # One page further the product wraps in Int32: the bound the
            # body asks of the bind before it walks in that type.
            _page, key_base = mapping(wide(held), wide(held + 1), wide(0), walk)
            assert key_base == 2**31
            page, wrapped = mapping(
                narrow(held), narrow(held + 1), narrow(0), walk, narrow
            )
            assert (page, wrapped) == (held, -(2**31))
        # The integers refuse another width: Int32 values without their
        # type meet the reversed walk's Int64 constant.
        with pytest.raises(AssertionError):
            mapping(narrow(0), narrow(9), narrow(0), ReversedWalk(8))


def test_the_loop_marks_what_the_group_shares_and_reloads_nothing():
    """Read from the template: where the bind bounds the trip count
    (``int32_steps``) the loop's head passes the trip count, the phase, the
    first step, the page range and the length through ``make_warp_uniform``,
    and no tile reloads the first step from the group's record; without
    the bound the loop is the one that reloads it. A tile takes its key
    base from the one mapping, in Int32 where the bind also bounds the
    table's keys (``int32_keys``) and in the mapping's Int64 otherwise.
    Neither softmax helper marks the vote's result."""
    pages = source_function("_compute_pages", template)
    assert [argument.arg for argument in pages.args.args][-2:] == [
        "int32_steps",
        "int32_keys",
    ]
    statements = pages.body[1:] if ast.get_docstring(pages) else pages.body
    implied, head, loop = statements
    # The key bound is no bound without the steps' bound.
    assert ast.unparse(implied) == (
        "assert cutlass.const_expr(int32_steps or not int32_keys)"
    )
    assert ast.unparse(head.test) == "cutlass.const_expr(int32_steps)"
    assert not head.orelse
    marked = {
        ast.unparse(node.targets[0]): ast.unparse(node.value) for node in head.body
    }
    assert marked == {
        "count": "cutlass.Int64(cute.arch.make_warp_uniform(cutlass.Int32(count)))",
        "phase": "cute.arch.make_warp_uniform(phase)",
        "step0": "cutlass.Int64(cute.arch.make_warp_uniform(cutlass.Int32(step0)))",
        "first_page": (
            "cutlass.Int64(cute.arch.make_warp_uniform(cutlass.Int32(first_page)))"
        ),
        "last_page": (
            "cutlass.Int64(cute.arch.make_warp_uniform(cutlass.Int32(last_page)))"
        ),
        "length": "cute.arch.make_warp_uniform(length)",
    }
    assert isinstance(loop, ast.For)
    assert ast.unparse(loop.iter) == "cutlass.range(_tile_steps(count, int32_steps))"
    walked = loop.body[0]
    assert ast.unparse(walked.test) == "cutlass.const_expr(int32_keys)"
    # Under the key bound: one call of the mapping with the range and the
    # step in Int32 and that type named, no read of the record.
    (narrow,) = walked.body
    assert ast.unparse(narrow) == (
        "_logical_page, logical_key_base = attention_page_coordinates("
        "cutlass.Int32(first_page), cutlass.Int32(last_page), "
        "cutlass.Int32(step0) + cutlass.Int32(step), page_walk, "
        "integer=cutlass.Int32)"
    )
    # Without it the mapping's Int64 product, and the reload of the first
    # step only where the head could not mark it.
    reload, mapped = walked.orelse
    assert ast.unparse(reload) == (
        "if cutlass.const_expr(walk is not None and (not int32_steps)):\n"
        "    assert walk is not None\n"
        "    step0 = cutlass.Int64(cutlass.Int32(walk[WALK_STEP]))"
    )
    assert ast.unparse(mapped) == (
        "_logical_page, logical_key_base = attention_page_coordinates("
        "first_page, last_page, step0 + step, page_walk)"
    )
    assert not any("WALK_STEP" in ast.unparse(node) for node in loop.body[1:])
    # Neither softmax helper hears of the form, and the vote's result is
    # the event branch's condition as the barrier's reduction returns it
    # (marked warp-uniform too it cost a one-unit kernel 0.2 to 0.4 us).
    for name, module in (
        ("online_softmax_reference", reference),
        ("online_softmax_per_key", per_key),
    ):
        (call,) = source_calls(pages, name)
        assert "int32" not in ast.unparse(call)
        helper = source_function(name, module)
        assert "make_warp_uniform" not in ast.unparse(helper)
        vote = next(
            index
            for index, node in enumerate(helper.body)
            if isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == "maximum_changed"
            and source_calls(node, "barrier_cta_red")
        )
        assert ast.unparse(helper.body[vote + 1].test) == "maximum_changed"
    # Both drivers hand the two bounds on as the body computed them.
    (units_call,) = source_calls(
        source_function("_compute_units", template), "_compute_pages"
    )
    assert [ast.unparse(argument) for argument in units_call.args[-2:]] == [
        "int32_steps",
        "int32_keys",
    ]
    (head_call,) = source_calls(
        source_function("_compute_head", template), "_compute_pages"
    )
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in head_call.keywords
    } == {"int32_steps": "int32_steps", "int32_keys": "int32_keys"}
    body = source_function("paged_attention_paired_body", template)
    for driver in ("_compute_units", "_compute_head"):
        (call,) = source_calls(body, driver)
        passed = [ast.unparse(argument) for argument in call.args]
        steps = passed.index("int32_steps")
        assert passed[steps + 1] == "int32_keys", driver


def _bounds(**values: object) -> tuple[bool, bool]:
    """The two bounds the body computes from its keywords, by running its
    own statements."""
    body = source_function("paged_attention_paired_body", template)
    wanted = ("int32_steps", "page_extent", "int32_keys")
    statements = [
        node
        for node in body.body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) in wanted
    ]
    assert [ast.unparse(node.targets[0]) for node in statements] == list(wanted)
    scope = dict(values)
    exec(compile(ast.Module(statements, []), "<bounds>", "exec"), {}, scope)
    return bool(scope["int32_steps"]), bool(scope["int32_keys"])


def test_the_key_base_is_int32_only_where_the_tables_keys_fit_one():
    """The steps' bound, units times the page table's extent, does not
    bound a page's index times its 128 keys: the Int32 walk needs the
    table's extent below 2**24 pages, which the body derives from the
    bound of a CTA of one unit and from a persistent CTA's page limit. A
    table of dynamic extent states neither bound."""
    item = {"partition": "item", "page_limit": 0}
    assert _bounds(stream_page_bound=17, **item) == (True, True)
    assert _bounds(stream_page_bound=2**24 - 1, **item) == (True, True)
    assert _bounds(stream_page_bound=2**24, **item) == (True, False)
    assert _bounds(stream_page_bound=2**31 - 1, **item) == (True, False)
    assert _bounds(stream_page_bound=0, **item) == (False, False)
    for partition in ("whole", "balanced"):
        assert _bounds(
            partition=partition, page_limit=17, stream_page_bound=300 * 17
        ) == (True, True)
        assert _bounds(
            partition=partition, page_limit=2**24 - 1, stream_page_bound=2**24 - 1
        ) == (True, True)
        # Two units of a table of 2**24 pages: an Int32 counts the steps
        # and not the keys.
        assert _bounds(
            partition=partition, page_limit=2**24, stream_page_bound=2 * 2**24
        ) == (True, False)
        assert _bounds(partition=partition, page_limit=17, stream_page_bound=0) == (
            False,
            False,
        )


def _unbounded(source: str) -> str:
    """The render of a page table of dynamic extent, which states no bound:
    the loop that marks nothing and reloads the first step."""
    unbounded, replaced = re.subn(r", stream_page_bound=\d+", "", source)
    assert replaced == 1
    return unbounded


def _loops(
    bound: BoundKernel[object],
    args: tuple[object, ...],
    sources: dict[str, str],
    directory: Path,
) -> dict[str, dict[str, int]]:
    """The forms of the softmax loop each source compiles to without a
    device, each source another kernel."""
    compiled = {
        name: compile_offline(bound, args, source, directory / name)
        for name, source in sources.items()
    }
    assert len({kernel.ptx for kernel in compiled.values()}) == len(sources)
    if any(kernel.sass is None for kernel in compiled.values()):
        pytest.skip("the disassembler is not installed")
    return {
        name: softmax_loop_forms(cast("str", kernel.sass))
        for name, kernel in compiled.items()
    }


def _check_forms(
    loop: dict[str, int],
    plain: dict[str, int],
    waits: int,
    ceilings: dict[str, int],
) -> None:
    """What the tile loop of a kernel with the bind's bound shows beside
    the loop of the same kernel without it (``plain``, which marks
    nothing), whichever assembler compiled the two: no yield, fewer
    convergence regions, fewer try-waits inside one, and no more warp
    synchronizations or divergent branches. Each loop holds the
    recurrence's ``waits`` try-waits a tile.

    ``ceilings`` are the counts the assembler of CUDA 13.3 leaves the loop
    with (the README has them). They are that assembler's choices, so they
    bound from above only: a toolkit that leaves fewer passes."""
    for forms in (loop, plain):
        assert forms["TRYWAIT"] >= waits, forms
    assert loop["YIELD"] == 0, loop
    assert loop["BSSY"] < plain["BSSY"], (loop, plain)
    assert loop[LOOP_WAITS_IN_REGIONS] < plain[LOOP_WAITS_IN_REGIONS], (loop, plain)
    for form in ("WARPSYNC", "BRA.DIV"):
        assert loop[form] <= plain[form], (form, loop, plain)
    for form, count in ceilings.items():
        assert loop[form] <= count, (form, loop)


def test_the_compiled_loop_has_no_yield_and_no_region_around_its_waits(tmp_path):
    """Compiled without a device. The loop around the softmax's vote of a
    balanced CTA, whose trip count, phase and first step come from words of
    shared memory, has its two try-waits and neither a yield nor a
    convergence region around them; the same kernel without the bind's
    bound, whose loop marks nothing, has regions and waits inside them.
    Whole units and one unit per CTA compile to a loop without a yield or
    a region as well.

    The recurrence over reference scores under the ``early_tmem``
    probability schedule, which the kernels of the parity campaign run,
    votes twice a tile and has three try-waits: the QK and, on either side
    of the vote's branch, the previous PV. At this fixture's eight query
    columns its loop has no yield and no wait inside a region, and fewer
    regions than the kernel without the bound: at most the one the
    assembler of CUDA 13.3 keeps, around the group barrier of an event."""
    bound = stream_bound(WAVES, input_banks=2)
    args = partition_attention_args(
        units=WAVES, rows=8, window=PAGES * PAGE - 1, table_pages=PAGES, pool_pages=13
    )
    sources = {
        "balanced": stream_render(bound, stream_config(**BALANCED)),
        "whole": stream_render(bound, stream_config(**WHOLE)),
        "item": stream_render(bound, stream_config()),
    }
    for name, source in sources.items():
        # The bound is the stream's: every unit of a persistent CTA, one
        # unit's table otherwise.
        assert body_keywords(source)["stream_page_bound"] == str(
            PAGES if name == "item" else WAVES * PAGES
        ), name
    sources["unbounded"] = _unbounded(sources["balanced"])
    forms = _loops(bound, args, sources, tmp_path / "per_key")
    none = {"BSSY": 0, LOOP_WAITS_IN_REGIONS: 0}
    _check_forms(forms["balanced"], forms["unbounded"], 2, {**none, "BRA.DIV": 1})
    for name in ("whole", "item"):
        loop = forms[name]
        assert loop["TRYWAIT"] >= 2, (name, loop)
        assert (loop["YIELD"], loop["BSSY"]) == (0, 0), (name, loop)
    # Reference scores under the TMEM schedule: the bind that is offered it.
    reference = stream_bound(WAVES, input_banks=2, reference_scores=True)
    reference_args = partition_attention_args(
        units=WAVES,
        rows=8,
        reference_scores=True,
        window=PAGES * PAGE - 1,
        table_pages=PAGES,
        pool_pages=13,
    )
    marked = stream_render(reference, stream_config(**BALANCED, **EARLY_TMEM))
    assert body_keywords(marked)["probability_schedule"] == "'early_tmem'"
    assert body_keywords(marked)["stream_page_bound"] == str(WAVES * PAGES)
    forms = _loops(
        reference,
        reference_args,
        {"balanced": marked, "unbounded": _unbounded(marked)},
        tmp_path / "reference",
    )
    _check_forms(
        forms["balanced"],
        forms["unbounded"],
        3,
        {"BSSY": 1, LOOP_WAITS_IN_REGIONS: 0, "BRA.DIV": 1},
    )


def test_the_loop_at_32_columns_has_no_yield_and_fewer_regions_and_waits_in_them(
    tmp_path,
):
    """The form at a width the parity campaign measures, compiled without
    a device: the 32 query columns of its draft kernels, with the
    recurrence over reference scores under ``early_tmem``. Beside the same
    kernel without the bind's bound the loop has no yield, fewer
    convergence regions and fewer try-waits inside one.

    The vote's result is not marked, so the assembler does not take the
    code under the vote's branch for warp-uniform, and the one of CUDA
    13.3 keeps a region there that it drops at eight columns: around the
    previous-PV wait of a tile without an event (no row's maximum
    changed), one of the two the loop that marks nothing has around it.
    That region is allowed here, as a ceiling, and not asked for: it is
    the assembler's choice, and the README has its counts."""
    bound = stream_bound(WAVES, input_banks=2, rows=32, reference_scores=True)
    args = partition_attention_args(
        units=WAVES,
        rows=32,
        reference_scores=True,
        window=PAGES * PAGE - 1,
        table_pages=PAGES,
        pool_pages=13,
    )
    marked = stream_render(bound, stream_config(32, **BALANCED, **EARLY_TMEM))
    assert body_keywords(marked)["query_columns"] == "32"
    assert body_keywords(marked)["probability_schedule"] == "'early_tmem'"
    forms = _loops(
        bound, args, {"balanced": marked, "unbounded": _unbounded(marked)}, tmp_path
    )
    # At most the region of that wait and the one around the event's
    # group barrier.
    _check_forms(
        forms["balanced"],
        forms["unbounded"],
        3,
        {"BSSY": 2, LOOP_WAITS_IN_REGIONS: 1, "BRA.DIV": 1},
    )
