"""The shared memory ``smem_footprint`` bounds, and the per-block check (CPU).

``smem_footprint`` reads a rendered body the way the DSL traces it: an
allocation in a ``range_constexpr`` loop or a loop over a Python sequence is
made once per iteration, one in a dynamic loop once, one in either branch of
a dynamic ``if``, a ``const_expr`` branch only when it is taken.  Its lower
bound keeps only what the compiler cannot drop as dead code: a dynamic loop
whose range may have a trip, the lesser branch of a dynamic ``if`` (the root
dispatch of a multi-root launch keeps both), no ``while`` body.  The
shared-memory grouped reduces allocate one buffer per traced call, sized as
their dispatcher sizes it: the expected sizes below evaluate the helper
bodies' own ``alloc_smem`` arithmetic from ``reduce_helpers.py`` and the
routing comes from the dispatch predicates of ``reduce_call``.
``check_smem_budget`` makes a kernel past the per-block maximum
``BackendUnsupported`` at codegen instead of a libNVVM failure.
"""

from __future__ import annotations

import ast
import contextlib
from itertools import product
from pathlib import Path
import time

import pytest
import torch

import helion
from helion._compiler.cute import reduce_call
from helion._compiler.cute import target_facts
from helion._compiler.cute.pdl_release import spread_pad_bytes
from helion._compiler.cute.reduce_call import GROUPED_REDUCE_HELPERS
from helion._compiler.cute.reduce_call import use_interleaved_shared_reduce
from helion._compiler.cute.reduce_call import use_serial_block_reduce
from helion._compiler.cute.smem_footprint import RootDispatch
from helion._compiler.cute.smem_footprint import SmemFootprint
from helion._compiler.cute.smem_footprint import smem_footprint
from helion._testing import cpu_render_target
from helion._testing import patch_target_facts
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only

TWO_STAGE = "_cute_grouped_reduce_shared_two_stage"


def _footprint(
    source: str,
    runtime_scalars: frozenset[str] = frozenset(),
    dispatch: RootDispatch | None = None,
    **constexpr_values: int,
) -> SmemFootprint:
    """The bounds ``smem_footprint`` gives ``source`` (its padding aside)."""
    footprint = smem_footprint(
        ast.parse(source).body,
        constexpr_values,
        runtime_scalars=runtime_scalars,
        dispatch=dispatch,
    )
    return SmemFootprint(footprint.low, footprint.high)


# 256 fp32 values: 1024 bytes, whose pointer the body reads.
_ALLOC = "p = cute.arch.alloc_smem(cutlass.Float32, 256)\nuse(p)\n"


def _loop(header: str, body: str = _ALLOC) -> str:
    return f"{header}:\n" + "".join(f"    {line}\n" for line in body.splitlines())


def test_unrolled_loops_allocate_per_copy() -> None:
    assert _footprint(_ALLOC) == SmemFootprint(1024, 1024)
    # The DSL unrolls these at trace time: one allocation per iteration.
    for header, copies in (
        ("for v in cutlass.range_constexpr(16)", 16),
        ("for v in cutlass.range_constexpr(2, 10, 2)", 4),
        ("for v in cutlass.range_constexpr(_V // 2)", 4),
        ("for v in (a, b, c)", 3),
    ):
        expected = SmemFootprint(1024 * copies, 1024 * copies)
        assert _footprint(_loop(header), _V=8) == expected, header
    nested = _loop("for v in cutlass.range_constexpr(4)", _loop("for w in (a, b)"))
    assert _footprint(nested) == SmemFootprint(8192, 8192)
    # A trip count the walk cannot read bounds nothing but zero from below.
    for header in ("for v in cutlass.range_constexpr(n)", "for v in values"):
        assert _footprint(_loop(header)) == SmemFootprint(0, None), header
    assert _footprint(_loop("for v in values", "x = v")) == SmemFootprint(0, 0)
    # An unrolled loop keeps Python's early exits: its trips bound the
    # allocations from above only (an inner loop's own exits do not count).
    for body in (
        _ALLOC + "break",
        "if cutlass.const_expr(v == 3):\n    continue\n" + _ALLOC,
    ):
        unrolled = _loop("for v in cutlass.range_constexpr(16)", body)
        assert _footprint(unrolled) == SmemFootprint(0, 16384), body
    inner = _loop("for w in cutlass.range_constexpr(2)", "break") + _ALLOC
    unrolled = _loop("for v in cutlass.range_constexpr(16)", inner)
    assert _footprint(unrolled) == SmemFootprint(16384, 16384)


def test_a_dynamic_loop_is_certain_only_when_its_range_may_have_a_trip() -> None:
    # The DSL traces a dynamic loop's body once, and the compiler keeps it
    # unless it proves the range empty, which it cannot when static bounds
    # have a trip or when bounds it cannot know (the kernel's runtime scalar
    # arguments: the sizes of a kernel without static shapes, each read
    # once) can make one.  ``unroll`` and the other hints of
    # ``cutlass.range`` leave the trip count alone.
    scalars = frozenset({"r", "n_rows"})
    for header in (
        "for i in range(8)",
        "for i in cutlass.range(8, unroll_full=True)",
        "for i in cutlass.range(0, 64, 16, unroll=1, prefetch_stages=2)",
        "for i in range(cutlass.Int32(0), cutlass.Int32(64), cutlass.Int32(_B))",
        "for i in range(cutlass.Int32(0), cutlass.Int32(cutlass.Int32(r)), _B)",
        "for i in range(r, n_rows)",
    ):
        footprint = _footprint(_loop(header), scalars, _B=16)
        assert footprint == SmemFootprint(1024, 1024), header
    # Elsewhere its body counts in ``high`` only: an empty static range (an
    # empty dimension), bounds the compiler may fold (a block index, a load,
    # a local: ``tile_offset_0 * 0``, one value twice), a keyword the walk
    # does not know, and every ``while`` (its test may be false on entry).
    for header in (
        "for i in range(0)",
        "for i in range(cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(_B))",
        "for i in cutlass.range(_B, 0)",
        "for i in cutlass.range(0, 0, 16, unroll=2)",
        "for i in range(n)",
        "for i in range(tile_offset_0, tile_offset_0, _B)",
        "for i in range(cutlass.Int32(cute.arch.block_idx()[0]), total, _B)",
        "for i in range(r, r)",
        "for i in cutlass.range(8, trips=8)",
        "while i < n",
        "while i < r",
    ):
        footprint = _footprint(_loop(header), scalars, _B=16)
        assert footprint == SmemFootprint(0, 1024), header


def test_a_dynamic_loop_is_certain_only_where_the_enclosing_code_allows() -> None:
    # Inside a loop the compiler knows its range has a trip: within
    # ``range(0, r)``, ``r >= 1``, so ``range(r, 0)`` never runs there and
    # the compiler drops it, though each range alone may have a trip.  A
    # scalar two loops' bounds relate, or one under a runtime test (which
    # may tell the compiler anything), makes no loop certain; a sibling loop
    # proves nothing.
    scalars = frozenset({"r", "n_rows"})
    certain, possible = SmemFootprint(1024, 1024), SmemFootprint(0, 1024)
    for outer, inner, expected in (
        ("for i in range(0, r, 16)", "for j in range(r, 0, 16)", possible),
        ("for i in range(0, r)", "for j in range(0, r)", certain),
        ("for i in range(r, 64)", "for j in range(64, r, -16)", certain),
        ("for i in range(r, 64)", "for j in range(64, r)", possible),
        ("for i in range(0, r)", "for j in range(r, n_rows)", certain),
        ("for i in range(r, n_rows)", "for j in range(0, r)", possible),
        ("for i in range(0, 64)", "for j in range(r, 0)", certain),
    ):
        nested = _loop(outer, _loop(inner))
        assert _footprint(nested, scalars) == expected, (outer, inner)
    sibling = _loop("for i in range(0, r)", "x = i") + _loop("for j in range(r, 0)")
    assert _footprint(sibling, scalars) == certain
    loop = "".join(
        f"    {line}\n" for line in _loop("for i in range(0, r)").splitlines()
    )
    guarded = f"if flag:\n{loop}else:\n{loop}"
    assert _footprint(guarded, scalars) == SmemFootprint(0, 2048)


def test_an_allocation_nothing_reads_is_not_certain() -> None:
    # The DSL drops an allocation whose pointer nothing reads (the spread pad
    # keeps its own live through a helper): one bound to a name the body
    # never loads, or one whose pointer the statement drops.
    unread = "p = cute.arch.alloc_smem(cutlass.Float32, 256)\n"
    assert _footprint(unread) == SmemFootprint(0, 1024)
    assert _footprint("cute.arch.alloc_smem(cutlass.Float32, 256)\n") == (
        SmemFootprint(0, 1024)
    )
    # A read in code the lower bound counts keeps it: a copy of an unrolled
    # loop with a trip, not a loop over a sequence that may be empty.
    assert _footprint(
        unread + _loop("for v in cutlass.range_constexpr(2)", "use(p)")
    ) == (SmemFootprint(1024, 1024))
    assert _footprint(unread + _loop("for v in values", "use(p)")) == (
        SmemFootprint(0, 1024)
    )


def test_an_allocation_read_only_through_dead_views_is_not_certain() -> None:
    # The pointer is used only once a view of it is read (``cute.make_tensor``,
    # ``cute.recast_ptr``, an offset): an unread view, or one read only in code
    # the compiler may drop, leaves the allocation unread, and the DSL drops
    # it (70000 fp32 values: past the 232448-byte maximum when counted).
    alloc = "p = cute.arch.alloc_smem(cutlass.Float32, 70000)\n"
    view = "t = cute.make_tensor(p, cute.make_layout(70000))\n"
    assert _footprint(alloc + view) == SmemFootprint(0, 280000)
    assert _footprint(
        alloc + view + _loop("if cutlass.const_expr(False)", "use(t)")
    ) == (SmemFootprint(0, 280000))
    assert _footprint(alloc + view + _loop("if flag", "use(t)")) == (
        SmemFootprint(0, 280000)
    )
    # Read through a chain of views, or in both branches of a runtime test.
    chain = "q = cute.recast_ptr(p, dtype=cutlass.Int32)\nt = cute.make_tensor(q + 4, layout)\n"
    assert _footprint(alloc + chain + "use(t[0])\n") == SmemFootprint(280000, 280000)
    both = _loop("if flag", "use(t)") + _loop("else", "keep(t)")
    assert _footprint(alloc + view + both) == SmemFootprint(280000, 280000)


def test_stores_and_dead_loads_keep_no_allocation() -> None:
    # The compiler drops the stores to storage no kept code loads, and an
    # assignment to names nothing reads with its loads, and then the
    # allocation.  Offline for sm_100a, the first bodies compile with their
    # 70000 fp32 values, past the 232448-byte maximum; the others fail.
    alloc = (
        "p = cute.arch.alloc_smem(cutlass.Float32, 70000)\n"
        "shared = cute.make_tensor(p, cute.make_layout(70000))\n"
        "shared[tid] = x[0]\n"
        "cute.arch.sync_threads()\n"
    )
    for dead in (
        "",
        "unused = shared[0]\n",
        "first, second = shared[0], shared[1]\n",
        "acc = cutlass.Float32(0.0)\nacc += shared[0]\n",
    ):
        assert _footprint(alloc + dead) == SmemFootprint(0, 280000), dead
    for read in (
        "used = shared[0]\nout[1] = used\n",
        "acc = cutlass.Float32(0.0)\nacc += shared[0]\nout[1] = acc\n",
        "out[1] = shared[0]\n",
        "shared[1] = shared[0]\n",
        "shared[1] += 1.0\n",
    ):
        assert _footprint(alloc + read) == SmemFootprint(280000, 280000), read


def test_method_stores_and_metadata_reads_keep_no_allocation() -> None:
    # A store through a view's ``store`` method keeps nothing, as a subscript
    # store does, and reading a view's size or shape reads no storage.
    # Offline for sm_100a, the first two bodies compile with their 70000 fp32
    # values, past the 232448-byte maximum.
    alloc = "p = cute.arch.alloc_smem(cutlass.Float32, 70000)\n"
    view = "shared = cute.make_tensor(p, cute.make_layout(70000))\nshared[tid] = x[0]\n"
    for unread in (
        alloc + "(p + tid).store(x[0])\ncute.arch.sync_threads()\n",
        alloc + view + "out[0] = cute.size(shared)\n",
        alloc + view + "out[0] = shared.shape[0]\n",
    ):
        assert _footprint(unread) == SmemFootprint(0, 280000), unread
    # A load through a view, also one a store stores, keeps it.
    for read in (
        alloc + "out[0] = (p + tid).load()\n",
        alloc + "(p + tid).store((p + 1).load())\n",
        alloc + view + "out[0] = cute.size(shared) + shared[1]\n",
    ):
        assert _footprint(read) == SmemFootprint(280000, 280000), read


def test_an_allocation_bound_another_way_or_after_a_return_is_not_certain() -> None:
    # An allocation counts in ``low`` bound to a name, directly or through
    # views, in code the walk can show traced: not as another call's argument
    # (the call may drop it), a list's element or an annotated assignment's
    # value, nor after a statement that may return.  A store through a view's
    # iterator reads nothing: offline for sm_100a, that body compiles with its
    # 70000 fp32 values.
    alloc = "cute.arch.alloc_smem(cutlass.Float32, 70000)"
    for body in (
        f"x = f({alloc})\nuse(x)\n",
        f"buffers = [{alloc}]\nuse(buffers)\n",
        f"p: cute.Pointer = {alloc}\nuse(p)\n",
        f"if cutlass.const_expr(flag):\n    return\np = {alloc}\nuse(p)\n",
        (
            f"p = {alloc}\nshared = cute.make_tensor(p, layout)\n"
            "shared.iterator[tid] = x[0]\n"
        ),
    ):
        assert _footprint(body) == SmemFootprint(0, 280000), body
    assert _footprint(f"p = {alloc}\nuse(p)\n") == SmemFootprint(280000, 280000)
    assert _footprint(
        f"p = {alloc}\nshared = cute.make_tensor(p, layout)\n"
        "out[0] = shared.iterator[tid]\n"
    ) == SmemFootprint(280000, 280000)


def test_a_read_counts_for_the_binding_it_sees() -> None:
    # Two 160000-byte buffers, of which the code reads one: counted both,
    # they would pass the 232448-byte maximum.
    buffers = (
        "a = cute.arch.alloc_smem(cutlass.Uint8, 160000)\n"
        "b = cute.arch.alloc_smem(cutlass.Uint8, 160000)\n"
    )
    # The branches of a foldable test view different buffers: the kernel
    # keeps the one its remaining branch reads, so neither is certain.
    for header in ("if cutlass.const_expr(flag)", "if flag"):
        chosen = _loop(header, "t = cute.make_tensor(a, layout)") + _loop(
            "else", "t = cute.make_tensor(b, layout)"
        )
        assert _footprint(buffers + chosen + "use(t)\n") == SmemFootprint(0, 320000)
    # A view bound again before the read: the read sees the second binding.
    rebound = (
        "t = cute.make_tensor(a, layout)\nt = cute.make_tensor(b, layout)\nuse(t)\n"
    )
    assert _footprint(buffers + rebound) == SmemFootprint(160000, 320000)
    # A read ahead of the allocation reads an older binding.
    unread_yet = "use(p)\np = cute.arch.alloc_smem(cutlass.Float32, 256)\n"
    assert _footprint(unread_yet) == SmemFootprint(0, 1024)


def test_unrolled_copies_count_where_their_own_trip_reads_them() -> None:
    # 16 copies of a 16384-byte buffer.  Each copy a trip reads itself
    # survives; a read after the loop sees the last copy only, which the
    # lower bound leaves out rather than count all 16.
    alloc = "p = cute.arch.alloc_smem(cutlass.Float32, 4096)"
    unrolled = "for v in cutlass.range_constexpr(16)"
    assert _footprint(_loop(unrolled, f"{alloc}\nuse(p)")) == SmemFootprint(
        262144, 262144
    )
    assert _footprint(_loop(unrolled, alloc) + "use(p)\n") == SmemFootprint(0, 262144)
    # One copy, of one unrolled trip or a dynamic loop's traced body, is the
    # one a read after the loop sees.
    for header in ("for v in cutlass.range_constexpr(1)", "for v in range(4)"):
        assert _footprint(_loop(header, alloc) + "use(p)\n") == SmemFootprint(
            16384, 16384
        ), header


def test_a_straight_line_body_is_walked_in_linear_time() -> None:
    # The walk visits each statement of a straight-line body once and keeps
    # one set of the names read after it, so its time is linear in the
    # statements (each ``if`` and loop copies that set).
    body = ast.parse(
        "\n".join(
            ["a_0 = b_0", *(f"a_{i} = a_{i - 1} + b_{i}" for i in range(1, 10000))]
        )
    ).body
    started = time.perf_counter()
    assert smem_footprint(body) == SmemFootprint()
    assert time.perf_counter() - started < 10


def _branches(header: str, prelude: str = "", orelse_prelude: str = "") -> str:
    return (
        f"{prelude}{header}:\n"
        "    p = cute.arch.alloc_smem(cutlass.Float32, 256)\n"
        "    use(p)\n"
        "else:\n"
        f"{orelse_prelude}"
        "    q = cute.arch.alloc_smem(cutlass.Float32, 512)\n"
        "    use(q)\n"
    )


def test_only_the_root_dispatch_keeps_both_branches_of_a_runtime_test() -> None:
    # The DSL traces both branches of a test it cannot evaluate, and the
    # compiler drops the branch a test it folds never takes (``tile_offset_0
    # < 0``: a block index times the block size is never negative), so only
    # the lesser branch is certain, whatever the test reads; a
    # ``const_expr`` test traces one branch.
    either = SmemFootprint(1024, 3072)
    for prelude, header in (
        ("tid = cutlass.Int32(cute.arch.thread_idx()[0])\n", "if tid == 0"),
        ("v = (x.iterator + 1).load()\n", "if v > 0"),
        ("lt = tile_offset_0 < 0\n", "if lt"),
        ("", "if flag"),
    ):
        assert _footprint(_branches(header, prelude)) == either, header
    static = _branches("if cutlass.const_expr(flag)")
    assert _footprint(static) == SmemFootprint(1024, 2048)
    # A multi-root launch dispatches its roots on the block index; when
    # every root gets blocks (codegen hands the walk the accumulated block
    # counts, ``root_dispatch``) each branch runs on some block, and the
    # kernel keeps both.
    dispatch = _branches(
        "if pid_shared < 4",
        "pid_shared = cutlass.Int32(cute.arch.block_idx()[0])\n",
        "    pid_shared = pid_shared - cutlass.Int32(4)\n",
    )
    roots = RootDispatch("pid_shared", frozenset({4}))
    assert _footprint(dispatch, dispatch=roots) == SmemFootprint(3072, 3072)
    nested = "if pid_shared < 12:\n" + "".join(
        f"    {line}\n" for line in dispatch.splitlines()[1:]
    )
    three = RootDispatch("pid_shared", frozenset({4, 12}))
    assert _footprint(nested, dispatch=three) == SmemFootprint(3072, 3072)
    # Without the dispatch, or for a test that is not one of its thresholds
    # (an empty first root's ``pid_shared < 0`` leaves its branch dead), a
    # test on the block index is any other runtime test.
    assert _footprint(dispatch) == either
    empty_first = dispatch.replace("pid_shared < 4", "pid_shared < 0")
    assert _footprint(empty_first, dispatch=roots) == either


def test_storage_the_walk_cannot_size_leaves_the_high_bound_open() -> None:
    # The spread pad is sized against the high bound: a body that reserves
    # shared memory through the DSL's ``SmemAllocator`` (the fa4 flash
    # families, the tcgen05 GEMMs: a struct carved out at trace time, already
    # at the per-block maximum), the dynamic shared memory base, or a size
    # that is not a literal has none and gets no pad at all: a pad sized
    # against storage of unknown size would fail every launch past the
    # maximum.
    assert _footprint("x = 1") == SmemFootprint(0, 0)
    literal = (
        "a = cute.arch.alloc_smem(cutlass.Uint8, 25600, alignment=128)\n"
        "b = cute.arch.alloc_smem(cutlass.Float32, 512, alignment=16)\n"
        "c = cute.arch.alloc_smem(cutlass.BFloat16, 8192)\n"
        "d = cute.arch.alloc_smem(cutlass.Int64, cutlass.Int32(2))\n"
        "use(a, b, c, d)\n"
    )
    nbytes = 25600 + 512 * 4 + 8192 * 2 + 2 * 8
    assert _footprint(literal) == SmemFootprint(nbytes, nbytes)
    for opaque in (
        (
            "smem = cutlass_utils_flash.SmemAllocator()\n"
            "storage = smem.allocate(_fbwd_storage_cls)\n"
        ),
        "alloc = cutlass.utils.SmemAllocator()\n",
        "t = smem.allocate_tensor(cutlass.Float32, layout, 128)\n",
        "p = smem.allocate_array(cutlass.Int64, 4)\n",
        "base = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=128)\n",
        "a = cute.arch.alloc_smem(cutlass.Uint8, size, alignment=128)\n",
        "a = cute.arch.alloc_smem(dtype, 1024)\n",
    ):
        assert _footprint(opaque).high is None, opaque


WARP = "_cute_grouped_reduce_warp"
TREE = "_cute_grouped_reduce_shared_tree"
CLUSTER = "_cute_grouped_reduce_cluster"
BLOCK = "_cute_grouped_reduce_block"
COLUMNS = "_cute_grouped_reduce_shared_columns"
FRAGMENT = "_cute_grouped_reduce_shared_two_stage_fragment"
PAIR = "_cute_grouped_reduce_cluster_online_pair"
WARP_PAIR = "_cute_grouped_reduce_cluster_warp_online_pair"

_HELPERS = {
    node.name: node
    for node in ast.parse(
        Path(reduce_call.__file__).with_name("reduce_helpers.py").read_text()
    ).body
    if isinstance(node, ast.FunctionDef)
}


def _helper_buffer(function: str, index: int = 0, **values: int) -> int:
    """The elements of the ``index``-th ``alloc_smem`` (in source order) of
    ``reduce_helpers.<function>``: its size expression evaluated over
    ``values`` and the integer assignments ahead of it (the helper's own
    arithmetic, read as text so the DSL is not imported, with the
    ``reduce_call`` sizes it calls)."""
    namespace: dict[str, object] = {
        "__builtins__": {},
        "cooperative_column_slots": reduce_call.cooperative_column_slots,
        **values,
    }
    pending: list[ast.stmt] = list(_HELPERS[function].body)
    seen = 0
    while pending:
        statement = pending.pop(0)
        if isinstance(statement, (ast.If, ast.For, ast.While)):
            pending[:0] = [*statement.body, *statement.orelse]
            continue
        for node in ast.walk(statement):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "alloc_smem"
            ):
                if seen == index:
                    expression = ast.Expression(node.args[1])
                    return eval(compile(expression, function, "eval"), namespace)
                seen += 1
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            # Lines over traced values (``type(identity)``, a lane index) do
            # not evaluate here and are not sizes.
            with contextlib.suppress(NameError):
                expression = ast.Expression(statement.value)
                value = eval(compile(expression, function, "eval"), namespace)
                namespace[statement.targets[0].id] = value
    raise AssertionError(f"{function} has no alloc_smem number {index}")


_SERIAL = "_cute_grouped_reduce_shared_serial_body"
_TWO_STAGE_BODY = "_cute_grouped_reduce_shared_two_stage_body"
_INTERLEAVED_WARP = "_cute_grouped_reduce_shared_interleaved_warp_body"
_INTERLEAVED_BLOCK = "_cute_grouped_reduce_shared_interleaved_block_body"

# A call of every grouped-reduce dispatcher (behind ``acc = [a, b, c]``, the
# column reduce's three accumulators) and the bytes its buffer takes: the
# element width times the elements the helper body it routes to allocates.
_REDUCE_BUFFERS = {
    # The serial form: one slot per warp.
    f"{TWO_STAGE}(v, 'sum', cutlass.Float32(0), l, g, m, "
    "pre=1, group_span=256, group_count=1)": 4
    * _helper_buffer(_SERIAL, group_span=256),
    # The interleaved block form: every thread's value plus the results.
    f"{TWO_STAGE}(v, 'max', cutlass.Float32(float('-inf')), l, g, m, "
    "pre=256, group_span=1024, group_count=1)": 4
    * _helper_buffer(_INTERLEAVED_BLOCK, pre=256, group_span=1024, group_count=1),
    # The interleaved warp form and the two-stage form: every sibling's
    # per-warp partials plus the results.
    f"{TWO_STAGE}(v, 'sum', cutlass.Float32(0), l, g, m, "
    "pre=4, group_span=128, group_count=2)": 4
    * _helper_buffer(_INTERLEAVED_WARP, pre=4, group_span=128, group_count=2),
    f"{TWO_STAGE}(v, 'sum', cutlass.Int64(0), l, g, m, "
    "pre=3, group_span=96, group_count=1)": 8
    * _helper_buffer(_TWO_STAGE_BODY, pre=3, group_span=96, group_count=1),
    f"{TREE}(v, 'prod', cutlass.Float32(1.0), l, g, m, "
    "pre=3, group_span=48, num_threads=96, group_count=2)": 4
    * _helper_buffer(
        "_cute_grouped_reduce_shared_tree_body", pre=3, num_threads=96, group_count=2
    ),
    # A whole-CTA group of 16 warps: the two-stage form.
    f"{BLOCK}(v, 'sum', cutlass.Float32(0), lane, group_span=512)": 4
    * _helper_buffer(_TWO_STAGE_BODY, pre=1, group_span=512, group_count=1),
    f"{COLUMNS}(acc, 'sum', cutlass.Float32(0), l, g, "
    "pre=1, group_span=256, group_count=1)": 4
    * _helper_buffer(COLUMNS, 0, width=3, group_span=256, group_count=1),
    f"{COLUMNS}(acc, 'sum', cutlass.Float32(0), l, g, "
    "pre=4, group_span=128, group_count=1, fold='warp')": 4
    * _helper_buffer(
        "_cute_warp_fold_column_sums", width=3, pre=4, group_span=128, group_count=1
    ),
    # The cooperative fold: the staged partials plus an output slot per lane
    # block and pass of every group.
    f"{COLUMNS}(acc, 'sum', cutlass.Float32(0), l, g, "
    "pre=4, group_span=128, group_count=2, fold='cooperative')": 4
    * _helper_buffer(
        "_cute_cooperative_column_sums", width=3, pre=4, group_span=128, group_count=2
    ),
    f"{COLUMNS}(acc, 'sum', cutlass.Float32(0), l, g, "
    "pre=4, group_span=128, group_count=1)": 4
    * _helper_buffer(COLUMNS, 1, width=3, pre=4, group_span=128, group_count=1),
    f"{FRAGMENT}(frag, out, 'sum', cutlass.Float32(0), l, g, m, "
    "count=4, pre=2, group_span=64, group_count=1)": 4
    * _helper_buffer(
        "_cute_grouped_reduce_shared_two_stage_fragment_body",
        count=4,
        pre=2,
        group_span=64,
        group_count=1,
    ),
    # The block reduce of the per-thread sum ahead of the exchange (serial,
    # four warps); the cluster exchanges receive into buffers the body
    # allocates.
    f"{PAIR}(s, mx, lane, buf, mbar, group_span=128, cluster_n=2, scale=1.0)": 4
    * _helper_buffer(_SERIAL, group_span=128),
    f"{WARP_PAIR}(s, mx, lane, buf, mbar, group_span=128, cluster_n=2, scale=1.0)": 0,
    f"{CLUSTER}(v, 'sum', cutlass.Float32(0.0), lane, buf, mbar, "
    "group_span=256, cluster_n=4)": 0,
    f"{WARP}(v, 'sum', cutlass.Float32(0), lane, pre=2, group_span=16)": 0,
}


def test_every_grouped_reduce_buffer_is_sized_like_its_helper() -> None:
    assert {call.split("(")[0] for call in _REDUCE_BUFFERS} == GROUPED_REDUCE_HELPERS
    for call, nbytes in _REDUCE_BUFFERS.items():
        footprint = _footprint(f"acc = [a, b, c]\nr = {call}\nuse(r)\n")
        assert footprint == SmemFootprint(nbytes, nbytes), call
        # A reduce whose result nothing reads keeps its buffer in the forms
        # measured to keep it (``_UNREAD_KEPT_SCOPES``): offline for sm_100a,
        # an unread column reduce over 64 fp32 values (262144 bytes)
        # compiles, where 64 unread two-stage reduces fail in libNVVM.
        kept = call.startswith((f"{TWO_STAGE}(", f"{TREE}(", f"{FRAGMENT}("))
        for unread in (f"r = {call}\n", f"{call}\n"):
            footprint = _footprint(f"acc = [a, b, c]\n{unread}")
            assert footprint == SmemFootprint(nbytes if kept else 0, nbytes), unread
    # A column reduce over a list whose length is unknown or changes.
    columns = next(call for call in _REDUCE_BUFFERS if call.startswith(COLUMNS))
    unknown = SmemFootprint(0, None)
    assert _footprint(f"r = {columns}\n") == unknown
    assert _footprint(f"acc = [a, b, c]\nacc.append(d)\nr = {columns}\n") == unknown
    # Inside an unrolled lane loop every copy takes its own buffer.
    two_stage = next(iter(_REDUCE_BUFFERS))
    unrolled = _loop("for v in cutlass.range_constexpr(16)", f"r = {two_stage}\nuse(r)")
    assert _footprint(unrolled) == SmemFootprint(16 * 32, 16 * 32)


def test_the_sized_dispatchers_are_the_ones_that_allocate() -> None:
    # Read as text, so the DSL is not imported: a dispatcher allocates when
    # it reaches an ``alloc_smem`` through the module's functions (by name:
    # the interleaved dispatcher picks its body into a variable first).
    source = Path(reduce_call.__file__).with_name("reduce_helpers.py").read_text()
    functions = {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }
    allocating = {
        name
        for name, function in functions.items()
        if any(
            isinstance(node, ast.Attribute) and node.attr == "alloc_smem"
            for node in ast.walk(function)
        )
    }
    references = {
        name: {
            node.id
            for node in ast.walk(function)
            if isinstance(node, ast.Name) and node.id in functions
        }
        for name, function in functions.items()
    }
    while (
        grown := {name for name, names in references.items() if names & allocating}
        - allocating
    ):
        allocating |= grown
    sized = {call.split("(")[0] for call, nbytes in _REDUCE_BUFFERS.items() if nbytes}
    assert sized == allocating & GROUPED_REDUCE_HELPERS


def test_the_two_stage_buffer_follows_the_dispatcher_routing() -> None:
    # The dispatcher routes on the predicates of ``reduce_call`` (read as
    # text: it calls them by name), and so does the walk.
    source = Path(reduce_call.__file__).with_name("reduce_helpers.py").read_text()
    dispatcher = ast.unparse(_HELPERS[TWO_STAGE])
    assert "use_serial_block_reduce(pre, group_span, group_count)" in dispatcher
    assert "use_interleaved_shared_reduce(pre, group_span)" in dispatcher
    assert "from .reduce_call import use_serial_block_reduce" in source
    assert "from .reduce_call import use_interleaved_shared_reduce" in source
    # Every routing outcome: whole-CTA and grouped, power-of-two and odd
    # sibling counts below and above a warp, spans up to a full CTA.
    for pre, span, count in product(
        (1, 2, 3, 4, 16, 32, 48, 64, 256), (32, 64, 96, 256, 512, 1024), (1, 2)
    ):
        if use_serial_block_reduce(pre, span, count):
            body = _SERIAL
        elif use_interleaved_shared_reduce(pre, span):
            body = _INTERLEAVED_WARP if pre < 32 else _INTERLEAVED_BLOCK
        else:
            body = _TWO_STAGE_BODY
        elements = _helper_buffer(body, pre=pre, group_span=span, group_count=count)
        call = (
            f"{TWO_STAGE}(v, 'sum', cutlass.Float32(0), l, g, m, "
            f"pre={pre}, group_span={span}, group_count={count})"
        )
        assert _footprint(f"r = {call}\nuse(r)\n").low == 4 * elements, (
            pre,
            span,
            count,
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def chunked_sum(x: torch.Tensor) -> torch.Tensor:
    s, r, c = x.shape
    out = torch.empty([s, c], dtype=torch.float32, device=x.device)
    for tile_s, tile_c in hl.tile([s, c]):
        acc = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        for tile_r in hl.tile(r):
            acc = acc + torch.sum(x[tile_s, tile_r, tile_c].to(torch.float32), dim=1)
        out[tile_s, tile_c] = acc
    return out


# Without static shapes the reduce loop runs to the size argument ``r``.
dynamic_chunked_sum = helion.kernel(
    backend="cute", static_shapes=False, autotune_effort="none"
)(chunked_sum.fn)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def two_chunked_sums(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``chunked_sum`` of ``x`` and of ``y``: two roots the launch dispatches
    on the block index, so the DSL traces both."""
    s, r, c = x.shape
    s_y, r_y, c_y = y.shape
    out_x = torch.empty([s, c], dtype=torch.float32, device=x.device)
    out_y = torch.empty([s_y, c_y], dtype=torch.float32, device=y.device)
    for tile_s, tile_c in hl.tile([s, c]):
        acc = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        for tile_r in hl.tile(r):
            acc = acc + torch.sum(x[tile_s, tile_r, tile_c].to(torch.float32), dim=1)
        out_x[tile_s, tile_c] = acc
    for tile_s, tile_c in hl.tile([s_y, c_y]):
        acc = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        for tile_r in hl.tile(r_y):
            acc = acc + torch.sum(y[tile_s, tile_r, tile_c].to(torch.float32), dim=1)
        out_y[tile_s, tile_c] = acc
    return out_x, out_y


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def guarded_chunked_sum(x: torch.Tensor) -> torch.Tensor:
    """``chunked_sum`` with its reduce behind a test the compiler proves false:
    a tile's offset, a block index times the block size, is never negative."""
    s, r, c = x.shape
    out = torch.empty([s, c], dtype=torch.float32, device=x.device)
    for tile_s, tile_c in hl.tile([s, c]):
        acc = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        for tile_r in hl.tile(r):
            if tile_s.begin < 0:
                acc = acc + torch.sum(
                    x[tile_s, tile_r, tile_c].to(torch.float32), dim=1
                )
        out[tile_s, tile_c] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def stepped_chunked_sum(x: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    """``chunked_sum`` in a ``while`` whose test is false on entry: its
    limit is a load times zero."""
    s, r, c = x.shape
    out = torch.empty([s, c], dtype=torch.float32, device=x.device)
    for tile_s, tile_c in hl.tile([s, c]):
        acc = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        step = hl.full([], 0, dtype=torch.int32)
        limit = steps[0] * 0
        while step < limit:
            for tile_r in hl.tile(r):
                acc = acc + torch.sum(
                    x[tile_s, tile_r, tile_c].to(torch.float32), dim=1
                )
            step = step + 1
        out[tile_s, tile_c] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def offset_chunked_sum(x: torch.Tensor) -> torch.Tensor:
    """``chunked_sum`` over ``hl.tile(tile_s.begin, tile_s.begin)``: a dynamic
    loop between two runtime bounds that are one value, an empty range."""
    s, r, c = x.shape
    out = torch.empty([s, c], dtype=torch.float32, device=x.device)
    for tile_s, tile_c in hl.tile([s, c]):
        acc = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        for tile_r in hl.tile(tile_s.begin, tile_s.begin):
            acc = acc + torch.sum(x[tile_s, tile_r, tile_c].to(torch.float32), dim=1)
        out[tile_s, tile_c] = acc
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def inner_chunked_sum(x: torch.Tensor) -> torch.Tensor:
    """``chunked_sum`` reducing over ``hl.tile(r, 0)`` inside ``hl.tile(r)``:
    the inner range is empty wherever the outer one runs."""
    s, r, c = x.shape
    out = torch.empty([s, c], dtype=torch.float32, device=x.device)
    for tile_s, tile_c in hl.tile([s, c]):
        acc = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        for _tile_r in hl.tile(r):
            for tile_q in hl.tile(r, 0):
                acc = acc + torch.sum(
                    x[tile_s, tile_q, tile_c].to(torch.float32), dim=1
                )
        out[tile_s, tile_c] = acc
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def live_inner_chunked_sum(x: torch.Tensor) -> torch.Tensor:
    """``inner_chunked_sum`` over ``hl.tile(0, r)``, which runs."""
    s, r, c = x.shape
    out = torch.empty([s, c], dtype=torch.float32, device=x.device)
    for tile_s, tile_c in hl.tile([s, c]):
        acc = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        for _tile_r in hl.tile(r):
            for tile_q in hl.tile(0, r):
                acc = acc + torch.sum(
                    x[tile_s, tile_q, tile_c].to(torch.float32), dim=1
                )
        out[tile_s, tile_c] = acc
    return out


def _config(vector: int, roots: int = 1) -> helion.Config:
    """``chunked_sum``'s config at cute_vector_widths [16, ``vector``, 1],
    each per-block list repeated for every root."""
    return helion.Config(
        block_sizes=[128, 64, 16] * roots,
        num_threads=[8, 8, 16] * roots,
        cute_vector_widths=[16, vector, 1] * roots,
    )


def test_a_kernel_past_the_per_block_maximum_is_rejected_at_codegen() -> None:
    # The 16 threads along each chunk's 64 rows combine them with an
    # interleaved shared-memory reduce (pre=64, group_span=1024: 4352
    # bytes), traced once per unrolled lane of cute_vector_widths
    # [16, V, 1], 16 * V copies.
    x = torch.empty(256, 512, 128, dtype=torch.bfloat16)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(chunked_sum, (x,))
        # 32 copies, 139264 bytes: within the modelled B200's 232448.
        code = bound.to_code(_config(2))
        assert code.count(f"{TWO_STAGE}(") == 1
        assert "cutlass.range_constexpr(16)" in code
        # 64 copies, 278528 bytes: libNVVM fails on this kernel.
        with pytest.raises(
            helion.exc.BackendUnsupported,
            match="278528 bytes of shared memory, over the 232448-byte",
        ):
            bound.to_code(_config(4))
        # Over an empty dimension every copy sits in a dynamic loop over an
        # empty range, which the DSL folds away with its buffer: no
        # rejection.
        empty = cpu_bind(chunked_sum, (torch.empty(256, 0, 128, dtype=x.dtype),))
        code = empty.to_code(_config(4))
        assert "range(cutlass.Int32(0), cutlass.Int32(0), " in code
        assert code.count(f"{TWO_STAGE}(") == 1
        # Without static shapes the loop runs to the size argument ``r``,
        # which the compiler cannot know: the 64 copies stay.
        dynamic = cpu_bind(dynamic_chunked_sum, (x,))
        with pytest.raises(
            helion.exc.BackendUnsupported, match="278528 bytes of shared memory"
        ):
            dynamic.to_code(_config(4))
        # Two such roots hold both roots' copies: 2 x 16 of them fit, 2 x 32
        # do not (libNVVM fails on that kernel too).
        two_roots = cpu_bind(two_chunked_sums, (x, x.clone()))
        assert two_roots.to_code(_config(1, roots=2)).count(f"{TWO_STAGE}(") == 2
        with pytest.raises(
            helion.exc.BackendUnsupported, match="278528 bytes of shared memory"
        ):
            two_roots.to_code(_config(2, roots=2))
        # The compiler drops the branch of a root without blocks (an empty
        # dimension of the first or the second root; both kernels compile):
        # only the other root's 32 copies are certain.
        for shapes in (
            ((0, 512, 128), (256, 512, 128)),
            ((256, 512, 128), (0, 512, 128)),
        ):
            args = tuple(torch.empty(*shape, dtype=x.dtype) for shape in shapes)
            code = cpu_bind(two_chunked_sums, args).to_code(_config(2, roots=2))
            assert code.count(f"{TWO_STAGE}(") == 2, shapes
    # The maximum is the target's: a 99 KiB part declines the 32 copies.
    with cpu_render_target(
        smem_capacity_bytes=101376, reduction_smem_budget_bytes=101376
    ):
        bound = cpu_bind(chunked_sum, (x,))
        with pytest.raises(
            helion.exc.BackendUnsupported,
            match="139264 bytes of shared memory, over the 101376-byte",
        ):
            bound.to_code(_config(2))
    # Below sm_90 libNVVM compiles 48 KiB of static allocations, under an
    # A100's 166912-byte opt-in maximum: its 16 copies, 69632 bytes, fail.
    with cpu_render_target(
        capability=(8, 0),
        smem_capacity_bytes=166912,
        reduction_smem_budget_bytes=166912,
    ):
        bound = cpu_bind(chunked_sum, (x,), capability=(8, 0))
        with pytest.raises(
            helion.exc.BackendUnsupported,
            match="69632 bytes of shared memory, over the 49152-byte",
        ):
            bound.to_code(_config(1))


def test_reduce_copies_in_code_the_compiler_drops_are_not_counted() -> None:
    # 64 reduce copies (278528 bytes) in code that never runs: behind a test
    # the compiler folds, in a ``while`` whose test is false on entry, in a
    # dynamic loop over an empty runtime range.  libNVVM drops them and
    # compiles each kernel, so codegen must not reject one.
    x = torch.empty(256, 512, 128, dtype=torch.bfloat16)
    steps = torch.zeros(4, dtype=torch.int32)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        for kernel, args, form in (
            (guarded_chunked_sum, (x,), "if "),
            (stepped_chunked_sum, (x, steps), "while "),
            (offset_chunked_sum, (x,), "range(cutlass.Int32(cutlass.Int32(tile_"),
        ):
            code = cpu_bind(kernel, args).to_code(_config(4))
            assert code.count(f"{TWO_STAGE}(") == 1, kernel.name
            assert form in code, kernel.name
        # Without static shapes: an inner loop over ``range(r, 0)`` inside the
        # reduce loop over ``range(0, r)`` never runs, though either range
        # alone may have a trip; over ``range(0, r)`` it runs (and libNVVM
        # fails on its 64 copies).
        inner = helion.Config(
            block_sizes=[128, 64, 16, 16],
            num_threads=[8, 8, 1, 16],
            cute_vector_widths=[16, 4, 1, 1],
        )
        code = cpu_bind(inner_chunked_sum, (x,)).to_code(inner)
        assert code.count(f"{TWO_STAGE}(") == 1
        with pytest.raises(
            helion.exc.BackendUnsupported, match="278528 bytes of shared memory"
        ):
            cpu_bind(live_inner_chunked_sum, (x,)).to_code(inner)


def test_the_spread_pad_reaches_the_footprint_past_dropped_reduce_copies() -> None:
    # One CTA per SM (4 CTAs on 148 SMs): the pad raises the kernel to the
    # 115713 bytes at which two CTAs no longer fit an SM.  Its 16 reduce
    # copies (69632 bytes) sit behind a test the compiler folds and drops,
    # so the pad is the whole footprint: sized against those copies, it left
    # the pad alone in the kernel, and two CTAs fit an SM.
    x = torch.empty(256, 512, 128, dtype=torch.bfloat16)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(guarded_chunked_sum, (x,))
        config = bound.config_spec.normalized_config(
            helion.Config(**_config(1).config, cute_pdl=True, cute_pdl_spread=True)
        )
        code = bound.to_code(config)
    assert code.count(f"{TWO_STAGE}(") == 1
    assert (
        "pdl_spread_pad = cute.arch.alloc_smem(cutlass.Uint8, 115713, alignment=128)"
    ) in code


def test_the_spread_pad_counts_the_alignment_padding_ahead_of_each_allocation() -> None:
    # Each allocation may start up to its alignment less one past the end of
    # the one before it: an explicit ``alignment``, or its element's width.
    footprint = smem_footprint(
        ast.parse(
            "a = cute.arch.alloc_smem(cutlass.Uint8, 100, alignment=128)\n"
            "b = cute.arch.alloc_smem(cutlass.Float32, 25)\n"
            "use(a, b)\n"
        ).body
    )
    assert footprint == SmemFootprint(200, 200, 127 + 3)
    # Eight 14568-byte buffers read under a runtime test, at one CTA per SM
    # (128 CTAs on 148 SMs): a 115713-byte pad would take the kernel to
    # 232257 bytes before padding, but past the 232448-byte maximum with the
    # 128-byte alignment of each buffer and of the pad (libNVVM failed).
    names = ", ".join(f"b{i}" for i in range(8))
    eight = "".join(
        f"b{i} = cute.arch.alloc_smem(cutlass.Uint8, 14568, alignment=128)\n"
        for i in range(8)
    ) + _loop("if flag", f"use({names})")
    footprint = smem_footprint(ast.parse(eight).body)
    assert footprint == SmemFootprint(0, 116544, 8 * 127)
    assert spread_pad_bytes(128, 148, footprint.low, footprint.high) == 115713
    assert (
        spread_pad_bytes(
            128, 148, footprint.low, footprint.high, own_padding=footprint.padding
        )
        is None
    )
    # Ten 4-byte buffers aligned to 128 bytes at 14 CTAs per SM: with their
    # 1270 bytes of padding the 14501-byte pad lays out past what 14 CTAs, each
    # with its reserved slice and slack, fit in (13 would), so it is declined.
    assert spread_pad_bytes(148 * 14, 148, 40, 40) == 14501
    assert spread_pad_bytes(148 * 14, 148, 40, 40, own_padding=1270) is None


def test_a_persistent_loop_is_entered() -> None:
    # Block 0 starts at the first of the launch's 1024 virtual program ids in
    # either persistent form, so the compiler keeps the loop's body: its 16
    # reduce copies (69632 bytes) are certain.  The one-CTA-per-SM pad then
    # stops at the 115713-byte footprint (46081 bytes of pad, not the whole
    # footprint on top of the copies), and 64 copies (278528 bytes) are
    # rejected at codegen, where libNVVM would otherwise fail.
    x = torch.empty(256, 512, 128, dtype=torch.bfloat16)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(chunked_sum, (x,))
        for pid_type in ("persistent_interleaved", "persistent_blocked"):
            config = bound.config_spec.normalized_config(
                helion.Config(
                    **_config(1).config,
                    pid_type=pid_type,
                    cute_pdl=True,
                    cute_pdl_spread=True,
                )
            )
            code = bound.to_code(config)
            assert "for virtual_pid in range(" in code, pid_type
            assert code.count(f"{TWO_STAGE}(") == 1, pid_type
            assert (
                "pdl_spread_pad = cute.arch.alloc_smem(cutlass.Uint8, 46081, "
                "alignment=128)"
            ) in code, pid_type
            with pytest.raises(
                helion.exc.BackendUnsupported, match="278528 bytes of shared memory"
            ):
                bound.to_code(
                    bound.config_spec.normalized_config(
                        helion.Config(**_config(4).config, pid_type=pid_type)
                    )
                )


def test_the_sm8x_limit_does_not_follow_the_planner_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # libNVVM compiles 48 KiB of static allocations below sm_90: a toolchain
    # limit, kept apart from the planners' static budget, so retuning that
    # budget leaves the per-block maximum the check holds kernels to.
    monkeypatch.setattr(target_facts, "_STATIC_SMEM_BUDGET_BYTES", 96 * 1024)
    assert target_facts.static_smem_budget() == 96 * 1024
    with patch_target_facts(smem_capacity_bytes=166912):
        device = torch.device("cuda")
        assert target_facts.max_smem_per_block(device, (8, 0)) == 48 * 1024
        assert target_facts.max_smem_per_block(device, (8, 6)) == 48 * 1024
        assert target_facts.max_smem_per_block(device, (9, 0)) == 166912
        assert target_facts.max_smem_per_block(device, None) == 166912


def test_an_unqueried_device_takes_its_capabilitys_maximum() -> None:
    # Without a device to query (a CPU bind, an offline compile), the maximum
    # follows the target's capability: 227 KiB on sm_90 and sm_100, 99 KiB on
    # sm_120, the sm_90 / sm_100 one when the capability is unknown.
    with patch_target_facts(smem_capacity_bytes=0):
        device = torch.device("cpu")
        assert target_facts.max_smem_per_block(device, (12, 0)) == 101376
        assert target_facts.max_smem_per_block(device, (10, 0)) == 232448
        assert target_facts.max_smem_per_block(device, (9, 0)) == 232448
        assert target_facts.max_smem_per_block(device, None) == 232448
        assert target_facts.max_smem_per_block(device, (8, 0)) == 48 * 1024
