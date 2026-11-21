"""Check ``helion._testing.cute_ptx_waits``, the static search of a compiled
kernel's PTX for global accesses that a path from the entry reaches without
a dependency wait, on small PTX snippets: which instructions count as
writes to global memory, those through a generic address among them, and
which wait ends a path. And its search for what a path reaches from one
instruction without executing another: both ways of a branch, a loop's
back edge, and what stands behind the second on every path. Both refuse
a kernel with a call or an indexed branch, which they do not follow."""

from __future__ import annotations

import re

import pytest

from helion._testing.cute_ptx_waits import global_accesses_ahead_of_a_wait
from helion._testing.cute_ptx_waits import instructions_reached_without

pytestmark = pytest.mark.cpu_only


def _kernel(body: str) -> str:
    """A PTX entry whose last parameter is the wait mode: zero waits at
    entry, nonzero branches past that wait to ``body``."""
    lines = "\n".join(f"\t{line}" for line in body.strip().splitlines())
    return f"""
.version 8.5
.target sm_100a
.address_size 64

.visible .entry kernel(
	.param .u64 kernel_param_0,
	.param .u32 kernel_param_1
)
{{
	.reg .pred 	%p<4>;
	.reg .b32 	%r<8>;
	.reg .b64 	%rd<4>;

	ld.param.u64 	%rd1, [kernel_param_0];
	ld.param.u32 	%r1, [kernel_param_1];
	setp.ne.b32 	%p1, %r1, 0;
	@%p1 bra 	$L__BB0_2;
	griddepcontrol.wait;
$L__BB0_2:
{lines}
	ret;
}}
"""


def _writes(body: str, *, mode_nonzero: bool = True) -> list[str]:
    writes, _reads = global_accesses_ahead_of_a_wait(
        _kernel(body), mode_nonzero=mode_nonzero
    )
    return writes


@pytest.mark.parametrize(
    "store",
    [
        "st.relaxed.gpu.global.u32 	[%rd1], %r2;",
        "st.release.sys.global.u32 	[%rd1], %r2;",
        "st.volatile.global.u32 	[%rd1], %r2;",
        "st.weak.global.L1::no_evict.u32 	[%rd1], %r2;",
        "st.mmio.relaxed.sys.global.u32 	[%rd1], %r2;",
        (
            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group "
            "[%rd1, {%r2, %r3}], [%r4];"
        ),
        (
            "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group "
            "[%rd1, {%r2, %r3}], [%r4];"
        ),
    ],
    ids=lambda store: store.split()[0],
)
def test_a_store_with_qualifiers_and_a_tensor_bulk_store_are_global_writes(store):
    """A store that names qualifiers ahead of its state space (an ordering
    with its scope, volatile, weak, a memory-mapped one) and the bulk
    copy from shared memory through a tensor map, plain and as a reduction,
    write global memory: reached without a wait they are reported, also
    under a predicate, and behind the wait they are not."""
    assert _writes(store) == [store]
    assert _writes(f"@%p2 {store}") == [f"@%p2 {store}"]
    assert _writes(f"griddepcontrol.wait;\n{store}") == []
    # Mode zero waits at the entry, ahead of everything.
    assert _writes(store, mode_nonzero=False) == []


@pytest.mark.parametrize(
    "write",
    [
        "st.u32 	[%rd1], %r2;",
        "st.v4.f32 	[%rd1], {%r2, %r3, %r4, %r5};",
        "st.relaxed.gpu.b64 	[%rd1], %rd2;",
        "atom.add.u32 	%r3, [%rd1], %r2;",
        "atom.acq_rel.gpu.cas.b32 	%r3, [%rd1], %r2, %r4;",
        "red.add.u32 	[%rd1], %r2;",
        "red.relaxed.gpu.max.s64 	[%rd1], %rd2;",
    ],
    ids=lambda write: write.split()[0],
)
def test_a_write_through_a_generic_address_may_be_a_global_write(write):
    """A store, an atomic and a reduction that name no state space, plain
    and with a type, a vector or an ordering, address generically: the
    address may be one of global memory, so reached without a wait they
    are reported, also under a predicate and as a second statement of one
    line (inline assembly), and behind the wait they are not."""
    assert _writes(write) == [write]
    for predicated in (f"@%p2 {write}", f"@!%p2 {write}"):
        assert _writes(predicated) == [predicated]
    inline = f"mov.u32 	%r2, 0; {write}"
    assert _writes(inline) == [inline]
    assert _writes(f"griddepcontrol.wait;\n{write}") == []
    # Mode zero waits at the entry, ahead of everything.
    assert _writes(write, mode_nonzero=False) == []


def test_a_write_that_names_another_state_space_is_no_global_write():
    """A store, an atomic and a reduction that name shared, local or
    parameter memory write there, and the instructions whose names hold
    ``st`` or ``red`` behind another word (the tensor-memory store, a
    barrier's reduction) are neither: none is reported, reached without a
    wait, alone or behind another statement of its line."""
    elsewhere = [
        "st.shared.u32 	[%r4], %r2;",
        "st.shared::cta.v4.b32 	[%r4], {%r2, %r3, %r5, %r6};",
        "st.volatile.shared::cluster.u32 	[%r4], %r2;",
        "st.local.u32 	[%rd1], %r2;",
        "st.param.b32 	[param0+0], %r2;",
        "atom.shared.add.u32 	%r3, [%r4], %r2;",
        "atom.acq_rel.cta.shared::cta.cas.b32 	%r3, [%r4], %r2, %r5;",
        "red.shared.add.u32 	[%r4], %r2;",
        "tcgen05.st.sync.aligned.16x256b.b32 	[%r4], {%r2, %r3};",
        "barrier.red.or.pred 	%p3, %r4, %p2;",
        "bar.red.popc.u32 	%r3, %r4, %p2;",
    ]
    assert _writes("\n".join(elsewhere)) == []
    assert _writes("\n".join(f"@%p2 {line}" for line in elsewhere)) == []
    assert _writes("\n".join(f"mov.u32 	%r2, 0; {line}" for line in elsewhere)) == []
    # Beside them a generic store is still seen.
    store = "st.u32 	[%rd1], %r2;"
    assert _writes("\n".join([*elsewhere, store])) == [store]


def test_a_wait_under_a_predicate_ends_no_path():
    """``griddepcontrol.wait`` under a predicate lets the threads the
    predicate excludes go on: the stores behind it are still reached
    without a wait. The wait as an instruction of its own ends the path."""
    store = "st.global.u32 	[%rd1], %r2;"
    for wait in ("@%p2 griddepcontrol.wait;", "@!%p2 griddepcontrol.wait;"):
        assert _writes(f"{wait}\n{store}") == [store]
    assert _writes(f"griddepcontrol.wait;\n{store}") == []
    assert _writes(f"@%p2 griddepcontrol.wait;\ngriddepcontrol.wait;\n{store}") == []


def test_what_counted_before_still_counts_and_loads_are_no_writes():
    """The forms the kernels have today: a plain and a hinted store, the
    atomics and the bulk copy to global memory are writes; stores to shared
    memory, loads, the bulk copy from global into shared memory and the
    tensor-map load into shared memory are none, and the loads and those
    copies are reported as reads."""
    written = [
        "st.global.b32 	[%rd1], %r2;",
        "st.global.L2::cache_hint.v4.f32 	[%rd1], {%r2, %r3, %r4, %r5};",
        "atom.add.acq_rel.gpu.global.u32 	%r3, [%rd1], %r2;",
        "red.relaxed.gpu.global.add.u32 	[%rd1], %r2;",
        "cp.async.bulk.global.shared::cta.bulk_group 	[%rd1], [%r2], %r3;",
        (
            "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 "
            "[%rd1], [%r2], %r3;"
        ),
    ]
    read = [
        "ld.relaxed.gpu.global.u32 	%r3, [%rd1];",
        "ld.global.cg.v2.f32 	{%r2, %r3}, [%rd1];",
        (
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx"
            "::bytes [%r4], [%rd1, {%r2, %r3}], [%r5];"
        ),
        (
            "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
            "[%r4], [%rd1], %r2, [%r5];"
        ),
    ]
    neither = [
        "st.shared.b32 	[%r4], %r2;",
        "st.shared::cta.v4.b32 	[%r4], {%r2, %r3, %r5, %r6};",
        "ld.shared.b32 	%r3, [%r4];",
    ]
    body = "\n".join([*written, *read, *neither])
    writes, reads = global_accesses_ahead_of_a_wait(_kernel(body), mode_nonzero=True)
    assert writes == written
    assert reads == read
    writes, reads = global_accesses_ahead_of_a_wait(_kernel(body), mode_nonzero=False)
    assert (writes, reads) == ([], [])


def _reached(body: str, stop: str = r"^bar\.sync\s+%r\d+, 128;") -> list[str]:
    """What a path reaches from the load of the word at offset 64 without
    a barrier of 128 threads, less the branches."""
    reached = instructions_reached_without(
        _kernel(body),
        start=re.compile(r"^ld\.shared\.b32\s+%r\d+, \[%r\d+\+64\];"),
        stop=re.compile(stop),
    )
    return [text for text in reached if "bra" not in text and text != "ret;"]


def test_a_barrier_on_every_path_leaves_nothing_behind_it_reached():
    """From a load, a branch is taken both ways: a store behind a barrier
    on one way and bare on the other is reached, one behind a barrier on
    both ways is not, and neither is anything ahead of the load. The load
    itself is not listed."""
    store = "st.shared.b32 \t[%r3+128], %r4;"
    load = "ld.shared.b32 \t%r2, [%r3+64];"
    join = "bar.sync \t%r5, 128;"
    assert _reached(f"{store}\n{load}\n{join}\n{store}") == []
    assert _reached(f"{load}\n{store}\n{join}") == [store]
    one_way = f"""
{load}
setp.ne.b32 \t%p2, %r2, 4;
@%p2 bra \t$L__BB0_4;
{join}
$L__BB0_4:
{store}
"""
    assert _reached(one_way) == ["setp.ne.b32 \t%p2, %r2, 4;", store]
    other_way = f"""
{load}
setp.ne.b32 \t%p2, %r2, 4;
@%p2 bra \t$L__BB0_4;
{store}
bra.uni \t$L__BB0_5;
$L__BB0_4:
{join}
$L__BB0_5:
add.s32 \t%r6, %r6, 1;
"""
    assert _reached(other_way) == [
        "setp.ne.b32 \t%p2, %r2, 4;",
        store,
        "add.s32 \t%r6, %r6, 1;",
    ]
    both_ways = f"""
{load}
setp.ne.b32 \t%p2, %r2, 4;
@%p2 bra \t$L__BB0_4;
{join}
bra.uni \t$L__BB0_5;
$L__BB0_4:
{join}
$L__BB0_5:
{store}
"""
    assert _reached(both_ways) == ["setp.ne.b32 \t%p2, %r2, 4;"]
    # A barrier of another count, or of the other form, is no such stop.
    for other in ("bar.sync \t%r5, 96;", "barrier.arrive \t%r5, 160;"):
        assert _reached(f"{load}\n{other}\n{store}") == [other, store]
    # Another load is no start, and a module without one is refused.
    with pytest.raises(AssertionError):
        _reached(f"ld.shared.b32 \t%r2, [%r3+68];\n{store}")


def test_a_loops_back_edge_and_an_unconditional_branch_are_followed():
    """A store that stands above the load inside a loop is reached from
    the load only around the loop's back edge. Without a barrier in the
    loop it is reached, and so is the load itself, which the way round
    comes back to. With a barrier between the store and the load the store
    is still reached, since the way round passes it first, and the load is
    not. Both hold for a loop of one block that branches to itself and for
    one of two blocks whose latch is an unconditional branch back to the
    header, an earlier block. With a barrier between the load and the back
    edge, or at the loop's top, the store is not reached. What an
    unconditional branch jumps over is not reached either, and its target
    is."""
    store = "st.shared.b32 \t[%r3+128], %r4;"
    load = "ld.shared.b32 \t%r2, [%r3+64];"
    join = "bar.sync \t%r5, 128;"
    step = "add.s32 \t%r6, %r6, 1;"
    test = "setp.lt.s32 \t%p2, %r6, %r7;"
    after = "sub.s32 \t%r6, %r6, 1;"
    looped = f"""
$L__BB0_4:
{{top}}
{store}
{{above}}
{load}
{step}
{{below}}
{test}
@%p2 bra \t$L__BB0_4;
{after}
"""
    assert _reached(looped.format(top="", above="", below="")) == [
        store,
        load,
        step,
        test,
        after,
    ]
    assert _reached(looped.format(top="", above=join, below="")) == [
        store,
        step,
        test,
        after,
    ]
    assert _reached(looped.format(top="", above="", below=join)) == [step]
    assert _reached(looped.format(top=join, above="", below="")) == [step, test, after]
    # A loop of two blocks: the exit test ends the first, and an
    # unconditional branch behind the step goes back to it.
    latched = f"""
$L__BB0_4:
{store}
{{above}}
{load}
{test}
@%p2 bra \t$L__BB0_5;
{step}
bra.uni \t$L__BB0_4;
$L__BB0_5:
{after}
"""
    assert _reached(latched.format(above="")) == [store, load, test, step, after]
    assert _reached(latched.format(above=join)) == [store, test, step, after]
    skipped = f"""
{load}
bra.uni \t$L__BB0_4;
{store}
$L__BB0_4:
{step}
"""
    assert _reached(skipped) == [step]


@pytest.mark.parametrize(
    "unfollowed",
    [
        "call \thelper;",
        "call.uni \t(retval0), helper, (param0);",
        # As the NVPTX backend writes one: its operands on their own lines.
        "call.uni (retval0), \nhelper, \n(\nparam0\n);",
        "brx.idx \t%r2, $L_brx_0;",
        "brx.idx.uni \t%r2, $L_brx_0;",
    ],
    ids=["call", "call.uni", "call.uni-lines", "brx.idx", "brx.idx.uni"],
)
def test_a_call_and_an_indexed_branch_are_refused(unfollowed):
    """The searches follow ``bra``, ``ret`` and ``trap``. A call to a
    device function that was not inlined would hide what the function
    writes, and an indexed branch where it goes: both searches refuse a
    kernel with either, where they would otherwise pass it as an ordinary
    instruction. Also under a predicate, as a second statement of a line
    (inline assembly) and behind the wait. An operand that only names one
    is no such instruction."""
    load = "ld.shared.b32 \t%r2, [%r3+64];"
    for body in (
        unfollowed,
        f"@%p2 {unfollowed}",
        f"@!%p2 {unfollowed}",
        f"mov.u32 \t%r2, 0; {unfollowed}",
        f"griddepcontrol.wait;\n{unfollowed}",
    ):
        for search in (_writes, _reached):
            with pytest.raises(AssertionError, match="unsupported control flow"):
                search(f"{load}\n{body}")
    named = "mov.u64 \t%rd2, call_table;"
    assert _writes(f"{load}\n{named}") == []
    assert _reached(f"{load}\n{named}") == [named]
