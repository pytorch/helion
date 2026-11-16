"""The complete chunks of a looped sweep run in a loop of their own
(``helion/_compiler/cute/peel_partial_chunks.py``)."""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest
import torch

import helion
from helion._compiler.cute.peel_partial_chunks import merge_unproven_peels
from helion._compiler.cute.peel_partial_chunks import peel_partial_chunks
from helion._compiler.cute.proven_loop_bounds import simplify_proven_loop_bounds
from helion._compiler.cute.reduce_call import GROUPED_REDUCE_HELPERS
from helion._compiler.cute.vector_reduction_packets import rolled_sums_split_after
from helion._compiler.cute.vector_reduction_packets import split_rolled_vector_sums
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only

# A rolled sweep of 1000 elements in chunks of 256 over 64 threads, 2 lanes of
# 2-wide vectors each: three complete chunks and one of 232.
SWEEP = """
total = 0
for roffset in range(cutlass.Int32(0), cutlass.Int32(1000), cutlass.Int32(CHUNK)):
    for lane in range(2):
        base = roffset + cutlass.Int32(cute.arch.thread_idx()[0]) * 2 + cutlass.Int32(lane) * 128
        mask = base < 1000
        for vec in cutlass.range_constexpr(2):
            if base + vec < 1000:
                record(base + vec)
        total = total + (base if mask else 0)
"""


def _peeled(
    source: str,
    *,
    thread_block_dims: tuple[int, int, int] | None = (64, 1, 1),
) -> list[ast.stmt]:
    """``source`` through the peel, the proof and the rejoin, as the pass
    pipeline runs them."""
    body = peel_partial_chunks(ast.parse(source).body, {"CHUNK": 256})
    body = simplify_proven_loop_bounds(
        body,
        enabled=True,
        thread_block_dims=thread_block_dims,
        constexpr_values={"CHUNK": 256},
    )
    return merge_unproven_peels(body)


def _split(source: str) -> list[ast.stmt]:
    return peel_partial_chunks(ast.parse(source).body, {"CHUNK": 256})


def _loops(body: list[ast.stmt]) -> list[ast.For]:
    return [statement for statement in body if isinstance(statement, ast.For)]


def _run(body: list[ast.stmt], thread: int) -> tuple[list[int], int]:
    seen: list[int] = []
    namespace = {
        "CHUNK": 256,
        "record": seen.append,
        "cutlass": SimpleNamespace(Int32=int, range_constexpr=range),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(thread_idx=lambda: (thread, 0, 0))
        ),
    }
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "<sweep>", "exec"), namespace)
    return seen, namespace["total"]


def test_complete_chunks_lose_their_bounds_predicates() -> None:
    loops = _loops(_peeled(SWEEP))
    assert [ast.unparse(loop.iter) for loop in loops] == [
        "range(cutlass.Int32(0), cutlass.Int32(768), cutlass.Int32(CHUNK))",
        "range(cutlass.Int32(768), cutlass.Int32(1000), cutlass.Int32(CHUNK))",
    ]
    full, partial = (ast.unparse(loop) for loop in loops)
    assert "< 1000" not in full, full
    assert "< 1000" in partial


@pytest.mark.parametrize("thread", [0, 17, 63])
def test_peeled_sweep_runs_the_same_iterations(thread: int) -> None:
    assert _run(_peeled(SWEEP), thread) == _run(ast.parse(SWEEP).body, thread)


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(SWEEP.replace("1000), cutlass", "1024), cutlass"), id="divisible"),
        pytest.param(SWEEP.replace("< 1000", "< limit"), id="no_end_predicate"),
        pytest.param(
            SWEEP.replace(
                "record(base + vec)", "total = cute.arch.warp_reduction_sum(total)"
            ),
            id="warp_reduction",
        ),
        pytest.param(
            SWEEP.replace("record(base + vec)", "break"), id="control_transfer"
        ),
    ],
)
def test_other_loops_stay_whole(source: str) -> None:
    assert len(_loops(_split(source))) == 1


# One call per kind of shared memory ``smem_footprint`` sizes outside the
# grouped reduces (one case each below).
@pytest.mark.parametrize(
    "call",
    [
        "cute.arch.alloc_smem(cutlass.Float32, 4)",
        "cutlass.utils.SmemAllocator()",
        "smem.allocate(Storage)",
        "smem.allocate_tensor(cutlass.Float32, layout)",
        "smem.allocate_array(cutlass.Float32, 4)",
        "cute.arch.get_dyn_smem(cutlass.Float32)",
        "_cute_resident_sums(partials, 64)",
    ],
)
def test_a_loop_allocating_shared_memory_stays_whole(call: str) -> None:
    """Every traced copy of the call would allocate again: the resident
    reduction's 64-thread sums, for one, take a ``SmemAllocator`` tensor."""
    source = SWEEP.replace("record(base + vec)", f"buf = {call}")
    assert len(_loops(_split(source))) == 1


def test_a_one_warp_resident_sum_keeps_its_loop_whole_too() -> None:
    """At one warp the resident sums reduce through shuffles alone, but
    ``allocates_smem`` names them at any thread count: the loop stays whole,
    as conservatively as beside the warp reduces."""
    source = SWEEP.replace(
        "record(base + vec)", "buf = _cute_resident_sums(partials, 32)"
    )
    assert len(_loops(_split(source))) == 1


# One case per grouped-reduce dispatcher: a new one must keep its loop whole
# too.
@pytest.mark.parametrize("helper", sorted(GROUPED_REDUCE_HELPERS))
def test_a_loop_reducing_across_threads_stays_whole(helper: str) -> None:
    """A copy of the loop would run the reduce at a second site, with a
    second shared buffer for the buffered forms; the warp and cluster forms
    claim nothing per call site and keep the loop whole conservatively."""
    source = SWEEP.replace("record(base + vec)", f"total = {helper}(total)")
    assert len(_loops(_split(source))) == 1


def test_calls_that_claim_no_site_of_their_own_leave_the_loop_peelable() -> None:
    """An async copy's commit or a select is repeated by the copies like any
    statement: besides the calls that may allocate shared memory, only an
    mbarrier's initialization and the reduces across threads keep a loop
    whole."""
    source = SWEEP.replace(
        "record(base + vec)",
        "record(cutlass.select_(mask, base + vec, 0))\n"
        "                cute.arch.cp_async_commit_group()",
    )
    assert len(_loops(_split(source))) == 2


def test_a_loop_ending_at_a_name_stays_whole() -> None:
    """The body must compare against the literal end, so a loop ending at a
    constexpr name stays whole, even where its predicates spell the name's
    value."""
    source = SWEEP.replace(
        "cutlass.Int32(1000), cutlass.Int32(CHUNK)",
        "cutlass.Int32(N), cutlass.Int32(CHUNK)",
    )
    body = peel_partial_chunks(ast.parse(source).body, {"CHUNK": 256, "N": 1000})
    assert len(_loops(body)) == 1


def test_int64_bounds_keep_their_type() -> None:
    """An Int64 loop keeps its induction variable's range: the copies cast
    their bounds as the loop did."""
    source = SWEEP.replace(
        "cutlass.Int32(0), cutlass.Int32(1000)", "cutlass.Int64(0), cutlass.Int64(1000)"
    )
    assert [ast.unparse(loop.iter) for loop in _loops(_split(source))] == [
        "range(cutlass.Int64(0), cutlass.Int64(768), cutlass.Int32(CHUNK))",
        "range(cutlass.Int64(768), cutlass.Int64(1000), cutlass.Int32(CHUNK))",
    ]


NEST = """
for i in range(cutlass.Int32(0), cutlass.Int32(1000), cutlass.Int32(CHUNK)):
    for j in range(cutlass.Int32(0), cutlass.Int32(1000), cutlass.Int32(CHUNK)):
        for k in range(cutlass.Int32(0), cutlass.Int32(1000), cutlass.Int32(CHUNK)):
            if i + j + k < 1000:
                record(k)
"""


def test_a_loop_holding_a_peeled_loop_stays_whole() -> None:
    """Every loop of the nest qualifies, but only the innermost is split:
    splitting its parents would copy the copies again, doubling the code
    per level."""
    (outer,) = _loops(_split(NEST))
    (middle,) = _loops(outer.body)
    assert len(_loops(middle.body)) == 2


def test_a_split_the_proof_does_not_use_is_rejoined() -> None:
    """Without the launch's thread counts the proof removes nothing, and
    neither does it where the comparisons read a value it cannot bound, nor
    in the complete chunks where it removes a test only the partial chunk
    passes (``roffset >= 700``): the loop comes back whole, as it was."""
    unbounded = SWEEP.replace("base < 1000", "limit < 1000").replace(
        "base + vec < 1000", "limit + vec < 1000"
    )
    late = unbounded.replace(
        "        total = total",
        "        if roffset >= 700:\n            record(roffset)\n        total = total",
    )
    for source, dims in ((SWEEP, None), (unbounded, (64, 1, 1)), (late, (64, 1, 1))):
        assert len(_loops(_split(source))) == 2
        (loop,) = _loops(_peeled(source, thread_block_dims=dims))
        assert ast.dump(loop) == ast.dump(_loops(ast.parse(source).body)[0])


ROLLED_SUM = """
accumulator = cutlass.Float32(0)
for offset in range(cutlass.Int32(0), cutlass.Int32(40), cutlass.Int32(16)):
    for lane in range(2):
        base = offset + lane * 8
        for element in cutlass.range_constexpr(8):
            index = base + element
            value = values[index] if index < 40 else cutlass.Float32(0)
            accumulator = accumulator + value
answer = accumulator
"""
# The same sweep inside a loop: ``split_rolled_vector_sums`` matches the nest
# through any depth of wrapping loops.
WRAPPED_ROLLED_SUM = """
accumulator = cutlass.Float32(0)
for outer in range(cutlass.Int32(0), cutlass.Int32(2), cutlass.Int32(1)):
    for offset in range(cutlass.Int32(0), cutlass.Int32(40), cutlass.Int32(16)):
        for lane in range(2):
            base = offset + lane * 8
            for element in cutlass.range_constexpr(8):
                index = base + element
                value = values[index] if index < 40 else cutlass.Float32(0)
                accumulator = accumulator + value
answer = accumulator
"""
# Guards the proof removes over 64 threads, which the matcher refuses: on a
# thread, on an element (which counts to 8), and around the whole sum, whose
# removal lifts the sum to the top level, where the split looks.
THREAD_GUARDED_ROLLED_SUM = """
accumulator = cutlass.Float32(0)
for offset in range(cutlass.Int32(0), cutlass.Int32(40), cutlass.Int32(16)):
    if cute.arch.thread_idx()[0] < 64:
        for lane in range(2):
            base = offset + lane * 8
            for element in cutlass.range_constexpr(8):
                index = base + element
                value = values[index] if index < 40 else cutlass.Float32(0)
                accumulator = accumulator + value
answer = accumulator
"""
ELEMENT_GUARDED_ROLLED_SUM = """
accumulator = cutlass.Float32(0)
for offset in range(cutlass.Int32(0), cutlass.Int32(40), cutlass.Int32(16)):
    for lane in range(2):
        base = offset + lane * 8
        for element in cutlass.range_constexpr(8):
            if element < 8:
                index = base + element
                value = values[index] if index < 40 else cutlass.Float32(0)
                accumulator = accumulator + value
answer = accumulator
"""
LIFTED_ROLLED_SUM = """
if cute.arch.thread_idx()[0] < 64:
    accumulator = cutlass.Float32(0)
    for offset in range(cutlass.Int32(0), cutlass.Int32(40), cutlass.Int32(16)):
        for lane in range(2):
            base = offset + lane * 8
            for element in cutlass.range_constexpr(8):
                index = base + element
                value = values[index] if index < 40 else cutlass.Float32(0)
                accumulator = accumulator + value
    answer = accumulator
"""


def _prove(body: list[ast.stmt]) -> list[ast.stmt]:
    """``body`` after the bounds proof over 64 threads."""
    return simplify_proven_loop_bounds(
        body, enabled=True, thread_block_dims=(64, 1, 1), constexpr_values={}
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(ROLLED_SUM, id="top_level"),
        pytest.param(WRAPPED_ROLLED_SUM, id="wrapped"),
        pytest.param(THREAD_GUARDED_ROLLED_SUM, id="thread_guard"),
        pytest.param(ELEMENT_GUARDED_ROLLED_SUM, id="element_guard"),
        pytest.param(LIFTED_ROLLED_SUM, id="lifted"),
    ],
)
def test_a_rolled_sum_split_into_independent_sums_stays_whole(source: str) -> None:
    """Under ``cute_independent_reduction`` the pipeline keeps whole the
    rolled sums ``split_rolled_vector_sums`` splits, which matches the loop
    whole (its peeled copies split into no sums at all), with the loops
    nested in the nest it names.  The split runs after the bounds proofs,
    and so is the question asked: a guard they remove or inline can make a
    sum splittable."""

    def sums(body: list[ast.stmt]) -> int:
        return sum(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id.startswith("_cute_rolled_sum_")
            for statement in body
            for node in ast.walk(statement)
        )

    def loops(body: list[ast.stmt]) -> int:
        return sum(
            isinstance(node, ast.For)
            for statement in body
            for node in ast.walk(statement)
        )

    whole = sums(split_rolled_vector_sums(_prove(ast.parse(source).body)))
    assert whole == 16
    assert sums(split_rolled_vector_sums(_prove(_split(source)))) == 0
    body = ast.parse(source).body
    kept = peel_partial_chunks(
        body, {}, keep_whole=rolled_sums_split_after(body, _prove)
    )
    assert loops(kept) == loops(ast.parse(source).body)
    assert sums(split_rolled_vector_sums(_prove(kept))) == whole


def _row_update_fn(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_r in hl.tile(x.size(0)):
        row = x[tile_r, :].to(torch.float32)
        out[tile_r, :] = (row * scale[tile_r][:, None]).to(out.dtype)
    return out


def test_rolled_row_sweep_peels_under_proven_bounds() -> None:
    """A 1000-wide row in chunks of 256: the complete chunks load and store
    without a predicate, the last one keeps its own."""
    kernel = helion.kernel(
        _row_update_fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    args = (torch.randn(8, 1000).to(torch.bfloat16), torch.randn(8))
    config = {
        "block_sizes": [1],
        "reduction_loops": [256],
        "num_threads": [1, 64],
    }
    with cpu_render_target():
        bound = cpu_bind(kernel, args)
        spec = bound.config_spec
        whole, peeled = (
            bound.to_code(
                spec.normalized_config(helion.Config(**config, cute_proven_bounds=flag))
            )
            for flag in (False, True)
        )
    assert whole.count("for roffset_1 in range(") == 1, whole
    full_header = "range(cutlass.Int32(0), cutlass.Int32(768),"
    partial_header = "range(cutlass.Int32(768), cutlass.Int32(1000),"
    full = peeled[peeled.index(full_header) : peeled.index(partial_header)]
    partial = peeled[peeled.index(partial_header) :]
    assert "< 1000" not in full, peeled
    assert "< 1000" in partial


@pytest.mark.parametrize(("width", "peeled"), [(5000, True), (8192, False)])
def test_loads_first_row_seed_peels_a_partial_last_chunk(
    width: int, peeled: bool
) -> None:
    """The 2048-element chunks of the loads-first row seed leave a partial
    last chunk of a 5000-wide row: the seed comes again under
    ``cute_proven_bounds``, which runs the complete chunks apart."""
    kernel = helion.kernel(
        _row_update_fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    args = (torch.randn(8, width).to(torch.bfloat16), torch.randn(8))
    with cpu_render_target():
        bound = cpu_bind(kernel, args)
        seeds = [
            seed
            for seed in bound.config_spec.compiler_seed_configs
            if seed.config.get("cute_lane_unroll") == 16
        ]
        assert [seed.config.get("cute_proven_bounds", False) for seed in seeds] == (
            [False, True] if peeled else [False]
        )
        if peeled:
            # The partial chunk runs in a loop of its own.
            assert "range(cutlass.Int32(4096), cutlass.Int32(5000)," in bound.to_code(
                seeds[1]
            )


@pytest.mark.cuda
@skipIfNotCUDA()
@onlyBackends(["cute"])
class TestPeelPartialChunks(TestCase):
    def test_peeled_row_update_is_bitwise_the_whole_sweep(self) -> None:
        x = torch.randn(64, 1000, device=DEVICE).to(torch.bfloat16)
        scale = torch.randn(64, device=DEVICE)
        config = {"block_sizes": [1], "reduction_loops": [256], "num_threads": [1, 64]}
        outputs = []
        for flag in (False, True):
            code, out = code_and_output(
                helion.kernel(
                    _row_update_fn,
                    backend="cute",
                    static_shapes=True,
                    config=helion.Config(**config, cute_proven_bounds=flag),
                ),
                (x, scale),
            )
            self.assertEqual("cutlass.Int32(768)" in code, flag)
            outputs.append(out)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        torch.testing.assert_close(
            outputs[1], (x.float() * scale[:, None]).to(x.dtype), rtol=0, atol=0
        )
