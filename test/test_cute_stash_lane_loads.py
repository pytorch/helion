"""CPU codegen tests: a split lane loop's later passes reuse its loads.

The lane split runs dependent reductions over a lane loop as one pass each
(an online softmax over a tile: the block's max, then the sum of
``exp(x - max)``), every pass recomputing its inputs, loads included.  Once
the passes are trace-time loops, ``cute/stash_lane_loads.py`` keeps the
first pass's loaded values in a trace-time list and the later passes read
them from it, when nothing between the loads may write the memory they read.
"""

from __future__ import annotations

import ast
import functools
import itertools
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute.autotune.heuristics.cute import CuteTileVecHeuristic
from helion._compiler.cute.stash_lane_loads import _load_sites
from helion._compiler.cute.stash_lane_loads import stash_sibling_lane_loads
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.lane_stash_kernels import online_softmax_stats

if TYPE_CHECKING:
    import contextlib

    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only


def _target() -> contextlib.AbstractContextManager[None]:
    # The CTA-wide reduction sizes its shared-memory groups by the budget.
    return cpu_render_target(reduction_smem_budget_bytes=232448)


@functools.cache
def _bound() -> BoundKernel:
    kernel = helion.kernel(
        online_softmax_stats, backend="cute", static_shapes=True, autotune_effort="none"
    )
    with _target():
        # Rows end mid-vector (8195 % 8): the loads are per element.
        return cpu_bind(kernel, (torch.empty(8, 8200, dtype=torch.bfloat16)[:, :8195],))


def _code(block: int, threads: int) -> str:
    bound = _bound()
    with _target():
        return bound.to_code(
            bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[1, block],
                    num_threads=[1, threads],
                    cute_vector_widths=[1, 8],
                    cute_lane_layouts=["blocked", "strided"],
                )
            )
        )


def test_online_softmax_sum_pass_reads_the_max_passs_loads() -> None:
    # 2048 columns over 64 threads: four lanes of 8-wide vectors per thread,
    # one CTA-wide reduction (with its barrier) between the two passes.
    code = _code(2048, 64)
    tree = ast.parse(code)
    lane_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and ast.unparse(node.iter) == "cutlass.range_constexpr(4)"
    ]
    assert len(lane_loops) == 2
    first, second = map(ast.unparse, lane_loops)
    assert "x.iterator" in first
    assert "_lane_stash_loads_.append(load)" in first
    assert "load = _lane_stash_loads_[lane_1 * 8 + vec_lane_1]" in second


_TWO_PASSES = """
stash = []
for lane in cutlass.range_constexpr(2):
    base = offset + cutlass.Int32(lane) * 64
    for vec in cutlass.range_constexpr(4):
        index = base + cutlass.Int32(vec)
        value = (x.iterator + index).load()
        acc = acc + value
{between}
for lane in cutlass.range_constexpr(2):
    base = offset + cutlass.Int32(lane) * {stride}
    for vec in cutlass.range_constexpr(4):
        index = base + cutlass.Int32(vec)
        value = (x.iterator + index).load()
        total = total + value * acc
"""


def _stash(source: str | list[ast.stmt], renames: dict[str, str] | None = None) -> str:
    names = (f"_s{i}" for i in itertools.count())
    body = ast.parse(source).body if isinstance(source, str) else source
    return ast.unparse(
        ast.Module(
            stash_sibling_lane_loads(
                body,
                lambda _: next(names),
                rename_groups=renames or {},
                kernel_tensors=("x", "y"),
            ),
            [],
        )
    )


def test_later_pass_reuses_the_earlier_load() -> None:
    code = _stash(_TWO_PASSES.format(between="acc = acc * 2.0", stride=64))
    assert code.count("(x.iterator + index).load()") == 1
    assert "_s0.append(value)" in code
    assert "value = _s0[lane * 4 + vec]" in code


def test_a_renamed_write_between_the_passes_keeps_the_later_load() -> None:
    # ``offset_new`` is the loop-output name the final rename folds into
    # ``offset``: the later pass reads the new value.
    source = _TWO_PASSES.format(between="offset_new = offset + 2", stride=64)
    assert "_s0" not in _stash(source, {"offset_new": "offset"})
    assert "_s0" in _stash(source)


def _two_passes(earlier: str, later: str) -> str:
    """Two passes over the same lanes whose inner bodies are ``earlier`` and
    ``later`` (four-space indented below the ``vec`` loop)."""
    return f"""
for lane in cutlass.range_constexpr(2):
    k = offset + cutlass.Int32(lane) * 64
    for vec in cutlass.range_constexpr(4):
{earlier}
acc = acc * 2.0
for lane in cutlass.range_constexpr(2):
    k = offset + cutlass.Int32(lane) * 64
    for vec in cutlass.range_constexpr(4):
{later}
"""


@pytest.mark.parametrize(
    ("earlier", "later"),
    [
        # The address reads ``k`` before the pass assigns it: the later pass
        # reads two elements further on.
        (
            "        k = k + 1\n        value = (x.iterator + k).load()",
            "        k = k + 1\n        value = (x.iterator + k).load()",
        ),
        # ``k`` moves on after each load, by 1 in one pass and 2 in the other.
        (
            "        value = (x.iterator + k).load()\n        k = k + 1",
            "        value = (x.iterator + k).load()\n        k = k + 2",
        ),
        # The address comes from a call not proven pure.
        (
            "        k = next_index(k)\n        value = (x.iterator + k).load()",
            "        k = next_index(k)\n        value = (x.iterator + k).load()",
        ),
        # The pointer is built outside the passes: which tensor it reads is
        # unknown.
        (
            "        value = (pointer + vec).load()",
            "        value = (pointer + vec).load()",
        ),
        # The path variable the stash is indexed by changes before the load.
        (
            "        vec = vec + 1\n        value = (x.iterator + k + vec).load()",
            "        vec = vec + 1\n        value = (x.iterator + k + vec).load()",
        ),
    ],
    ids=[
        "read_before_assigned",
        "moved_after_the_load",
        "impure_index",
        "pointer",
        "reassigned_path_variable",
    ],
)
def test_an_address_the_passes_may_not_share_keeps_the_later_load(
    earlier: str, later: str
) -> None:
    code = _stash(_two_passes(earlier, later))
    assert "_s0" not in code


def test_a_loop_rebinding_an_input_before_the_load_keeps_the_later_load() -> None:
    # The loop leaves ``k`` at 3: the earlier pass loads x[3 + lane], the
    # later one x[offset + lane].
    source = """
for lane in cutlass.range_constexpr(2):
    k = offset
    for k in cutlass.range_constexpr(4):
        acc = acc + 1.0
    value = (x.iterator + k + lane).load()
    acc = acc + value
acc = acc * 2.0
for lane in cutlass.range_constexpr(2):
    k = offset
    value = (x.iterator + k + lane).load()
    total = total + value * acc
"""
    assert "_s0" not in _stash(source)
    assert "_s0" in _stash(source.replace("for k in", "for j in"))


def test_a_list_mutated_between_the_passes_keeps_the_later_load() -> None:
    # The address reads a trace-time list that ``reverse()`` updates in place.
    load = "        value = (x.iterator + offs[vec]).load()"
    source = _two_passes(load, load)
    assert "_s0" in _stash(source)
    assert "_s0" not in _stash(source.replace("acc = acc * 2.0", "offs.reverse()"))


@pytest.mark.parametrize("statement", ["continue", "break"])
def test_a_pass_that_ends_an_iteration_early_keeps_its_loads(statement: str) -> None:
    # The stash is filled by appends and read by lane number: a lane the
    # earlier pass skips would shift every later read.
    load = "        value = (x.iterator + k + vec).load()"
    early = f"        if vec == 1:\n            {statement}\n{load}"
    assert "_s0" in _stash(_two_passes(load, load))
    assert "_s0" not in _stash(_two_passes(early, load))
    assert "_s0" not in _stash(_two_passes(load, early))


_THREE_LEVELS = """
for lane in cutlass.range_constexpr(2):
    for vec in cutlass.range_constexpr(4):
        for e in cutlass.range_constexpr(2):
{skip}            value = (x.iterator + offset + lane * 8 + vec * 2 + e).load()
            acc = acc + value
acc = acc * 2.0
for lane in cutlass.range_constexpr(2):
    for vec in cutlass.range_constexpr(4):
        for e in cutlass.range_constexpr(2):
            value = (x.iterator + offset + lane * 8 + vec * 2 + e).load()
            total = total + value * acc
"""

_SIBLING_INNER_LOOPS = """
for lane in cutlass.range_constexpr(2):
    for vec in cutlass.range_constexpr(4):
        acc = acc + 1.0
    for vec in cutlass.range_constexpr(4):
{skip}        value = (x.iterator + offset + lane * 4 + vec).load()
        acc = acc + value
acc = acc * 2.0
for lane in cutlass.range_constexpr(2):
    for vec in cutlass.range_constexpr(4):
        acc = acc + 1.0
    for vec in cutlass.range_constexpr(4):
        value = (x.iterator + offset + lane * 4 + vec).load()
        total = total + value * acc
"""


@pytest.mark.parametrize(
    ("source", "indent"),
    [(_THREE_LEVELS, 12), (_SIBLING_INNER_LOOPS, 8)],
    ids=["third_level", "sibling_inner_loops"],
)
def test_an_early_exit_at_any_depth_keeps_the_loads(source: str, indent: int) -> None:
    # A ``continue`` in a loop below the pass's inner loop, or in the second
    # of two inner loops, skips appends too: the later pass's reads would
    # take the wrong elements, and run past the stash.
    pad = " " * indent
    skip = f"{pad}if {'e' if indent == 12 else 'vec'} == 1:\n{pad}    continue\n"
    assert "_s0" in _stash(source.format(skip=""))
    assert "_s0" not in _stash(source.format(skip=skip))


@pytest.mark.parametrize(
    ("refused", "reused"),
    [
        # Read before its loop starts, ``vec`` holds what the loop before
        # the pass left in the first lane and 3 in the second.
        (
            "for vec in cutlass.range_constexpr(2):\n    pre = vec\n{before_both}",
            "{inside_both}",
        ),
        # The same text inside the loop in one pass and before it in the
        # other.
        ("{moved}", "{inside_both}"),
        # The first pass moves ``lane`` on after the load: its loads drift.
        ("{drifting}", "{plain_both}"),
        # Two path loops share a target: the stash index names it twice.
        ("{shadowed}", "{shadowed_renamed}"),
        # The address reads ``vec`` alone; ``lane`` changes after the load,
        # and the inner loop's next iterations would read the stash at it.
        ("{assigned_after}", "{vec_only_both}"),
    ],
    ids=[
        "read_before_its_loop",
        "at_another_depth",
        "updated_after_the_load",
        "repeated_target",
        "outer_variable_assigned_after_the_load",
    ],
)
def test_a_path_variable_that_is_not_the_elements_index_keeps_the_later_load(
    refused: str, reused: str
) -> None:
    def passes(first: str, second: str) -> str:
        return (
            f"for lane in cutlass.range_constexpr(2):\n{first}"
            "acc = acc * 2.0\n"
            f"for lane in cutlass.range_constexpr(2):\n{second}"
        )

    inside = (
        "    for vec in cutlass.range_constexpr(4):\n"
        "        k = offset + vec\n"
        "        value = (x.iterator + k + lane * 4).load()\n"
        "        acc = acc + value\n"
    )
    before = (
        "    k = offset + vec\n"
        "    for vec in cutlass.range_constexpr(4):\n"
        "        value = (x.iterator + k + lane * 4).load()\n"
        "        acc = acc + value\n"
    )
    plain = (
        "    for vec in cutlass.range_constexpr(4):\n"
        "        value = (x.iterator + offset + lane * 4 + vec).load()\n"
        "        acc = acc + value\n"
    )
    drifting = plain + "        lane = lane + 1\n"
    vec_only = (
        "    for vec in cutlass.range_constexpr(4):\n"
        "        value = (x.iterator + offset + vec).load()\n"
        "        acc = acc + value\n"
    )
    shadowed = (
        "    for lane in cutlass.range_constexpr(4):\n"
        "        value = (x.iterator + offset + lane).load()\n"
        "        acc = acc + value\n"
    )
    shadowed_renamed = shadowed.replace("for lane in", "for vec in").replace(
        "+ lane)", "+ lane * 4 + vec)"
    )
    sources = {
        "before_both": passes(before, before),
        "inside_both": passes(inside, inside),
        "plain_both": passes(plain, plain),
        "moved": passes(inside, before),
        "drifting": passes(drifting, plain),
        "shadowed": passes(shadowed, shadowed),
        "shadowed_renamed": passes(shadowed_renamed, shadowed_renamed),
        "assigned_after": passes(vec_only, vec_only + "        lane = 100\n"),
        "vec_only_both": passes(vec_only, vec_only),
    }
    assert "_s0" not in _stash(refused.format(**sources))
    assert "_s0" in _stash(reused.format(**sources))


@pytest.mark.parametrize(
    ("refused", "reused"),
    [
        # Each pass builds its offsets with an append of its own.
        (
            "{each_pass}",
            "{each_pass_same}",
        ),
        # An alias of the offsets is appended to between the passes.
        ("{through_an_alias}", "{alias_unchanged}"),
        # ... extended by an augmented assignment.
        ("{augmented_alias}", "{alias_unchanged}"),
        # ... taken from a list the offsets were appended to.
        ("{through_a_container}", "{alias_unchanged}"),
        # The offsets are the results a fragment grouped reduce stores.
        ("{reduced_into}", "{reduced_elsewhere}"),
        # The address reads an attribute stored between the passes.
        ("{attribute_stored}", "{other_attribute_stored}"),
    ],
    ids=[
        "appended_in_each_pass",
        "appended_through_an_alias",
        "extended_through_an_alias",
        "appended_through_a_container",
        "a_fragment_reduces_results",
        "an_attribute_store",
    ],
)
def test_a_container_the_address_reads_changed_in_place_keeps_the_later_load(
    refused: str, reused: str
) -> None:
    def passes(first: str, between: str, second: str) -> str:
        return (
            f"for lane in cutlass.range_constexpr(2):\n{first}"
            f"{between}\n"
            f"for lane in cutlass.range_constexpr(2):\n{second}"
        )

    def body(build: str, use: str) -> str:
        return f"{build}    value = (x.iterator + offs[-1] + lane).load()\n    {use}\n"

    sources = {
        "each_pass": passes(
            body("    offs = [0]\n    offs.append(offset)\n", "acc = acc + value"),
            "acc = acc * 2.0",
            body(
                "    offs = [0]\n    offs.append(offset + 9)\n", "total = total + value"
            ),
        ),
        "each_pass_same": passes(
            body("    offs = [offset]\n", "acc = acc + value"),
            "acc = acc * 2.0",
            body("    offs = [offset]\n", "total = total + value"),
        ),
        "through_an_alias": "offs = [offset]\nalias = offs\n"
        + passes(
            body("", "acc = acc + value"),
            "alias.append(offset + 9)",
            body("", "total = total + value"),
        ),
        "alias_unchanged": "offs = [offset]\nalias = offs\n"
        + passes(
            body("", "acc = acc + value"),
            "other = [0]\nother.append(offset + 9)",
            body("", "total = total + value"),
        ),
        "augmented_alias": "offs = [offset]\nalias = offs\n"
        + passes(
            body("", "acc = acc + value"),
            "alias += [offset + 9]",
            body("", "total = total + value"),
        ),
        "through_a_container": "offs = [offset]\nrows = []\nrows.append(offs)\n"
        "alias = rows[0]\n"
        + passes(
            body("", "acc = acc + value"),
            "alias.append(offset + 9)",
            body("", "total = total + value"),
        ),
        "reduced_into": "offs = [offset]\n"
        + passes(
            body("", "acc = acc + value"),
            "acc = _cute_grouped_reduce_shared_two_stage_fragment(acc, offs, smem, 1, 64)",
            body("", "total = total + value"),
        ),
        "reduced_elsewhere": "offs = [offset]\n"
        + passes(
            body("", "acc = acc + value"),
            "acc = _cute_grouped_reduce_shared_two_stage_fragment(acc, maxes, smem, 1, 64)",
            body("", "total = total + value"),
        ),
        "attribute_stored": passes(
            body("", "acc = acc + value").replace("offs[-1]", "st.off"),
            "st.off = offset + 9",
            body("", "total = total + value").replace("offs[-1]", "st.off"),
        ),
        "other_attribute_stored": passes(
            body("", "acc = acc + value").replace("offs[-1]", "st.off"),
            "other.off = offset + 9",
            body("", "total = total + value").replace("offs[-1]", "st.off"),
        ),
    }
    assert "_s0" not in _stash(refused.format(**sources))
    assert "_s0" in _stash(reused.format(**sources))


def test_an_alias_bound_under_its_loop_output_name_keeps_the_later_load() -> None:
    # ``alias_out`` is the loop-output name the final rename folds into
    # ``alias``: the append changes the list the address reads.
    load = "    value = (x.iterator + offs[-1] + lane).load()\n    acc = acc + value\n"
    source = (
        "offs = [offset]\nalias_out = offs\n"
        f"for lane in cutlass.range_constexpr(2):\n{load}"
        "alias.append(offset + 9)\n"
        f"for lane in cutlass.range_constexpr(2):\n{load}"
    )
    assert "_s0" not in _stash(source, {"alias_out": "alias"})
    assert "_s0" in _stash(source)


def test_an_offset_added_to_a_pointer_points_nowhere() -> None:
    # The address is bound to a name: its integer offset, the mask computed
    # from it and the masked accumulator a CTA-wide reduction is handed
    # between the passes do not point into ``x``.
    source = """
for lane in cutlass.range_constexpr(2):
    for vec in cutlass.range_constexpr(4):
        index = offset + lane * 4 + vec
        ptr = x.iterator + index
        mask = index < n
        value = ptr.load()
        acc = acc + (value if mask else 0.0)
acc = _cute_grouped_reduce_shared_two_stage(acc, smem_groups, 1, 64)
for lane in cutlass.range_constexpr(2):
    for vec in cutlass.range_constexpr(4):
        index = offset + lane * 4 + vec
        ptr = x.iterator + index
        mask = index < n
        value = ptr.load()
        total = total + (value * acc if mask else 0.0)
"""
    assert "value = _s0[lane * 4 + vec]" in _stash(source)


def test_a_later_load_matches_the_first_of_two_identical_earlier_loads() -> None:
    # The earlier pass loads the same text in two sibling loops, the second
    # from a base moved on between them (its slice holds both bindings); the
    # later pass loads what the first loop loads.
    source = """
for lane in cutlass.range_constexpr(2):
    base = offset + lane * 4
    for vec in cutlass.range_constexpr(4):
        value = (x.iterator + base + vec).load()
        acc = acc + value
    base = base + 64
    for vec in cutlass.range_constexpr(4):
        value = (x.iterator + base + vec).load()
        acc = acc + value * 2.0
acc = acc * 3.0
for lane in cutlass.range_constexpr(2):
    base = offset + lane * 4
    for vec in cutlass.range_constexpr(4):
        value = (x.iterator + base + vec).load()
        total = total + value * acc
"""
    code = _stash(source)
    assert code.count("value = (x.iterator + base + vec).load()") == 2, code
    assert "value = _s0[lane * 4 + vec]" in code, code


@pytest.mark.parametrize(
    ("call", "reused"),
    [
        ("_opaque(p)", False),
        (
            (
                "_cute_inline_asm_elementwise(p, acc, asm='st.global.f32 [$0], $1;',"
                " constraints='l,f', dtype=cutlass.Float32, is_pure=False)"
            ),
            False,
        ),
        # Declared pure: lowered without side effects.
        (
            (
                "r = _cute_inline_asm_elementwise(p, acc, asm='add.f32 $0, $1, $2;',"
                " constraints='=f,l,f', dtype=cutlass.Float32, is_pure=True)"
            ),
            True,
        ),
    ],
    ids=["opaque_call", "impure_asm", "pure_asm"],
)
def test_a_call_writing_through_a_loaded_address_keeps_the_later_load(
    call: str, reused: bool
) -> None:
    # The address is loaded from memory, a value the pointer analysis does
    # not follow: a call that may write memory it does not name may write
    # what the loads read, whatever it is handed.
    between = f"p = (addresses.iterator + tid).load()\n{call}"
    code = _stash(_TWO_PASSES.format(between=between, stride=64))
    assert ("_s0" in code) is reused


def test_a_stride_scaled_index_reaches_a_reduction_without_pointing_anywhere() -> None:
    # A layout read scales the index; the mask computed from it masks the
    # accumulator a CTA-wide reduction between the passes is handed.  The
    # index points nowhere, so neither do the mask and the accumulator.
    source = """
for lane in cutlass.range_constexpr(2):
    for vec in cutlass.range_constexpr(4):
        index = offset + (lane * 4 + vec) * x.layout.stride[1]
        mask = index < n
        value = (x.iterator + index).load()
        acc = acc + (value if mask else 0.0)
acc = _cute_grouped_reduce_shared_two_stage(acc, smem_groups, 1, 64)
for lane in cutlass.range_constexpr(2):
    for vec in cutlass.range_constexpr(4):
        index = offset + (lane * 4 + vec) * x.layout.stride[1]
        mask = index < n
        value = (x.iterator + index).load()
        total = total + (value * acc if mask else 0.0)
"""
    assert "value = _s0[lane * 4 + vec]" in _stash(source)


@pytest.mark.parametrize(
    ("setup", "between"),
    [
        (
            (
                "p = cute.arch.alloc_smem(cutlass.Float32, 64)\n"
                "smem = cute.make_tensor(p, layout)"
            ),
            "_opaque(p)",
        ),
        (
            "smem = cute.make_tensor(ptr, layout)\nsm2 = cute.make_tensor(ptr, layout)",
            "cute.copy(frag, sm2)",
        ),
        (
            (
                "p = cute.arch.alloc_smem(cutlass.Float32, 64)\n"
                "q = p + 4\n"
                "smem = cute.make_tensor(q, layout)"
            ),
            "_opaque(p)",
        ),
        (
            (
                "p = cute.arch.alloc_smem(cutlass.Float32, 64)\n"
                "smem = cute.make_tensor(p.align(16), layout)"
            ),
            "_opaque(p)",
        ),
        (
            "whole = cute.make_tensor(ptr, layout)\nsmem = whole[None, 0]",
            "cute.copy(frag, whole)",
        ),
    ],
    ids=[
        "the_tensors_pointer",
        "a_tensor_on_the_same_pointer",
        "a_pointer_offset_into",
        "a_pointer_aligned_into",
        "the_tensor_viewed",
    ],
)
def test_a_tensor_sharing_a_pointer_keeps_the_later_load(
    setup: str, between: str
) -> None:
    # The passes load ``smem``; a call handed the pointer it was made on (or
    # computed from), another tensor made on that pointer or the tensor
    # ``smem`` views may write what they read.
    source = setup + _TWO_PASSES.replace("x.iterator", "smem.iterator")
    assert "_s0" in _stash(source.format(between="acc = acc * 2.0", stride=64))
    assert "_s0" not in _stash(source.format(between=between, stride=64))


def test_a_register_tensor_the_load_reads_keeps_the_later_load() -> None:
    # Each pass fills its own register tensor; the later pass's loads read
    # what its own earlier iterations stored there.
    source = """
for lane in cutlass.range_constexpr(2):
    buf = cute.make_rmem_tensor(8, cutlass.Float32)
    for vec in cutlass.range_constexpr(4):
        value = (buf.iterator + vec).load()
        acc = acc + value
for lane in cutlass.range_constexpr(2):
    buf = cute.make_rmem_tensor(8, cutlass.Float32)
    for vec in cutlass.range_constexpr(4):
        value = (buf.iterator + vec).load()
        buf[vec + 1] = value * 2.0
"""
    assert "_s0" not in _stash(source)


def test_a_register_tensor_the_loads_do_not_read_may_be_written() -> None:
    source = _TWO_PASSES.format(
        between="scratch = cute.make_rmem_tensor(8, cutlass.Float32)", stride=64
    ).replace("total = total + value * acc", "scratch[vec] = value * acc")
    assert "value = _s0[lane * 4 + vec]" in _stash(source)


def test_three_passes_read_the_first_passs_loads() -> None:
    source = _TWO_PASSES.format(between="acc = acc * 2.0", stride=64)
    third = source[source.index("for lane", source.index("acc = acc * 2.0")) :]
    code = _stash(source + "total = total * 0.5\n" + third)
    assert code.count("(x.iterator + index).load()") == 1
    assert code.count("value = _s0[lane * 4 + vec]") == 2


def test_the_stashes_of_a_span_fit_a_register_budget() -> None:
    # Each 16-byte packet of a pass is eight elements of four registers: the
    # first two stashes (64 registers) fit, the third keeps its loads.
    loads = "\n".join(
        f"        p{i} = cute.arch.load(t{i}.iterator + index, "
        f"ir.VectorType.get([8], cutlass.Uint16.mlir_type))"
        for i in range(3)
    )
    source = f"""
for lane in cutlass.range_constexpr(2):
    for vec in cutlass.range_constexpr(4):
        index = offset + (cutlass.Int32(lane) * 4 + cutlass.Int32(vec)) * 8
{loads}
acc = acc * 2.0
for lane in cutlass.range_constexpr(2):
    for vec in cutlass.range_constexpr(4):
        index = offset + (cutlass.Int32(lane) * 4 + cutlass.Int32(vec)) * 8
{loads}
"""
    code = _stash(source)
    assert code.count("_s0.append(p0)") == 1
    assert code.count("_s1.append(p1)") == 1
    assert "_s2" not in code
    assert code.count("p2 = cute.arch.load(") == 2


def test_stashes_inside_and_around_a_pass_share_the_register_budget() -> None:
    # Inside the first pass, two loops over ``vec`` share three 16-byte
    # packet loads (16 registers each over four iterations); around them, the
    # passes would share ``q`` (32 registers over eight), past the thread's
    # 64 with those live: ``q`` keeps its loads.
    def loads(*names: str) -> str:
        return "\n".join(
            f"        {name} = cute.arch.load(t{name}.iterator + index, "
            "ir.VectorType.get([8], cutlass.Uint16.mlir_type))"
            for name in names
        )

    index = (
        "        index = offset + (cutlass.Int32(lane) * 4 + cutlass.Int32(vec)) * 8"
    )
    source = f"""
for lane in cutlass.range_constexpr(2):
    for vec in cutlass.range_constexpr(4):
{index}
{loads("p0", "p1", "p2", "q")}
    acc = acc * 2.0
    for vec in cutlass.range_constexpr(4):
{index}
{loads("p0", "p1", "p2")}
total = total * 0.5
for lane in cutlass.range_constexpr(2):
    for vec in cutlass.range_constexpr(4):
{index}
{loads("q")}
"""
    code = _stash(source)
    assert all(f"_s{i}.append(p{i})" in code for i in range(3))
    assert "_s3" not in code
    assert code.count("q = cute.arch.load(") == 2


def test_a_statement_placed_in_two_passes_is_rewritten_in_one() -> None:
    # The lane split may put one statement object in both passes.
    body = ast.parse(_TWO_PASSES.format(between="acc = acc * 2.0", stride=64)).body
    first, second = (stmt for stmt in body if isinstance(stmt, ast.For))
    first_vec, second_vec = first.body[1], second.body[1]
    assert isinstance(first_vec, ast.For) and isinstance(second_vec, ast.For)
    second_vec.body[1] = first_vec.body[1]
    code = _stash(body)
    assert code.count("(x.iterator + index).load()") == 1
    assert "value = _s0[lane * 4 + vec]" in code


@pytest.mark.parametrize(
    ("between", "stride"),
    [
        # A store between the passes may write what the loads read.
        ("(y.iterator + offset).store(acc)", 64),
        # A call that is not a known load or pure computation is handed x.
        ("_opaque(x, acc)", 64),
        # The later pass reads other addresses.
        ("acc = acc * 2.0", 128),
        # The base the loads add to changes between the passes.
        ("offset = offset + 1", 64),
        # A call that is not a known load or pure computation is handed y,
        # which may share memory with x.
        ("_opaque(y)", 64),
        # The same call handed a name pointing into y.
        ("p = y.iterator + offset\n_opaque(p)", 64),
        # ... bound by a tuple assignment.
        ("p, q = (y.iterator + offset, 0)\n_opaque(p)", 64),
        # A store through a subscript or an attribute of a name: a view of
        # a tensor that may be made on x's pointer, of y, of an object.
        ("v[None, 0][1] = acc", 64),
        ("y[0][1] = acc", 64),
        ("st.buf[0] = acc", 64),
    ],
)
def test_later_pass_keeps_its_load(between: str, stride: int) -> None:
    code = _stash(_TWO_PASSES.format(between=between, stride=stride))
    assert code.count("(x.iterator + index).load()") == 2
    assert "_s0" not in code


@pytest.mark.parametrize(
    "between",
    [
        "sp = smem.iterator + tid\n_store_helper(sp, acc)",
        "p = smem.iterator\n_opaque(p)",
    ],
    ids=["store_helper", "opaque_call"],
)
def test_a_pointer_into_a_local_tensor_keeps_the_later_load(between: str) -> None:
    # The passes load a shared-memory tensor, not a kernel argument; a call
    # handed a pointer into it may write what they read.
    source = _TWO_PASSES.replace("x.iterator", "smem.iterator")
    assert "_s0" in _stash(source.format(between="acc = acc * 2.0", stride=64))
    assert "_s0" not in _stash(source.format(between=between, stride=64))


@pytest.mark.parametrize(
    "header",
    [
        "if cute.arch.atomic_add(y.iterator + index, 1.0) > 0:",
        "if _opaque(y, lane):",
        "for j in range(_opaque(y)):",
        "while _opaque(y):",
    ],
    ids=[
        "atomic_in_if_test",
        "call_in_if_test",
        "call_in_for_iter",
        "call_in_while_test",
    ],
)
def test_an_effect_in_a_pass_header_keeps_the_later_load(header: str) -> None:
    # A branch test or loop iterable inside the first pass writes memory too.
    source = _TWO_PASSES.format(between="acc = acc * 2.0", stride=64).replace(
        "        acc = acc + value\n",
        f"        acc = acc + value\n        {header}\n            acc = acc + 1.0\n",
    )
    code = _stash(source)
    assert code.count("(x.iterator + index).load()") == 2
    assert "_s0" not in code


def test_a_loop_before_the_path_may_bind_a_path_variable() -> None:
    # Each pass runs a loop over ``vec`` before the one holding the load,
    # which binds ``vec`` afresh: the reuse still holds.
    source = """
for lane in cutlass.range_constexpr(2):
    k = offset + cutlass.Int32(lane) * 64
    for vec in cutlass.range_constexpr(4):
        pre = vec * 2
    for vec in cutlass.range_constexpr(4):
        value = (x.iterator + k + vec).load()
        acc = acc + value
acc = acc * 2.0
for lane in cutlass.range_constexpr(2):
    k = offset + cutlass.Int32(lane) * 64
    for vec in cutlass.range_constexpr(4):
        pre = vec * 2
    for vec in cutlass.range_constexpr(4):
        value = (x.iterator + k + vec).load()
        total = total + value * acc
"""
    assert "value = _s0[lane * 4 + vec]" in _stash(source)


def test_a_store_in_a_pass_keeps_the_later_load() -> None:
    # The first pass's later lanes store after the element's load: anything in
    # either pass counts as running between the two loads of an element.
    source = _TWO_PASSES.format(between="acc = acc * 2.0", stride=64).replace(
        "        acc = acc + value\n",
        "        acc = acc + value\n        (y.iterator + index).store(acc)\n",
    )
    code = _stash(source)
    assert code.count("(x.iterator + index).load()") == 2
    assert "_s0" not in code


def _lane_base_strides(code: str) -> tuple[int, int]:
    """Element strides of the first lane loop's chunk base: per thread (along
    ``thread_idx``) and per lane."""
    base = next(
        node.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("lane_base_")
    )

    class Substitute(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            text = ast.unparse(node)
            if text == "cutlass.Int32(cute.arch.thread_idx()[0])":
                return ast.Name("thread", ast.Load())
            if text.startswith("cutlass.Int32(") and len(node.args) == 1:
                return self.visit(node.args[0])
            return self.generic_visit(node)

    expr = ast.Expression(Substitute().visit(base))
    names = {node.id for node in ast.walk(expr) if isinstance(node, ast.Name)}
    (lane,) = {name for name in names if name.startswith("lane_")}
    (offset,) = {name for name in names if name.startswith("tile_offset_")}
    compiled = compile(ast.fix_missing_locations(expr), "<lane_base>", "eval")

    def at(thread: int, lane_index: int) -> int:
        return eval(compiled, {}, {"thread": thread, lane: lane_index, offset: 0})

    return at(1, 0) - at(0, 0), at(0, 1) - at(0, 0)


def test_tile_rows_seed_the_two_warp_block() -> None:
    """``CuteTileVecHeuristic`` seeds Helion-Triton's 2048-element block over
    two warps beside its one-vector-per-thread block: four 8-wide vectors per
    thread, strided by the thread count so a warp's load reads 32 adjacent
    vectors (consecutive threads 16 bytes apart, not 64)."""
    bound = _bound()
    with _target():
        spec = bound.config_spec
        tile = spec.block_sizes[1].block_id
        seeds = {
            (
                spec.block_sizes.config_get(seed.config["block_sizes"], tile),
                spec.num_threads.config_get(seed.config["num_threads"], tile),
            ): seed
            for seed in spec.compiler_seed_configs
            if spec.cute_vector_widths.config_get(
                seed.config.get("cute_vector_widths", []), tile, 1
            )
            == 8
            and "cute_reduction_sequence" not in seed.config
        }
        assert (1024, 128) in seeds
        code = bound.to_code(spec.normalized_config(seeds[2048, 64]))
    assert _lane_base_strides(code) == (8, 512)


def test_tile_rows_without_a_lane_layout_slot_seed_one_vector_per_thread() -> None:
    """The two-warp block needs strided lanes: without the knob its lanes
    would stay blocked, so only the one-vector-per-thread block is seeded."""
    bound = _bound()
    with _target():
        spec = bound.config_spec
        host = bound.host_function
        assert host is not None
        tile = spec.block_sizes[1].block_id
        with (
            bound.env,
            host,
            patch.object(spec.cute_lane_layouts, "valid_block_ids", return_value=[]),
        ):
            seeds = CuteTileVecHeuristic.get_seed_configs(bound.env, host.device_ir)
    assert [
        spec.block_sizes.config_get(seed.config["block_sizes"], tile) for seed in seeds
    ] == [1024]


def test_a_stash_counts_the_registers_of_the_assignments_result() -> None:
    # Two lanes of each load: a bf16 packet of eight takes four registers per
    # lane, bitcast to Int32 too; an fp64 scalar takes two, a plain load
    # one, and two again converted to fp64.
    loop = ast.parse(
        """
for lane in cutlass.range_constexpr(2):
    a = cute.arch.load(p, ir.VectorType.get([8], cutlass.BFloat16.mlir_type))
    b = cute.arch.load(p, ir.VectorType.get([8], cutlass.BFloat16.mlir_type)).bitcast(cutlass.Int32)
    c = cute.arch.load(p, cutlass.Float64)
    d = (x.iterator + lane).load()
    e = cutlass.Float64((x.iterator + lane).load())
"""
    ).body[0]
    assert isinstance(loop, ast.For)
    registers = {
        ast.unparse(site.stmt.targets[0]): site.registers()
        for site in _load_sites(loop)
    }
    assert registers == {"a": 8, "b": 8, "c": 4, "d": 2, "e": 4}
