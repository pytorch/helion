from __future__ import annotations

import ast
import itertools
import operator
import re
import types

import pytest
import torch

import helion
from helion._compiler.ast_extension import convert
from helion._compiler.cute.lane_invariant_division import _Rewriter
from helion._compiler.cute.lane_invariant_division import divide_vector_bases
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_index_math import flat_channel_code
from helion._testing.cute_index_math import flat_channel_scale
from helion._testing.cute_index_math import split_row_code
from helion._testing.cute_index_math import split_row_config
from helion._testing.cute_index_math import split_row_input
from helion._testing.cute_index_math import split_row_moments
from helion._testing.cute_index_math import split_row_moments_dynamic
import helion.language as hl

pytestmark = pytest.mark.cpu_only

PROLOGUE = """
pid_flat = cutlass.Int32(cute.arch.block_idx()[0])
offsets_base_0 = pid_flat * _BLOCK_SIZE_0
hw = {hw}
for lane_0 in cutlass.range_constexpr(4):
    lane_base_0 = offsets_base_0 + (
        cutlass.Int32(lane_0) * 256 + cutlass.Int32(cute.arch.thread_idx()[0])
    ) * {step}
    for vec_lane_0 in {vector_loop}:
        offsets_0 = lane_base_0 + cutlass.Int32(vec_lane_0)
        indices_0 = offsets_0
{body}
"""


def _divide(
    source: str,
    constexpr_values: dict[str, int] | None = None,
    rename_groups: dict[str, str] | None = None,
    *,
    compare_bases: bool = False,
    extended: bool = False,
) -> str:
    module = ast.parse(source)
    if extended:
        # Codegen's nodes: subclasses of the ast classes (``ExtendedAST``).
        module = convert(module)
        assert isinstance(module, ast.Module)
    result = divide_vector_bases(
        module.body,
        constexpr_values=constexpr_values or {},
        rename_groups=rename_groups,
        compare_bases=compare_bases,
    )
    return ast.unparse(ast.Module(body=result, type_ignores=[]))


def _rewrite(
    body: str,
    *,
    hw: int = 784,
    step: int = 8,
    vector_loop: str = "cutlass.range_constexpr(8)",
    block_size: int = 8192,
    compare_bases: bool = False,
    extended: bool = False,
) -> str:
    source = PROLOGUE.format(hw=hw, step=step, vector_loop=vector_loop, body=body)
    return _divide(
        source,
        {"_BLOCK_SIZE_0": block_size},
        compare_bases=compare_bases,
        extended=extended,
    )


QUOTIENT = "        channel = indices_0 // hw\n"


def test_a_divisor_multiple_of_the_vector_divides_the_vector_base() -> None:
    """784 = 98 * 8 and every lane base is a multiple of 8: no lane of a vector
    crosses a multiple of 784, so all eight share ``lane_base_0 // 784``."""
    assert "channel = lane_base_0 // hw" in _rewrite(QUOTIENT)


@pytest.mark.parametrize(
    ("hw", "step", "vector_loop", "block_size"),
    [
        # 196 = 24.5 * 8: a vector may straddle a channel boundary.
        (196, 8, "cutlass.range_constexpr(8)", 8192),
        # Lane bases 4 apart are not multiples of the vector width.
        (784, 4, "cutlass.range_constexpr(8)", 8192),
        # An odd block size leaves the base's alignment unknown.
        (784, 8, "cutlass.range_constexpr(8)", 8191),
        # A staged loop is not an unrolled vector of lanes.
        (784, 8, "range(8)", 8192),
    ],
)
def test_unproven_identities_keep_the_per_lane_quotient(
    hw: int, step: int, vector_loop: str, block_size: int
) -> None:
    result = _rewrite(
        QUOTIENT, hw=hw, step=step, vector_loop=vector_loop, block_size=block_size
    )
    assert "channel = indices_0 // hw" in result


def test_a_rebound_base_is_not_divided() -> None:
    """The quotient reads the lane offset after its base was rebound: the
    recorded base no longer describes it."""
    result = _rewrite("        lane_base_0 = lane_base_0 + 1\n" + QUOTIENT)
    assert "channel = indices_0 // hw" in result


def test_remainders_and_other_lanes_are_left_alone() -> None:
    result = _rewrite(
        "        within = indices_0 % hw\n"
        "        lane_term = (cutlass.Int32(vec_lane_0) + 8) // hw\n"
    )
    assert "within = indices_0 % hw" in result
    assert "lane_term = (cutlass.Int32(vec_lane_0) + 8) // hw" in result


REMAINDER = (
    "        within = _cute_python_mod(cutlass.Int32(indices_0), cutlass.Int32(hw))\n"
)


@pytest.mark.parametrize(("hw", "divided"), [(784, True), (196, False)])
def test_a_python_remainder_is_the_base_remainder_plus_the_lane(
    hw: int, divided: bool
) -> None:
    """The base's remainder by 784 is a multiple of 8 below 784, so the lanes'
    remainders follow it one by one; by 196 a vector can wrap around."""
    result = _rewrite(REMAINDER, hw=hw)
    rewritten = (
        "within = _cute_python_mod(cutlass.Int32(lane_base_0), cutlass.Int32(hw))"
        " + vec_lane_0"
    )
    assert (rewritten in result) is divided, result


@pytest.mark.parametrize(("row_factor", "divided"), [("hw", True), ("4", False)])
def test_a_multiple_of_the_vector_plus_an_offset_shifts_its_base(
    row_factor: str, divided: bool
) -> None:
    """``row * 784 + offset``, the flat index of a de-linearized ``x[e // d, e
    % d]``, is the lane offset of ``row * 784 + lane_base``; a row stride of 4
    is no multiple of the vector width."""
    result = _rewrite(
        f"        flat = pid_flat * {row_factor} + indices_0\n"
        "        channel = flat // hw\n"
        "        within = _cute_python_mod(flat, hw)\n"
    )
    base = f"pid_flat * {row_factor} + lane_base_0"
    assert (f"channel = ({base}) // hw" in result) is divided, result
    assert (
        f"within = _cute_python_mod({base}, hw) + vec_lane_0" in result
    ) is divided, result


@pytest.mark.parametrize(
    ("comparison", "rewritten"),
    [
        ("indices_0 < 8192", "lane_base_0 < 8192"),
        ("indices_0 >= 8192", "lane_base_0 >= 8192"),
        ("8192 > indices_0", "8192 > lane_base_0"),
        ("8192 <= indices_0", "8192 <= lane_base_0"),
        ("operator.lt(indices_0, hw)", "operator.lt(lane_base_0, hw)"),
        ("operator.ge(indices_0, hw)", "operator.ge(lane_base_0, hw)"),
        ("operator.gt(hw, indices_0)", "operator.gt(hw, lane_base_0)"),
        ("operator.le(hw, indices_0)", "operator.le(hw, lane_base_0)"),
    ],
)
def test_a_comparison_with_a_multiple_of_the_vector_compares_the_base(
    comparison: str, rewritten: str
) -> None:
    """No lane of a vector based at a multiple of 8 reaches the next multiple
    of 8, so a tile mask against 8192 (or 784) answers alike for all eight.
    Without ``compare_bases`` (``cute_uniform_lane_compares``) each lane
    still compares its own index."""
    result = _rewrite(f"        inside = {comparison}\n", compare_bases=True)
    assert f"inside = {rewritten}" in result, result
    unchanged = _rewrite(f"        inside = {comparison}\n")
    assert f"inside = {comparison}" in unchanged, unchanged


def test_codegen_nodes_compare_the_base() -> None:
    """The kernel body codegen builds holds ``ExtendedAST`` subclasses of the
    ast node classes, comparison operators included."""
    result = _rewrite(
        "        inside = indices_0 < 8192\n"
        "        called = operator.ge(indices_0, hw)\n",
        compare_bases=True,
        extended=True,
    )
    assert "inside = lane_base_0 < 8192" in result, result
    assert "called = operator.ge(lane_base_0, hw)" in result, result


@pytest.mark.parametrize(
    "comparison",
    [
        # A bound a vector may straddle.
        "indices_0 < 8191",
        "operator.lt(indices_0, 8191)",
        # The base is at most 8192 while its last lane is above.
        "indices_0 <= 8192",
        "8192 > indices_0 + 1",
        "indices_0 == 8192",
        # A chain of comparisons.
        "0 <= indices_0 < 8192",
        # A float bound: 8192.5 lies inside a vector.
        "indices_0 < 8192.5",
        "indices_0 < loaded * 8",
    ],
)
def test_other_comparisons_keep_the_lane(comparison: str) -> None:
    result = _rewrite(f"        inside = {comparison}\n", compare_bases=True)
    assert f"inside = {comparison}" in result, result


@pytest.mark.parametrize(("hw", "rewritten"), [(784, True), (196, False)])
def test_a_remainder_bound_compares_the_base_remainder(
    hw: int, rewritten: bool
) -> None:
    """The bound of a de-linearized coordinate, ``e % 784 < 784``: the base's
    remainder is a multiple of 8, so the lanes' remainders stay below 784
    together.  By 196 the remainder is not rewritten and nor is its bound."""
    result = _rewrite(
        REMAINDER + "        inside = cutlass.Int32(within) < hw\n",
        hw=hw,
        compare_bases=True,
    )
    base_bound = (
        "inside = cutlass.Int32(_cute_python_mod(cutlass.Int32(lane_base_0), "
        "cutlass.Int32(hw))) < hw"
    )
    assert (base_bound in result) is rewritten, result


@pytest.mark.parametrize(("hw", "rewritten"), [(784, True), (196, False)])
def test_a_remainder_bound_compares_the_base_remainder_spelled_in_place(
    hw: int, rewritten: bool
) -> None:
    """The remainder inside the comparison: rewritten to the base's
    remainder plus the lane, an offset no name holds, whose base the
    comparison takes all the same."""
    remainder = "_cute_python_mod(cutlass.Int32({}), cutlass.Int32(hw))"
    result = _rewrite(
        f"        inside = {remainder.format('indices_0')} < hw\n",
        hw=hw,
        compare_bases=True,
    )
    assert (f"inside = {remainder.format('lane_base_0')} < hw" in result) is (
        rewritten
    ), result


def test_a_bound_cast_from_a_remainder_keeps_its_factor() -> None:
    """``cutlass.Int32(e % 784)`` of an 8-aligned base is a multiple of 8
    under the cast as well, so a lane compared with it compares the base."""
    result = _rewrite(
        "        bound = cutlass.Int32(_cute_python_mod(cutlass.Int32(lane_base_0), 784))\n"
        "        inside = indices_0 < bound\n",
        compare_bases=True,
    )
    assert "inside = lane_base_0 < bound" in result, result


NESTED = """
base = cutlass.Int32(raw) * 8
for v in cutlass.range_constexpr(8):
    idx = base + v
    within = _cute_python_mod(idx, 64)
    row = within // 32
    col = _cute_python_mod(within, 32)
"""


@pytest.mark.parametrize("compare_bases", [False, True])
def test_a_remainder_has_a_factor_only_under_the_knob(compare_bases: bool) -> None:
    """``(e % 64) // 32`` and ``(e % 64) % 32`` of an 8-aligned lane offset:
    the remainder's base is a multiple of 8, but without ``compare_bases`` a
    remainder has factor 1 as before, so nothing below it is rewritten and
    the code is the knob-off code."""
    result = _divide(NESTED, compare_bases=compare_bases)
    assert "within = _cute_python_mod(base, 64) + v" in result, result
    divided = (
        "row = _cute_python_mod(base, 64) // 32" in result,
        "col = _cute_python_mod(_cute_python_mod(base, 64), 32) + v" in result,
    )
    assert divided == (compare_bases, compare_bases), result


COMPARED = """
base = cutlass.Int32(raw) * 8
bound = cutlass.Int32(limit) * 8
for v in cutlass.range_constexpr(8):
    idx = base + v
    within = _cute_python_mod(idx, 24)
    below = idx < bound
    at_or_above = idx >= bound
    above_bound = bound > idx
    bound_below = bound <= idx
    called = (
        operator.lt(idx, bound),
        operator.ge(idx, bound),
        operator.gt(bound, idx),
        operator.le(bound, idx),
    )
    in_row = within < 24
    in_place = _cute_python_mod(idx, 24) < 24
    cast_bound = cutlass.Int32(_cute_python_mod(base, 24))
    below_cast = idx < cast_bound
    seen.append(
        (below, at_or_above, above_bound, bound_below, called, in_row, in_place)
    )
    seen.append(below_cast)
"""


def test_rewritten_comparisons_answer_like_the_per_lane_ones() -> None:
    """Run the vector loop before and after the rewrite with Python ints for
    the DSL scalars, over negative and positive multiples of 8."""
    rewritten = _divide(COMPARED, compare_bases=True)
    assert "below = base < bound" in rewritten, rewritten
    assert "in_row = _cute_python_mod(base, 24) < 24" in rewritten, rewritten
    assert "in_place = _cute_python_mod(base, 24) < 24" in rewritten, rewritten
    assert "below_cast = base < cast_bound" in rewritten, rewritten
    fake_cutlass = types.SimpleNamespace(range_constexpr=range, Int32=int)
    for raw, limit in itertools.product(range(-3, 4), repeat=2):
        results = []
        for source in (COMPARED, rewritten):
            namespace = {
                "cutlass": fake_cutlass,
                "operator": operator,
                "_cute_python_mod": operator.mod,
                "raw": raw,
                "limit": limit,
                "seen": [],
            }
            exec(source, namespace)
            results.append(namespace["seen"])
        assert results[0] == results[1], (raw, limit)


STAGED = """
for tile_offset in range(
    cutlass.Int32({start}), cutlass.Int32(1024), cutlass.Int32(_BLOCK_SIZE_0)
):
    lane_base = tile_offset + cutlass.Int32(cute.arch.thread_idx()[0]) * 8
    for v in cutlass.range_constexpr(8):
        idx = lane_base + cutlass.Int32(v)
        channel = idx // 784
"""


@pytest.mark.parametrize(("start", "divided"), [(0, True), (4, False)])
def test_a_staged_loop_counter_has_the_factor_of_its_start_and_step(
    start: int, divided: bool
) -> None:
    """A device loop's tile offset steps by the block size from its start."""
    result = _divide(STAGED.format(start=start), {"_BLOCK_SIZE_0": 128})
    assert ("channel = lane_base // 784" in result) is divided, result


PIPELINED = """
next_base = cutlass.Int32(cute.arch.thread_idx()[0]) * 8
for tile_offset in range(
    cutlass.Int32(0), cutlass.Int32(1024), cutlass.Int32(_BLOCK_SIZE_0)
):
    lane_base = next_base
    next_base = (
        tile_offset
        + cutlass.Int32(_BLOCK_SIZE_0)
        + cutlass.Int32(cute.arch.thread_idx()[0]) * {stride}
    )
    for v in cutlass.range_constexpr(8):
        idx = lane_base + cutlass.Int32(v)
        channel = idx // 784
"""


@pytest.mark.parametrize(("stride", "divided"), [(8, True), (4, False)])
def test_a_lane_base_carried_across_iterations_keeps_its_factor(
    stride: int, divided: bool
) -> None:
    """A software-pipelined device loop computes the next iteration's lane
    base before the vector loop of this one: a multiple of 8 on entry and on
    every update is one on every iteration; a multiple of 4 on update is not."""
    result = _divide(PIPELINED.format(stride=stride), {"_BLOCK_SIZE_0": 128})
    assert ("channel = lane_base // 784" in result) is divided, result


def _nested_pipelined(depth: int) -> str:
    """``depth`` staged loops, each advancing the lane base by 8, around a
    vector loop that divides it."""
    lines = ["base = cutlass.Int32(cute.arch.thread_idx()[0]) * 8"]
    indent = ""
    for level in range(depth):
        lines += [
            f"{indent}for step_{level} in range(4):",
            f"{indent}    base = base + 8",
        ]
        indent += "    "
    lines += [
        f"{indent}for v in cutlass.range_constexpr(8):",
        f"{indent}    idx = base + cutlass.Int32(v)",
        f"{indent}    channel = idx // 784",
    ]
    return "\n".join(lines)


def test_nested_carried_loops_pass_over_their_bodies_a_linear_number_of_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each loop finds the facts it carries in passes over its own body,
    where a nested loop takes its carried facts as first found instead of
    finding them again: six nested loops carrying the base pass over loop
    bodies fewer than three times each (not once per enclosing pass),
    and the vector still divides its base."""
    passes = 0
    iteration = _Rewriter.iteration

    def counted(self: _Rewriter, loop: ast.For, counter: object) -> None:
        nonlocal passes
        passes += 1
        iteration(self, loop, counter)

    monkeypatch.setattr(_Rewriter, "iteration", counted)
    result = _divide(_nested_pipelined(6))
    assert "channel = base // 784" in result, result
    assert passes < 3 * 6, passes


@pytest.mark.parametrize("cast", ["cutlass.Int64", "cutlass.Uint32"])
def test_the_dividend_keeps_its_cast(cast: str) -> None:
    """Only the lane-offset name under the cast is replaced: the quotient keeps
    its type and signedness."""
    result = _rewrite(f"        channel = {cast}(indices_0) // hw\n")
    assert f"channel = {cast}(lane_base_0) // hw" in result


VECTOR = """
pid_flat = cutlass.Int32(cute.arch.block_idx()[0])
base = pid_flat * 8
hw = 784
{source}
"""
LANES = """
for v in cutlass.range_constexpr(8):
    idx = base + cutlass.Int32(v)
    channel = idx // hw
"""


@pytest.mark.parametrize(
    ("source", "divided"),
    [
        (LANES, True),
        # A staged loop updates the base after the vector loop: the next
        # iteration's base is no multiple of 8.
        (
            "for step in range(4):\n"
            + LANES.replace("\n", "\n    ")
            + "base = base + 1\n",
            False,
        ),
        # So does the vector loop itself, for the next lane.
        (LANES + "    base = base + 1\n", False),
        # The base is a multiple of 8 on one branch only.
        ("base = pid_flat + 1\nif flag:\n    base = pid_flat * 8\n" + LANES, False),
        # An augmented assignment (a statement the pass does not model).
        ("base += 1\n" + LANES, False),
        # The lane no longer counts 0..7.
        (LANES.replace("    idx", "    v = v * 4\n    idx"), False),
        # 7 divides the base and 784, but is no power of two: a wrapped base
        # need not stay a multiple of it.
        ("base = pid_flat * 56\n" + LANES.replace("(8)", "(7)"), False),
    ],
)
def test_facts_hold_for_every_iteration_and_branch(source: str, divided: bool) -> None:
    result = _divide(VECTOR.format(source=source))
    assert ("channel = base // hw" in result) is divided, result


STAGED_LANES = "for step in range(4):\n" + LANES.replace("\n", "\n    ")


@pytest.mark.parametrize(
    ("source", "rename_groups"),
    [
        # The staged loop carries the base: the next iteration's base is no
        # multiple of 8.
        (
            STAGED_LANES + "base_before_loop_join = base + 1\n",
            {"base": "base", "base_before_loop_join": "base"},
        ),
        # It carries the divisor: the next iteration divides by 785.
        (STAGED_LANES + "hw_1 = hw + 1\n", {"hw": "hw", "hw_1": "hw"}),
        # The vector loop updates its lane: the lanes run 0, 4, ..., 28.
        (LANES.replace("    idx", "    v_1 = v * 4\n    idx"), {"v": "v", "v_1": "v"}),
    ],
)
@pytest.mark.parametrize("merged", [False, True])
def test_a_value_carried_under_an_alias_counts_as_reassigned(
    source: str, rename_groups: dict[str, str], merged: bool
) -> None:
    """``ast_rename`` merges the alias with the value it updates, so the outer
    loop reassigns that value and its facts at loop entry do not hold in
    later iterations, and a lane updated under an alias no longer counts
    0..7.  Unmerged, the loops leave them alone and the vector divides its
    base."""
    result = _divide(
        VECTOR.format(source=source), rename_groups=rename_groups if merged else None
    )
    assert ("channel = base // hw" in result) is not merged, result


def test_a_name_merged_with_the_loop_counter_carries_no_facts() -> None:
    """``j`` starts a multiple of 8 and each iteration leaves it ``i * 8``,
    but a rename that merges ``j`` with the counter ``i`` makes it the
    counter at the top of every iteration: ``base = j`` is ``i`` there, no
    multiple of 8, so the lanes keep their quotients."""
    source = """
pid_flat = cutlass.Int32(cute.arch.block_idx()[0])
hw = 784
j = pid_flat * 8
for i in range(1, 100):
    base = j
    for v in cutlass.range_constexpr(8):
        idx = base + cutlass.Int32(v)
        channel = idx // hw
    j = i * 8
"""
    assert "channel = base // hw" in _divide(source)
    merged = _divide(source, rename_groups={"i": "i", "j": "i"})
    assert "channel = base // hw" not in merged, merged


@pytest.mark.parametrize(
    ("base", "lane", "divided"),
    [
        # A Python int base and lane add up to a Python int, like the base.
        ("64", "v", True),
        # With an Int32 lane the sum is an Int32 the Python int base is not.
        ("64", "cutlass.Int32(v)", False),
        # A float multiple of 8 is no integer multiple of 8.
        ("loaded * 8", "cutlass.Int32(v)", False),
    ],
)
def test_only_an_integer_base_of_the_offsets_type_is_divided(
    base: str, lane: str, divided: bool
) -> None:
    result = _divide(
        f"base = {base}\nhw = 784\nfor v in cutlass.range_constexpr(8):\n"
        f"    idx = base + {lane}\n    channel = idx // hw\n"
    )
    assert ("channel = base // hw" in result) is divided, result


@pytest.mark.parametrize("side", [14, 28])
def test_flat_channel_gather_divides_once_per_vector(side: int) -> None:
    code = flat_channel_code(side)
    assert ("lane_base_0 // v_" in code) is (side == 28), code


@pytest.mark.parametrize("uniform_compares", [False, True])
def test_split_row_masks_compare_each_vector_once(uniform_compares: bool) -> None:
    """``split_row_moments`` reads rows of 192 (24 vectors of 8) through
    ``x[r // 192, r % 192, c]``, 4 splits of 288 in 128-row chunks.  Under
    ``cute_uniform_lane_compares`` the chunk's tile mask ``index < 288``, the
    split's ``r < 1152`` and the in-row bound ``r % 192 < 192`` compare each
    vector's lane base once; without it each lane compares its own index, in
    the code the config renders without the knob."""
    with mock_cuda_unavailable():
        code = split_row_code(uniform_compares=uniform_compares)
        if not uniform_compares:
            assert code == split_row_code(uniform_compares=None), code
    lane = "lane_base_2" if uniform_compares else "indices_2"
    flat = r"v_\d+ \+ lane_base_2" if uniform_compares else r"v_\d+"
    tile_masks = re.findall(r"mask_2 = .* and (\w+) < 288\n", code)
    assert tile_masks and set(tile_masks) == {lane}, code
    # ``tile_r.index < rows`` and ``r < total``, ``r = s * 288 + tile_r.index``.
    assert f"= operator.lt({lane}, v_" in code, code
    assert re.search(rf"= operator\.lt\({flat}, v_\d+\)\n", code), code
    in_row = re.compile(
        r"cutlass\.Int32\(_cute_python_mod\(cutlass\.Int32\(v_\d+ \+ lane_base_2\), "
        r"cutlass\.Int32\(v_\d+\)\)\) < 192"
    )
    assert bool(in_row.search(code)) is uniform_compares, code


def test_dynamic_shapes_keep_per_lane_compares() -> None:
    """Under dynamic shapes the split length and the row bounds are kernel
    arguments whose factors nothing proves, so the knob changes no compare:
    the identity is applied only where both multiples are proven."""
    with mock_cuda_unavailable():
        bound = cpu_bind(split_row_moments_dynamic, (split_row_input(6, 192),))
        off = bound.to_code(split_row_config(bound, uniform_compares=False))
        on = bound.to_code(split_row_config(bound, uniform_compares=True))
    assert on == off
    assert "and indices_2 < rows\n" in on, on


def test_vector_seeds_carry_uniform_compare_witnesses() -> None:
    """The first four compiler seeds with vector lanes carry knob-on
    witnesses (``flat_channel_scale``'s pointwise seeds, 8-wide)."""
    x = torch.empty(4, 24, 28, 28, dtype=torch.bfloat16)
    scale = torch.empty(24, dtype=torch.float32)
    with cpu_render_target():
        spec = cpu_bind(flat_channel_scale, (x, scale)).config_spec
    (group,) = [
        group
        for group in spec.compiler_coverage_groups
        if group.key == "cute_uniform_lane_compares"
    ]
    assert group.mechanism == "cute.uniform_lane_compares"
    assert group.domain == (False, True)
    assert len(group.witnesses) == 4
    for witness in group.witnesses:
        assert witness.value is True
        assert max(witness.carrier.config["cute_vector_widths"]) >= 2


@pytest.mark.parametrize(
    ("batch", "hw", "block", "threads"),
    [
        # Splits of 25088 rows: 128 threads over two 16-wide vectors each.
        (32, 3136, 4096, 128),
        # Splits of 288 rows: the chunk shrinks to the loop, then the threads.
        (6, 192, 512, 16),
    ],
)
def test_delinearized_rows_seed_one_compare_per_vector(
    batch: int, hw: int, block: int, threads: int
) -> None:
    """``cute_delinearized_row_reduction`` seeds a device loop that reduces
    rows reached through ``r % hw``: one (split, channel) per CTA, two
    32-byte vectors per thread, ``cute_lane_unroll`` and
    ``cute_uniform_lane_compares`` on at every shape.  The seed's chunk
    masks compare each vector's lane base."""
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(split_row_moments, (split_row_input(batch, hw),))
        spec = bound.config_spec
        assert spec.autotuner_heuristics == ["cute_delinearized_row_reduction"]
        (seed,) = spec.compiler_seed_configs
        assert seed.config["block_sizes"] == [1, 1, block]
        assert seed.config["num_threads"] == [1, 1, threads]
        assert seed.config["cute_vector_widths"] == [1, 1, 16]
        assert seed.config["cute_lane_unroll"] == 2
        assert seed.config["cute_uniform_lane_compares"] is True
        # The seed sets the knob itself: it carries no witness.
        assert not [
            group
            for group in spec.compiler_coverage_groups
            if group.key == "cute_uniform_lane_compares"
        ]
        code = bound.to_code(spec.normalized_config(seed))
    assert re.search(rf"mask_2 = .*lane_base_2 < {(batch * hw) // 4}\n", code), code


@pytest.mark.parametrize(
    ("batch", "hw", "dynamic", "capability", "block", "threads", "vec"),
    [
        # hw = 196 is a multiple of 4, not of 8 (splits of 1568 rows are
        # multiples of 32).
        (32, 196, False, (10, 0), 1024, 128, 4),
        # hw = 49 is odd: one-element lanes.
        (32, 49, False, (10, 0), 256, 128, 1),
        # A dynamic hw and split length are kernel arguments, whose factors
        # the lowering cannot see even where the input metadata pins them.
        (6, 192, True, (10, 0), 256, 128, 1),
        # Before SM100 a 32-byte vector is two 16-byte accesses.
        (32, 3136, False, (9, 0), 2048, 128, 8),
        # Splits of 4 rows: the chunk shrinks to the loop, then the vector.
        (1, 16, False, (10, 0), 4, 1, 2),
    ],
)
def test_delinearized_rows_seed_narrows_the_vector(
    batch: int,
    hw: int,
    dynamic: bool,
    capability: tuple[int, int],
    block: int,
    threads: int,
    vec: int,
) -> None:
    """The seed's vector halves until ``hw`` and the split offsets are
    static multiples of it, as the lowering proves a packet on the generated
    expression, and starts from 16 bytes before SM100; the vector renders
    as one packet load, and the lane unroll and the knob stay on."""
    kernel = split_row_moments_dynamic if dynamic else split_row_moments
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(kernel, (split_row_input(batch, hw),), capability=capability)
        spec = bound.config_spec
        assert spec.autotuner_heuristics == ["cute_delinearized_row_reduction"]
        (seed,) = spec.compiler_seed_configs
        assert seed.config["block_sizes"] == [1, 1, block]
        assert seed.config["num_threads"] == [1, 1, threads]
        assert seed.config["cute_vector_widths"] == [1, 1, vec]
        assert seed.config["cute_lane_unroll"] == 2
        assert seed.config["cute_uniform_lane_compares"] is True
        code = bound.to_code(spec.normalized_config(seed))
    assert ("cute.arch.load(" in code) is (vec > 1), code


def _split_row_sums_from_row_3(x: torch.Tensor) -> torch.Tensor:
    """``split_row_moments``'s sums with the row loop starting at row 3."""
    batch, hw, channels = x.shape
    total = batch * hw
    rows = (total + 3) // 4
    sums = torch.empty([4, channels], dtype=torch.float32, device=x.device)
    for tile_s, tile_c in hl.tile([4, channels]):
        acc = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        for tile_r in hl.tile(3, rows):
            r = tile_s.index[:, None] * rows + tile_r.index[None, :]
            valid = (r < total) & (tile_r.index[None, :] < rows)
            values = hl.load(
                x, [r // hw, r % hw, tile_c], extra_mask=valid[:, :, None]
            ).to(torch.float32)
            acc = acc + torch.sum(values, dim=1)
        sums[tile_s, tile_c] = acc
    return sums


def test_delinearized_rows_seed_takes_one_element_lanes_off_row_0() -> None:
    """A row loop starting at row 3 puts its lane bases off the multiples of
    the vector, where no packet loads: the seed takes one-element lanes."""
    kernel = helion.kernel(
        _split_row_sums_from_row_3, backend="cute", autotune_effort="none"
    )
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(kernel, (split_row_input(32, 3136),), capability=(10, 0))
        spec = bound.config_spec
        assert spec.autotuner_heuristics == ["cute_delinearized_row_reduction"]
        (seed,) = spec.compiler_seed_configs
        assert seed.config["cute_vector_widths"] == [1, 1, 1]


def test_delinearized_rows_seed_takes_one_element_lanes_off_aligned_rows() -> None:
    """A channel stride of 3137 (each NCHW plane padded by one element) puts
    the rows' starts off the multiples of the vector in memory, where no
    packet loads: the seed takes one-element lanes."""
    planes = torch.randn(32, 16, 3137).to(torch.bfloat16)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(
            split_row_moments,
            (planes[:, :, :3136].permute(0, 2, 1),),
            capability=(10, 0),
        )
        spec = bound.config_spec
        assert spec.autotuner_heuristics == ["cute_delinearized_row_reduction"]
        (seed,) = spec.compiler_seed_configs
        assert seed.config["cute_vector_widths"] == [1, 1, 1]


def test_delinearized_rows_seed_skips_a_size_one_dims_stride() -> None:
    """A batch of one may have any stride (7 here), which the lowering's
    packet proof skips, as the seed does: 16-wide lanes load packets."""
    x = torch.as_strided(
        torch.randn(16 * 3136).to(torch.bfloat16), (1, 3136, 16), (7, 1, 3136)
    )
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(split_row_moments, (x,), capability=(10, 0))
        spec = bound.config_spec
        assert spec.autotuner_heuristics == ["cute_delinearized_row_reduction"]
        (seed,) = spec.compiler_seed_configs
        assert seed.config["cute_vector_widths"] == [1, 1, 16]
        code = bound.to_code(spec.normalized_config(seed))
    assert "cute.arch.load(" in code, code


@pytest.mark.parametrize(("offset", "vec"), [(4, 4), (1, 1)])
def test_delinearized_rows_seed_narrows_the_vector_to_the_base(
    offset: int, vec: int
) -> None:
    """A view ``offset`` elements into its storage starts each row's packets
    off the multiples of a wider vector in memory, where the lowering loads
    no packet: the seed narrows the vector to the base address's alignment,
    and a 4-wide one loads packets."""
    storage = torch.randn(offset + 6 * 16 * 192).to(torch.bfloat16)
    x = storage[offset:].view(6, 16, 192).permute(0, 2, 1)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(split_row_moments, (x,), capability=(10, 0))
        spec = bound.config_spec
        assert spec.autotuner_heuristics == ["cute_delinearized_row_reduction"]
        (seed,) = spec.compiler_seed_configs
        assert seed.config["cute_vector_widths"] == [1, 1, vec]
        code = bound.to_code(spec.normalized_config(seed))
    assert ("cute.arch.load(" in code) is (vec > 1), code


def _split_row_sums_of_12_rows(x: torch.Tensor) -> torch.Tensor:
    """Sums of splits of 12 rows that start 16 rows apart."""
    batch, hw, channels = x.shape
    total = batch * hw
    splits = (total + 15) // 16
    sums = torch.empty([splits, channels], dtype=torch.float32, device=x.device)
    for tile_s, tile_c in hl.tile([splits, channels]):
        acc = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        for tile_r in hl.tile(12):
            r = tile_s.index[:, None] * 16 + tile_r.index[None, :]
            values = hl.load(
                x, [r // hw, r % hw, tile_c], extra_mask=(r < total)[:, :, None]
            ).to(torch.float32)
            acc = acc + torch.sum(values, dim=1)
        sums[tile_s, tile_c] = acc
    return sums


def test_delinearized_rows_seed_narrows_the_vector_to_the_loop() -> None:
    """``hw = 48`` and splits 16 rows apart are multiples of 16, but a packet
    needs the loop's 12 rows a multiple of its width as well, as the
    lowering does: the seed takes 4-wide lanes, which load packets."""
    kernel = helion.kernel(
        _split_row_sums_of_12_rows, backend="cute", autotune_effort="none"
    )
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(kernel, (split_row_input(4, 48),), capability=(10, 0))
        spec = bound.config_spec
        assert spec.autotuner_heuristics == ["cute_delinearized_row_reduction"]
        (seed,) = spec.compiler_seed_configs
        assert seed.config["cute_vector_widths"] == [1, 1, 4]
        code = bound.to_code(spec.normalized_config(seed))
    assert "cute.arch.load(" in code, code


def test_channels_last_rows_are_the_column_heuristics() -> None:
    """With the channels innermost (``[N, H*W, C]`` contiguous) the stride-1
    axis is the column tile's: the split column heuristic takes the kernel,
    not the de-linearized row one."""
    x = torch.empty(6, 192, 16, dtype=torch.bfloat16)
    with cpu_render_target():
        heuristics = cpu_bind(split_row_moments, (x,)).config_spec.autotuner_heuristics
    assert "cute_delinearized_row_reduction" not in heuristics, heuristics
    assert "cute_split_column_reduction" in heuristics, heuristics
