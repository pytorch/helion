from __future__ import annotations

import ast

import pytest

from helion._compiler.cute.lane_invariant_division import divide_vector_bases
from helion._testing.cute_index_math import flat_channel_code

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
) -> str:
    module = ast.parse(source)
    result = divide_vector_bases(
        module.body,
        constexpr_values=constexpr_values or {},
        rename_groups=rename_groups,
    )
    return ast.unparse(ast.Module(body=result, type_ignores=[]))


def _rewrite(
    body: str,
    *,
    hw: int = 784,
    step: int = 8,
    vector_loop: str = "cutlass.range_constexpr(8)",
    block_size: int = 8192,
) -> str:
    source = PROLOGUE.format(hw=hw, step=step, vector_loop=vector_loop, body=body)
    return _divide(source, {"_BLOCK_SIZE_0": block_size})


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
    from helion._compiler.cute.lane_invariant_division import _Rewriter

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
