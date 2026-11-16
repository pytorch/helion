from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.ast_extension import convert
from helion._compiler.cute.passes import pipeline
from helion._compiler.cute.simplify_proven_bounds import simplify_proven_bounds
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_index_math import flat_channel_code
from helion._testing.cute_index_math import row_scale

if TYPE_CHECKING:
    from helion._compiler.device_function import DeviceFunction

pytestmark = pytest.mark.cpu_only

SOURCE = """
pid_0 = cutlass.Int32(cute.arch.block_idx()[0])
pid_1 = cutlass.Int32(cute.arch.block_idx()[1])
pid_2 = cutlass.Int32(cute.arch.block_idx()[2])
tile_offset_2 = pid_0
tile_offset_1 = pid_1
tile_offset_0 = pid_2 * _BLOCK_SIZE_0
state_index = index_tensor.load()
inactive = operator.lt(state_index, 0)
if inactive:
    output.store(cutlass.BFloat16(0))
else:
    producer_base = (
        (
            cutlass.Int32(cute.arch.thread_idx()[0])
            + cutlass.Int32(cute.arch.thread_idx()[1]) * 16
        )
        % 32
        * 4
    )
    a_vector = cute.arch.load(
        a.iterator
        + (
            producer_base + 128 * tile_offset_2
            if producer_base + 3 + 128 * tile_offset_2 < a_size_1
            else 0
        ),
        vector4,
    )
    for lane in cutlass.range_constexpr(4):
        feature = producer_base + lane + 128 * tile_offset_2
        a_value = a_vector[lane] if feature < a_size_1 else zero
        dt_value = dt[feature] if feature < dt_bias_size_0 else zero
    column = cutlass.Int32(cute.arch.thread_idx()[0]) * 8
    state_pair = (
        loaded_state
        if cutlass.Int32(state_index) < initial_state_size_0
        else zero_pair
    )
    if (
        cutlass.Int32(state_index) < initial_state_size_0
        and column < 128
        and column + 7 < 128
    ):
        store_state(state_pair)
    else:
        pass
"""


def _transform(
    source: str = SOURCE,
    *,
    enabled: bool = True,
    exact_grid: bool = True,
    thread_block_dims: tuple[int, int, int] | None = (16, 8, 1),
    block_grid_dims: tuple[int | None, int | None, int | None] = (12, 256, 1),
    constexpr_values: dict[str, int] | None = None,
    proven_tensor_size_values: (dict[tuple[str, int], tuple[str, int]] | None) = None,
    rename_groups: dict[str, str] | None = None,
    extended: bool = False,
) -> str:
    module = ast.parse(source)
    if extended:
        module = cast("ast.Module", convert(module))
    body = simplify_proven_bounds(
        module.body,
        enabled=enabled,
        exact_grid=exact_grid,
        thread_block_dims=thread_block_dims,
        block_grid_dims=block_grid_dims,
        constexpr_values=(
            {
                "_BLOCK_SIZE_0": 128,
                "_BLOCK_SIZE_1": 1,
                "_BLOCK_SIZE_2": 1,
            }
            if constexpr_values is None
            else constexpr_values
        ),
        proven_tensor_size_values=(
            {
                ("a", 1): ("a_size_1", 1536),
                ("dt", 0): ("dt_bias_size_0", 1536),
                ("state", 0): ("initial_state_size_0", 256),
            }
            if proven_tensor_size_values is None
            else proven_tensor_size_values
        ),
        rename_groups=rename_groups,
    )
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


@pytest.mark.parametrize("extended", [False, True])
def test_generated_full_row_bounds_are_removed_but_dynamic_state_guard_remains(
    extended: bool,
) -> None:
    result = _transform(extended=extended)

    assert result.count("_cute_proven_bounds_0_abi_version = 1") == 1
    assert "cute.arch.block_idx()[2]" not in result
    assert "< a_size_1" not in result
    assert "< dt_bias_size_0" not in result
    assert "if inactive:" in result
    assert result.count("cutlass.Int32(state_index) < initial_state_size_0") == 2
    assert "and column < 128" not in result
    assert "and column + 7 < 128" not in result
    assert "if cutlass.Int32(state_index) < initial_state_size_0:" in result


def test_unproven_launch_shape_returns_the_body_unchanged() -> None:
    # ``None`` (an argument-sized or otherwise unproven launch axis) proves
    # nothing; no ``(1, 1, 1)`` placeholder stands in for it.
    result = _transform(thread_block_dims=None)

    assert result == ast.unparse(ast.parse(SOURCE))
    assert "< a_size_1" in result
    assert "_cute_proven_bounds_0_abi_version" not in result


def test_missing_size_proof_retains_only_that_tensor_guard() -> None:
    result = _transform(
        proven_tensor_size_values={
            ("dt", 0): ("dt_bias_size_0", 1536),
            ("state", 0): ("initial_state_size_0", 256),
        }
    )

    assert "< a_size_1" in result
    assert "< dt_bias_size_0" not in result
    assert "cutlass.Int32(state_index) < initial_state_size_0" in result


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"enabled": False}, "cute.arch.block_idx()[2]"),
        ({"exact_grid": False}, "cute.arch.block_idx()[2]"),
        ({"block_grid_dims": (12, None, 1)}, "cute.arch.block_idx()[2]"),
        ({"block_grid_dims": (12, 256, 2)}, "cute.arch.block_idx()[2]"),
        ({"thread_block_dims": (32, 33, 1)}, "< a_size_1"),
    ],
)
def test_launch_proof_near_misses_retain_guards(
    kwargs: dict[str, object], needle: str
) -> None:
    result = _transform(**kwargs)  # type: ignore[arg-type]

    assert needle in result


@pytest.mark.parametrize(
    "source",
    [
        "a_size_1 = 1536\nvalue = x if thread < a_size_1 else zero\n",
        "_BLOCK_SIZE_0 = 128\nvalue = cute.arch.block_idx()[2]\n",
        "left = right = 0\nvalue = cute.arch.block_idx()[2]\n",
        "while keep_going:\n    value = cute.arch.block_idx()[2]\n",
    ],
)
def test_ambiguous_or_unsupported_reassignment_fails_closed(source: str) -> None:
    result = _transform(source)

    assert "_cute_proven_bounds_" not in result
    assert ast.dump(ast.parse(result), include_attributes=False) == ast.dump(
        ast.parse(source), include_attributes=False
    )


def test_loop_carried_value_is_not_used_to_remove_later_iteration_guard() -> None:
    result = _transform(
        """
x = 0
for lane in cutlass.range_constexpr(2):
    value = source[lane] if x < 1 else zero
    x = 10
"""
    )

    assert "if x < 1 else zero" in result
    assert "_cute_proven_bounds_" not in result


def test_unknown_loop_range_does_not_reuse_old_target_bound() -> None:
    result = _transform(
        """
lane = 0
for lane in runtime_range:
    value = source[lane] if lane < 1 else zero
"""
    )

    assert "if lane < 1 else zero" in result
    assert "_cute_proven_bounds_" not in result


def test_constexpr_loop_range_can_prove_each_iteration() -> None:
    result = _transform(
        """
for lane in cutlass.range_constexpr(4):
    value = source[lane] if lane < 4 else zero
"""
    )

    assert "if lane < 4 else zero" not in result
    assert "value = source[lane]" in result


def test_proven_true_if_is_flattened() -> None:
    result = _transform(
        """
column = cutlass.Int32(cute.arch.thread_idx()[0]) * 8
if column + 7 < 128:
    vector_store()
else:
    scalar_fallback()
"""
    )

    assert "_cute_proven_bounds_0_abi_version = 1" in result
    assert "vector_store()" in result
    assert "scalar_fallback()" not in result
    assert not any(isinstance(node, ast.If) for node in ast.walk(ast.parse(result)))


@pytest.mark.parametrize(
    "expression",
    [
        "base + 16 < tensor_size",
        "(thread << 1) < tensor_size",
        "thread - 32 < tensor_size",
        "thread + 32 < tensor_size",
    ],
)
def test_unsupported_overflow_or_out_of_range_math_is_not_proven(
    expression: str,
) -> None:
    source = f"""
base = 2147483640
thread = cutlass.Int32(cute.arch.thread_idx()[0])
value = source[thread] if {expression} else zero
"""
    result = _transform(
        source,
        proven_tensor_size_values={
            ("tensor", 0): ("tensor_size", 32),
        },
    )

    (conditional,) = [
        node for node in ast.walk(ast.parse(result)) if isinstance(node, ast.IfExp)
    ]
    assert ast.dump(conditional.test, include_attributes=False) == ast.dump(
        ast.parse(expression, mode="eval").body, include_attributes=False
    )
    assert "_cute_proven_bounds_" not in result


def test_conflicting_exact_names_fail_closed() -> None:
    result = _transform(
        "value = cute.arch.block_idx()[2]\n",
        constexpr_values={"size": 12},
        proven_tensor_size_values={("tensor", 0): ("size", 13)},
    )

    assert "cute.arch.block_idx()[2]" in result
    assert "_cute_proven_bounds_" not in result


def test_generated_abi_marker_does_not_clobber_an_existing_name() -> None:
    result = _transform(
        """
_cute_proven_bounds_0_abi_version = sentinel
value = cute.arch.block_idx()[2]
consumer = consume(_cute_proven_bounds_0_abi_version)
"""
    )

    assert "_cute_proven_bounds_1_abi_version = 1" in result
    assert "consumer = consume(_cute_proven_bounds_0_abi_version)" in result


CHANNEL_GATHER = """
pid_flat = cutlass.Int32(cute.arch.block_idx()[0])
offsets_base_0 = pid_flat * _BLOCK_SIZE_0
v_0 = 196
v_2 = 384
for lane_0 in cutlass.range_constexpr(4):
    lane_base_0 = offsets_base_0 + (
        cutlass.Int32(lane_0) * 256 + cutlass.Int32(cute.arch.thread_idx()[0])
    ) * 8
    for vec_lane_0 in cutlass.range_constexpr(8):
        indices_0 = lane_base_0 + cutlass.Int32(vec_lane_0)
        v_1 = indices_0 // v_0
        v_3 = _cute_python_mod(cutlass.Int32(v_1), cutlass.Int32(v_2))
        weight = w[v_3] if cutlass.Int32(v_3) < 384 else zero
        half_lane = lane_0 // 2
        unproven = loaded // 3
"""


@pytest.mark.parametrize("extended", [False, True])
def test_proven_channel_index_drops_its_guard_and_signed_corrections(
    extended: bool,
) -> None:
    """``(i // HW) % C`` of a flat NCHW index: the Python remainder bounds the
    channel below ``C``, so the gather's guard goes, and the division and
    remainder of the proven nonnegative index run unsigned (no sign
    corrections).  Trace-time Python ints and unproven values keep their
    operators."""
    result = _transform(
        CHANNEL_GATHER,
        thread_block_dims=(256, 1, 1),
        block_grid_dims=(1176, 1, 1),
        constexpr_values={"_BLOCK_SIZE_0": 8192},
        proven_tensor_size_values={},
        extended=extended,
    )

    assert "v_1 = cutlass.Int32(cutlass.Uint32(indices_0) // cutlass.Uint32(v_0))" in (
        result
    )
    assert (
        "v_3 = cutlass.Int32(cutlass.Uint32(cutlass.Int32(v_1)) % "
        "cutlass.Uint32(cutlass.Int32(v_2)))"
    ) in result
    assert "weight = w[v_3]\n" in result + "\n"
    assert "half_lane = lane_0 // 2" in result
    assert "unproven = loaded // 3" in result


def test_unproven_launch_keeps_the_channel_guard() -> None:
    result = _transform(
        CHANNEL_GATHER,
        thread_block_dims=(256, 1, 1),
        block_grid_dims=(None, 1, 1),
        constexpr_values={"_BLOCK_SIZE_0": 8192},
        proven_tensor_size_values={},
    )

    assert "_cute_python_mod(cutlass.Int32(v_1), cutlass.Int32(v_2))" in result
    assert "if cutlass.Int32(v_3) < 384" in result


UNSIGNED_OPERANDS = """
thread = cutlass.Int32(cute.arch.thread_idx()[0])
wide = cutlass.Int64(thread)
narrow = thread
value = {expression}
"""


@pytest.mark.parametrize(
    ("expression", "unsigned"),
    [
        ("narrow // 4", True),
        ("cutlass.Int32(wide) % 4", True),
        # A wider or unsigned operand: the division's result is no Int32, so
        # the rewrite's Int32 would change its type.
        ("wide // 4", False),
        ("cutlass.Int64(thread) // 4", False),
        ("cutlass.Uint32(thread) % 4", False),
        ("thread // cutlass.Int64(4)", False),
        # A trace-time Python int stays one.
        ("12 // 4", False),
    ],
)
def test_unsigned_division_keeps_the_type_of_the_signed_one(
    expression: str, unsigned: bool
) -> None:
    result = _transform(UNSIGNED_OPERANDS.format(expression=expression))

    if unsigned:
        assert "value = cutlass.Int32(cutlass.Uint32(" in result, result
    else:
        assert f"value = {expression}" in result, result


def test_each_branch_starts_from_the_integer_kinds_before_the_if() -> None:
    """The ``else`` branch divides the Int64 bound before the ``if``, not the
    Int32 the ``if`` branch binds."""
    result = _transform(
        """
thread = cutlass.Int32(cute.arch.thread_idx()[0])
value = cutlass.Int64(thread)
if flag:
    value = thread
else:
    quotient = value // 4
"""
    )

    assert "quotient = value // 4" in result


@pytest.mark.parametrize(
    ("stop", "unsigned"), [("8", True), ("cutlass.Int64(8)", False)]
)
def test_a_staged_loop_counts_in_the_type_of_its_bounds(
    stop: str, unsigned: bool
) -> None:
    result = _transform(f"stop = {stop}\nfor k in range(stop):\n    half = k // 2\n")

    assert ("half = k // 2" not in result) is unsigned, result


def test_a_python_bound_past_int32_widens_a_staged_counter() -> None:
    """The counter's only value, 2**30, fits Int32, but the Python stop 2**32
    makes the DSL count in Int64: an unsigned Int32 half would narrow it."""
    result = _transform(
        "for k in range(1073741824, 4294967296, 4294967296):\n    half = k // 2\n"
    )

    assert "half = k // 2" in result, result


@pytest.mark.parametrize(
    "update",
    [
        # The loop carries ``d``: its second iteration divides by -1.
        "for i in range(2):\n    q = col // d\n    d_1 = d - 9\n",
        # Straight-line code reads the updated ``d`` once renamed.
        "d_1 = d - 9\nq = col // d\n",
    ],
)
@pytest.mark.parametrize("merged", [False, True])
def test_a_value_updated_under_an_alias_counts_as_updated(
    update: str, merged: bool
) -> None:
    """``d_1`` is ``d`` once ``ast_rename`` merges them, so writing it writes
    ``d``: ``col // d`` keeps its signed division (column 7 over -1 is -7, not
    the unsigned 0).  Unmerged, ``d`` stays 8 and the division runs
    unsigned."""
    result = _transform(
        "col = cutlass.Int32(cute.arch.thread_idx()[0])\nd = cutlass.Int32(8)\n"
        + update,
        rename_groups={"d": "d", "d_1": "d"} if merged else {},
    )

    assert ("q = col // d" in result) is merged, result


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_unsigned_division_needs_int32_indices(index_dtype: torch.dtype) -> None:
    """``(tile.index // 8) * 2**20`` with int64 indexing multiplies an Int64
    quotient; an unsigned Int32 one would wrap from ``tile.index`` 2**14 on."""
    bound = cpu_bind(row_scale[index_dtype], (torch.empty(65536),))
    config = bound.config_spec.normalized_config(
        helion.Config(block_sizes=[1024], num_threads=[128], cute_proven_bounds=True)
    )
    code = bound.to_code(config)
    assert ("cutlass.Uint32(" in code) is (index_dtype == torch.int32), code


@pytest.mark.parametrize("proven", [False, True])
def test_flat_launch_proves_its_grid(proven: bool) -> None:
    """A flat one-dimensional launch has an exact grid too: with the knob on,
    the channel gather loses its guard and its index math runs unsigned."""
    code = flat_channel_code(14, proven_bounds=proven)
    assert ("< 24 else" in code) is not proven, code
    assert ("cutlass.Uint32(" in code) is proven, code


def test_a_host_wrapper_launch_proves_no_flat_grid() -> None:
    """A host wrapper plan launches its own ``grid_x``, so the flat program
    ids' static grid proves nothing about ``block_idx``.  The plan is
    registered only while the grid is proved."""
    exact_flat_grid = pipeline._exact_flat_grid

    def with_wrapper_plan(fn: DeviceFunction) -> int | None:
        plans = pipeline.cute_codegen_state(fn.codegen).wrapper_plans
        plans.append({"kind": "own_grid"})
        try:
            return exact_flat_grid(fn)
        finally:
            plans.pop()

    with patch.object(pipeline, "_exact_flat_grid", with_wrapper_plan):
        code = flat_channel_code(14, proven_bounds=True)
    assert "< 24 else" in code, code
    assert "cutlass.Uint32(" not in code, code
