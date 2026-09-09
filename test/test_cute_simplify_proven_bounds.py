from __future__ import annotations

import ast
from typing import cast

import pytest

from helion._compiler.ast_extension import convert
from helion._compiler.cute.simplify_proven_bounds import simplify_proven_bounds

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
    xyz_grid: bool = True,
    thread_block_dims: tuple[int, int, int] = (16, 8, 1),
    block_grid_dims: tuple[int | None, int | None, int | None] = (12, 256, 1),
    constexpr_values: dict[str, int] | None = None,
    proven_tensor_size_values: (dict[tuple[str, int], tuple[str, int]] | None) = None,
    extended: bool = False,
) -> str:
    module = ast.parse(source)
    if extended:
        module = cast("ast.Module", convert(module))
    body = simplify_proven_bounds(
        module.body,
        enabled=enabled,
        xyz_grid=xyz_grid,
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
        ({"xyz_grid": False}, "cute.arch.block_idx()[2]"),
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
