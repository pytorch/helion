from __future__ import annotations

import ast
import math
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import collective_matmul
from helion._compiler.cute.passes import pipeline
from helion._compiler.cute.proven_loop_bounds import exact_static_grid
from helion._compiler.cute.proven_loop_bounds import simplify_proven_loop_bounds
import helion._compiler.cute.simplify_proven_bounds as proven_bounds
from helion._compiler.cute.simplify_proven_bounds import simplify_proven_bounds
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _rewrite(source: str, *, threads: tuple[int, int, int] | None = (1, 2, 64)) -> str:
    return ast.unparse(
        ast.Module(
            body=simplify_proven_loop_bounds(
                ast.parse(source).body,
                enabled=True,
                thread_block_dims=threads,
                constexpr_values={"BLOCK": 64},
            ),
            type_ignores=[],
        )
    )


SOURCE = """
pid = cutlass.Int32(cute.arch.block_idx()[0]) // 16 % 2
offset = pid * BLOCK
end = BLOCK + BLOCK * (offset // BLOCK)
for k in range(cutlass.Int32(0), cutlass.Int32(end), cutlass.Int32(BLOCK)):
    for slot in cutlass.range_constexpr(4):
        flat = (cutlass.Int32(cute.arch.thread_idx()[1]) + cutlass.Int32(cute.arch.thread_idx()[2]) * 2 + slot * 128) * 8
        local = flat % BLOCK
        remat = BLOCK + BLOCK * (offset // BLOCK)
        if k + local < remat and k + local + 7 < remat:
            record(k, local, 8)
        else:
            for lane in cutlass.range_constexpr(8):
                if k + local + lane < remat:
                    record(k, local + lane, 1)
"""


def _execute(
    source: str, block: int, thread: tuple[int, int, int]
) -> list[tuple[int, int]]:
    output = []

    def record(k: int, local: int, width: int) -> None:
        output.extend((k, local + lane) for lane in range(width))

    namespace = {
        "BLOCK": 64,
        "cutlass": SimpleNamespace(
            Int32=int, Int64=int, Uint32=int, Uint64=int, range_constexpr=range
        ),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                block_idx=lambda: (block, 0, 0), thread_idx=lambda: thread
            )
        ),
        "record": record,
    }
    exec(compile(source, "<proof-oracle>", "exec"), namespace)
    return output


def test_general_pass_preserves_a_mask_if_its_input_is_phi_renamed() -> None:
    body = ast.parse(
        "row = 0\nfor i in range(2):\n    record(x[row] if row < 600 else -1)\n    row_next = 600"
    ).body
    state = MagicMock()
    ctx = state.ctx
    ctx.fn.cute_state.lanes.collective_mma_sites = ()
    ctx.config.config = {"cute_proven_bounds": True}
    ctx.exact_thread_block_dims = (1, 1, 1)
    ctx.constexpr_values = {}
    ctx.rename_groups = {"row_next": "row"}
    result = pipeline._proven_loop_bounds(state, body)
    source = ast.unparse(ast.Module(body=result, type_ignores=[]))
    values = []
    exec(source.replace("row_next", "row"), {"record": values.append, "x": [5] * 600})
    assert values == [5, -1]


def test_complete_tile_rematerialized_bounds_and_all_thread_coordinates() -> None:
    transformed = _rewrite(SOURCE)
    assert "else:" not in transformed
    assert "if k +" not in transformed
    for block in (0, 16, 31, 1023):
        for z in range(64):
            for y in range(2):
                assert _execute(transformed, block, (0, y, z)) == _execute(
                    SOURCE, block, (0, y, z)
                )


@pytest.mark.parametrize("pipeline", ["loop", "launch", "both"])
@pytest.mark.parametrize(
    "expression",
    [
        "7 and k < 2",
        "0 and k < 2",
        "prefix and k < 2",
        "k < 2 and prefix",
        "prefix and k < 2 and True",
        "not prefix and k < 2",
        "(prefix or 7) and k < 2",
        "(prefix and k < 2) or 9",
        "prefix and k < 2 and middle and k < 2",
        "prefix and (k < 2 and middle) and k < 2",
        "prefix and (k < 2 or middle) and k < 2",
    ],
)
def test_bound_simplification_preserves_boolean_operand_values(
    pipeline: str, expression: str
) -> None:
    source = f"for k in range(2):\n    record({expression})\n"
    body = ast.parse(source).body
    if pipeline in {"loop", "both"}:
        body = simplify_proven_loop_bounds(
            body,
            enabled=True,
            thread_block_dims=(1, 1, 1),
            constexpr_values={},
        )
    if pipeline in {"launch", "both"}:
        body = simplify_proven_bounds(
            body,
            enabled=True,
            exact_grid=True,
            thread_block_dims=(1, 1, 1),
            block_grid_dims=(1, 1, 1),
            constexpr_values={},
            proven_tensor_size_values={},
        )
    transformed = ast.unparse(ast.Module(body=body, type_ignores=[]))
    for prefix in (0, 7, 0.0, -0.0, 2.5, False, True):
        for middle in (0, 3, False, True):
            results = []
            for program in (source, transformed):
                output = []
                exec(
                    program,
                    {"prefix": prefix, "middle": middle, "record": output.append},
                )
                results.append(
                    [(type(value), value, math.copysign(1, value)) for value in output]
                )
            assert results[0] == results[1]


@pytest.mark.parametrize("extra", [1, 7, 31, 63])
def test_partial_tiles_keep_tail_predicates_and_execute_identically(extra: int) -> None:
    source = SOURCE.replace(
        "end = BLOCK + BLOCK * (offset // BLOCK)",
        f"end = BLOCK + BLOCK * (offset // BLOCK) + {extra}",
    ).replace("remat = BLOCK + BLOCK * (offset // BLOCK)", "remat = end")
    transformed = _rewrite(source)
    assert "else:" in transformed
    for block in (0, 16):
        for thread in ((0, 0, 0), (0, 1, 3), (0, 1, 63)):
            assert _execute(transformed, block, thread) == _execute(
                source, block, thread
            )


@pytest.mark.parametrize("assignment", ["end = 1", "end = runtime", "offset = runtime"])
def test_reassigned_stop_or_rematerialization_is_not_a_stale_proof(
    assignment: str,
) -> None:
    source = SOURCE.replace("    for slot", f"    {assignment}\n    for slot").replace(
        "remat = BLOCK + BLOCK * (offset // BLOCK)",
        "remat = end"
        if assignment.startswith("end")
        else "remat = BLOCK + BLOCK * (offset // BLOCK)",
    )
    transformed = _rewrite(source)
    assert "else:" in transformed


def test_snapshot_alias_keeps_original_stop_after_reassignment() -> None:
    source = (
        SOURCE.replace("for k in range", "snapshot = end\nfor k in range")
        .replace("    for slot", "    end = runtime\n    for slot")
        .replace("remat = BLOCK + BLOCK * (offset // BLOCK)", "remat = snapshot")
    )
    transformed = _rewrite(source)
    assert "else:" not in transformed


@pytest.mark.parametrize(
    "expression",
    [
        "cutlass.Int32(2147483640) + 64",
        "cutlass.Int64(2147483640) + 64",
        "cutlass.Int32(2147483640)",
        "cutlass.Int32(-1)",
        "cutlass.Int32(input_tensor.load())",
        "cutlass.Int32(runtime)",
    ],
)
def test_overflow_negative_and_memory_derived_end_fail_closed(expression: str) -> None:
    source = f"""
end = {expression}
for k in range(0, end, BLOCK):
    if k + 63 < end:
        vector()
    else:
        scalar()
"""
    assert "scalar()" in _rewrite(source)


@pytest.mark.parametrize(
    "source",
    [
        "for k in range(128, 0, -64):\n    out = yes if k + 63 < 128 else no\n",
        "for k in runtime_range:\n    out = yes if k < 1 else no\n",
        "k = 0\nfor k in runtime_range:\n    out = yes if k < 1 else no\n",
        "x = 0\nfor k in range(4):\n    out = yes if x < 1 else no\n    x = 10\n",
        "for k in range(0, 128, 64):\n    if runtime:\n        k = 128\n    out = yes if k + 63 < 128 else no\n",
        "for k in range(0, 128, 64):\n    k = 128\n    out = yes if k + 63 < 128 else no\n",
    ],
)
def test_loop_carried_and_unknown_ranges_retain_checks(source: str) -> None:
    assert "else no" in _rewrite(source)


def test_enclosing_with_does_not_remove_runtime_branch_or_effects() -> None:
    source = "with elect():\n" + "\n".join(
        "    " + line for line in SOURCE.splitlines()
    )
    transformed = _rewrite(source)
    assert "with elect():" in transformed
    assert "else:" not in transformed
    assert "record(k, local, 8)" in transformed


@pytest.mark.parametrize("threads", [None, (0, 1, 1), (1024, 2, 1)])
def test_missing_or_invalid_launch_facts_do_not_transform(
    threads: tuple[int, int, int] | None,
) -> None:
    assert "else:" in _rewrite(SOURCE, threads=threads)


@pytest.mark.parametrize(
    "prefix", ["BLOCK = 64\n", "cute = something\n", "range = something\n"]
)
def test_protected_bindings_fail_closed(prefix: str) -> None:
    assert "_cute_proven_loop_bounds" not in _rewrite(prefix + SOURCE)


def test_bounded_proof_growth_leaves_unknown_arithmetic_unchanged() -> None:
    source = "x = cutlass.Int32(cute.arch.block_idx()[0]) % 2\n"
    for index in range(40):
        source += f"v{index} = x % {index + 2}\n"
    expression = " + ".join(f"v{index}" for index in range(40))
    source += (
        f"end = {expression}\nfor k in range(end):\n    out = yes if k < end else no\n"
    )
    assert "else no" in _rewrite(source)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(16 * ((128 + BLOCK - 1) // BLOCK) * 2 * 16,)", (1024, 1, 1)),
        ("(12, 256, 1)", (12, 256, 1)),
        ("(runtime,)", None),
        ("(tensor.size(0),)", None),
        ("(0,)", None),
        ("(2**31,)", None),
        ("(1, 65536)", None),
    ],
)
def test_exact_grid_accepts_only_bounded_constant_arithmetic(
    expression: str, expected: tuple[int, int, int] | None
) -> None:
    assert (
        exact_static_grid(ast.parse(expression, mode="eval").body, {"BLOCK": 64})
        == expected
    )


def test_exact_grid_can_prove_last_flat_axis_but_unknown_grid_cannot() -> None:
    source = "row = cutlass.Int32(cute.arch.block_idx()[0]) // 64\nvalue = source[row] if row < 16 else zero\n"
    assert "else zero" in _rewrite(source)
    body = simplify_proven_loop_bounds(
        ast.parse(source).body,
        enabled=True,
        thread_block_dims=(128, 1, 1),
        constexpr_values={},
        block_grid_dims=(1024, 1, 1),
    )
    assert "else zero" not in ast.unparse(ast.Module(body=body, type_ignores=[]))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _doubled(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        out[tile_m, tile_n] = x[tile_m, tile_n] * 2
    return out


@pytest.mark.parametrize("planned", [False, True])
def test_a_host_wrapper_launch_proves_no_xyz_grid(planned: bool) -> None:
    """The proven-bounds pass gets an xyz launch's grid from its program
    ids, (256 / 16, 512 / 64, 1), unless a host wrapper plan launches a grid
    of its own.  Such a plan can arrive as late as the prologue's collective
    lowering (the gathered-MMA route), where this one is registered."""
    lower = collective_matmul.lower_collective_matmul
    simplify = proven_bounds.simplify_proven_bounds
    grids = []

    def lower_with_plan(body, fn, **kwargs):
        if planned:
            plans = pipeline.cute_codegen_state(fn.codegen).wrapper_plans
            plans.append({"kind": "own_grid"})
        return lower(body, fn, **kwargs)

    def recording(body, **kwargs):
        grids.append(kwargs["block_grid_dims"])
        return simplify(body, **kwargs)

    bound = cpu_bind(_doubled, (torch.empty(256, 512),))
    config = bound.config_spec.normalized_config(
        helion.Config(
            block_sizes=[16, 64],
            num_threads=[1, 64],
            pid_type="xyz",
            cute_proven_bounds=True,
        )
    )
    with (
        patch.object(collective_matmul, "lower_collective_matmul", lower_with_plan),
        patch.object(proven_bounds, "simplify_proven_bounds", recording),
    ):
        bound.to_code(config)
    assert grids == [(None, None, None) if planned else (16, 8, 1)]


_MASKS = """
for k in range(cutlass.Int32(0), cutlass.Int32(512), cutlass.Int32(BLOCK)):
    row = k + cutlass.Int32(cute.arch.thread_idx()[2])
    in_rows = operator.lt(row, 512)
    in_tile = operator.ge(BLOCK, cute.arch.thread_idx()[2])
    valid = in_rows & in_tile
    value = load(row, valid and row < 512) if valid else 0
    record(value if in_rows else 0)
"""


def _propagated(source: str, *, propagate_truths: bool = True) -> str:
    return ast.unparse(
        ast.Module(
            body=simplify_proven_loop_bounds(
                ast.parse(source).body,
                enabled=True,
                thread_block_dims=(1, 2, 64),
                constexpr_values={"BLOCK": 64},
                propagate_truths=propagate_truths,
            ),
            type_ignores=[],
        )
    )


def test_proven_masks_follow_the_names_they_are_bound_to() -> None:
    # 64-row tiles of 512 rows on 64 threads: every row is in range, which
    # the kernel's ``operator.lt`` mask states, and the masks' ``&`` and
    # every read of it are true with it.
    assert _propagated(_MASKS) == (
        "_cute_proven_loop_bounds_0_abi_version = 1\n"
        "for k in range(cutlass.Int32(0), cutlass.Int32(512), cutlass.Int32(BLOCK)):\n"
        "    row = k + cutlass.Int32(cute.arch.thread_idx()[2])\n"
        "    in_rows = True\n"
        "    in_tile = True\n"
        "    valid = True\n"
        "    value = load(row, True)\n"
        "    record(value)"
    )
    # Without the option the proof reads ``<`` comparisons only.
    plain = _propagated(_MASKS, propagate_truths=False)
    assert "in_rows = operator.lt(row, 512)" in plain
    assert "value = load(row, valid and True) if valid else 0" in plain


@pytest.mark.parametrize(
    ("old", "new", "kept"),
    [
        ("operator.lt(row, 512)", "operator.lt(row, 500)", "if in_rows else 0"),
        ("operator.lt(row, 512)", "operator.ne(row, 512)", "if in_rows else 0"),
        ("in_rows & in_tile", "in_rows & other", "if valid else 0"),
        ("in_rows & in_tile", "in_rows | in_tile", "if valid else 0"),
        (
            "    value = load",
            "    if row:\n        valid = other\n    value = load",
            "if valid else 0",
        ),
        (
            "    value = load",
            "    for valid in range(2):\n        pass\n    value = load",
            "if valid else 0",
        ),
        (
            "    value = load",
            "    valid, extra = pair()\n    value = load",
            "if valid else 0",
        ),
    ],
    ids=[
        "unproven_comparison",
        "another_operator",
        "and_of_an_unknown",
        "or_of_masks",
        "bound_again_in_a_branch",
        "bound_again_by_a_loop",
        "bound_again_in_a_tuple",
    ],
)
def test_an_unproven_or_rebound_mask_keeps_its_reads(
    old: str, new: str, kept: str
) -> None:
    assert old in _MASKS
    assert kept in _propagated(_MASKS.replace(old, new))


def test_a_mask_proven_in_one_branch_is_not_true_in_the_other() -> None:
    source = """
for k in range(cutlass.Int32(0), cutlass.Int32(512), cutlass.Int32(BLOCK)):
    row = k + cutlass.Int32(cute.arch.thread_idx()[2])
    valid = other
    if row:
        valid = operator.lt(row, 512)
        record(1 if valid else 0)
    else:
        record(2 if valid else 0)
    record(3 if valid else 0)
"""
    proven = _propagated(source)
    assert "record(1)" in proven
    assert "record(2 if valid else 0)" in proven
    assert "record(3 if valid else 0)" in proven
