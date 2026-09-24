from __future__ import annotations

import ast
from copy import copy
from dataclasses import replace
from itertools import product
from typing import Any
from unittest.mock import patch

import pytest
import torch

from test.test_cute_chained_initialized_accumulator import KEY
from test.test_cute_chained_initialized_accumulator import _cpu

import helion
from helion import exc
from helion._compiler.cute import chained_initialized_accumulator as initialized
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])
MAJORS = tuple(product(("K", "MN"), repeat=2))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _major_pair(a, b, c, d, rows, columns, family: hl.constexpr):
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, col in hl.tile([m, n], block_size=[None, n]):
        kk = hl.arange(k)
        left = (a[row, kk].float() * 1.01).to(a.dtype)
        first = hl.dot(left, b[kk, col])
        if family == "column":
            coefficient = columns[col][None, :]
        else:
            coefficient = torch.exp(rows[row])[:, None]
            if family == "scan":
                prefix = hl.cumsum(rows[kk], dim=0)
                coefficient = torch.exp(prefix[row])[:, None]
        seed = first * coefficient
        right = (d[kk, col].float() * rows[kk][:, None]).to(a.dtype)
        second = hl.dot(c[row, kk], right)
        out[row, col] = (seed + second).to(out.dtype)
    return out


def _major_args(
    dtype: torch.dtype = torch.bfloat16,
    n: int = 64,
    first: tuple[str, str] = ("K", "MN"),
    second: tuple[str, str] = ("K", "MN"),
    family: str = "row",
    m: int = 128,
) -> tuple[Any, ...]:
    values = []
    for major_a, major_b in (first, second):
        a = torch.empty((m, 128), dtype=dtype)
        b = torch.empty((128, n), dtype=dtype)
        if major_a == "MN":
            a = torch.empty((128, m), dtype=dtype).T
        if major_b == "K":
            b = torch.empty((n, 128), dtype=dtype).T
        values.extend((a, b))
    return (*values, torch.empty(max(m, 128)), torch.empty(n), family)


def _major_config(enabled: object = True, **extra: Any) -> helion.Config:
    result: dict[str, Any] = {
        "block_sizes": [128],
        "num_warps": 4,
        "cute_chained_mma_schedule": "tcgen05_tmem",
    }
    if enabled != "missing":
        result[KEY] = enabled
    return helion.Config.from_dict(result | extra)


def _major_code(args: tuple[Any, ...], **extra: Any) -> str:
    with _cpu():
        bound = _major_pair._bind_isolated(args)
        raw = _major_config(**extra)
        normalized = bound._normalized_config_copy(raw)
        code = bound.to_code(raw)
        assert bound.to_code(normalized) == code
        return code


def _assert_majors(code: str, first: tuple[str, str], second: tuple[str, str]) -> None:
    calls = [
        n
        for n in ast.walk(ast.parse(code))
        if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "chain_sm100.make_trivial_tiled_mma"
    ]
    assert [
        [ast.unparse(n.args[i]).rsplit(".", 1)[-1] for i in (2, 3)] for n in calls
    ] == [list(first), list(second)]
    assert all(ast.unparse(n.args[4]) == "cutlass.Float32" for n in calls)
    assert all(ast.unparse(n.args[5]) == "tcgen05.CtaGroup.ONE" for n in calls)
    assert all(ast.unparse(n.args[7]) == "tcgen05.OperandSource.SMEM" for n in calls)
    assert "St32x32bOp(tcgen05.Repetition(32))" in code
    store = code.index("cute.copy(chain_seed_copy, chain_0_values, chain_seed_target)")
    assert code[store:].splitlines()[1:3] == [
        "    cute.arch.fence_view_async_tmem_store()",
        "    cute.arch.sync_threads()",
    ]
    assert code.index("cute.arch.fence_view_async_tmem_load()") < store
    assert store < code.index("chain_1_mma =")
    assert "chain_0_mma.set(tcgen05.Field.ACCUMULATE, False)" in code
    assert "chain_1_mma.set(tcgen05.Field.ACCUMULATE, True)" in code


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("n", range(32, 257, 32))
@pytest.mark.parametrize("major", MAJORS)
def test_all_column_geometries_real_raw_normalized_sources(
    dtype: torch.dtype, n: int, major: tuple[str, str]
) -> None:
    _assert_majors(_major_code(_major_args(dtype, n, major, major)), major, major)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("family", ("column", "scan"))
@pytest.mark.parametrize("first,second", tuple(product(MAJORS, repeat=2)))
def test_independent_stage_major_pairs_and_coefficient_families(
    dtype: torch.dtype, family: str, first: tuple[str, str], second: tuple[str, str]
) -> None:
    _assert_majors(
        _major_code(_major_args(dtype, 64, first, second, family)), first, second
    )


@pytest.mark.parametrize("major", MAJORS)
def test_genuine_compile_config_stops_before_generated_module(
    major: tuple[str, str],
) -> None:
    class StopBeforeLoad(BaseException):
        pass

    with _cpu():
        bound = _major_pair._bind_isolated(_major_args(first=major, second=major))
        raw = _major_config()
        normalized = bound._normalized_config_copy(raw)
        expected = bound.to_code(raw)
        seen = []

        def stop(code: str, *, extra: str) -> None:
            assert extra == "" and code == expected
            seen.append(code)
            raise StopBeforeLoad

        with (
            patch.object(type(bound.env.backend), "setup_compile_cache_dir"),
            patch("helion.runtime.kernel.PyCodeCache.load", side_effect=stop),
        ):
            for request in (raw, normalized):
                with pytest.raises(StopBeforeLoad):
                    bound.compile_config(request, allow_print=False)
        assert seen == [expected, expected] and not bound._compile_cache


@pytest.fixture(scope="module")
def geometry_plan() -> tuple[Any, dict[tuple[int, str], int]]:
    original = initialized._seed_geometry_supported
    captured = []

    def capture(plan: Any, axes: dict[tuple[int, str], int]) -> bool:
        captured.append((plan, axes))
        return original(plan, axes)

    with patch.object(initialized, "_seed_geometry_supported", side_effect=capture):
        _major_code(_major_args())
    return captured[0]


@pytest.mark.parametrize(
    "change",
    [
        {"threads": 64},
        {"threads": 256},
        {"strategy": "warp"},
        {"dtype": torch.float32},
        {"shapes": ((64, 64, 128), (64, 64, 128))},
        {"shapes": ((128, 64, 128), (128, 32, 128))},
        {"shapes": ((128, 48, 128), (128, 48, 128))},
        {"shapes": ((128, 288, 128), (128, 288, 128))},
        {"shapes": ((128, 64, 0), (128, 64, 128))},
        {"shapes": ((128, 64, 24), (128, 64, 128))},
        {"axes": ((0, 129, 128),)},
        {"axes": ((0, 0, 128),)},
        {"axes": ()},
    ],
)
def test_unproved_geometry_rejects(
    geometry_plan: tuple[Any, dict], change: dict
) -> None:
    plan, axes = geometry_plan
    assert not initialized._seed_geometry_supported(replace(plan, **change), axes)


@pytest.mark.parametrize("bad", (-1, 2, 3))
def test_unknown_operand_major_rejects(
    geometry_plan: tuple[Any, dict], bad: int
) -> None:
    plan, axes = geometry_plan
    assert not initialized._seed_geometry_supported(plan, axes | {(0, "a"): bad})
    assert not initialized._seed_geometry_supported(plan, {(0, "a"): 1})


def test_narrow_accumulator_rejects(geometry_plan: tuple[Any, dict]) -> None:
    plan, axes = geometry_plan
    dot = copy(plan.dots[0])
    dot.meta = dict(dot.meta) | {"val": torch.empty((128, 64), dtype=torch.bfloat16)}
    assert not initialized._seed_geometry_supported(
        replace(plan, dots=(dot, plan.dots[1])), axes
    )


@pytest.mark.parametrize("m", (64, 129))
def test_real_partial_and_m64_roots_remain_rejected(m: int) -> None:
    with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
        _major_code(_major_args(m=m, first=("MN", "K")))


def test_shared_capacity_is_still_authoritative() -> None:
    from helion._compiler.cute.tcgen05_config import CuteTcgen05Config

    with (
        _cpu(),
        patch.object(CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=1),
        pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)),
    ):
        _major_pair._bind_isolated(_major_args(first=("MN", "K"))).to_code(
            _major_config()
        )
