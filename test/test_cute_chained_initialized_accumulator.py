from __future__ import annotations

import ast
from contextlib import contextmanager
from copy import copy
from dataclasses import replace
from itertools import product
from typing import Any
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute import chained_initialized_accumulator as initialized
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])
KEY = "cute_chained_initialized_accumulator"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _pair(a, b, c, d, scale, mode: hl.constexpr):
    m, k = a.shape
    n = b.size(1)
    out = torch.empty(
        (m, n), dtype=torch.float32 if mode == "fp32out" else a.dtype, device=a.device
    )
    for row, col in hl.tile([m, n], block_size=[None, n if mode == "fixed" else None]):
        kk = hl.arange(k)
        left = (a[row, kk].float() * 1.01).to(a.dtype)
        first = hl.dot(left, b[kk, col])
        coefficient = torch.exp(scale[row])[:, None]
        if mode == "scan":
            prefix = hl.cumsum(scale[kk], dim=0)
            coefficient = torch.exp(prefix[row])[:, None]
        if mode == "sin":
            coefficient = torch.sin(scale[row])[:, None]
        if mode == "bf16":
            first = first.to(torch.bfloat16).float()
        if mode == "fp16":
            first = first.to(torch.float16).float()
        if mode == "fp64":
            first = first.to(torch.float64).float()
        if mode == "transpose":
            first = first.T
        seed = first * coefficient
        if mode == "seed_cast":
            seed = seed.to(torch.bfloat16).float()
        right = (d[kk, col].float() * scale[kk][:, None]).to(a.dtype)
        second = hl.dot(c[row, kk], right)
        result = seed + second
        if mode == "extra_first":
            result = result + first
        if mode == "extra_seed":
            result = result + seed
        if mode == "extra_second":
            result = result + second
        if mode == "swapped":
            result = second + seed
        if mode == "alpha":
            result = torch.add(seed, second, alpha=2)
        if mode == "join_view":
            result = result.T
        if mode == "join_gather":
            result = result[(row.index + 1) % m, :]
        out[row, col] = result.to(out.dtype)
    return out


def _args(dtype=torch.bfloat16, n=64, m=128, mode="plain", kind="dense"):
    if mode == "plain" and n & (n - 1):
        mode = "fixed"
    values = [
        torch.empty(shape, dtype=dtype)
        for shape in ((m, 128), (128, n), (m, 128), (128, n))
    ]
    scale = torch.empty(max(m, 128), dtype=torch.float32)
    if kind == "offset":
        scale = torch.empty(scale.numel() + 4, dtype=torch.float32)[4:]
    if kind == "stride":
        scale = torch.empty(scale.numel() * 2, dtype=torch.float32)[::2]
    if kind == "major":
        values[1] = torch.empty((n, 128), dtype=dtype).T
    return (*values, scale, mode)


@contextmanager
def _cpu():
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        yield


def _config(n=64, enabled: object = True, **extra):
    values: dict[str, Any] = {
        "block_sizes": [128] if n & (n - 1) else [128, n],
        "num_warps": 4,
        "cute_chained_mma_schedule": "tcgen05_tmem",
    }
    if enabled != "missing":
        values[KEY] = enabled
    return helion.Config.from_dict(values | extra)


def _code(args, enabled: object = True, **extra):
    with _cpu():
        return _pair._bind_isolated(args).to_code(
            _config(args[1].size(1), enabled, **extra)
        )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("n", (32, 64, 96, 128, 160, 192, 224, 256))
def test_typed_emitted_seed_and_join(dtype, n):
    args = _args(dtype, n)
    old = None
    if n <= 192:
        old = _code(args, "missing")
        assert _code(args, False) == old
    else:
        # The full old first-C arena exceeds the unchanged SMEM capacity;
        # initialized reuse can admit the genuinely smaller effective plan.
        with pytest.raises(exc.BackendUnsupported):
            _code(args, "missing")
    new = _code(args)
    if old is not None:
        assert "chain_0_c =" in old
    assert not any(
        isinstance(node, ast.Name) and node.id == "chain_0_c"
        for node in ast.walk(ast.parse(new))
    )
    assert "chain_seed_copy =" in new
    assert "chain_0_mma.set(tcgen05.Field.ACCUMULATE, False)" in new
    assert "chain_1_mma.set(tcgen05.Field.ACCUMULATE, True)" in new
    assert "chain_1_mma.set(tcgen05.Field.ACCUMULATE, False)" not in new
    store = new.index("cute.copy(chain_seed_copy, chain_0_values, chain_seed_target)")
    assert new[store:].splitlines()[1:3] == [
        "    cute.arch.fence_view_async_tmem_store()",
        "    cute.arch.sync_threads()",
    ]
    assert new.index("cute.arch.fence_view_async_tmem_load()") < store
    assert store < new.index("chain_1_mma =")
    assert new.count("cute.gemm(") == 2
    assert new.count("cute.arch.mbarrier_wait(") == 2
    if old is not None:
        assert old.count("cute.gemm(") == old.count("cute.arch.mbarrier_wait(") == 2
    # Operand and final dtype conversions remain explicit. No BF16/FP16 seed.
    dtype_name = "BFloat16" if dtype == torch.bfloat16 else "Float16"
    assert new.count(f"cutlass.{dtype_name}(") >= 3
    seed_loop = next(
        node
        for node in ast.walk(ast.parse(new))
        if isinstance(node, ast.For) and ast.unparse(node.target) == "chain_seed_index"
    )
    assert f"cutlass.{dtype_name}(" not in ast.unparse(seed_loop)
    assert "cutlass.Float32(" in ast.unparse(seed_loop)
    epi = new.split("chain_epi_values =", 1)[1]
    assert (
        "exp2(" not in epi and "chain_seed" not in epi and "chain_0_values" not in epi
    )


@pytest.mark.parametrize(
    "mode",
    (
        "bf16",
        "fp16",
        "fp64",
        "seed_cast",
        "transpose",
        "extra_first",
        "extra_seed",
        "extra_second",
        "swapped",
        "alpha",
        "sin",
        "join_view",
        "join_gather",
    ),
)
def test_real_fx_rejections(mode):
    with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
        _code(_args(n=128, mode=mode))


@pytest.mark.parametrize("kind", ("dense", "offset", "stride"))
def test_scan_and_readonly_coefficient_views(kind):
    assert "chain_seed_copy =" in _code(_args(mode="scan", kind=kind))


def test_supported_major_and_partial_root_reject():
    assert "chain_seed_copy =" in _code(_args(kind="major"))
    with pytest.raises(exc.InvalidConfig, match="invalid chained MMA schedule"):
        _code(_args(m=129))


@pytest.mark.parametrize("value", (0, 1, 2, "true", None))
def test_bool_schema_rejects_nonbool(value):
    with pytest.raises(exc.InvalidConfig, match="must be bool"):
        _code(_args(), value)


def test_other_schedule_rejects_true_false_canonicalizes():
    with pytest.raises(exc.InvalidConfig, match="independent FP32"):
        _code(_args(), cute_chained_mma_schedule="coalesced")
    with _cpu():
        spec = _pair._bind_isolated(_args()).config_spec
        config = spec.normalized_config(_config(enabled=False))
        assert KEY not in config.config


def test_unavailable_scan_boundary_is_fail_closed():
    from helion._compiler.cute import chained_matmul as chain

    original = chain._codegen_scans

    def missing(cg, plan, boundaries, scans, cache):
        result = original(cg, plan, boundaries, scans, cache)
        boundaries.pop(plan.scans[0])
        return result

    with (
        patch.object(chain, "_codegen_scans", side_effect=missing),
        pytest.raises(exc.BackendUnsupported, match="boundary"),
    ):
        _code(_args(mode="scan"))


def test_effective_plan_and_exact_shared_cache_budget():
    from helion._compiler.cute import chained_tcgen05 as tcgen

    original_bytes = tcgen._shared_memory_bytes
    original_cache = tcgen.make_early_auxiliary_cache
    calls = []
    budgets = []

    def accounting(plan, *, startup=False):
        value = original_bytes(plan, startup=startup)
        calls.append((plan, value))
        return value

    def cache(cg, plan, scans, remaining):
        budgets.append((plan, remaining))
        return original_cache(cg, plan, scans, remaining)

    with patch.object(tcgen, "_shared_memory_bytes", side_effect=accounting):
        old = _code(_args(), False)
        old_size = calls[-1][1]
        calls.clear()
        with patch.object(tcgen, "make_early_auxiliary_cache", side_effect=cache):
            new = _code(
                _args(),
                True,
                cute_chained_auxiliary_cache=True,
            )
    assert "chain_0_c =" in old and "chain_seed_copy =" in new
    assert calls and budgets
    assert all(plan.initialized_accumulator is not None for plan, _ in calls)
    assert old_size - calls[-1][1] == 4 * 128 * 64
    assert all(plan is calls[-1][0] for plan, _ in budgets)
    assert budgets[-1][1] == 232448 - calls[-1][1]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _external_pair(a, b, c, d, scale, out):
    m, k = a.shape
    n = b.size(1)
    for row, col in hl.tile([m, n]):
        kk = hl.arange(k)
        first = hl.dot(a[row, kk], b[kk, col])
        seed = first * scale[row][:, None]
        second = hl.dot(c[row, kk], d[kk, col])
        out[row, col] = (seed + second).to(out.dtype)
    return out


@pytest.mark.parametrize("alias", (False, True))
def test_external_and_aliased_output_not_admitted(alias):
    args = _args(n=128)[:-1]
    out = args[0] if alias else torch.empty_like(args[0])
    with _cpu(), pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
        _external_pair._bind_isolated((*args, out)).to_code(_config(128))


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_fp32_final_store_keeps_seed_and_no_extra_conversion(dtype):
    code = _code(_args(dtype, mode="fp32out"))
    assert "chain_seed_copy =" in code
    assert (
        "chain_epi_values = cute.make_rmem_tensor(chain_1_coords.shape, cutlass.Float32)"
        in code
    )


@pytest.mark.parametrize("enabled", (False, True))
def test_real_compile_config_normalization_before_module_import(enabled):
    class StopBeforeModuleImport(BaseException):
        pass

    with _cpu():
        bound = _pair._bind_isolated(_args())
        raw = _config(enabled=enabled)
        raw_before = dict(raw.config)
        normalized = bound._normalized_config_copy(raw)
        original = bound.to_code
        sources = []
        configs = []

        def intercept(config, **kwargs):
            configs.append(dict(config.config))
            sources.append(original(config, **kwargs))
            raise StopBeforeModuleImport

        with (
            patch.object(bound.env.backend, "setup_compile_cache_dir"),
            patch.object(bound, "to_code", side_effect=intercept),
            patch(
                "helion.runtime.kernel.PyCodeCache.load",
                side_effect=AssertionError("module import forbidden"),
            ),
        ):
            for request in (raw, normalized):
                with pytest.raises(StopBeforeModuleImport):
                    bound.compile_config(request)
        assert raw.config == raw_before
        assert configs == [normalized.config, normalized.config]
        assert sources[0] == sources[1]
        assert ("chain_seed_copy =" in sources[0]) is enabled


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
