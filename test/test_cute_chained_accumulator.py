from __future__ import annotations

import ast
from contextlib import contextmanager
from copy import copy
import dataclasses
from dataclasses import replace
import itertools
from itertools import product
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch

from test.test_cute_chained_tcgen05 import _tcgen_chain
from test.test_cute_chained_tcgen05 import _tcgen_inputs
from test.test_cute_chained_tcgen05 import _tcgen_single

from ._cute_aux import _cpu_codegen
import helion
from helion import exc
from helion._compiler.autotuner_heuristics import cute as heuristics
from helion._compiler.backend import TritonBackend
from helion._compiler.cute import chained_initialized_accumulator as initialized
from helion._compiler.cute import chained_matmul
from helion._compiler.cute import chained_tcgen05
from helion._compiler.cute.chained_late_rhs import _layout_bytes
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import default_cute_mma_support
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_spec import CUTE_CHAINED_TMEM_EARLY_RELEASE_KEY
from helion.autotuner.config_spec import ConfigSpec
import helion.language as hl

if TYPE_CHECKING:
    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan

pytestmark = skipUnlessBackends(["cute"])


# Initialized accumulator.

INITIALIZED_KEY = "cute_chained_initialized_accumulator"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _initialized_pair(a, b, c, d, scale, mode: hl.constexpr):
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


def _initialized_args(dtype=torch.bfloat16, n=64, m=128, mode="plain", kind="dense"):
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


def _initialized_config(n=64, enabled: object = True, **extra):
    values: dict[str, Any] = {
        "block_sizes": [128] if n & (n - 1) else [128, n],
        "num_warps": 4,
        "cute_chained_mma_schedule": "tcgen05_tmem",
    }
    if enabled != "missing":
        values[INITIALIZED_KEY] = enabled
    return helion.Config.from_dict(values | extra)


def _initialized_code(args, enabled: object = True, **extra):
    with _cpu():
        return _initialized_pair._bind_isolated(args).to_code(
            _initialized_config(args[1].size(1), enabled, **extra)
        )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("n", (32, 64, 96, 128, 160, 192, 224, 256))
def test_typed_emitted_seed_and_join(dtype, n):
    args = _initialized_args(dtype, n)
    old = None
    if n <= 192:
        old = _initialized_code(args, "missing")
        assert _initialized_code(args, False) == old
    else:
        # The full old first-C arena exceeds the unchanged SMEM capacity;
        # initialized reuse can admit the genuinely smaller effective plan.
        with pytest.raises(exc.BackendUnsupported):
            _initialized_code(args, "missing")
    new = _initialized_code(args)
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
        _initialized_code(_initialized_args(n=128, mode=mode))


@pytest.mark.parametrize("kind", ("dense", "offset", "stride"))
def test_scan_and_readonly_coefficient_views(kind):
    assert "chain_seed_copy =" in _initialized_code(
        _initialized_args(mode="scan", kind=kind)
    )


def test_supported_major_and_partial_root_reject():
    assert "chain_seed_copy =" in _initialized_code(_initialized_args(kind="major"))
    with pytest.raises(exc.InvalidConfig, match="invalid chained MMA schedule"):
        _initialized_code(_initialized_args(m=129))


@pytest.mark.parametrize("value", (0, 1, 2, "true", None))
def test_bool_schema_rejects_nonbool(value):
    with pytest.raises(exc.InvalidConfig, match="must be bool"):
        _initialized_code(_initialized_args(), value)


def test_other_schedule_rejects_true_false_canonicalizes():
    with pytest.raises(exc.InvalidConfig, match="independent FP32"):
        _initialized_code(_initialized_args(), cute_chained_mma_schedule="coalesced")
    with _cpu():
        spec = _initialized_pair._bind_isolated(_initialized_args()).config_spec
        config = spec.normalized_config(_initialized_config(enabled=False))
        assert INITIALIZED_KEY not in config.config


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
        _initialized_code(_initialized_args(mode="scan"))


def test_effective_plan_and_exact_shared_cache_budget():
    from helion._compiler.cute import chained_tcgen05 as tcgen

    original_bytes = tcgen._shared_memory_bytes
    original_cache = tcgen.make_early_auxiliary_cache
    calls = []
    budgets = []

    def accounting(plan):
        value = original_bytes(plan)
        calls.append((plan, value))
        return value

    def cache(cg, plan, scans, remaining):
        budgets.append((plan, remaining))
        return original_cache(cg, plan, scans, remaining)

    with patch.object(tcgen, "_shared_memory_bytes", side_effect=accounting):
        old = _initialized_code(_initialized_args(), False)
        old_size = calls[-1][1]
        calls.clear()
        with patch.object(tcgen, "make_early_auxiliary_cache", side_effect=cache):
            new = _initialized_code(
                _initialized_args(),
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
    args = _initialized_args(n=128)[:-1]
    out = args[0] if alias else torch.empty_like(args[0])
    with _cpu(), pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
        _external_pair._bind_isolated((*args, out)).to_code(_initialized_config(128))


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_fp32_final_store_keeps_seed_and_no_extra_conversion(dtype):
    code = _initialized_code(_initialized_args(dtype, mode="fp32out"))
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
        bound = _initialized_pair._bind_isolated(_initialized_args())
        raw = _initialized_config(enabled=enabled)
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
        result[INITIALIZED_KEY] = enabled
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


# Late rhs.

LATE_RHS_KEY = "cute_chained_late_rhs_reuse"

INIT = "cute_chained_initialized_accumulator"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _late_rhs_pair(a, b, c, d, scale, weights, mode: hl.constexpr):
    m, k = a.shape
    n = b.size(1)
    out = torch.empty(
        (m, n), dtype=torch.float32 if mode == "fp32out" else a.dtype, device=a.device
    )
    for row, col in hl.tile([m, n], block_size=[None, n]):
        kk = hl.arange(k)
        ll = hl.arange(c.size(1))
        first = hl.dot(a[row, kk], b[kk, col])
        seed = first * torch.exp(scale[row])[:, None]
        prefix = hl.cumsum(weights[ll], dim=0)
        left = (c[row, ll].float() * torch.exp(prefix)[None, :]).to(c.dtype)
        if mode == "a_reader":
            left = c[row, ll]
        right = d[ll, col]
        if mode == "computed_rhs":
            right = (right.float() * 0.5).to(d.dtype)
        second = hl.dot(left, right)
        result = seed + second
        if mode == "extra_first":
            result = result + first
        if mode == "a_reader":
            result = result + c[row, col].float()
        else:
            result = result + d[row, col].float()
        out[row, col] = result.to(out.dtype)
    return out


def _late_rhs_args(dtype=torch.bfloat16, n=64, kind="dense", mode="plain"):
    m, k = 128, max(128, n)
    second_k = n if mode == "a_reader" else 128
    a = torch.empty((m, k), dtype=dtype)
    b = torch.empty((k, n), dtype=dtype)
    c = torch.empty((m, second_k), dtype=dtype)
    d = torch.empty((max(m, second_k), n), dtype=dtype)
    if kind == "offset":
        d = torch.empty(d.numel() + 1, dtype=dtype)[1:].view(d.shape)
    elif kind == "stride":
        d = torch.empty((d.size(0), n * 2), dtype=dtype)[:, ::2]
    elif kind == "tail":
        d = torch.empty((d.size(0) - 1, n), dtype=dtype)
    return a, b, c, d, torch.empty(m), torch.empty(second_k), mode


def _late_rhs_config(value: object = True, **extra):
    config: dict[str, object] = {
        "block_sizes": [128],
        "num_warps": 4,
        "cute_chained_mma_schedule": "tcgen05_tmem",
        INIT: True,
    }
    if value != "missing":
        config[LATE_RHS_KEY] = value
    return helion.Config.from_dict(config | extra)


def _late_rhs_code(args, value: object = True, capacity=232448, **extra):
    with (
        _cpu(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=capacity
        ),
    ):
        return _late_rhs_pair._bind_isolated(args).to_code(
            _late_rhs_config(value, **extra)
        )


def _without_schedule_delta(old: str, new: str) -> None:
    start = old.index("    chain_1_b_ptr = cute.arch.alloc_smem(")
    # Only the RHS pointer/layout/copy statements, not following early caches.
    parsed = ast.parse(old)
    kernel = next(
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(item, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "chain_1_b_ptr"
                for target in item.targets
            )
            for item in node.body
        )
    )
    pointer = next(
        i
        for i, node in enumerate(kernel.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "chain_1_b_ptr"
    )
    copy_end = next(
        node.end_lineno
        for node in kernel.body[pointer + 1 :]
        if isinstance(node, (ast.If, ast.For))
    )
    assert copy_end is not None
    end = len("".join(old.splitlines(keepends=True)[:copy_end]))
    block = old[start:end]
    late = block.replace(
        block.splitlines(keepends=True)[0], "    chain_1_b_ptr = chain_b_workspace\n", 1
    )
    seed_ready = (
        "    cute.arch.fence_view_async_tmem_store()\n    cute.arch.sync_threads()\n"
    )
    assert new.count(seed_ready + late) == 1
    original = old.replace(block, "", 1).replace(
        "    chain_output_ptr = chain_b_workspace\n",
        "    chain_output_ptr = chain_a_workspace\n",
        1,
    )
    assert new.replace(late, "", 1) == original


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("n", (32, 64, 96, 256))
def test_generic_exact_delta_and_typed_boundaries(dtype, n):
    args = _late_rhs_args(dtype, n)
    old = _late_rhs_code(args, "missing", capacity=500000)
    assert old == _late_rhs_code(args, False, capacity=500000)
    new = _late_rhs_code(args)
    _without_schedule_delta(old, new)
    assert "chain_0_c =" not in new
    assert "chain_1_mma.set(tcgen05.Field.ACCUMULATE, True)" in new
    assert new.count("cute.gemm(") == 2
    assert new.count("cute.arch.mbarrier_wait(") == 2
    epi = new[new.index("chain_epi_values =") :]
    assert "Float32(" in epi


@pytest.mark.parametrize("kind", ("dense", "offset", "stride", "tail"))
def test_guard_and_scalar_fallback_whole_block(kind):
    args = _late_rhs_args(kind=kind)
    old = _late_rhs_code(args, False)
    new = _late_rhs_code(args)
    _without_schedule_delta(old, new)
    tail = new.split("chain_1_b_ptr =", 1)[1].split("chain_1_mma =", 1)[0]
    assert "chain_1_b_step" in tail
    if kind != "stride":
        assert "toint() % 16 == 0" in tail and "else:" in tail
    else:
        assert "cpasync.CopyG2SOp" not in tail


@pytest.mark.parametrize("cache", (False, True))
@pytest.mark.parametrize("raw", (False, True))
@pytest.mark.parametrize("aux", (False, True))
def test_raw_readcache_auxiliary_interactions(cache, raw, aux):
    args = _late_rhs_args()
    flags = {
        "cute_chained_pointwise_vectorize": True,
        "cute_chained_pointwise_read_cache": cache,
        "cute_chained_pointwise_inplace_async": raw,
        "cute_chained_auxiliary_cache": aux,
        "cute_chained_pointwise_unroll": 8,
    }
    _without_schedule_delta(
        _late_rhs_code(args, False, **flags), _late_rhs_code(args, **flags)
    )


@pytest.mark.parametrize("mode", ("computed_rhs", "extra_first", "fp32out", "a_reader"))
def test_unsafe_or_unsupported_pair_rejects(mode):
    with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
        _late_rhs_code(_late_rhs_args(mode=mode))


@pytest.mark.parametrize("value", (0, 1, None, "true", 2))
def test_nonboolean_rejects(value):
    with pytest.raises(exc.InvalidConfig, match="must be bool"):
        _late_rhs_code(_late_rhs_args(), value)


@pytest.mark.parametrize(
    "extra", ({INIT: False}, {INIT: True, "cute_chained_mma_schedule": "coalesced"})
)
def test_configuration_dependency_rejects(extra):
    with pytest.raises(exc.InvalidConfig, match="late RHS"):
        _late_rhs_code(_late_rhs_args(), **extra)


def test_resolved_accounting_and_exact_capacity():
    captured = []
    original = chained_tcgen05.supported_plan

    def capture(plan):
        old = dataclasses.replace(plan, late_rhs_reuse=None)
        captured.append(
            (
                plan,
                chained_tcgen05._shared_memory_bytes(plan),
                chained_tcgen05._shared_memory_bytes(old),
            )
        )
        return original(plan)

    with patch.object(chained_tcgen05, "supported_plan", capture):
        _late_rhs_code(_late_rhs_args())
    plan, new_bytes, old_bytes = captured[-1]
    arena = plan.late_rhs_reuse
    assert arena is not None
    assert (arena.a_bytes, arena.b_bytes, arena.rhs_bytes, arena.output_bytes) == (
        32768,
        16384,
        16384,
        16384,
    )
    assert old_bytes - new_bytes == 16384
    _late_rhs_code(_late_rhs_args(), capacity=new_bytes)
    with pytest.raises(exc.BackendUnsupported):
        _late_rhs_code(_late_rhs_args(), capacity=new_bytes - 1)


@pytest.mark.parametrize(
    "shape,inner", (((0, 64), 1), ((128, 7), 1), ((63, 128), 0), ((128, 4), 1))
)
def test_partial_swizzle_atom_rejects(shape, inner):
    assert _layout_bytes(shape, inner) is None


def test_no_cuda_initialization():
    before = torch.cuda.is_initialized()
    _late_rhs_code(_late_rhs_args())
    assert torch.cuda.is_initialized() == before


def test_actual_a_epilogue_reader_rejects():
    with pytest.raises(exc.BackendUnsupported, match="epilogue reader"):
        _late_rhs_code(_late_rhs_args(mode="a_reader"))


def test_auxiliary_cache_budget_is_legacy_capped():
    budgets = []
    original = chained_tcgen05.make_early_auxiliary_cache

    def capture(cg, plan, scans, budget):
        budgets.append(budget)
        return original(cg, plan, scans, budget)

    with patch.object(chained_tcgen05, "make_early_auxiliary_cache", capture):
        _late_rhs_code(_late_rhs_args(), False, cute_chained_auxiliary_cache=True)
        _late_rhs_code(_late_rhs_args(), True, cute_chained_auxiliary_cache=True)
    assert len(budgets) == 2 and budgets[0] == budgets[1]


@pytest.mark.parametrize("enabled", (False, True))
def test_real_compile_config_raw_and_normalized(enabled):
    class StopBeforeModuleImport(BaseException):
        pass

    with _cpu():
        bound = _late_rhs_pair._bind_isolated(_late_rhs_args())
        raw = _late_rhs_config(enabled)
        before = dict(raw.config)
        normalized = bound._normalized_config_copy(raw)
        original = bound.to_code
        sources = []

        def intercept(config, **kwargs):
            assert config.config == normalized.config
            sources.append(original(config, **kwargs))
            raise StopBeforeModuleImport

        with (
            patch.object(bound.env.backend, "setup_compile_cache_dir"),
            patch.object(bound, "to_code", side_effect=intercept),
            patch(
                "helion.runtime.kernel.PyCodeCache.load",
                side_effect=AssertionError("module load forbidden"),
            ),
        ):
            for config in (raw, normalized):
                with pytest.raises(StopBeforeModuleImport):
                    bound.compile_config(config)
        assert raw.config == before
        assert sources[0] == sources[1]
        assert ("chain_output_ptr = chain_a_workspace" in sources[0]) is enabled


@pytest.mark.parametrize("alias", (False, True))
def test_external_output_or_alias_stays_rejected(alias):
    from test.test_cute_chained_accumulator import _external_pair

    a, b, c, d, scale, weights, mode = _late_rhs_args(n=128)
    out = a if alias else torch.empty_like(a)
    config = _late_rhs_config(True, block_sizes=[128, 128])
    with _cpu(), pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
        _external_pair._bind_isolated((a, b, c, d, scale, out)).to_code(config)


def test_output_byte_span_does_not_fit_a():
    # Two M128 x K64 operands provide only 16KiB A, less than N128 output.
    args = list(_late_rhs_args(n=128))
    args[0] = torch.empty((128, 64), dtype=torch.bfloat16)
    args[1] = torch.empty((64, 128), dtype=torch.bfloat16)
    args[2] = torch.empty((128, 64), dtype=torch.bfloat16)
    args[5] = torch.empty(64)
    with pytest.raises(exc.BackendUnsupported, match="late RHS"):
        _late_rhs_code(tuple(args))


# Tmem free.

TMEM_FREE_KEY = "cute_chained_tmem_free"


def _tmem_free_config(mode: str | None = "last_read", n: int = 64) -> helion.Config:
    result = helion.Config(
        block_sizes=[128, n],
        num_warps=4,
        cute_chained_mma_schedule="tcgen05_tmem",
    )
    if mode is not None:
        result.config[TMEM_FREE_KEY] = mode
    return result


def _tmem_free_bound(
    kind: str = "scan", dtype: torch.dtype = torch.bfloat16, n: int = 64
):
    if kind == "single":
        values = (
            torch.empty(2, 128, 128, dtype=dtype),
            torch.empty(2, 128, n, dtype=dtype),
            torch.empty(2, 128, dtype=dtype),
            True,
            True,
        )
        return _tcgen_single._bind_isolated(values)
    return _tcgen_chain._bind_isolated((*_tcgen_inputs("cpu", dtype, n=n), kind))


def _inverse(source: str) -> str:
    device = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    assert device.end_lineno is not None
    lines = source.splitlines(keepends=True)
    free = next(
        i
        for i, line in enumerate(lines)
        if line.strip() == "chain_allocator.free(chain_tptr)"
    )
    assert lines[free - 2].strip() == "cute.arch.fence_view_async_tmem_load()"
    assert lines[free - 1].strip() == "cute.arch.sync_threads()"
    statement = lines.pop(free)
    # Removing the early statement shifts the final device line by one.
    end = device.end_lineno - 1
    lines[end:end] = ["    cute.arch.sync_threads()\n", statement]
    return "".join(lines)


@pytest.mark.parametrize("kind", ["single", "plain", "scan", "three"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_whole_source_inverse_and_defaults(kind, dtype):
    with _cpu_codegen():
        bound = _tmem_free_bound(kind, dtype)
        legacy = bound.to_code(_tmem_free_config(None))
        assert bound.to_code(_tmem_free_config("legacy")) == legacy
        raw = _tmem_free_config()
        early = bound.to_code(raw)
        assert early == bound.to_code(bound._normalized_config_copy(raw))
        assert _inverse(early) == legacy
        assert early.count("chain_allocator.free(chain_tptr)") == 1
        assert early.count("cute.arch.sync_threads()") + 1 == legacy.count(
            "cute.arch.sync_threads()"
        )


@pytest.mark.parametrize("n", [32, 64, 128, 256])
def test_all_ordinary_widths_and_config_roundtrip(n):
    with _cpu_codegen():
        bound = _tmem_free_bound("single", n=n)
        raw = _tmem_free_config(n=n)
        normalized = bound._normalized_config_copy(raw)
        assert normalized.config[TMEM_FREE_KEY] == "last_read"
        assert bound.to_code(normalized) == bound.to_code(raw)
        assert bound.config_spec.flatten_missing_field_default(TMEM_FREE_KEY, {}) == (
            True,
            "legacy",
        )
        fragment = bound.config_spec._flat_fields()[TMEM_FREE_KEY]
        assert isinstance(fragment, EnumFragment)
        assert fragment.choices == ("legacy", "last_read")


@pytest.mark.parametrize("value", [None, True, False, 0, 1, "early", "LEGACY", [], {}])
@pytest.mark.parametrize("repair", [False, True])
def test_invalid_values_never_repair(value, repair):
    with _cpu_codegen():
        bound = _tmem_free_bound()
        config = _tmem_free_config().config | {TMEM_FREE_KEY: value}
        with pytest.raises(exc.InvalidConfig, match="must be legacy or last_read"):
            bound.config_spec.normalize(config, _fix_invalid=repair)


@pytest.mark.parametrize(
    "delta",
    [
        {"cute_chained_mma_schedule": "coalesced"},
        {"num_warps": 8},
        {"pid_type": "persistent_interleaved"},
        {"cute_cluster_n": 2},
    ],
)
@pytest.mark.parametrize("repair", [False, True])
def test_unsupported_route_never_repair(delta, repair):
    with _cpu_codegen():
        bound = _tmem_free_bound()
        with pytest.raises(exc.InvalidConfig, match="resident one-CTA"):
            bound.config_spec.normalize(
                _tmem_free_config().config | delta, _fix_invalid=repair
            )


def test_non_cute_rejects_enabled_and_omits_legacy():
    spec = ConfigSpec(backend=TritonBackend())
    with pytest.raises(exc.InvalidConfig, match="resident one-CTA"):
        spec.normalize({TMEM_FREE_KEY: "last_read"}, _fix_invalid=True)
    default: dict[str, object] = {}
    legacy: dict[str, object] = {TMEM_FREE_KEY: "legacy"}
    spec.normalize(default)
    spec.normalize(legacy)
    assert legacy == default


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy(x):
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = x[tile] + 1
    return out


def test_unrelated_kernel_rejects_enabled():
    with _cpu_codegen():
        bound = _copy._bind_isolated((torch.empty(128),))
        with pytest.raises(exc.InvalidConfig, match="resident one-CTA"):
            bound.to_code(
                helion.Config.from_dict(
                    {"block_sizes": [128], TMEM_FREE_KEY: "last_read"}
                )
            )


@pytest.mark.parametrize("m,n", [(64, 64), (127, 64), (128, 96)])
def test_existing_shape_admission_remains_strict(m, n):
    with _cpu_codegen():
        values = (
            torch.empty(2, 128, m, dtype=torch.bfloat16),
            torch.empty(2, 128, n, dtype=torch.bfloat16),
            torch.empty(2, 128, dtype=torch.bfloat16),
            True,
            True,
        )
        bound = _tcgen_single._bind_isolated(values)
        config = _tmem_free_config(n=n)
        config.config["block_sizes"] = [m, n]
        if m == 64:
            source = bound.to_code(config)
            legacy = helion.Config.from_dict(
                {
                    key: value
                    for key, value in config.config.items()
                    if key != TMEM_FREE_KEY
                }
            )
            assert _inverse(source) == bound.to_code(legacy)
        else:
            with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
                bound.to_code(config)


def test_physical_capacity_is_not_waived():
    with _cpu_codegen():
        bound = _tmem_free_bound()
        with (
            patch.object(
                CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=1024
            ),
            pytest.raises(exc.BackendUnsupported),
        ):
            bound.to_code(_tmem_free_config())


def _body():
    return [
        "chain_0_acc = cute.make_tensor(chain_tptr, layout)",
        "chain_0_copy = tcgen05.make_tmem_copy(atom, chain_0_acc)",
        "chain_0_thread = chain_0_copy.get_slice(chain_thread)",
        "chain_0_source = chain_0_thread.partition_S(chain_0_acc)",
        "chain_0_values = cute.make_rmem_tensor(shape, cutlass.Float32)",
        "cute.copy(chain_0_copy, chain_0_source, chain_0_values)",
        "cute.arch.fence_view_async_tmem_load()",
        "cute.arch.sync_threads()",
    ]


@pytest.mark.parametrize("index", [-1, -2, -3])
def test_final_snapshot_fence_and_rendezvous_are_required(index):
    body = _body()
    body.pop(index)
    with pytest.raises(chained_matmul._UnsupportedChain, match="final TMEM load"):
        chained_tcgen05._last_read_epilogue(body, ["x = chain_0_values[0]"], 0)


@pytest.mark.parametrize(
    "use",
    [
        "x = chain_tptr",
        "x = chain_0_acc[0]",
        "cute.copy(atom, chain_0_source, values)",
        "tcgen05.commit(bar)",
        "cute.gemm(mma, a, b, c)",
        "cute.arch.fence_view_async_tmem_store()",
        "chain_allocator.free(chain_tptr)",
    ],
)
def test_later_tmem_use_is_rejected(use):
    with pytest.raises(chained_matmul._UnsupportedChain, match="later TMEM use"):
        chained_tcgen05._last_read_epilogue(_body(), [use], 0)


def test_indirect_prior_alias_is_rejected():
    body = _body()
    body[1:1] = [
        "other = cute.make_tensor(chain_0_acc.iterator, layout)",
        "offset = other.iterator + 8",
        "nested = cute.make_tensor(offset, layout)",
    ]
    with pytest.raises(chained_matmul._UnsupportedChain, match="later TMEM use"):
        chained_tcgen05._last_read_epilogue(body, ["x = nested[0]"], 0)


@pytest.mark.parametrize(
    "alias",
    [
        "hidden: object = chain_0_source",
        "hidden, other = (chain_0_source, 0)",
        "hidden += chain_tptr",
        "if (hidden := chain_0_source):\n    pass",
    ],
)
def test_unsupported_tmem_alias_forms_fail_closed(alias):
    body = _body()
    body.insert(-3, alias)
    with pytest.raises(chained_matmul._UnsupportedChain, match="unproven TMEM alias"):
        chained_tcgen05._last_read_epilogue(body, ["x = hidden[0]"], 0)


def test_register_snapshot_and_layout_metadata_remain_usable():
    epi = [
        "coords = chain_0_thread.partition_D(shared)",
        "x = chain_0_values[0]",
        "shared[0] = x",
        "cute.arch.sync_threads()",
        "output[0] = shared[0]",
    ]
    assert chained_tcgen05._last_read_epilogue(_body(), epi, 0) == [
        "chain_allocator.free(chain_tptr)",
        *epi,
    ]


def test_real_codegen_rejects_injected_future_tmem_read():
    original = chained_tcgen05._epilogue

    def changed(*args, **kwargs):
        return [*original(*args, **kwargs), "later = chain_tptr.toint()"]

    with (
        _cpu_codegen(),
        patch.object(chained_tcgen05, "_epilogue", side_effect=changed),
        pytest.raises(exc.BackendUnsupported, match="later TMEM use"),
    ):
        _tmem_free_bound().to_code(_tmem_free_config())


# Tmem early release.

EARLY_RELEASE_KEY = CUTE_CHAINED_TMEM_EARLY_RELEASE_KEY

RELEASE = "    chain_allocator.relinquish_alloc_permit()\n"

RETRIEVE = "    chain_tptr = chain_allocator.retrieve_ptr(cutlass.Float32)\n"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _early_release_single(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for row, col in hl.tile([m, n], block_size=[None, n]):
        kk = hl.arange(k)
        out[row, col] = hl.dot(a[row, kk], b[kk, col])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _pointwise(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    for row in hl.tile(a.numel()):
        out[row] = a[row] + 1
    return out


def _early_release_values(
    dtype: torch.dtype, n: int, m: int = 128
) -> tuple[torch.Tensor, ...]:
    return torch.empty((m, 128), dtype=dtype), torch.empty((128, n), dtype=dtype)


def _early_release_config(value: object = "missing", **extra: object) -> helion.Config:
    config: dict[str, object] = {
        "block_sizes": [128],
        "num_warps": 4,
        "cute_chained_mma_schedule": "tcgen05_tmem",
    }
    if value != "missing":
        config[EARLY_RELEASE_KEY] = value
    return helion.Config.from_dict(config | extra)


def _assert_only_release(old: str, new: str) -> None:
    assert old.count(RELEASE) == new.count(RELEASE) == 1
    assert old.count(RETRIEVE) == new.count(RETRIEVE) == 1
    assert RETRIEVE + RELEASE in old
    assert RETRIEVE + RELEASE not in new
    assert new.replace(RELEASE, "", 1).replace(RETRIEVE, RETRIEVE + RELEASE, 1) == old
    for source, early in ((old, False), (new, True)):
        module = ast.parse(source)
        body = next(
            node.body
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and any(
                isinstance(item, ast.Expr)
                and ast.unparse(item).startswith("chain_allocator.allocate(")
                for item in node.body
            )
        )
        statements = [ast.unparse(node) for node in body]
        allocate = next(
            i
            for i, text in enumerate(statements)
            if text.startswith("chain_allocator.allocate(")
        )
        release = statements.index(RELEASE.strip())
        wait = statements.index("chain_allocator.wait_for_alloc()")
        retrieve = statements.index(RETRIEVE.strip())
        free = next(
            i
            for i, text in enumerate(statements)
            if text.startswith("chain_allocator.free(")
        )
        assert allocate < wait < retrieve < free
        assert (release == allocate + 1) is early
        assert retrieve == wait + 1
        assert statements[free - 1] == "cute.arch.sync_threads()"
        for method in (
            "allocate",
            "relinquish_alloc_permit",
            "wait_for_alloc",
            "retrieve_ptr",
            "free",
        ):
            calls = [
                node
                for node in ast.walk(module)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and ast.unparse(node.func) == f"chain_allocator.{method}"
            ]
            assert len(calls) == 1


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("n", (32, 64, 96, 128, 256))
def test_single_allocation_domain_and_default_identity(
    dtype: torch.dtype, n: int
) -> None:
    before = torch.cuda.is_initialized()
    with _cpu():
        bound = _early_release_single._bind_isolated(_early_release_values(dtype, n))
        old = bound.to_code(_early_release_config())
        assert old == bound.to_code(_early_release_config(False))
        new = bound.to_code(_early_release_config(True))
    _assert_only_release(old, new)
    columns = max(32, 1 << (n - 1).bit_length())
    assert f"chain_allocator.allocate({columns})" in new
    assert new.count("cute.gemm(") == 1
    assert torch.cuda.is_initialized() == before


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("n", (32, 64, 128, 256))
def test_packed_bridge_keeps_doubled_columns(dtype: torch.dtype, n: int) -> None:
    args = (*_tcgen_inputs("cpu", dtype, n=n), "plain")
    config = helion.Config(
        block_sizes=[128, n], num_warps=4, cute_chained_mma_schedule="tcgen05_tmem"
    )
    with _cpu():
        bound = _tcgen_chain._bind_isolated(args)
        old = bound.to_code(config)
        new = bound.to_code(
            helion.Config.from_dict(config.config | {EARLY_RELEASE_KEY: True})
        )
    _assert_only_release(old, new)
    assert "OperandSource.TMEM" in new
    assert f"chain_allocator.allocate({max(128, n) * 2})" in new


@pytest.mark.parametrize("initialized", (False, True))
@pytest.mark.parametrize(
    "raw,cache,aux", tuple(itertools.product((False, True), repeat=3))
)
def test_two_dot_interactions_exact_inverse(
    initialized: bool, raw: bool, cache: bool, aux: bool
) -> None:
    config = _late_rhs_config(
        False,
        cute_chained_initialized_accumulator=initialized,
        cute_chained_pointwise_vectorize=True,
        cute_chained_pointwise_inplace_async=raw,
        cute_chained_pointwise_read_cache=cache,
        cute_chained_auxiliary_cache=aux,
        cute_chained_pointwise_unroll=8,
    )
    with _cpu():
        bound = _late_rhs_pair._bind_isolated(_late_rhs_args())
        old = bound.to_code(config)
        new = bound.to_code(
            helion.Config.from_dict(config.config | {EARLY_RELEASE_KEY: True})
        )
    _assert_only_release(old, new)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("kind", ("dense", "offset", "stride", "tail"))
def test_late_rhs_guard_and_lifetime_unchanged(dtype: torch.dtype, kind: str) -> None:
    config = _late_rhs_config(
        cute_chained_pointwise_vectorize=True,
        cute_chained_pointwise_read_cache=True,
        cute_chained_pointwise_inplace_async=True,
    )
    with _cpu():
        bound = _late_rhs_pair._bind_isolated(_late_rhs_args(dtype, kind=kind))
        old = bound.to_code(config)
        new = bound.to_code(
            helion.Config.from_dict(config.config | {EARLY_RELEASE_KEY: True})
        )
    _assert_only_release(old, new)
    assert "chain_output_ptr = chain_a_workspace" in new


@pytest.mark.parametrize("value", (0, 1, None, "true", 2))
@pytest.mark.parametrize("fix", (False, True))
def test_strict_bool_even_fix_invalid(value: object, fix: bool) -> None:
    with _cpu():
        spec = _early_release_single._bind_isolated(
            _early_release_values(torch.bfloat16, 64)
        ).config_spec
        with pytest.raises(exc.InvalidConfig, match="must be bool"):
            spec.normalize(_early_release_config(value), _fix_invalid=fix)


@pytest.mark.parametrize("fix", (False, True))
@pytest.mark.parametrize("schedule", ("coalesced", "cp_async", "unknown"))
def test_unsupported_true_never_repaired_away(fix: bool, schedule: str) -> None:
    with _cpu():
        spec = _early_release_single._bind_isolated(
            _early_release_values(torch.bfloat16, 64)
        ).config_spec
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(
                _early_release_config(True, cute_chained_mma_schedule=schedule),
                _fix_invalid=fix,
            )
        spec.cute_chained_tcgen05_search_enabled = False
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(_early_release_config(True), _fix_invalid=fix)


def test_ordinary_pointwise_false_identity_true_rejection() -> None:
    with _cpu():
        bound = _pointwise._bind_isolated((torch.empty(128),))
        old = helion.Config(block_sizes=[128])
        assert bound.to_code(old) == bound.to_code(
            helion.Config.from_dict(old.config | {EARLY_RELEASE_KEY: False})
        )
        assert EARLY_RELEASE_KEY not in bound.config_spec._flat_fields()
        for fix in (False, True):
            with pytest.raises(exc.InvalidConfig, match="early TMEM"):
                bound.config_spec.normalize(
                    helion.Config.from_dict(old.config | {EARLY_RELEASE_KEY: True}),
                    _fix_invalid=fix,
                )


@pytest.mark.parametrize("m", (64, 129))
def test_no_geometry_admission_expansion(m: int) -> None:
    with _cpu(), pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
        _early_release_single._bind_isolated(
            _early_release_values(torch.bfloat16, 64, m)
        ).to_code(_early_release_config(True))


def test_unavailable_hardware_rejects() -> None:
    with (
        _cpu(),
        patch_cute_mma_support(default_cute_mma_support(tcgen05_f16bf16=False)),
    ):
        bound = _early_release_single._bind_isolated(
            _early_release_values(torch.bfloat16, 64)
        )
        assert EARLY_RELEASE_KEY not in bound.config_spec._flat_fields()
        with pytest.raises(exc.InvalidConfig):
            bound.to_code(_early_release_config(True))


def test_real_resource_boundary_and_no_plan_rejection() -> None:
    footprints: list[int] = []
    original = chained_tcgen05.supported_plan

    def capture(plan: ChainedMatmulPlan) -> bool:
        footprints.append(chained_tcgen05._shared_memory_bytes(plan))
        return original(plan)

    with _cpu(), patch.object(chained_tcgen05, "supported_plan", side_effect=capture):
        bound = _early_release_single._bind_isolated(
            _early_release_values(torch.bfloat16, 256)
        )
        old = bound.to_code(_early_release_config())
        new = bound.to_code(_early_release_config(True))
    _assert_only_release(old, new)
    assert len(set(footprints)) == 1
    for enabled in (False, True):
        with (
            _cpu(),
            patch.object(
                CuteTcgen05Config,
                "per_cta_smem_capacity_bytes",
                return_value=footprints[0],
            ),
        ):
            _early_release_single._bind_isolated(
                _early_release_values(torch.bfloat16, 256)
            ).to_code(_early_release_config(enabled))
        with (
            _cpu(),
            patch.object(
                CuteTcgen05Config,
                "per_cta_smem_capacity_bytes",
                return_value=footprints[0] - 1,
            ),
            pytest.raises(exc.BackendUnsupported),
        ):
            _early_release_single._bind_isolated(
                _early_release_values(torch.bfloat16, 256)
            ).to_code(_early_release_config(enabled))
    with (
        _cpu(),
        patch(
            "helion._compiler.cute.chained_matmul.plan_chained_matmul",
            return_value=None,
        ),
        pytest.raises(exc.BackendUnsupported, match="early TMEM"),
    ):
        _early_release_single._bind_isolated(
            _early_release_values(torch.bfloat16, 64)
        ).to_code(_early_release_config(True))


@pytest.mark.parametrize("enabled", (False, True))
def test_real_compile_config_raw_normalized_stops_at_load(enabled: bool) -> None:
    class StopBeforeModuleLoad(BaseException):
        pass

    sources: list[str] = []

    def stop(source: str, *args: object, **kwargs: object) -> None:
        sources.append(source)
        raise StopBeforeModuleLoad

    with _cpu():
        bound = _early_release_single._bind_isolated(
            _early_release_values(torch.bfloat16, 128)
        )
        raw = _early_release_config(enabled)
        before = dict(raw.config)
        normalized = bound._normalized_config_copy(raw)
        with (
            patch.object(bound.env.backend, "setup_compile_cache_dir"),
            patch("helion.runtime.kernel.PyCodeCache.load", side_effect=stop),
        ):
            for config in (raw, normalized):
                with pytest.raises(StopBeforeModuleLoad):
                    bound.compile_config(config)
        assert raw.config == before
        assert sources[0] == sources[1] == bound.to_code(raw)


def test_one_seed_twin_preserves_objects_and_all_old_priority() -> None:
    seeds = [
        helion.Config(cute_chained_mma_schedule="coalesced"),
        helion.Config(cute_chained_mma_schedule="tcgen05_tmem", block_sizes=[128, 64]),
        helion.Config(cute_chained_mma_schedule="tcgen05_tmem", block_sizes=[128, 32]),
    ]
    seeds.append(seeds[1])
    result = heuristics._with_early_tmem_release_seed(seeds)
    assert len(result) == len(seeds) + 1
    assert result[2].config == seeds[1].config | {EARLY_RELEASE_KEY: True}
    assert all(
        a is b
        for a, b in zip(
            (x for x in result if not x.config.get(EARLY_RELEASE_KEY)),
            seeds,
            strict=True,
        )
    )
    assert heuristics._with_early_tmem_release_seed([]) == []
    assert heuristics._with_early_tmem_release_seed(seeds[:1]) == seeds[:1]
