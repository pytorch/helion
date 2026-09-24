from __future__ import annotations

import ast
import dataclasses
from unittest.mock import patch

import pytest
import torch

from test.test_cute_chained_initialized_accumulator import _cpu

import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_tcgen05
from helion._compiler.cute.chained_late_rhs import _layout_bytes
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])
KEY = "cute_chained_late_rhs_reuse"
INIT = "cute_chained_initialized_accumulator"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _pair(a, b, c, d, scale, weights, mode: hl.constexpr):
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


def _args(dtype=torch.bfloat16, n=64, kind="dense", mode="plain"):
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


def _config(value: object = True, **extra):
    config: dict[str, object] = {
        "block_sizes": [128],
        "num_warps": 4,
        "cute_chained_mma_schedule": "tcgen05_tmem",
        INIT: True,
    }
    if value != "missing":
        config[KEY] = value
    return helion.Config.from_dict(config | extra)


def _code(args, value: object = True, capacity=232448, **extra):
    with (
        _cpu(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=capacity
        ),
    ):
        return _pair._bind_isolated(args).to_code(_config(value, **extra))


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
    args = _args(dtype, n)
    old = _code(args, "missing", capacity=500000)
    assert old == _code(args, False, capacity=500000)
    new = _code(args)
    _without_schedule_delta(old, new)
    assert "chain_0_c =" not in new
    assert "chain_1_mma.set(tcgen05.Field.ACCUMULATE, True)" in new
    assert new.count("cute.gemm(") == 2
    assert new.count("cute.arch.mbarrier_wait(") == 2
    epi = new[new.index("chain_epi_values =") :]
    assert "Float32(" in epi


@pytest.mark.parametrize("kind", ("dense", "offset", "stride", "tail"))
def test_guard_and_scalar_fallback_whole_block(kind):
    args = _args(kind=kind)
    old = _code(args, False)
    new = _code(args)
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
    args = _args()
    flags = {
        "cute_chained_pointwise_vectorize": True,
        "cute_chained_pointwise_read_cache": cache,
        "cute_chained_pointwise_inplace_async": raw,
        "cute_chained_auxiliary_cache": aux,
        "cute_chained_pointwise_unroll": 8,
    }
    _without_schedule_delta(_code(args, False, **flags), _code(args, **flags))


@pytest.mark.parametrize("mode", ("computed_rhs", "extra_first", "fp32out", "a_reader"))
def test_unsafe_or_unsupported_pair_rejects(mode):
    with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
        _code(_args(mode=mode))


@pytest.mark.parametrize("value", (0, 1, None, "true", 2))
def test_nonboolean_rejects(value):
    with pytest.raises(exc.InvalidConfig, match="must be bool"):
        _code(_args(), value)


@pytest.mark.parametrize(
    "extra", ({INIT: False}, {INIT: True, "cute_chained_mma_schedule": "coalesced"})
)
def test_configuration_dependency_rejects(extra):
    with pytest.raises(exc.InvalidConfig, match="late RHS"):
        _code(_args(), **extra)


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
        _code(_args())
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
    _code(_args(), capacity=new_bytes)
    with pytest.raises(exc.BackendUnsupported):
        _code(_args(), capacity=new_bytes - 1)


@pytest.mark.parametrize(
    "shape,inner", (((0, 64), 1), ((128, 7), 1), ((63, 128), 0), ((128, 4), 1))
)
def test_partial_swizzle_atom_rejects(shape, inner):
    assert _layout_bytes(shape, inner) is None


def test_actual_seed_old_order_and_useful_first100():
    with _cpu():
        bound = _pair._bind_isolated(_args())
        spec = bound.config_spec
        assert bound.host_function is not None
        with bound.env:
            new = CuteChainedMatmulHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
            spec.cute_chained_late_rhs_reuse_search_enabled = False
            try:
                old = CuteChainedMatmulHeuristic.get_seed_configs(
                    bound.env, bound.host_function.device_ir
                )
            finally:
                spec.cute_chained_late_rhs_reuse_search_enabled = True
            assert new is not None and old is not None
            assert [seed for seed in new if not seed.config.get(KEY)] == old
            assert new[0] == old[0]
            generation = ConfigGeneration(spec)
            population = [
                generation.unflatten(item)
                for item in generation.random_population_flat(100)
            ]
        useful = [
            value
            for value in population
            if value.config.get(KEY)
            and value.config.get("cute_chained_pointwise_unroll") == 8
            and value.config.get("cute_chained_pointwise_read_cache")
        ]
        assert useful
        source = bound.to_code(useful[0])
        assert "chain_output_ptr = chain_a_workspace" in source


def test_no_cuda_initialization():
    before = torch.cuda.is_initialized()
    _code(_args())
    assert torch.cuda.is_initialized() == before


def test_actual_a_epilogue_reader_rejects():
    with pytest.raises(exc.BackendUnsupported, match="epilogue reader"):
        _code(_args(mode="a_reader"))


def test_auxiliary_cache_budget_is_legacy_capped():
    budgets = []
    original = chained_tcgen05.make_early_auxiliary_cache

    def capture(cg, plan, scans, budget):
        budgets.append(budget)
        return original(cg, plan, scans, budget)

    with patch.object(chained_tcgen05, "make_early_auxiliary_cache", capture):
        _code(_args(), False, cute_chained_auxiliary_cache=True)
        _code(_args(), True, cute_chained_auxiliary_cache=True)
    assert len(budgets) == 2 and budgets[0] == budgets[1]


@pytest.mark.parametrize("enabled", (False, True))
def test_real_compile_config_raw_and_normalized(enabled):
    class StopBeforeModuleImport(BaseException):
        pass

    with _cpu():
        bound = _pair._bind_isolated(_args())
        raw = _config(enabled)
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
    from test.test_cute_chained_initialized_accumulator import _external_pair

    a, b, c, d, scale, weights, mode = _args(n=128)
    out = a if alias else torch.empty_like(a)
    config = _config(True, block_sizes=[128, 128])
    with _cpu(), pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
        _external_pair._bind_isolated((a, b, c, d, scale, out)).to_code(config)


def test_output_byte_span_does_not_fit_a():
    # Two M128 x K64 operands provide only 16KiB A, less than N128 output.
    args = list(_args(n=128))
    args[0] = torch.empty((128, 64), dtype=torch.bfloat16)
    args[1] = torch.empty((64, 128), dtype=torch.bfloat16)
    args[2] = torch.empty((128, 64), dtype=torch.bfloat16)
    args[5] = torch.empty(64)
    with pytest.raises(exc.BackendUnsupported, match="late RHS"):
        _code(tuple(args))
