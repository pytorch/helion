from __future__ import annotations

import ast
import dataclasses
from itertools import product
from typing import cast
from unittest.mock import patch

import pytest
import torch

from test.test_cute_chained_initialized_accumulator import _cpu
from test.test_cute_chained_late_rhs import _args
from test.test_cute_chained_late_rhs import _code
from test.test_cute_chained_late_rhs import _config
from test.test_cute_chained_late_rhs import _pair

import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_tcgen05
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])
KEY = "cute_chained_k_schedule"
FLAGS = {
    "cute_chained_pointwise_vectorize": True,
    "cute_chained_pointwise_unroll": 8,
    "cute_chained_pointwise_read_cache": True,
}


def source(mode="serial64", args=None, **extra):
    return _code(_args() if args is None else args, **(FLAGS | {KEY: mode} | extra))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _earlier_raw(a, b, c, d, scale):
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, col in hl.tile([m, n], block_size=[None, n]):
        kk = hl.arange(k)
        first_left = (a[row, kk].float() * scale[kk][None, :]).to(a.dtype)
        first = hl.dot(first_left, b[kk, col])
        seed = first * torch.exp(scale[row])[:, None]
        final_left = (c[row, kk].float() * scale[kk][None, :]).to(a.dtype)
        second = hl.dot(final_left, d[kk, col])
        out[row, col] = (seed + second + d[row, col].float()).to(a.dtype)
    return out


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("mode", ("serial64", "overlap64"))
def test_raw_on_earlier_operand_only_is_preserved(dtype, mode):
    # FP32 final leaf cannot be raw-preloaded into a BF16/FP16 arena; the
    # already-supported same-dtype first leaf still activates the raw option.
    args = (
        torch.empty((128, 128), dtype=dtype),
        torch.empty((128, 64), dtype=dtype),
        torch.empty((128, 128)),
        torch.empty((128, 64), dtype=dtype),
        torch.empty(128),
    )
    with _cpu():
        bound = _earlier_raw._bind_isolated(args)
        config = _config(
            **(
                FLAGS
                | {
                    KEY: mode,
                    "cute_chained_pointwise_inplace_async": True,
                    "cute_chained_pointwise_read_cache": False,
                }
            )
        )
        result = bound.to_code(config)
    assert "chain_0_a_raw_copy" in result
    assert "chain_1_a_raw_copy" not in result
    assert "chain_k_half" in result


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize(
    "first_a,first_b,final_b", tuple(product(("K", "MN"), repeat=3))
)
def test_independent_major_combinations(dtype, first_a, first_b, final_b):
    args = list(_args(dtype))
    if first_a == "MN":
        args[0] = torch.empty((128, 128), dtype=dtype).T
    if first_b == "K":
        args[1] = torch.empty((64, 128), dtype=dtype).T
    if final_b == "K":
        args[3] = torch.empty((64, 128), dtype=dtype).T
    result = source(args=tuple(args))
    calls = [
        n
        for n in ast.walk(ast.parse(result))
        if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "chain_sm100.make_trivial_tiled_mma"
    ]
    assert [
        [ast.unparse(n.args[i]).rsplit(".", 1)[-1] for i in (2, 3)] for n in calls
    ] == [[first_a, first_b], ["K", final_b]]


def test_unsupported_reduction_has_no_k_seed_family():
    args = list(_args())
    args[2] = torch.empty((128, 64), dtype=torch.bfloat16)
    args[5] = torch.empty(64)
    with _cpu():
        bound = _pair._bind_isolated(tuple(args))
        assert not bound.config_spec.cute_chained_k_schedule_search_enabled
        assert not any(KEY in c.config for c in bound.config_spec.compiler_seed_configs)


def test_new_siblings_preserve_duplicate_object_order():
    from helion._compiler.autotuner_heuristics.cute import _with_k_schedule_seeds

    anchor = _config(cute_chained_pointwise_vectorize=False)
    parent = _config(**FLAGS)
    other = _config(**(FLAGS | {"cute_chained_auxiliary_cache": True}))
    seeds = [anchor, other, parent, other, parent]
    result = _with_k_schedule_seeds(seeds, enabled=True)
    assert [id(c) for c in result if KEY not in c.config] == [id(c) for c in seeds]
    assert result[0] is seeds[0]
    assert [c.config[KEY] for c in result[1:3]] == ["serial64", "overlap64"]
    for child in result[1:3]:
        assert {k: v for k, v in child.config.items() if k != KEY} == other.config


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("n", range(32, 257, 32))
@pytest.mark.parametrize("mode", ("serial64", "overlap64"))
def test_source_halves_seed_and_phases(dtype, n, mode):
    args = _args(dtype, n)
    control = source("full", args)
    new = source(mode, args)
    assert control == _code(args, **FLAGS)
    assert "cute.local_tile(chain_1_a, (128, 64), (0, chain_k_half))" in new
    assert "chain_k_half * 64 + chain_thread % 8 * 8" in new
    assert "chain_1_local_kk in cutlass.range_constexpr(4)" in new
    assert "chain_k_half * 4 + chain_1_local_kk" in new
    assert "cutlass.range(8, unroll=8)" in new
    assert "cutlass.range(64, unroll=1)" in new
    assert new.count("cute.gemm(") == control.count("cute.gemm(") == 2
    assert new.count("chain_1_mma.set(tcgen05.Field.ACCUMULATE, True)") == 2
    assert "chain_0_c =" not in new
    start = "    for chain_k_half in cutlass.range_constexpr(2):"
    end = "    chain_1_copy ="
    body = new[new.index(start) : new.index(end)]
    assert body.count("cute.arch.cp_async_commit_group()") == 1
    assert body.count("cute.arch.cp_async_wait_group(0)") == 1
    assert body.count("cute.arch.fence_view_async_shared()") == 1
    assert body.count("tcgen05.commit(chain_bars + 1)") == 1
    if mode == "serial64":
        assert "mbarrier_wait(chain_bars + 1, chain_k_half)" in body
        assert body.count("cute.arch.sync_threads()") == 2
    else:
        assert "mbarrier_wait(chain_bars + 1, 0)" in body
        assert "    if chain_warp == 0:" in body
        assert body.count("cute.arch.sync_threads()") == 1
    # Full allocations, initial dot/FP32 seed, final result/epilogue are exact.
    assert (
        new[: new.index("    chain_1_mma =")]
        == control[: control.index("    chain_1_mma =")]
    )
    assert new[new.index(end) :] == control[control.index(end) :]

    def allocations(text):
        return [
            ast.dump(node)
            for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "cute.arch.alloc_smem"
        ]

    assert allocations(new) == allocations(control)


@pytest.mark.parametrize("value", (None, False, True, 0, 1, "", "serial", "FULL"))
@pytest.mark.parametrize("repair", (False, True))
def test_strict_enum_even_repair(value, repair):
    with _cpu():
        bound = _pair._bind_isolated(_args())
        config = _config(**(FLAGS | {KEY: value}))
        with pytest.raises(exc.InvalidConfig, match="must be full"):
            bound.config_spec.normalize(config, _fix_invalid=repair)
        assert config.config[KEY] == value


@pytest.mark.parametrize(
    "override",
    (
        {"num_warps": 8},
        {"cute_chained_initialized_accumulator": False},
        {"cute_chained_late_rhs_reuse": False},
        {"cute_chained_pointwise_vectorize": False},
        {"cute_chained_mma_schedule": "coalesced"},
        {"cute_chained_direct_output": True},
    ),
)
@pytest.mark.parametrize("repair", (False, True))
def test_prerequisites_are_not_repaired(override, repair):
    with _cpu():
        bound = _pair._bind_isolated(_args())
        config = _config(**(FLAGS | {KEY: "overlap64"} | override))
        with pytest.raises(exc.InvalidConfig):
            bound.config_spec.normalize(config, _fix_invalid=repair)
        assert config.config[KEY] == "overlap64"


@pytest.mark.parametrize("mode", ("serial64", "overlap64"))
def test_selected_final_raw_preload_rejects(mode):
    with pytest.raises(exc.BackendUnsupported, match="raw preload"):
        source(mode, cute_chained_pointwise_inplace_async=True)


@pytest.mark.parametrize("mode", ("computed_rhs", "extra_first", "fp32out", "a_reader"))
def test_unsupported_pair_rejects(mode):
    with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
        source(args=_args(mode=mode))


def test_exact_shared_accounting():
    plans = []
    original = chained_tcgen05.supported_plan

    def capture(plan):
        assert chained_tcgen05._shared_memory_bytes(
            plan
        ) == chained_tcgen05._shared_memory_bytes(
            dataclasses.replace(plan, k_schedule=None)
        )
        plans.append(plan)
        return original(plan)

    with patch.object(chained_tcgen05, "supported_plan", capture):
        source()
    plan = plans[-1]
    assert plan.k_schedule.byte_spans == ((0, 16384), (16384, 32768))


@pytest.mark.parametrize("kind", ("offset", "stride", "tail", "major"))
def test_full_leaf_guard_or_conservative_rejection(kind):
    args = list(_args())
    operand = cast("torch.Tensor", args[2])
    if kind == "offset":
        args[2] = torch.empty(128 * 128 + 1, dtype=operand.dtype)[1:].view(128, 128)
    elif kind == "stride":
        args[2] = torch.empty((128, 256), dtype=operand.dtype)[:, ::2]
    elif kind == "tail":
        args[2] = torch.empty((128, 127), dtype=operand.dtype)
        args[5] = torch.empty(127)
    else:
        args[2] = operand.T
    if kind != "offset":
        with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
            source(args=tuple(args))
    else:
        result = source(args=tuple(args))
        assert "toint() % 16 == 0" in result
        assert (
            "chain_1_a_load = (chain_thread + chain_1_a_step * 128) // 64 * 128"
            in result
        )


@pytest.mark.parametrize("k", (64, 256))
def test_final_reduction_extent_rejects(k):
    args = list(_args())
    args[2] = torch.empty((128, k), dtype=torch.bfloat16)
    args[3] = torch.empty((max(128, k), 64), dtype=torch.bfloat16)
    args[5] = torch.empty(k)
    with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
        source(args=tuple(args))


@pytest.mark.parametrize("mode", ("full", "serial64", "overlap64"))
def test_real_raw_canonical_preload_stops(mode):
    class StopBeforeLoad(BaseException):
        pass

    with _cpu():
        bound = _pair._bind_isolated(_args())
        raw = _config(**(FLAGS | {KEY: mode}))
        canonical = bound._normalized_config_copy(raw)
        expected = bound.to_code(raw)
        assert expected == bound.to_code(canonical)
        observed = []

        def stop(code, **kwargs):
            observed.append(code)
            raise StopBeforeLoad

        with (
            patch.object(bound.env.backend, "setup_compile_cache_dir"),
            patch("helion.runtime.kernel.PyCodeCache.load", side_effect=stop),
        ):
            for request in (raw, canonical):
                with pytest.raises(StopBeforeLoad):
                    bound.compile_config(request, allow_print=False)
        assert observed == [expected, expected] and not bound._compile_cache


def test_actual_pool_and_normalized_first100():
    from types import SimpleNamespace

    from helion.autotuner.base_search import PopulationBasedSearch

    with _cpu():
        bound = _pair._bind_isolated(_args())
        spec = bound.config_spec
        assert bound.host_function is not None
        with bound.env:
            pool = CuteChainedMatmulHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
            spec.cute_chained_k_schedule_search_enabled = False
            try:
                old = CuteChainedMatmulHeuristic.get_seed_configs(
                    bound.env, bound.host_function.device_ir
                )
            finally:
                spec.cute_chained_k_schedule_search_enabled = True
            assert pool is not None and old is not None
            assert [seed for seed in pool if KEY not in seed.config] == old
            assert len(pool) == len(old) + 2 and pool[0] == old[0]
            generation = ConfigGeneration(spec)
            flats = generation.random_population_flat(100)
            members = [
                PopulationBasedSearch.make_unbenchmarked(
                    cast(
                        "PopulationBasedSearch", SimpleNamespace(config_gen=generation)
                    ),
                    flat,
                )
                for flat in flats
            ]
            population = [member.config for member in members if member is not None]
            assert len(members) == 100
        for mode in ("serial64", "overlap64"):
            candidate = next(
                item for item in population if item.config.get(KEY) == mode
            )
            assert "chain_k_half" in bound.to_code(candidate)
            assert generation.unflatten(generation.flatten(candidate)) == candidate


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("cache", (False, True))
@pytest.mark.parametrize("unroll", (1, 2, 4, 8))
def test_half_cache_refresh_and_other_knobs(dtype, cache, unroll):
    new = source(
        args=_args(dtype),
        cute_chained_pointwise_read_cache=cache,
        cute_chained_pointwise_unroll=unroll,
        cute_chained_auxiliary_cache=True,
        cute_chained_tmem_early_release=True,
    )
    if cache:
        half = new.index("    for chain_k_half")
        refresh = new.index(
            "chain_k_half * 64 + chain_thread % 8 * 8 + chain_1_a_pointwise_cache_element"
        )
        consume = new.index(
            f"for chain_1_a_pointwise_step in cutlass.range(8, unroll={unroll})"
        )
        assert half < refresh < consume
    assert "chain_allocator.relinquish_alloc_permit()" in new


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_cpu_full_vs_half_typed_expression_and_poison(dtype):
    # Evaluate identical elementwise RHS on identical scalar CPU operands;
    # this is not a claim about GPU exp rounding or reduction association.
    raw = torch.linspace(-3, 3, 16384).view(128, 128).to(dtype)
    coefficient = torch.linspace(-8, 8, 128)
    coefficient[0], coefficient[63], coefficient[64], coefficient[-1] = (
        float("nan"),
        float("inf"),
        -float("inf"),
        -0.0,
    )
    expected = (raw.float() * torch.exp(coefficient)[None, :]).to(dtype)
    actual = torch.full_like(expected, float("nan"))
    old = torch.full((8,), 12345.0)
    for half in range(2):
        seen = set()
        for thread in range(128):
            columns = half * 64 + 8 * (thread % 8) + torch.arange(8)
            cache = torch.exp(coefficient[columns])
            assert not torch.equal(cache, old)
            for step in range(8):
                row = thread // 8 + 16 * step
                actual[row, columns] = (raw[row, columns].float() * cache).to(dtype)
                seen.update((row, int(col)) for col in columns)
        assert len(seen) == 8192
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
