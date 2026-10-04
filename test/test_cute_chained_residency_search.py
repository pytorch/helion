from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_loop_search import _loop_search
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion.autotuner.config_spec import CUTE_CHAINED_POINTWISE_CACHE_BYTES_KEY
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.compile_environment import CompileEnvironment
    from helion.runtime.kernel import BoundKernel


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _root_residency(a, b):
    m, k = a.shape
    n = b.shape[-1]
    output = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for rows, cols in hl.tile([m, n], block_size=[128, 32]):
        kk = hl.arange(k)
        repeated = torch.exp(a[rows, kk].float()).to(a.dtype)
        first = hl.dot(repeated, b[kk, cols], out_dtype=torch.float32)
        output[rows, cols] = hl.dot(
            repeated, b[kk, cols], acc=first, out_dtype=torch.float32
        )
    return output


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _loop_residency(a, b, initial):
    steps, m, k = a.shape
    n = b.shape[-1]
    history = torch.empty((steps, m, n), dtype=torch.float32, device=a.device)
    output = torch.empty_like(initial)
    for rows, cols in hl.tile([m, n], block_size=[128, 32]):
        state = initial[rows, cols]
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(k)
            coefficient = torch.exp(hl.cumsum(a[step.id, 0, kk].float(), dim=0))
            repeated = (a[step.id, rows, kk].float() * coefficient[None, :]).to(a.dtype)
            first = hl.dot(repeated, b[step.id, kk, cols], out_dtype=torch.float32)
            state = hl.dot(
                repeated,
                b[step.id, kk, cols],
                acc=state + first,
                out_dtype=torch.float32,
            )
            history[step.id, rows, cols] = state
        output[rows, cols] = state
    return history, output


@pytest.fixture(scope="module", params=["root", "loop"])
def residency_bound(request: pytest.FixtureRequest) -> Iterator[BoundKernel]:
    with _cpu_codegen():
        if request.param == "root":
            yield _root_residency._bind_isolated(
                (
                    torch.empty((128, 16), dtype=torch.bfloat16),
                    torch.empty((16, 32), dtype=torch.bfloat16),
                )
            )
        else:
            yield _loop_residency._bind_isolated(
                (
                    torch.empty((3, 128, 16), dtype=torch.bfloat16),
                    torch.empty((3, 16, 32), dtype=torch.bfloat16),
                    torch.empty((128, 32), dtype=torch.float32),
                )
            )


def _config(budget: object = 4096, **overrides: object) -> helion.Config:
    return helion.Config.from_dict(
        {
            "num_warps": 4,
            "cute_chained_mma_schedule": "coalesced",
            CUTE_CHAINED_POINTWISE_CACHE_BYTES_KEY: budget,
            **overrides,
        }
    )


def test_residency_field_and_budget_roundtrip(residency_bound: BoundKernel) -> None:
    spec = residency_bound.config_spec
    assert spec.cute_chained_pointwise_residency_search_enabled
    assert CUTE_CHAINED_POINTWISE_CACHE_BYTES_KEY in spec._flat_fields()
    generation = spec.create_config_generation()
    for budget in (0, 4096, 16384):
        config = spec.normalized_config(_config(budget))
        assert config.config.get(CUTE_CHAINED_POINTWISE_CACHE_BYTES_KEY, 0) == budget
        assert generation.unflatten(generation.flatten(config)) == config
    assert (
        CUTE_CHAINED_POINTWISE_CACHE_BYTES_KEY
        not in spec.normalized_config(_config(0)).config
    )
    override = spec.create_config_generation(
        overrides={CUTE_CHAINED_POINTWISE_CACHE_BYTES_KEY: 16384}
    )
    plain = spec.normalized_config(_config(0))
    assert (
        override.unflatten(override.flatten(plain))[
            CUTE_CHAINED_POINTWISE_CACHE_BYTES_KEY
        ]
        == 16384
    )


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("budget", [True, False, None, -1, 128, 4096.0, "4096"])
def test_residency_budget_strict_integer_before_repair(
    residency_bound: BoundKernel, budget: object, repair: bool
) -> None:
    with pytest.raises(exc.InvalidConfig, match="must be 0, 4096 or 16384"):
        residency_bound.config_spec.normalize(_config(budget), _fix_invalid=repair)


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "overrides",
    [
        {"cute_affine_scan_schedule": "warp"},
        {"cute_chained_direct_output": True},
    ],
)
def test_residency_cannot_disappear_in_other_lowering(
    residency_bound: BoundKernel, overrides: dict[str, object], repair: bool
) -> None:
    with pytest.raises(exc.InvalidConfig):
        residency_bound.config_spec.normalize(_config(**overrides), _fix_invalid=repair)


def test_residency_zero_preserves_existing_source(residency_bound: BoundKernel) -> None:
    plain = helion.Config(num_warps=4, cute_chained_mma_schedule="coalesced")
    assert residency_bound.to_code(_config(0)) == residency_bound.to_code(plain)


def test_residency_seed_suffix_is_bounded_and_preserves_objects(
    residency_bound: BoundKernel,
) -> None:
    assert residency_bound.host_function is not None
    original = CuteChainedMatmulHeuristic._with_pointwise_residency_seeds
    captured: list[list[helion.Config]] = []
    residency_results: list[list[helion.Config]] = []

    def record(
        env: CompileEnvironment, seeds: list[helion.Config]
    ) -> list[helion.Config]:
        captured.append(list(seeds))
        result = original(env, seeds)
        residency_results.append(result)
        return result

    with (
        residency_bound.env,
        residency_bound.host_function,
        patch.object(
            CuteChainedMatmulHeuristic,
            "_with_pointwise_residency_seeds",
            side_effect=record,
        ),
    ):
        seeds = CuteChainedMatmulHeuristic.get_seed_configs(
            residency_bound.env, residency_bound.host_function.device_ir
        )
    assert seeds is not None and len(captured) == 1
    assert len(residency_results) == 1
    seeds = residency_results[0]
    prefix = captured[0]
    assert all(left is right for left, right in zip(prefix, seeds, strict=False))
    suffix = seeds[len(prefix) :]
    assert len(suffix) == 2
    assert not set(prefix) & set(suffix)
    assert len(set(suffix)) == len(suffix)
    for seed in suffix:
        assert seed[CUTE_CHAINED_POINTWISE_CACHE_BYTES_KEY] == 4096
        assert not seed.config.get("cute_chained_initialized_accumulator")
        assert not seed.config.get("cute_chained_direct_output")
        residency_bound.config_spec.normalized_config(seed)
    if residency_bound.config_spec.cute_chained_loop_search_enabled:
        assert {seed["cute_chained_group_contractions"] for seed in suffix} == {
            False,
            True,
        }
        for seed in suffix:
            assert seed.num_warps == 16
            assert seed["cute_chained_pointwise_vectorize"] is True
            assert seed["cute_chained_scan_schedule"] == "warp"
    else:
        assert {seed["cute_chained_mma_schedule"] for seed in suffix} == {
            "coalesced",
            "tcgen05_tmem",
        }


@pytest.mark.parametrize("repair", [False, True])
def test_residency_legacy_family_rejects_and_keeps_seed_prefix(
    residency_bound: BoundKernel, repair: bool
) -> None:
    spec = residency_bound.config_spec
    with patch.object(spec, "cute_chunk_prefill_task_order", object()):
        with pytest.raises(exc.InvalidConfig, match="explicit shared prefill family"):
            spec.normalize(_config(), _fix_invalid=repair)
        seeds = [helion.Config(num_warps=4)]
        assert (
            CuteChainedMatmulHeuristic._with_pointwise_residency_seeds(
                residency_bound.env, seeds
            )
            is seeds
        )


def test_shared_loads_without_arithmetic_do_not_enable_residency() -> None:
    with _cpu_codegen():
        bound = _loop_search._bind_isolated(
            (
                torch.empty((3, 128, 32), dtype=torch.bfloat16),
                torch.empty((3, 32, 128), dtype=torch.bfloat16),
                torch.empty((3, 32, 128), dtype=torch.bfloat16),
                torch.empty((128, 128), dtype=torch.float32),
            )
        )
        spec = bound.config_spec
        assert not spec.cute_chained_pointwise_residency_search_enabled
        assert CUTE_CHAINED_POINTWISE_CACHE_BYTES_KEY not in spec._flat_fields()
        prefix = list(spec.compiler_seed_configs)
        assert (
            CuteChainedMatmulHeuristic._with_pointwise_residency_seeds(
                bound.env, prefix
            )
            is prefix
        )
        for repair in (False, True):
            with pytest.raises(exc.InvalidConfig, match="reused common-region"):
                spec.normalize(_config(), _fix_invalid=repair)
