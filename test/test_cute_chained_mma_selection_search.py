from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_residency_search import _loop_residency
from .test_cute_chained_residency_search import _root_residency
from .test_cute_chunk_recurrence import _CONFIG as recurrence_config
from .test_cute_chunk_recurrence import _bt16_fp32_chain
from .test_cute_chunk_recurrence import _fake_inputs
from .test_cute_chunk_recurrence import _real_dispatch_cpu
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion.autotuner.config_spec import CUTE_CHAINED_WARP_MMA_ROWS_KEY
from helion.autotuner.config_spec import EnumFragment
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.compile_environment import CompileEnvironment
    from helion.runtime.kernel import BoundKernel


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _selection_loop(a, b, initial, row_block: hl.constexpr, seeded: hl.constexpr):
    steps, m, k = a.shape
    n = b.size(-1)
    output = torch.empty_like(initial)
    history = torch.empty((steps, m, n), device=a.device, dtype=torch.float32)
    for rows, cols in hl.tile([m, n], block_size=[row_block, 32]):
        state = initial[rows, cols]
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(k)
            if seeded:
                product = hl.dot(
                    a[step.id, rows, kk],
                    b[step.id, kk, cols],
                    acc=state,
                    out_dtype=torch.float32,
                )
            else:
                product = hl.dot(
                    a[step.id, rows, kk],
                    b[step.id, kk, cols],
                    out_dtype=torch.float32,
                )
            state = product + state * 0.5
            history[step.id, rows, cols] = state
        output[rows, cols] = state
    return history, output


@pytest.fixture(scope="module")
def selection_bound() -> Iterator[BoundKernel]:
    with _cpu_codegen():
        yield _loop_residency._bind_isolated(
            (
                torch.empty((3, 128, 16), dtype=torch.bfloat16),
                torch.empty((3, 16, 32), dtype=torch.bfloat16),
                torch.empty((128, 32), dtype=torch.float32),
            )
        )


def _config(rows: object = 32, **overrides: object) -> helion.Config:
    return helion.Config.from_dict(
        {
            "num_warps": 16,
            "cute_chained_mma_schedule": "tcgen05_tmem",
            CUTE_CHAINED_WARP_MMA_ROWS_KEY: rows,
            **overrides,
        }
    )


def test_selection_field_roundtrip_and_override(selection_bound: BoundKernel) -> None:
    spec = selection_bound.config_spec
    assert spec.cute_chained_warp_mma_search_enabled
    field = spec._flat_fields()[CUTE_CHAINED_WARP_MMA_ROWS_KEY]
    assert isinstance(field, EnumFragment)
    assert field.search_values(10) == [0, 16, 32, 64, 128]
    generation = spec.create_config_generation()
    for rows in (0, 16, 32, 64, 128):
        config = spec.normalized_config(_config(rows))
        assert config.config.get(CUTE_CHAINED_WARP_MMA_ROWS_KEY, 0) == rows
        assert generation.unflatten(generation.flatten(config)) == config
    zero = spec.normalized_config(_config(0))
    assert CUTE_CHAINED_WARP_MMA_ROWS_KEY not in zero.config
    override = spec.create_config_generation(
        overrides={CUTE_CHAINED_WARP_MMA_ROWS_KEY: 64}
    )
    assert (
        override.unflatten(override.flatten(zero))[CUTE_CHAINED_WARP_MMA_ROWS_KEY] == 64
    )


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("rows", [True, False, None, -1, 1, 48, 256, 32.0, "32"])
def test_selection_strict_thresholds_before_repair(
    selection_bound: BoundKernel, rows: object, repair: bool
) -> None:
    with pytest.raises(exc.InvalidConfig, match="must be 0, 16, 32, 64 or 128"):
        selection_bound.config_spec.normalize(_config(rows), _fix_invalid=repair)


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "overrides",
    [
        {"cute_chained_mma_schedule": "coalesced"},
        {"cute_chained_mma_schedule": "cp_async"},
        {"cute_affine_scan_schedule": "warp"},
        {"cute_chained_direct_output": True},
    ],
)
def test_selection_rejects_other_emitters_before_repair(
    selection_bound: BoundKernel, overrides: dict[str, object], repair: bool
) -> None:
    with pytest.raises(exc.InvalidConfig, match="requires explicit tcgen05_tmem"):
        selection_bound.config_spec.normalize(_config(**overrides), _fix_invalid=repair)


@pytest.mark.parametrize("repair", [False, True])
def test_selection_requires_explicit_schedule_and_discovery(
    selection_bound: BoundKernel, repair: bool
) -> None:
    config = _config()
    config.config.pop("cute_chained_mma_schedule")
    with pytest.raises(exc.InvalidConfig, match="requires explicit tcgen05_tmem"):
        selection_bound.config_spec.normalize(config, _fix_invalid=repair)
    with (
        patch.object(
            selection_bound.config_spec, "cute_chained_warp_mma_search_enabled", False
        ),
        pytest.raises(exc.InvalidConfig, match="eligible uninitialized contraction"),
    ):
        selection_bound.config_spec.normalize(_config(), _fix_invalid=repair)


@pytest.mark.parametrize(
    "rows,seeded,eligible",
    [(16, False, True), (128, False, True), (256, False, False), (16, True, False)],
)
def test_selection_discovery_respects_fixed_rows_and_explicit_accumulators(
    rows: int, seeded: bool, eligible: bool
) -> None:
    with _cpu_codegen():
        bound = _selection_loop._bind_isolated(
            (
                torch.empty((3, rows, 16), dtype=torch.bfloat16),
                torch.empty((3, 16, 32), dtype=torch.bfloat16),
                torch.empty((rows, 32), dtype=torch.float32),
                rows,
                seeded,
            )
        )
        spec = bound.config_spec
        assert spec.cute_chained_loop_search_enabled
        assert spec.cute_chained_warp_mma_search_enabled is eligible
        assert (CUTE_CHAINED_WARP_MMA_ROWS_KEY in spec._flat_fields()) is eligible
        if not eligible:
            with pytest.raises(
                exc.InvalidConfig, match="eligible uninitialized contraction"
            ):
                spec.normalize(_config(), _fix_invalid=True)


def test_selection_not_advertised_for_root_and_zero_preserves_source() -> None:
    with _cpu_codegen():
        bound = _root_residency._bind_isolated(
            (
                torch.empty((128, 16), dtype=torch.bfloat16),
                torch.empty((16, 32), dtype=torch.bfloat16),
            )
        )
        assert not bound.config_spec.cute_chained_warp_mma_search_enabled
        assert CUTE_CHAINED_WARP_MMA_ROWS_KEY not in bound.config_spec._flat_fields()
        plain = helion.Config(num_warps=4, cute_chained_mma_schedule="coalesced")
        zero = helion.Config.from_dict(
            plain.config | {CUTE_CHAINED_WARP_MMA_ROWS_KEY: 0}
        )
        assert bound.to_code(zero) == bound.to_code(plain)
        for repair in (False, True):
            with pytest.raises(
                exc.InvalidConfig, match="requires explicit tcgen05_tmem"
            ):
                bound.config_spec.normalize(_config(num_warps=4), _fix_invalid=repair)


def test_selection_zero_does_not_claim_explicit_legacy_recurrence() -> None:
    with _real_dispatch_cpu():
        bound = _bt16_fp32_chain._bind_isolated(
            (*_fake_inputs(fp32_state=True), 128**-0.5)
        )
        plain = helion.Config.from_dict(
            recurrence_config.config | {"cute_chunk_recurrence_dv_partitions": 4}
        )
        zero = helion.Config.from_dict(
            plain.config | {CUTE_CHAINED_WARP_MMA_ROWS_KEY: 0}
        )
        normalized = bound.config_spec.normalized_config(zero)
        assert "cute_chained_mma_schedule" not in normalized.config
        assert CUTE_CHAINED_WARP_MMA_ROWS_KEY not in normalized.config
        assert bound.config_spec.normalized_config(normalized) == normalized
        assert bound.to_code(zero) == bound.to_code(plain)
        positive = helion.Config.from_dict(
            plain.config | {CUTE_CHAINED_WARP_MMA_ROWS_KEY: 32}
        )
        for repair in (False, True):
            with pytest.raises(
                exc.InvalidConfig, match="requires explicit tcgen05_tmem"
            ):
                bound.config_spec.normalize(positive, _fix_invalid=repair)


def test_selection_seed_suffix_preserves_complete_old_pool(
    selection_bound: BoundKernel,
) -> None:
    original = CuteChainedMatmulHeuristic._with_loop_warp_mma_seeds
    captured: list[list[helion.Config]] = []

    def record(
        env: CompileEnvironment, seeds: list[helion.Config]
    ) -> list[helion.Config]:
        captured.append(list(seeds))
        return original(env, seeds)

    assert selection_bound.host_function is not None
    with (
        selection_bound.env,
        selection_bound.host_function,
        patch.object(
            CuteChainedMatmulHeuristic, "_with_loop_warp_mma_seeds", side_effect=record
        ),
        patch.object(
            selection_bound.config_spec,
            "cute_chained_preparation_pipeline_search_enabled",
            False,
        ),
    ):
        seeds = CuteChainedMatmulHeuristic.get_seed_configs(
            selection_bound.env, selection_bound.host_function.device_ir
        )
    assert seeds is not None and len(captured) == 1
    prefix = captured[0]
    assert all(left is right for left, right in zip(prefix, seeds, strict=False))
    suffix = seeds[len(prefix) :]
    assert len(suffix) == len(set(suffix)) == 2
    assert not set(prefix) & set(suffix)
    assert all(CUTE_CHAINED_WARP_MMA_ROWS_KEY not in seed.config for seed in prefix)
    for seed in suffix:
        assert seed[CUTE_CHAINED_WARP_MMA_ROWS_KEY] == 32
        assert seed.num_warps == 16
        grouped = seed.config.get("cute_chained_group_contractions", False)
        parents = [
            parent
            for parent in prefix
            if parent.config.get("cute_chained_mma_schedule") == "tcgen05_tmem"
            and parent.num_warps == 16
            and parent.config.get("cute_chained_group_contractions", False) == grouped
        ]
        expected = max(
            reversed(parents), key=lambda parent: math.prod(parent.block_sizes)
        )
        assert seed.config == expected.config | {CUTE_CHAINED_WARP_MMA_ROWS_KEY: 32}
        assert seed["cute_chained_pointwise_unroll"] == 8
        assert seed["cute_chained_pointwise_cache_bytes"] == 4096
        assert seed["cute_chained_scan_schedule"] == "warp"
        selection_bound.config_spec.normalized_config(seed)
    spec = selection_bound.config_spec
    for fact in (
        "cute_chained_loop_search_enabled",
        "cute_chained_warp_mma_search_enabled",
    ):
        with patch.object(spec, fact, False):
            assert original(selection_bound.env, prefix) is prefix
    with patch.object(spec, "cute_chunk_prefill_task_order", object()):
        assert original(selection_bound.env, prefix) is prefix
