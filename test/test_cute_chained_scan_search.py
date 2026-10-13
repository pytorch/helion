from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_loop_collectives import _inputs
from .test_cute_chained_loop_collectives import _postdot_chain
from .test_cute_chained_loop_collectives import _postdot_inputs
from .test_cute_chained_loop_collectives import _prefix_coefficient_recurrence
from .test_cute_chained_loop_search import _loop_search
from .test_cute_chained_matmul import _scan_modified_chain
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion.autotuner.config_spec import CUTE_CHAINED_SCAN_SCHEDULE_KEY

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.compile_environment import CompileEnvironment
    from helion.runtime.kernel import BoundKernel


@pytest.fixture(scope="module", params=["loop", "root-rank1", "root-rank2"])
def scan_bound(request: pytest.FixtureRequest) -> Iterator[BoundKernel]:
    with _cpu_codegen():
        if request.param == "loop":
            yield _prefix_coefficient_recurrence._bind_isolated(_inputs("cpu", 3, 1))
        else:
            rank = 1 if request.param == "root-rank1" else 2
            yield _postdot_chain._bind_isolated(
                _postdot_inputs("cpu", loop=False, rank=rank, axis=1)
            )


def _config(schedule: object = "warp") -> helion.Config:
    return helion.Config.from_dict(
        {
            "num_warps": 4,
            "cute_chained_mma_schedule": "coalesced",
            CUTE_CHAINED_SCAN_SCHEDULE_KEY: schedule,
        }
    )


def test_general_scan_field_roundtrip_and_override(scan_bound: BoundKernel) -> None:
    spec = scan_bound.config_spec
    assert spec.cute_chained_scan_search_enabled
    assert CUTE_CHAINED_SCAN_SCHEDULE_KEY in spec._flat_fields()
    serial = spec.normalized_config(_config("serial"))
    assert CUTE_CHAINED_SCAN_SCHEDULE_KEY not in serial.config
    warp = spec.normalized_config(_config())
    generation = spec.create_config_generation()
    for config in (serial, warp):
        assert generation.unflatten(generation.flatten(config)) == config
    override = spec.create_config_generation(
        overrides={CUTE_CHAINED_SCAN_SCHEDULE_KEY: "warp"}
    )
    assert override.unflatten(override.flatten(serial))[
        CUTE_CHAINED_SCAN_SCHEDULE_KEY
    ] == ("warp")


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("value", [None, True, 0, 1, "auto", "WARP"])
def test_scan_schedule_strict_types_before_repair(
    scan_bound: BoundKernel, value: object, repair: bool
) -> None:
    with pytest.raises(exc.InvalidConfig, match="must be serial or warp"):
        scan_bound.config_spec.normalize(_config(value), _fix_invalid=repair)


def test_scan_siblings_append_after_all_original_seed_objects(
    scan_bound: BoundKernel,
) -> None:
    assert scan_bound.host_function is not None
    device_ir = scan_bound.host_function.device_ir
    original = CuteChainedMatmulHeuristic._with_collective_schedule_seeds
    captured: list[list[helion.Config]] = []
    collective_results: list[list[helion.Config]] = []

    def record(
        env: CompileEnvironment, seeds: list[helion.Config]
    ) -> list[helion.Config]:
        captured.append(list(seeds))
        result = original(env, seeds)
        collective_results.append(result)
        return result

    with (
        scan_bound.env,
        scan_bound.host_function,
        patch.object(
            CuteChainedMatmulHeuristic,
            "_with_collective_schedule_seeds",
            side_effect=record,
        ),
    ):
        seeds = CuteChainedMatmulHeuristic.get_seed_configs(scan_bound.env, device_ir)
    assert seeds is not None and len(captured) == 1
    assert len(collective_results) == 1
    seeds = collective_results[0]
    prefix = captured[0]
    assert all(first is second for first, second in zip(prefix, seeds, strict=False))
    suffix = seeds[len(prefix) :]
    assert (
        0
        < len(suffix)
        <= (6 if scan_bound.config_spec.cute_chained_loop_search_enabled else 2)
    )
    assert any(
        seed.config.get(CUTE_CHAINED_SCAN_SCHEDULE_KEY) == "warp" for seed in suffix
    )
    assert all(CUTE_CHAINED_SCAN_SCHEDULE_KEY not in seed.config for seed in prefix)
    assert not set(prefix) & set(suffix)
    assert len(set(suffix)) == len(suffix)
    for seed in suffix:
        scan_bound.config_spec.normalized_config(seed)


def test_scan_seed_remains_available_without_tcgen05(scan_bound: BoundKernel) -> None:
    spec = scan_bound.config_spec
    if not spec.cute_chained_loop_search_enabled:
        return
    with patch.object(spec, "cute_chained_tcgen05_search_enabled", False):
        prefix = CuteChainedMatmulHeuristic._loop_seed_configs(scan_bound.env)
        seeds = CuteChainedMatmulHeuristic._with_collective_schedule_seeds(
            scan_bound.env, prefix
        )
        assert len(seeds) == len(prefix) + 1
        assert seeds[-1].config["cute_chained_mma_schedule"] == "coalesced"
        assert seeds[-1].config[CUTE_CHAINED_SCAN_SCHEDULE_KEY] == "warp"
        assert "cute_chained_pointwise_vectorize" not in seeds[-1].config
        spec.normalized_config(seeds[-1])


@pytest.mark.parametrize("repair", [False, True])
def test_scan_cannot_be_ignored_by_affine_or_legacy_family(
    scan_bound: BoundKernel, repair: bool
) -> None:
    spec = scan_bound.config_spec
    affine = _config()
    affine.config["cute_affine_scan_schedule"] = "warp"
    with pytest.raises(exc.InvalidConfig, match="common contraction lowering"):
        spec.normalize(affine, _fix_invalid=repair)
    with patch.object(spec, "cute_chunk_prefill_task_order", object()):
        with pytest.raises(exc.InvalidConfig, match="explicit shared prefill family"):
            spec.normalize(_config(), _fix_invalid=repair)
        seeds = [helion.Config(num_warps=4)]
        assert (
            CuteChainedMatmulHeuristic._with_collective_schedule_seeds(
                scan_bound.env, seeds
            )
            is seeds
        )


def test_loop_vector_seeds_are_bounded_and_do_not_require_a_scan() -> None:
    args = (
        torch.empty((3, 128, 32), dtype=torch.bfloat16),
        torch.empty((3, 32, 128), dtype=torch.bfloat16),
        torch.empty((3, 32, 128), dtype=torch.bfloat16),
        torch.empty((128, 128), dtype=torch.float32),
    )
    with _cpu_codegen():
        bound = _loop_search._bind_isolated(args)
        spec = bound.config_spec
        assert not spec.cute_chained_scan_search_enabled
        assert CUTE_CHAINED_SCAN_SCHEDULE_KEY not in spec._flat_fields()
        prefix = CuteChainedMatmulHeuristic._loop_seed_configs(bound.env)
        seeds = CuteChainedMatmulHeuristic._with_collective_schedule_seeds(
            bound.env, prefix
        )
        assert all(
            first is second for first, second in zip(prefix, seeds, strict=False)
        )
        suffix = seeds[len(prefix) :]
        assert len(suffix) == 2
        assert {seed.config["cute_chained_group_contractions"] for seed in suffix} == {
            False,
            True,
        }
        for seed in suffix:
            assert seed.num_warps == 16
            assert seed.config["cute_chained_pointwise_vectorize"] is True
            assert seed.config["cute_chained_mma_schedule"] == "tcgen05_tmem"
            assert CUTE_CHAINED_SCAN_SCHEDULE_KEY not in seed.config
            spec.normalized_config(seed)
        for repair in (False, True):
            with pytest.raises(
                exc.InvalidConfig, match="eligible general contraction scan"
            ):
                spec.normalize(_config(), _fix_invalid=repair)


def test_legacy_rank_one_prelude_keeps_fields_and_seeds_unchanged() -> None:
    args = (
        torch.empty((2, 32, 16), dtype=torch.bfloat16),
        torch.empty((2, 32, 16), dtype=torch.bfloat16),
        torch.empty((2, 32, 16), dtype=torch.bfloat16),
        torch.empty((2, 32), dtype=torch.float32),
    )
    with _cpu_codegen():
        bound = _scan_modified_chain._bind_isolated(args)
        spec = bound.config_spec
        assert spec.cute_chained_matmul_search_enabled
        assert not spec.cute_chained_loop_search_enabled
        assert not spec.cute_chained_scan_search_enabled
        assert CUTE_CHAINED_SCAN_SCHEDULE_KEY not in spec._flat_fields()
        seeds = list(spec.compiler_seed_configs)
        assert (
            CuteChainedMatmulHeuristic._with_collective_schedule_seeds(bound.env, seeds)
            is seeds
        )
        plain = helion.Config(num_warps=4, cute_chained_mma_schedule="coalesced")
        assert spec.normalized_config(_config("serial")) == spec.normalized_config(
            plain
        )
        assert bound.to_code(_config("serial")) == bound.to_code(plain)
        for repair in (False, True):
            with pytest.raises(
                exc.InvalidConfig, match="eligible general contraction scan"
            ):
                spec.normalize(_config(), _fix_invalid=repair)
