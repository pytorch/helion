from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_warp_pair import _args
from .test_cute_chained_warp_pair import _pair
from .test_cute_chained_warp_result_transport import _code
from .test_cute_chained_warp_result_transport import _gemms
import helion
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute.chained_matmul import _UnsupportedChain
import helion.language as hl

PAIR = "cute_chained_warp_register_a"
TRANSPORT = "cute_chained_warp_result_transport"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _three(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    output = torch.empty(
        (a.size(0), a.size(1), d.size(2)), device=a.device, dtype=a.dtype
    )
    for batch in hl.tile(a.size(0), block_size=1):
        first = hl.dot(a[batch.id, :, :], b[batch.id, :, :])
        second = hl.dot((-first).to(a.dtype), c[batch.id, :, :])
        third = hl.dot(second.to(a.dtype), d[batch.id, :, :])
        output[batch.id, :, :] = third
    return output


@pytest.mark.parametrize("schedule", ["coalesced", "cp_async"])
def test_direct_preserves_internal_storage_math_and_reader_join(schedule: str) -> None:
    legacy, old = _code("legacy", cute_chained_mma_schedule=schedule)
    direct, plan = _code("direct", cute_chained_mma_schedule=schedule)
    assert plan.warp_workspace is old.warp_workspace is None
    transport = plan.warp_result_transport
    assert transport is not None and transport.workspace is None
    assert _gemms(legacy) == _gemms(direct)
    assert legacy.count("alloc_smem(") == direct.count("alloc_smem(") + 1
    assert legacy.count("cute.autovec_copy(") == direct.count("cute.autovec_copy(") + 1
    assert direct.count("sync_threads()") == legacy.count("sync_threads()") + 1
    assert "chain_2_c_ptr" not in direct and "chain_c_workspace" not in direct
    assert "chain_0_c_ptr" in direct and "chain_1_c_ptr" in direct
    with pytest.raises(_UnsupportedChain, match="proof changed"):
        replace(transport, mode="packed_direct").check(plan)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [16, 32])
def test_direct_composes_with_internal_register_pair(
    dtype: torch.dtype, rows: int
) -> None:
    args = [torch.empty((2, rows, 32), dtype=dtype)]
    args.extend(torch.empty((2, 32, 32), dtype=dtype) for _ in range(3))
    config = {
        "block_sizes": [],
        "num_warps": 2,
        "cute_chained_mma_schedule": "coalesced",
        "cute_chained_warp_register_a": True,
    }
    with _cpu_codegen():
        bound = _three._bind_isolated(tuple(args))
        legacy = bound.to_code(helion.Config.from_dict(config))
        direct = bound.to_code(
            helion.Config.from_dict(
                dict(config, cute_chained_warp_result_transport="direct")
            )
        )
    assert "chain_pair_0_a" in direct and "chain_1_copy_a =" not in direct
    assert "chain_0_c_ptr" in direct and "chain_1_c_ptr" in direct
    assert "chain_2_c_ptr" not in direct and "chain_2_terminal_coords" in direct
    assert "chain_c_workspace" not in direct
    assert _gemms(legacy) == _gemms(direct)
    assert f"< {rows}" in direct


def test_direct_terminal_preserves_narrow_internal_stages() -> None:
    args = [torch.empty((2, 16, 16), dtype=torch.bfloat16) for _ in range(3)]
    args.append(torch.empty((2, 16, 64), dtype=torch.bfloat16))
    request = {
        "block_sizes": [],
        "num_warps": 4,
        "cute_chained_mma_schedule": "coalesced",
        "cute_chained_warp_register_a": True,
    }
    with _cpu_codegen():
        bound = _three._bind_isolated(tuple(args))
        legacy = bound.to_code(helion.Config.from_dict(request))
        direct = bound.to_code(
            helion.Config.from_dict(
                dict(request, cute_chained_warp_result_transport="direct")
            )
        )
    assert _gemms(legacy) == _gemms(direct)
    assert (
        legacy.count("if chain_thread < 32:")
        == direct.count("if chain_thread < 32:")
        == 2
    )
    assert "chain_pair_0_a" in direct
    assert "chain_0_c_ptr" in direct and "chain_1_c_ptr" in direct
    assert "chain_2_c_ptr" not in direct and "chain_2_terminal_coords" in direct
    assert "chain_c_workspace" not in direct
    assert direct.count("sync_threads()") == legacy.count("sync_threads()") + 1


def test_static_small_unary_pair_is_searchable_and_seeds_append() -> None:
    args = tuple(torch.empty((2, 32, 32), dtype=torch.bfloat16) for _ in range(4))
    with _cpu_codegen():
        bound = _three._bind_isolated(args)
        spec = bound.config_spec
        assert spec.cute_chained_warp_register_a_search_enabled
        assert PAIR in spec._flat_fields()
        seeds = spec.compiler_seed_configs
        extras = [seed for seed in seeds if seed.config.get(PAIR)]
        assert len(extras) == 2
        assert extras[0].config.get(TRANSPORT, "legacy") == "legacy"
        assert extras[1].config[TRANSPORT] == "direct"
        base = seeds[:-2]
        assert not any(seed.config.get(PAIR) for seed in base)
        assert (
            CuteChainedMatmulHeuristic._with_warp_register_a_seeds(bound.env, base)
            == seeds
        )
        for seed in extras:
            # Fixed flat PID type is omitted by generic flattening.
            normalized = spec.normalized_config(
                {key: value for key, value in seed.config.items() if key != "pid_type"}
            )
            generation = spec.create_config_generation()
            assert generation.unflatten(generation.flatten(normalized)) == normalized
        for override in ({PAIR: False}, {TRANSPORT: "packed"}):
            with patch.dict(bound.env.settings.autotune_config_overrides, override):
                assert (
                    CuteChainedMatmulHeuristic._with_warp_register_a_seeds(
                        bound.env, base
                    )
                    == base
                )


def test_fanout_and_large_operands_do_not_expand_search() -> None:
    with _cpu_codegen():
        args, _ = _args(torch.bfloat16, 32, 32, 32)
        fanout = _pair._bind_isolated((*args, True))
        assert not fanout.config_spec.cute_chained_warp_register_a_search_enabled
        assert PAIR not in fanout.config_spec._flat_fields()
        large = _three._bind_isolated(
            tuple(torch.empty((2, 64, 64), dtype=torch.float16) for _ in range(4))
        )
        assert not large.config_spec.cute_chained_warp_register_a_search_enabled
        assert PAIR not in large.config_spec._flat_fields()
