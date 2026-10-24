from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _forbid_native_compile
from test.test_cute_bounded_cache_codegen import _cpu_target
from test.test_cute_full_chunk_row_seeds import bind
from test.test_cute_full_chunk_row_seeds import column_sum
from test.test_cute_full_chunk_row_seeds import seeds_for

import helion
from helion._compiler.autotuner_heuristics.cute_full_chunk_row import (
    CuteFullChunkRowHeuristic,
)
from helion._compiler.autotuner_heuristics.cute_rolled_cluster_packet import (
    CuteRolledClusterPacketHeuristic,
)
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.device_ir import DeviceIR

KEY = "cute_independent_reduction"


@pytest.fixture(autouse=True)
def cpu_only() -> Iterator[None]:
    initialized = torch.cuda.is_initialized()
    # The packet-cluster suite checks the appended family's integration; keep
    # this suite's original independent-carrier prefix assertions isolated.
    with (
        _cpu_target(),
        _forbid_native_compile() as native,
        patch.object(
            CuteRolledClusterPacketHeuristic, "get_seed_configs", return_value=[]
        ),
    ):
        yield
        assert all(mock.call_count == 0 for mock in native)
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize(
    ("size", "dtype", "static", "layout"),
    [
        (4, torch.bfloat16, True, "dense"),
        (129, torch.bfloat16, True, "dense"),
        (4096, torch.bfloat16, True, "dense"),
        (4097, torch.float32, True, "dense"),
        (7168, torch.float32, False, "dense"),
        (7168, torch.float64, True, "dense"),
        (4096, torch.bfloat16, True, "offset"),
        (4096, torch.bfloat16, True, "strided"),
    ],
)
def test_one_reference_carrier_preserves_all_prior_seeds_and_default(
    size: int, dtype: torch.dtype, static: bool, layout: str
) -> None:
    original = CuteFullChunkRowHeuristic.get_seed_configs

    def prior(env: CompileEnvironment, device_ir: DeviceIR) -> list[helion.Config]:
        with patch.object(
            CuteFullChunkRowHeuristic, "_low_thread_seeds", return_value=[]
        ):
            return [
                seed for seed in original(env, device_ir) if not seed.get(KEY, False)
            ]

    with patch.object(CuteFullChunkRowHeuristic, "get_seed_configs", side_effect=prior):
        old = bind(3, size, dtype, static=static, layout=layout)
    bound = bind(3, size, dtype, static=static, layout=layout)
    with bound.env:
        additions = CuteFullChunkRowHeuristic._low_thread_seeds(
            bound.env, bound.host_function.device_ir
        )
    with old.env:
        old_reference = old.config_spec.autotune_reference_config()
        old_default = old.config_spec.default_config()
        assert old.config_spec._flat_fields()[KEY].search_values() == [False]
    with bound.env:
        spec = bound.config_spec
        reference = spec.autotune_reference_config()
        carrier = helion.Config.from_dict(reference.config | {KEY: True})
        assert spec.compiler_seed_configs == [
            *old.config_spec.compiler_seed_configs,
            carrier,
            *additions,
        ]
        assert spec.compiler_default_config == old.config_spec.compiler_default_config
        assert reference == old_reference
        assert spec.default_config() == old_default
        field = spec._flat_fields()[KEY]
        assert field.search_values() == [False, True]
        assert field.pattern_neighbors(False) == [True]
        generation = ConfigGeneration(spec)
        flat, effective = generation.strict_config_pair(carrier)
        assert generation.strict_unflatten(flat) == effective
        # Toggling only this coordinate is legal on every prior compiler seed;
        # discovery is not confined to the carrier's reference geometry.
        for seed in old.config_spec.compiler_seed_configs:
            flat, effective = generation.strict_config_pair(
                helion.Config.from_dict(seed.config | {KEY: True})
            )
            assert generation.strict_unflatten(flat) == effective
    assert bound.to_code(old_default) == old.to_code(old_default)
    assert sum(seed.get(KEY, False) for seed in seeds_for(bound)) == 1 + len(additions)
    assert len(seeds_for(bound)) <= 17


@pytest.mark.parametrize("domain", [None, (False,), (False, True)])
def test_carrier_uses_live_legal_domain_not_current_search_subset(
    domain: tuple[bool, ...] | None,
) -> None:
    bound = bind(1, 4096, torch.bfloat16)
    spec = bound.config_spec
    with bound.env:
        fields = spec._flat_fields()
    if domain is None:
        fields.pop(KEY)
    else:
        fields[KEY] = EnumFragment(domain, search_choices=(False,))
    with patch.object(spec, "_flat_fields", return_value=fields):
        seeds = seeds_for(bound)
    assert sum(seed.get(KEY, False) for seed in seeds) == (
        9 if domain == (False, True) else 0
    )
    assert len(seeds) == (17 if domain == (False, True) else 8)


def test_narrowed_defaults_remain_reference_geometry() -> None:
    bound = bind(1, 4096, torch.bfloat16)
    spec = bound.config_spec
    thread = spec.num_threads.block_id_lookup(spec.reduction_loops[0].block_id)
    with (
        patch.object(thread, "_fragment", return_value=EnumFragment((32,))),
        patch.object(
            spec.cute_reduction_reloads[0],
            "_fragment",
            return_value=EnumFragment(("gmem",)),
        ),
    ):
        seeds = seeds_for(bound)
        carrier = seeds[-1]
        with bound.env:
            assert carrier == helion.Config.from_dict(
                spec.autotune_reference_config().config | {KEY: True}
            )
            generation = ConfigGeneration(spec)
            flat, effective = generation.strict_config_pair(carrier)
            assert generation.strict_unflatten(flat) == effective
            assert effective.num_threads[1] == 32
            assert effective["cute_reduction_reloads"] == ["gmem"]


def test_structural_carrier_does_not_require_row_memory_layout() -> None:
    bound = helion.kernel(
        column_sum, backend="cute", static_shapes=True
    )._bind_isolated((torch.empty((4096, 16)),))
    assert not seeds_for(bound, packets_only=True)
    with bound.env:
        expected = helion.Config.from_dict(
            bound.config_spec.autotune_reference_config().config | {KEY: True}
        )
    assert seeds_for(bound)[-1] == expected


def pointwise(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row, column in hl.tile(x.size()):
        out[row, column] = x[row, column] * 2
    return out


def test_no_reduction_has_no_carrier() -> None:
    bound = helion.kernel(pointwise, backend="cute", static_shapes=True)._bind_isolated(
        (torch.empty((3, 4096)),)
    )
    assert not seeds_for(bound)


def test_carrier_requires_strict_roundtrip() -> None:
    bound = bind(1, 4096, torch.bfloat16)
    with bound.env:
        reference = bound.config_spec.autotune_reference_config()
    with patch.object(ConfigGeneration, "strict_unflatten", return_value=reference):
        seeds = seeds_for(bound)
    assert len(seeds) == 8
    assert not any(seed.get(KEY, False) for seed in seeds)


def test_already_active_seed_is_not_duplicated() -> None:
    bound = bind(1, 4096, torch.bfloat16)
    with bound.env:
        carrier = helion.Config.from_dict(
            bound.config_spec.autotune_reference_config().config
            | {KEY: True, "cute_replicated_reduction": True}
        )
    with patch.object(
        CuteFullChunkRowHeuristic, "_packet_seeds", return_value=[carrier]
    ):
        assert seeds_for(bound) == [carrier]
