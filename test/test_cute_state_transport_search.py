from __future__ import annotations

from itertools import product
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._compiler.autotuner_heuristics.cute import CuteChunkPrefillHeuristic
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.triton.backend import TritonBackend
from helion.autotuner.config_spec import CUTE_CHUNK_PREFILL_SCHEDULE_KEY as SCHEDULE
from helion.autotuner.config_spec import CUTE_CHUNK_PREFILL_TASK_ORDER_KEY as ORDER
from helion.autotuner.config_spec import CUTE_STATE_TRANSFER_MAX_BITS_KEY as WIDTH
from helion.autotuner.config_spec import CUTE_STATE_TRANSFER_TRANSPORT_KEY as TRANSPORT
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import EnumFragment
from helion.exc import InvalidConfig

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.device_ir import DeviceIR


@pytest.fixture
def spec() -> Iterator[ConfigSpec]:
    with _cpu_codegen():
        yield ConfigSpec(
            backend=CuteBackend(),
            device=torch.device("cpu"),
            target_device_capability=(10, 3),
            num_sm=148,
        )


def _seeds(spec: ConfigSpec) -> list[helion.Config]:
    result = CuteChunkPrefillHeuristic.get_seed_configs(
        cast("CompileEnvironment", SimpleNamespace(config_spec=spec)),
        cast("DeviceIR", None),
    )
    assert result is not None
    return result


@pytest.mark.parametrize("prefill", (False, True))
def test_transport_default_is_omitted_and_coordinate_is_appended(
    spec: ConfigSpec, prefill: bool
) -> None:
    spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=64))
    if prefill:
        spec.enable_cute_chunk_prefill_task_order_search()
    spec.enable_cute_state_transfer_search()
    spec.user_defined_tunables["user_choice"] = EnumFragment((3, 5))
    old_fields = tuple(spec._flat_fields())
    old_default = spec.default_config()
    assert spec.cute_state_transfer_transport is None
    spec.enable_cute_state_transport_search()
    assert tuple(spec._flat_fields()) == (*old_fields, TRANSPORT)
    assert spec.default_config().to_json() == old_default.to_json()
    assert helion.Config().cute_state_transfer_transport == "register"
    assert TRANSPORT not in helion.Config()
    generation = spec.create_config_generation()
    for transport in ("register", "tma", "tma_pipelined", "tma_planar"):
        requested = helion.Config.from_dict({**old_default, TRANSPORT: transport})
        normalized = spec.normalized_config(requested)
        assert requested[TRANSPORT] == transport
        assert normalized.cute_state_transfer_transport == transport
        assert (TRANSPORT in normalized) is (transport != "register")
        flat = generation.flatten(normalized)
        assert flat[-1] == transport
        assert generation.unflatten(flat) == normalized
        assert helion.Config.from_json(normalized.to_json()) == normalized
        pinned = spec.create_config_generation(overrides={TRANSPORT: transport})
        assert (
            pinned.unflatten(pinned.default_flat()).cute_state_transfer_transport
            == transport
        )


@pytest.mark.parametrize("with_width", (False, True))
@pytest.mark.parametrize("schedules", (("single",), ("single", "prefix_tail_2")))
def test_each_old_seed_has_appended_transport_siblings(
    spec: ConfigSpec, with_width: bool, schedules: tuple[str, ...]
) -> None:
    spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=64))
    orders = ("identity", "longest_first", "longest_first_precompute")
    spec.enable_cute_chunk_prefill_task_order_search(
        schedules=schedules, task_orders=orders
    )
    if with_width:
        spec.enable_cute_state_transfer_search()
    old_seeds = _seeds(spec)
    old_serialized = [seed.to_json() for seed in old_seeds]
    spec.enable_cute_state_transport_search()
    seeds = _seeds(spec)
    count = len(old_seeds)
    assert len(seeds) == count * 4
    assert [seed.to_json() for seed in seeds[:count]] == old_serialized
    assert [dict(seed) for seed in seeds[count:]] == [
        {**seed, TRANSPORT: transport}
        for transport in ("tma", "tma_pipelined", "tma_planar")
        for seed in old_seeds
    ]
    assert (
        CuteChunkPrefillHeuristic.get_seed_config(
            cast("CompileEnvironment", SimpleNamespace(config_spec=spec)),
            cast("DeviceIR", None),
        )
        == old_seeds[0]
    )
    generation = spec.create_config_generation()
    assert generation.default_flat()[-1] == "register"
    actual = set()
    for seed in seeds:
        config = spec.normalized_config({"block_sizes": [64], **seed})
        flat = generation.flatten(config)
        assert generation.unflatten(flat) == config
        neighbors = [
            candidate
            for candidate in generation.coordinate_neighbor_projections(flat)
            if candidate.key == TRANSPORT
        ]
        assert len(neighbors) == 3
        assert {neighbor.to_value for neighbor in neighbors} == {
            "register",
            "tma",
            "tma_pipelined",
            "tma_planar",
        } - {config.cute_state_transfer_transport}
        actual.add((config[SCHEDULE], config[ORDER], config.get(WIDTH), flat[-1]))
    expected = set(
        product(
            schedules,
            orders,
            (256, 128) if with_width else (None,),
            ("register", "tma", "tma_pipelined", "tma_planar"),
        )
    )
    assert actual == expected
    population = generation.random_population_flat(len(seeds), user_seed_configs=seeds)
    assert {tuple(flat) for flat in population} == {
        tuple(generation.flatten(spec.normalized_config({"block_sizes": [64], **seed})))
        for seed in seeds
    }


@pytest.mark.parametrize("value", (None, False, True, 0, 1.0, "TMA", "", [], {}))
def test_invalid_transport_rejected_and_repaired_to_implicit_default(
    spec: ConfigSpec, value: object
) -> None:
    spec.enable_cute_state_transport_search()
    invalid = helion.Config.from_dict({TRANSPORT: value})
    with pytest.raises(InvalidConfig, match="must be one of"):
        spec.normalize(invalid)
    spec.normalize(invalid, _fix_invalid=True)
    assert TRANSPORT not in invalid


@pytest.mark.parametrize(
    "transport", ("register", "tma", "tma_pipelined", "tma_planar")
)
def test_transport_requires_separate_admission(
    spec: ConfigSpec, transport: str
) -> None:
    spec.enable_cute_state_transfer_search()
    assert spec.cute_state_transfer_transport is None
    assert TRANSPORT not in spec._flat_fields()
    config = helion.Config.from_dict({TRANSPORT: transport})
    with pytest.raises(InvalidConfig, match="bound state graph"):
        spec.normalized_config(config)
    spec.normalize(config, _fix_invalid=True)
    assert TRANSPORT not in config


def test_other_backend_rejects_transport(spec: ConfigSpec) -> None:
    triton = ConfigSpec(backend=TritonBackend(), device=torch.device("cpu"), num_sm=148)
    assert not triton.supports_config_key(TRANSPORT)
    with pytest.raises(InvalidConfig, match="not supported by backend"):
        triton.enable_cute_state_transport_search()
    for transport in ("register", "tma", "tma_pipelined", "tma_planar"):
        config = helion.Config.from_dict({TRANSPORT: transport})
        with pytest.raises(InvalidConfig, match=TRANSPORT):
            triton.normalized_config(config)
        triton.normalize(config, _fix_invalid=True)
        assert TRANSPORT not in config
