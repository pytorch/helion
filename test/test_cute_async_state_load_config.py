from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteAsyncStateLoadHeuristic
from helion._compiler.backend import CuteBackend
from helion._compiler.backend import TritonBackend
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import MemoryOpFact
from helion.runtime.kernel import _input_tensor_metadata

if TYPE_CHECKING:
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.device_ir import DeviceIR
    from helion.autotuner.config_fragment import EnumFragment


def _fake_env(spec: ConfigSpec) -> CompileEnvironment:
    return cast("CompileEnvironment", SimpleNamespace(config_spec=spec))


_UNUSED_DEVICE_IR = cast("DeviceIR", None)


def _memory_fact(kind: str, *, graph_id: int = 0) -> MemoryOpFact:
    return MemoryOpFact(
        indexing_index=0 if kind == "load" else 1,
        kind=kind,
        eviction_index=None,
        tensor_name="state",
        dtype=torch.bfloat16,
        ndim=4,
        num_reuses=0,
        matmul_operand=None,
        graph_id=graph_id,
        subscript_strides=(196864, 16384, 128, 1),
        inner_extent=128,
    )


def test_runtime_config_exposes_async_state_load_knobs() -> None:
    config = helion.Config(
        cute_async_load_stages=5,
        cute_async_load_lookahead=4,
        cute_async_load_group_rows=2,
        cute_async_load_cache="cg",
        cute_async_store_policy="l2_evict_last",
        cute_bf16x2_recurrence=True,
        cute_proven_bounds=True,
    )

    assert config.config == {
        "cute_async_load_stages": 5,
        "cute_async_load_lookahead": 4,
        "cute_async_load_group_rows": 2,
        "cute_async_load_cache": "cg",
        "cute_async_store_policy": "l2_evict_last",
        "cute_bf16x2_recurrence": True,
        "cute_proven_bounds": True,
    }
    assert config.cute_async_load_stages == 5
    assert config.cute_async_load_lookahead == 4
    assert config.cute_async_load_group_rows == 2
    assert config.cute_async_load_cache == "cg"
    assert config.cute_async_store_policy == "l2_evict_last"
    assert config.cute_bf16x2_recurrence is True
    assert config.cute_proven_bounds is True
    round_tripped = helion.Config.from_json(config.to_json())
    assert round_tripped.cute_async_store_policy == "l2_evict_last"
    assert round_tripped.cute_bf16x2_recurrence is True
    assert round_tripped.cute_proven_bounds is True
    assert helion.Config().cute_async_load_stages == 0
    assert helion.Config().cute_async_load_lookahead == 4
    assert helion.Config().cute_async_load_group_rows == 2
    assert helion.Config().cute_async_load_cache == "cg"
    assert helion.Config().cute_async_store_policy == "default"
    assert helion.Config().cute_bf16x2_recurrence is False
    assert helion.Config().cute_proven_bounds is False


@patch("helion.autotuner.config_spec.torch.version.cuda", "13.0")
def test_config_surface_is_fact_gated_and_normalized() -> None:
    spec = ConfigSpec(backend=CuteBackend(), target_device_capability=(10, 3))
    with pytest.raises(exc.InvalidConfig, match="compatible in-place"):
        spec.normalize({"cute_async_load_stages": 5})

    removed: dict[str, object] = {"cute_async_load_stages": 5}
    spec.normalize(removed, _fix_invalid=True)
    assert "cute_async_load_stages" not in removed

    with pytest.raises(exc.InvalidConfig, match="compatible BF16 recurrence"):
        spec.normalize({"cute_bf16x2_recurrence": True})
    removed = {"cute_bf16x2_recurrence": True}
    spec.normalize(removed, _fix_invalid=True)
    assert "cute_bf16x2_recurrence" not in removed

    with pytest.raises(exc.InvalidConfig, match="exact CuTe launch"):
        spec.normalize({"cute_proven_bounds": True})
    removed = {"cute_proven_bounds": True}
    spec.normalize(removed, _fix_invalid=True)
    assert "cute_proven_bounds" not in removed

    spec.enable_cute_async_load_pipeline()
    spec.enable_cute_bf16x2_recurrence()
    spec.enable_cute_proven_bounds()
    config: dict[str, object] = {
        "cute_async_load_stages": 5,
        "cute_async_load_lookahead": 4,
        "cute_async_load_group_rows": 2,
        "cute_async_load_cache": "cg",
        "cute_async_store_policy": "l2_evict_last",
        "cute_bf16x2_recurrence": True,
        "cute_proven_bounds": True,
    }
    spec.normalize(config)
    assert {key: config[key] for key in config if key.startswith("cute_async_")} == {
        "cute_async_load_stages": 5,
        "cute_async_load_lookahead": 4,
        "cute_async_load_group_rows": 2,
        "cute_async_load_cache": "cg",
        "cute_async_store_policy": "l2_evict_last",
    }
    assert config["cute_bf16x2_recurrence"] is True
    assert config["cute_proven_bounds"] is True

    with pytest.raises(exc.InvalidConfig, match="lookahead"):
        spec.normalize(
            {
                "cute_async_load_stages": 3,
                "cute_async_load_lookahead": 4,
            }
        )
    repaired: dict[str, object] = {
        "cute_async_load_stages": 3,
        "cute_async_load_lookahead": 4,
    }
    spec.normalize(repaired, _fix_invalid=True)
    assert repaired["cute_async_load_lookahead"] == 2


@pytest.mark.parametrize(("stages", "lookahead"), ((3, 2), (4, 3), (5, 4)))
def test_omitted_async_lookahead_is_derived_from_stages(
    stages: int, lookahead: int
) -> None:
    spec = ConfigSpec(backend=CuteBackend())
    spec.enable_cute_async_load_pipeline()
    config: dict[str, object] = {"cute_async_load_stages": stages}

    spec.normalize(config)

    assert config["cute_async_load_lookahead"] == lookahead


@pytest.mark.parametrize(
    ("key", "value", "repaired_value"),
    (
        ("cute_async_load_stages", False, 0),
        ("cute_async_load_stages", 3.0, 0),
        ("cute_async_load_lookahead", True, 4),
        ("cute_async_load_lookahead", 2.0, 4),
        ("cute_async_load_group_rows", True, 2),
        ("cute_async_load_group_rows", 2.0, 2),
    ),
)
def test_async_integer_knobs_require_exact_integers(
    key: str, value: object, repaired_value: int
) -> None:
    spec = ConfigSpec(backend=CuteBackend())
    spec.enable_cute_async_load_pipeline()

    with pytest.raises(exc.InvalidConfig, match=key):
        spec.normalize({key: value})

    repaired = {key: value}
    spec.normalize(repaired, _fix_invalid=True)
    assert repaired[key] == repaired_value


@pytest.mark.parametrize(
    ("capability", "cuda_version"),
    (((10, 0), "13.0"), ((10, 3), "12.8")),
)
def test_l2_store_policy_is_canonicalized_and_not_searched_when_unsupported(
    capability: tuple[int, int], cuda_version: str
) -> None:
    with patch("helion.autotuner.config_spec.torch.version.cuda", cuda_version):
        spec = ConfigSpec(backend=CuteBackend(), target_device_capability=capability)
        spec.enable_cute_async_load_pipeline()
        config: dict[str, object] = {
            "cute_async_load_stages": 5,
            "cute_async_store_policy": "l2_evict_last",
        }

        spec.normalize(config)
        field = cast("EnumFragment", spec._flat_fields()["cute_async_store_policy"])
        prior = CuteBackend().config_value_priors(spec)["cute_async_store_policy"]

    assert config["cute_async_store_policy"] == "default"
    assert field.choices == ("default",)
    assert prior(field, 0) == "default"


@patch("helion.autotuner.config_spec.torch.version.cuda", "13.0")
def test_l2_store_policy_is_preserved_and_searched_on_validated_target() -> None:
    spec = ConfigSpec(backend=CuteBackend(), target_device_capability=(10, 3))
    spec.enable_cute_async_load_pipeline()
    config: dict[str, object] = {
        "cute_async_load_stages": 5,
        "cute_async_store_policy": "l2_evict_last",
    }

    spec.normalize(config)

    assert config["cute_async_store_policy"] == "l2_evict_last"
    field = cast("EnumFragment", spec._flat_fields()["cute_async_store_policy"])
    assert field.choices == ("default", "l2_evict_last")


@pytest.mark.parametrize(
    "config",
    (
        {"cute_async_store_policy": "l2_evict_last"},
        {"cute_bf16x2_recurrence": True},
        {"cute_proven_bounds": True},
    ),
)
def test_new_knobs_are_cute_only(config: dict[str, object]) -> None:
    with pytest.raises(exc.InvalidConfig, match="Unsupported config keys"):
        ConfigSpec(backend=TritonBackend()).normalize(config)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    (
        ("cute_async_store_policy", "streaming", "cute_async_store_policy"),
        ("cute_bf16x2_recurrence", 1, "must be a boolean"),
        ("cute_proven_bounds", 1, "must be a boolean"),
    ),
)
def test_new_knobs_reject_invalid_values(key: str, value: object, message: str) -> None:
    spec = ConfigSpec(backend=CuteBackend())
    spec.enable_cute_async_load_pipeline()
    spec.enable_cute_bf16x2_recurrence()
    spec.enable_cute_proven_bounds()

    with pytest.raises(exc.InvalidConfig, match=message):
        spec.normalize({key: value})

    repaired = {key: value}
    spec.normalize(repaired, _fix_invalid=True)
    assert repaired[key] == ("default" if key == "cute_async_store_policy" else False)


def test_bf16x2_is_disabled_when_async_pipeline_is_disabled() -> None:
    spec = ConfigSpec(backend=CuteBackend())
    spec.enable_cute_async_load_pipeline()
    spec.enable_cute_bf16x2_recurrence()
    config: dict[str, object] = {
        "cute_async_load_stages": 0,
        "cute_bf16x2_recurrence": True,
    }

    spec.normalize(config)

    assert config["cute_bf16x2_recurrence"] is False


def test_heuristic_enables_surface_and_seeds_measured_schedule() -> None:
    spec = ConfigSpec(backend=CuteBackend(), target_device_capability=(8, 0))
    spec.memory_op_facts = [_memory_fact("load"), _memory_fact("store")]
    env = _fake_env(spec)

    specialization_facts = CuteAsyncStateLoadHeuristic.register_facts(
        env, _UNUSED_DEVICE_IR
    )
    seed = CuteAsyncStateLoadHeuristic.get_seed_config(env, _UNUSED_DEVICE_IR)

    assert spec.cute_async_load_pipeline_enabled
    assert spec.cute_bf16x2_recurrence_enabled
    assert spec.cute_proven_bounds_enabled
    assert specialization_facts == frozenset({"input_tensor_metadata"})
    assert seed is not None
    assert seed.config["cute_async_load_stages"] == 5
    assert seed.config["cute_async_load_lookahead"] == 4
    assert seed.config["cute_async_load_group_rows"] == 2
    assert seed.config["cute_async_load_cache"] == "cg"
    assert seed.config["cute_proven_bounds"] is False
    assert all(
        key in spec._flat_fields()
        for key in (
            "cute_async_load_stages",
            "cute_async_load_lookahead",
            "cute_async_load_group_rows",
            "cute_async_load_cache",
            "cute_async_store_policy",
            "cute_bf16x2_recurrence",
            "cute_proven_bounds",
        )
    )


def test_heuristic_requires_same_graph_load_and_store() -> None:
    spec = ConfigSpec(backend=CuteBackend(), target_device_capability=(8, 0))
    spec.memory_op_facts = [
        _memory_fact("load", graph_id=0),
        _memory_fact("store", graph_id=1),
    ]
    env = _fake_env(spec)

    CuteAsyncStateLoadHeuristic.register_facts(env, _UNUSED_DEVICE_IR)

    assert not spec.cute_async_load_pipeline_enabled
    assert not spec.cute_bf16x2_recurrence_enabled
    assert not spec.cute_proven_bounds_enabled


def test_heuristic_gate_does_not_depend_on_current_input_layout() -> None:
    spec = ConfigSpec(backend=CuteBackend(), target_device_capability=(8, 0))
    load = _memory_fact("load")
    store = _memory_fact("store")
    spec.memory_op_facts = [
        load._replace(subscript_strides=(1, 128, 1, 128), inner_extent=7),
        store._replace(subscript_strides=(1, 128, 1, 128), inner_extent=7),
    ]
    env = _fake_env(spec)

    specialization_facts = CuteAsyncStateLoadHeuristic.register_facts(
        env, _UNUSED_DEVICE_IR
    )

    assert spec.cute_async_load_pipeline_enabled
    assert spec.cute_bf16x2_recurrence_enabled
    assert spec.cute_proven_bounds_enabled
    assert specialization_facts == frozenset({"input_tensor_metadata"})


@pytest.mark.parametrize("capability", (None, (7, 0), (7, 5)))
def test_heuristic_requires_known_sm80_or_newer(
    capability: tuple[int, int] | None,
) -> None:
    spec = ConfigSpec(
        backend=CuteBackend(),
        target_device_capability=capability,
    )
    spec.memory_op_facts = [_memory_fact("load"), _memory_fact("store")]
    env = _fake_env(spec)

    specialization_facts = CuteAsyncStateLoadHeuristic.register_facts(
        env, _UNUSED_DEVICE_IR
    )
    seed = CuteAsyncStateLoadHeuristic.get_seed_config(env, _UNUSED_DEVICE_IR)

    assert not spec.cute_async_load_pipeline_enabled
    assert not spec.cute_bf16x2_recurrence_enabled
    assert not spec.cute_proven_bounds_enabled
    assert not specialization_facts
    assert seed is None

    spec.enable_cute_async_load_pipeline()
    assert not CuteAsyncStateLoadHeuristic.is_eligible(env, _UNUSED_DEVICE_IR)
    assert CuteAsyncStateLoadHeuristic.get_seed_config(env, _UNUSED_DEVICE_IR) is None


def test_input_metadata_specialization_separates_stride_reuse() -> None:
    contiguous = torch.empty((4, 8), dtype=torch.bfloat16)
    incompatible = torch.empty_strided((4, 8), (1, 4), dtype=torch.bfloat16)

    assert contiguous.shape == incompatible.shape
    assert _input_tensor_metadata((contiguous,)) != _input_tensor_metadata(
        (incompatible,)
    )


def test_backend_priors_favor_measured_schedule() -> None:
    priors = CuteBackend().config_value_priors(cast("ConfigSpec", None))

    assert {
        "cute_async_load_stages",
        "cute_async_load_lookahead",
        "cute_async_load_group_rows",
        "cute_async_load_cache",
        "cute_async_store_policy",
        "cute_bf16x2_recurrence",
        "cute_proven_bounds",
    } <= priors.keys()
