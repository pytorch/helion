from __future__ import annotations

import pytest
import torch

from test.test_cute_async_state_load_config import _UNUSED_DEVICE_IR
from test.test_cute_async_state_load_config import _fake_env
from test.test_cute_async_state_load_config import _memory_fact

from helion._compiler.autotuner_heuristics.cute import CuteAsyncStateLoadHeuristic
from helion._compiler.backend import CuteBackend
from helion.autotuner.config_spec import ConfigSpec


def test_fp32_bounds_facts_do_not_enable_half_transport() -> None:
    spec = ConfigSpec(backend=CuteBackend(), target_device_capability=(10, 3))
    spec.memory_op_facts = [
        _memory_fact(kind)._replace(dtype=torch.float32) for kind in ("load", "store")
    ]
    env = _fake_env(spec)
    facts = CuteAsyncStateLoadHeuristic.register_facts(env, _UNUSED_DEVICE_IR)
    assert facts == frozenset({"input_tensor_metadata"})
    assert spec.cute_proven_bounds_enabled
    assert not spec.cute_async_load_pipeline_enabled
    assert not spec.cute_bf16x2_recurrence_enabled
    assert CuteAsyncStateLoadHeuristic.get_seed_config(env, _UNUSED_DEVICE_IR) is None


@pytest.mark.parametrize(
    "mutation",
    (
        "different_graph",
        "different_owner",
        "load_only",
        "store_only",
        "unnamed",
        "rank1",
        "integer",
        "old_arch",
    ),
)
def test_fp32_bounds_requires_existing_structural_facts(mutation: str) -> None:
    spec = ConfigSpec(backend=CuteBackend(), target_device_capability=(10, 3))
    load = _memory_fact("load")._replace(dtype=torch.float32)
    store = _memory_fact("store")._replace(dtype=torch.float32)
    if mutation == "different_graph":
        store = store._replace(graph_id=1)
    elif mutation == "different_owner":
        store = store._replace(tensor_name="other")
    elif mutation == "unnamed":
        load, store = (fact._replace(tensor_name=None) for fact in (load, store))
    elif mutation == "rank1":
        load, store = (fact._replace(ndim=1) for fact in (load, store))
    elif mutation == "integer":
        load, store = (fact._replace(dtype=torch.int32) for fact in (load, store))
    elif mutation == "old_arch":
        spec.target_device_capability = (7, 5)
    spec.memory_op_facts = (
        [load]
        if mutation == "load_only"
        else [store]
        if mutation == "store_only"
        else [load, store]
    )
    facts = CuteAsyncStateLoadHeuristic.register_facts(
        _fake_env(spec), _UNUSED_DEVICE_IR
    )
    assert not facts and not spec.cute_proven_bounds_enabled
    assert not spec.cute_async_load_pipeline_enabled
