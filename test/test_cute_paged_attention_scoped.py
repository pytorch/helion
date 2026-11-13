from __future__ import annotations

import ast
import functools
from typing import TYPE_CHECKING
from typing import cast

import pytest

import helion
from helion._compiler.cute.cute_mma_dag import mma_dag_pass_warps_reason
from helion._compiler.cute.paged_attention_config import paged_attention_config_reason
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_optional_lse import optional_lse_fixture

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only


@functools.cache
def _bound(query_len: int, form: str = "mapped") -> BoundKernel:
    kernel, args = optional_lse_fixture(form, False, query_len=query_len)
    with cpu_render_target(fake_device_sm_count=148):
        return cpu_bind(cast("Kernel[object]", kernel), args)


@pytest.mark.parametrize("query_len", [1, 4, 5])
@pytest.mark.parametrize("scoped", [False, True])
def test_scoped_schedule_selects_launch_and_native_body(query_len, scoped):
    bound = _bound(query_len)
    config = optional_lse_config()
    config.config.update(
        block_sizes=[helion.next_power_of_2(query_len * 8)],
        cute_mma_dag_pass_warps="attention_scoped" if scoped else "attention",
        cute_min_blocks_per_mp=2,
    )
    spec = bound.config_spec
    assert spec.cute_mma_dag_pass_warps is not None
    assert "attention_scoped" in spec.cute_mma_dag_pass_warps.choices
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        normalized = spec.normalized_config(config)
        assert normalized.get("cute_mma_dag_pass_warps") == config.get(
            "cute_mma_dag_pass_warps"
        )
        assert bound.env.backend.autotune_config_is_viable(spec, normalized)
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(normalized)
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "paged_attention_partial_body"
    ]
    assert len(calls) == 1
    assert {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in calls[0].keywords
        if keyword.arg in ("query_columns", "scoped_registers")
    } == {"query_columns": query_len * 8, "scoped_registers": scoped}
    assert f"block=({384 if scoped else 320}, 1, 1)" in source
    assert "_helion_cute_preferred_smem_carveout" not in source


def test_scoped_schedule_rejects_reduced_denominator():
    bound = _bound(4, "reduced")
    spec = bound.config_spec
    assert spec.cute_mma_dag_pass_warps is not None
    assert "attention" in spec.cute_mma_dag_pass_warps.choices
    assert "attention_scoped" not in spec.cute_mma_dag_pass_warps.choices
    facts = spec.cute_mma_dag_facts
    assert facts is not None
    config = optional_lse_config()
    config.config["cute_mma_dag_pass_warps"] = "attention_scoped"
    sizes = {axis.block_id: axis.extent for axis in facts.axes}
    assert "per-key" in str(paged_attention_config_reason(facts, sizes, config))
    assert "per-key" in str(
        mma_dag_pass_warps_reason(facts, 4, 2, "tcgen05", "attention_scoped")
    )
    assert bound.host_function is not None
    with (
        bound.env,
        bound.host_function,
        pytest.raises(helion.exc.InvalidConfig, match="attention_scoped"),
    ):
        spec.normalized_config(config)


@pytest.mark.parametrize("scalar_policy", [False, True])
@pytest.mark.parametrize("invalid_occupancy", [1, 3])
def test_scoped_normalization_keeps_policy_and_rejects_invalid_topology(
    scalar_policy, invalid_occupancy
):
    bound = _bound(5)
    config = optional_lse_config()
    config.config.update(
        block_sizes=[64],
        cute_mma_dag_pass_warps="attention_scoped",
        cute_min_blocks_per_mp=2,
        load_eviction_policies="" if scalar_policy else [],
    )
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        normalized = bound.config_spec.normalized_config(config)
        assert normalized.get("cute_mma_dag_pass_warps") == "attention_scoped"
        assert bound.env.backend.autotune_config_is_viable(
            bound.config_spec, normalized
        )
        normalized.config["cute_min_blocks_per_mp"] = invalid_occupancy
        assert not bound.env.backend.autotune_config_is_viable(
            bound.config_spec, normalized
        )
        with pytest.raises(helion.exc.InvalidConfig, match="min_blocks_per_mp"):
            bound.config_spec.normalized_config(normalized)
        bound.config_spec.normalize(normalized, _fix_invalid=True)
        # The existing repair policy drops an unsupported attention topology;
        # it must not retain a register schedule with the wrong entry budget.
        assert normalized.get("cute_mma_dag_pass_warps") == "off"
        assert bound.env.backend.autotune_config_is_viable(
            bound.config_spec, normalized
        )
