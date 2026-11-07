"""The ordinary PDL entry wait dominates paged attention's scalar prelude."""

from __future__ import annotations

import ast
import functools
from typing import TYPE_CHECKING
from typing import cast

import pytest

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_paged_attention import paged_online_attention
from helion._testing.cute_paged_attention_schedule import paged_attention_args
from helion._testing.cute_pdl import WAIT_CALL
from helion._testing.cute_pdl import kernel_definition

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only


@functools.cache
def _bound(per_key: bool) -> BoundKernel[object]:
    with cpu_render_target(fake_device_sm_count=148):
        return cpu_bind(
            cast("Kernel[object]", paged_online_attention),
            (*paged_attention_args(rows=8 if per_key else 64), None, per_key),
        )


@pytest.mark.parametrize("per_key", [False, True])
@pytest.mark.parametrize("pdl", [False, True])
def test_paged_attention_uses_generic_entry_pdl(per_key, pdl):
    bound = _bound(per_key)
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[8 if per_key else 64],
                cute_mma_dag="tcgen05",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
                cute_mma_dag_pass_warps="attention",
                cute_min_blocks_per_mp=1,
                cute_pdl=pdl,
            )
        )
        assert config.get("cute_mma_dag_pass_warps") == "attention"
        assert config.get("cute_pdl") is pdl
        assert bound.env.backend.autotune_config_is_viable(bound.config_spec, config)
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    kernel = kernel_definition(source)
    wait = WAIT_CALL
    release = "cute.arch.griddepcontrol_launch_dependents()"
    calls = [
        ast.unparse(node) for node in ast.walk(kernel) if isinstance(node, ast.Call)
    ]
    assert ("._helion_cute_use_pdl = True" in source) is pdl
    assert calls.count(wait) == int(pdl)
    assert calls.count(release) == int(pdl)
    assert any("paged_attention" in call and "_body(" in call for call in calls)
    if pdl:
        # This precedes even the scalar tensor loads generated for lengths
        # and bounds, not just the opaque asynchronous attention body call.
        assert ast.unparse(kernel.body[0]) == wait
        assert ast.unparse(kernel.body[1]) == release
        assert any(call.startswith("_helion_mma_dag.ldg_s32(") for call in calls)
