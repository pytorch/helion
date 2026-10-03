from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_cache_set_integration import _args as independent_args
from .test_cute_chained_cache_set_integration import _config as independent_config
from .test_cute_chained_cache_set_integration import _root_cache_set
from .test_cute_chained_pointwise_cache_sets import _independent
from .test_cute_chained_pointwise_residency import _repeated
from .test_cute_chained_preparation_cut import _runtime_values
import helion
from helion import exc
from helion._compiler.cute import chained_pointwise_residency as residency
import helion.language as hl

KEY = "cute_chained_pointwise_cache_nested"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("limit", [2, 4])
def test_nested_rank_reduction_keeps_types_dependencies_and_marginal_cost(dtype, limit):
    plan, child = _repeated(dtype, rank=1)
    old = residency.plan_pointwise_cache(plan, 4096)
    selected = residency.plan_pointwise_cache(
        plan, 4096, max_entries=limit, nested=True
    )
    assert len(selected.entries) == 2
    nested, parent = selected.entries
    assert nested.node is child and parent.node is old.entries[0].node
    assert nested.dtype == dtype and parent.dtype == old.entries[0].dtype
    assert nested.name == "chain_pointwise_cache_1"
    assert parent.name == "chain_pointwise_cache_0"
    assert parent.dependencies == (child,)
    assert nested.dependencies == ()
    assert nested.estimated_saved_operations == (16 * 16 - 16) * 8
    assert (
        parent.estimated_saved_operations == old.entries[0].estimated_saved_operations
    )
    assert nested.first_stage == parent.first_stage == 0
    assert nested.last_stage == parent.last_stage == 1
    assert selected.shared_bytes == 512 + 128


@pytest.mark.parametrize("value", [1, 0, "true", None, [], 1.0])
def test_nested_policy_strict_bool(value):
    plan, _ = _repeated(rank=1)
    with pytest.raises(ValueError, match="bool"):
        residency.plan_pointwise_cache(plan, 4096, max_entries=2, nested=value)


def test_nested_requires_multiple_entries_and_respects_full_budget():
    plan, _ = _repeated(rank=1)
    with pytest.raises(ValueError, match="max_entries"):
        residency.plan_pointwise_cache(plan, 4096, nested=True)
    assert (
        len(
            residency.plan_pointwise_cache(
                plan, 639, max_entries=2, nested=True
            ).entries
        )
        == 1
    )
    assert (
        len(
            residency.plan_pointwise_cache(
                plan, 640, max_entries=2, nested=True
            ).entries
        )
        == 2
    )


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("limit", [1, 2, 4])
def test_default_false_preserves_original_selection(rank, limit):
    plan, _ = _repeated(rank=rank)
    assert residency.plan_pointwise_cache(
        plan, 4096, max_entries=limit
    ) == residency.plan_pointwise_cache(plan, 4096, max_entries=limit, nested=False)


def test_same_rank_overlap_remains_rejected_and_independent_order_is_unchanged():
    plan, _ = _repeated(rank=2)
    assert (
        len(
            residency.plan_pointwise_cache(
                plan, 4096, max_entries=4, nested=True
            ).entries
        )
        == 1
    )
    plan, _ = _independent()
    assert residency.plan_pointwise_cache(
        plan, 4096, max_entries=4, nested=True
    ) == residency.plan_pointwise_cache(plan, 4096, max_entries=4)


def test_publication_layers_reject_cycles_and_order_reversed_entries():
    plan, _ = _repeated(rank=1)
    child, parent = residency.plan_pointwise_cache(
        plan, 4096, max_entries=2, nested=True
    ).entries
    assert residency._publication_layers((parent, child)) == ((child,), (parent,))
    with pytest.raises(ValueError, match="cyclic"):
        residency._publication_layers(
            (replace(child, dependencies=(parent.node,)), parent)
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nested_root(a, b, beta, half_stage: hl.constexpr):
    m, k = a.shape
    n = b.size(-1)
    output = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for rows, cols in hl.tile([m, n], block_size=[128, 32]):
        kk = hl.arange(k)
        loaded = hl.load(beta, [rows], extra_mask=rows.index % 2 == 0)
        if half_stage:
            sigmoid = torch.sigmoid(loaded).float()
        else:
            sigmoid = torch.sigmoid(loaded.float())
        operand = (sigmoid[:, None] * a[rows, kk].float()).to(b.dtype)
        first = hl.dot(operand, b[kk, cols])
        output[rows, cols] = hl.dot(operand, b[kk, cols], acc=first)
    return output


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nested_loop(a, b, beta, initial, half_stage: hl.constexpr):
    steps, m, k = a.shape
    n = b.size(-1)
    output = torch.empty_like(initial)
    history = torch.empty((steps, m, n), dtype=torch.float32, device=a.device)
    for rows, cols in hl.tile([m, n], block_size=[128, 32]):
        state = initial[rows, cols]
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(k)
            loaded = hl.load(beta, [step.id, rows], extra_mask=rows.index % 2 == 0)
            if half_stage:
                sigmoid = torch.sigmoid(loaded).float()
            else:
                sigmoid = torch.sigmoid(loaded.float())
            operand = (sigmoid[:, None] * a[step.id, rows, kk].float()).to(b.dtype)
            first = hl.dot(operand, b[step.id, kk, cols])
            state = hl.dot(operand, b[step.id, kk, cols], acc=state + first)
            history[step.id, rows, cols] = state
        output[rows, cols] = state
    return history, output


def _args(loop, dtype, half_stage=False):
    prefix = (2,) if loop else ()
    m = 13 if loop else 128
    args = (
        torch.empty((*prefix, m, 32), dtype=dtype),
        torch.empty(
            (*prefix, 32, 32), dtype=torch.bfloat16 if dtype == torch.float32 else dtype
        ),
        torch.empty((*prefix, m), dtype=dtype),
    )
    return (
        (*args, torch.empty((m, 32), dtype=torch.float32), half_stage)
        if loop
        else (*args, half_stage)
    )


@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("half_stage", [False, True])
def test_public_root_and_loop_use_ordered_original_emission(loop, dtype, half_stage):
    kernel = _nested_loop if loop else _nested_root
    args = _args(loop, dtype, half_stage)
    config = {
        "num_warps": 16 if loop else 4,
        "cute_chained_mma_schedule": "tcgen05_tmem",
        "cute_chained_preparation_pipeline": False,
        "cute_chained_pointwise_cache_bytes": 16384,
        "cute_chained_pointwise_cache_entries": 2,
    }
    selected = []
    original = residency.plan_pointwise_cache

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("nested"):
            selected.append(result)
        return result

    with _cpu_codegen(), patch.object(residency, "plan_pointwise_cache", observe):
        bound = kernel._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
            source = bound.to_code(helion.Config.from_dict({**config, KEY: True}))
    assert selected and len(selected[-1].entries) == 2
    child, parent = selected[-1].entries
    assert child.dtype == (dtype if half_stage else torch.float32)
    assert child.node in parent.dependencies
    start = source.index(f"for {child.name}_step")
    end = source.index(f"for {parent.name}_step")
    assert start < end
    assert "sync" in source[start:end]
    assert child.name in source[end:]
    assert "cute.math.exp2" in source[start:end]
    assert ".load() if" in source[start:end]


@pytest.mark.parametrize("loop", [False, True])
def test_public_default_false_roundtrip_and_no_seed_injection(loop):
    with _cpu_codegen():
        kernel = _nested_loop if loop else _nested_root
        bound = kernel._bind_isolated(_args(loop, torch.bfloat16))
        spec = bound.config_spec
        assert spec.flatten_missing_field_default(KEY, {}) == (True, False)
        absent = spec.normalized_config(helion.Config())
        requested = helion.Config.from_dict({KEY: False})
        assert spec.normalized_config(requested) == absent
        assert requested[KEY] is False
        assert all(KEY not in seed.config for seed in spec.compiler_seed_configs)
        normalized = spec.normalized_config(
            helion.Config.from_dict(
                {
                    KEY: True,
                    "num_warps": 4,
                    "cute_chained_mma_schedule": "coalesced",
                    "cute_chained_pointwise_cache_bytes": 16384,
                    "cute_chained_pointwise_cache_entries": 2,
                }
            )
        )
        generation = spec.create_config_generation()
        assert generation.unflatten(generation.flatten(normalized)) == normalized


def test_public_nested_requires_an_effective_pair_not_just_independent_caches():
    args = independent_args("cpu", False, torch.bfloat16, 1)
    config = independent_config(False, 2)
    with _cpu_codegen():
        bound = _root_cache_set._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_root_cache_set, args)):
            assert "chain_pointwise_cache_1" in bound.to_code(config)
            with pytest.raises(
                exc.BackendUnsupported, match="effective typed residency"
            ):
                bound.to_code(helion.Config.from_dict({**config.config, KEY: True}))


@pytest.mark.parametrize("value", [1, 0, "true", None, [], 1.0])
def test_public_policy_rejects_nonbool_before_emission(value):
    with _cpu_codegen():
        bound = _nested_root._bind_isolated(_args(False, torch.bfloat16))
        with pytest.raises(exc.InvalidConfig, match="must be bool"):
            bound.config_spec.normalized_config(helion.Config.from_dict({KEY: value}))


@pytest.mark.parametrize("budget,count", [(0, 2), (4096, 1)])
def test_public_policy_never_repairs_prerequisites(budget, count):
    with _cpu_codegen():
        bound = _nested_root._bind_isolated(_args(False, torch.bfloat16))
        with pytest.raises(
            exc.InvalidConfig, match="requires positive (pointwise )?cache bytes"
        ):
            bound.config_spec.normalized_config(
                helion.Config.from_dict(
                    {
                        KEY: True,
                        "cute_chained_pointwise_cache_bytes": budget,
                        "cute_chained_pointwise_cache_entries": count,
                    }
                )
            )
