"""Artifact-only tests: actual compiled clients plus independent ledger negatives."""

from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

import helion
from helion import exc
from helion._compiler.cute import native_matmul_metadata as native
from helion.autotuner.config_spec import ConfigSpec
from helion.runtime.cute import launcher


def _case() -> tuple[Any, Any]:
    graph = Graph()
    a, b = graph.placeholder("a"), graph.placeholder("b")
    dot = graph.call_function(torch.mm, (a, b))
    graph.output(dot)
    for node in graph.nodes:
        node.meta["val"] = torch.empty((16, 16), dtype=torch.bfloat16)
    plan = SimpleNamespace(
        dots=(dot,),
        shapes=((16, 16, 16),),
        threads=128,
        scans=(),
        axes=((0, 16, 16),),
        dtype=torch.bfloat16,
        operand_dtype=lambda stage: torch.bfloat16,
    )
    cg = SimpleNamespace(
        device_function=SimpleNamespace(
            config=SimpleNamespace(config={native.KEY: True}), body=[]
        ),
        cute_uses_matmul=False,
        cute_matmul_layouts=None,
        cute_wrapper_plans=[],
    )
    return cg, plan


def _stage(cg, plan, lines=("native_stage()",)):
    native.record_native_stage(cg, plan, plan.dots, "warp", lines)


def _commit(cg, plan, lines=("native_stage()",)):
    cg.device_function.body = ast.parse("\n".join(lines)).body
    native.commit_body(cg, plan, lines)


def test_complete_native_body_and_default_no_records():
    cg, plan = _case()
    with native.body_attempt(cg):
        _stage(cg, plan)
        _commit(cg, plan)
    assert native.validate_native_metadata(cg)
    cg, plan = _case()
    cg.device_function.config.config.clear()
    with native.body_attempt(cg):
        _stage(cg, plan)
        _commit(cg, plan)
    assert cg.cute_uses_matmul and cg.cute_matmul_layouts is None
    assert not native.validate_native_metadata(cg)


@pytest.mark.parametrize("position", ("before", "after_stage", "after_commit"))
def test_unknown_monotonic_in_all_orders(position):
    cg, plan = _case()
    with native.body_attempt(cg):
        if position == "before":
            native.record_unknown(cg)
        _stage(cg, plan)
        if position == "after_stage":
            native.record_unknown(cg)
        if position != "after_commit":
            with pytest.raises(exc.BackendUnsupported, match="coverage"):
                _commit(cg, plan)
        else:
            _commit(cg, plan)
            native.record_unknown(cg)
    with pytest.raises(exc.BackendUnsupported):
        native.validate_native_metadata(cg)


@pytest.mark.parametrize("position", ("before", "after_stage", "after_commit"))
@pytest.mark.parametrize("disabled", (None, False))
def test_existing_obligation_survives_temporary_opt_out(position, disabled):
    cg, plan = _case()
    with native.body_attempt(cg):
        if position != "before":
            _stage(cg, plan)
        if position == "after_commit":
            _commit(cg, plan)
        if disabled is None:
            cg.device_function.config.config.pop(native.KEY)
        else:
            cg.device_function.config.config[native.KEY] = disabled
        native.record_unknown(cg)
        assert cg.cute_matmul_layouts.unknown
        cg.device_function.config.config[native.KEY] = True
        if position == "before":
            _stage(cg, plan)
        if position != "after_commit":
            with pytest.raises(exc.BackendUnsupported):
                _commit(cg, plan)
    with pytest.raises(exc.BackendUnsupported):
        native.validate_native_metadata(cg)


@pytest.mark.parametrize("disabled", (None, False))
def test_default_unknown_does_not_allocate_ledger(disabled):
    cg, plan = _case()
    if disabled is None:
        cg.device_function.config.config.pop(native.KEY)
    else:
        cg.device_function.config.config[native.KEY] = disabled
    native.record_unknown(cg)
    assert cg.cute_uses_matmul and cg.cute_matmul_layouts is None


def test_abandoned_successful_stage_cannot_authorize_same_dot_next_attempt():
    cg, plan = _case()
    with native.body_attempt(cg):
        _stage(cg, plan)
    with (
        native.body_attempt(cg),
        pytest.raises(exc.BackendUnsupported, match="coverage"),
    ):
        _commit(cg, plan)


def test_nested_discarded_root_attempt_never_supplies_outer_fallback_receipts():
    cg, plan = _case()
    with native.body_attempt(cg):
        with native.body_attempt(cg):
            _stage(cg, plan)
        with pytest.raises(exc.BackendUnsupported, match="coverage"):
            _commit(cg, plan)


def test_discarded_segment_does_not_authorize_different_installed_body():
    cg, plan = _case()
    with native.body_attempt(cg):
        _stage(cg, plan)
        with pytest.raises(exc.BackendUnsupported, match="coverage"):
            _commit(cg, plan, ("different_stage()",))


def test_duplicate_and_missing_stage_fail_before_authority():
    for count in (0, 2):
        cg, plan = _case()
        with native.body_attempt(cg):
            for _ in range(count):
                _stage(cg, plan)
            with pytest.raises(exc.BackendUnsupported, match="coverage"):
                _commit(cg, plan)


@pytest.mark.parametrize("changed_scope", (False, True))
def test_uniform_role_indent_only_never_changes_nested_control_scope(changed_scope):
    cg, plan = _case()
    original = ("if ready:", "    issue()", "publish()")
    installed = (
        "if role:",
        "    if ready:",
        "        issue()",
        "        publish()" if changed_scope else "    publish()",
    )
    with native.body_attempt(cg):
        _stage(cg, plan, original)
        if changed_scope:
            with pytest.raises(exc.BackendUnsupported, match="coverage"):
                _commit(cg, plan, installed)
        else:
            _commit(cg, plan, installed)
    if not changed_scope:
        assert native.validate_native_metadata(cg)


@pytest.mark.parametrize("phase", ("precommit", "finalize"))
@pytest.mark.parametrize(
    "field,value",
    (("nodes", ()), ("implementation", "universal"), ("source", ("different()",))),
)
def test_same_object_receipt_mutation_rejects(phase, field, value):
    cg, plan = _case()
    with native.body_attempt(cg):
        _stage(cg, plan)
        receipt = cg.cute_matmul_layouts.active.stages[0]
        if phase == "finalize":
            _commit(cg, plan)
        object.__setattr__(receipt, field, value)
        if phase == "precommit":
            with pytest.raises(exc.BackendUnsupported):
                _commit(cg, plan)
    if phase == "finalize":
        with pytest.raises(exc.BackendUnsupported):
            native.validate_native_metadata(cg)


@pytest.mark.parametrize("phase", ("precommit", "finalize"))
@pytest.mark.parametrize("mutation", ("dtype", "kwargs", "shape", "config", "body"))
def test_independent_deep_revision_and_body_snapshot(phase, mutation):
    cg, plan = _case()
    with native.body_attempt(cg):
        _stage(cg, plan)
        if phase == "finalize":
            _commit(cg, plan)
        if mutation == "dtype":
            plan.dtype = torch.float16
        elif mutation == "kwargs":
            plan.dots[0].kwargs = {"unexpected": [1, 2]}
        elif mutation == "shape":
            plan.dots[0].meta["val"] = torch.empty((16, 32), dtype=torch.bfloat16)
        elif mutation == "config":
            cg.device_function.config.config["other"] = {"nested": [True]}
        elif phase == "precommit":
            cg.cute_matmul_layouts.active.stages = ()
        else:
            cg.device_function.body[0].value.func.id = "different_stage"
        if phase == "precommit":
            with pytest.raises(exc.BackendUnsupported):
                _commit(cg, plan)
    if phase == "finalize":
        with pytest.raises(exc.BackendUnsupported):
            native.validate_native_metadata(cg)


@pytest.mark.parametrize("marker", (None, False, 0, 1, "true", True))
@pytest.mark.parametrize("disable", (False, True))
def test_rectangular_extension_requires_strict_optin_disable_always_wins(
    marker, disable
):
    kernel = SimpleNamespace(
        _helion_cute_wrapper_plans=[{"kind": "chained_rectangular_leaf_tma"}],
        _helion_cute_disable_bake_tensor_shapes=disable,
    )
    if marker is not None:
        kernel._helion_cute_native_metadata_specialization = marker
    assert launcher._cute_bake_tensor_shapes_guard(kernel) is (
        marker is True and not disable
    )


@pytest.mark.parametrize("orientation", ("mn", "nm"))
@pytest.mark.parametrize("partial", (False, True))
def test_original_oriented_full_tile_guard_unchanged(orientation, partial):
    plan = {
        "kind": "tcgen05_ab_tma",
        "orientation": orientation,
        "m_size": 128,
        "n_size": 256,
        "k_total_size": 64,
        "bm": 128 if orientation == "mn" else 256,
        "bn": 256 if orientation == "mn" else 128,
        "bk": 64,
    }
    if partial:
        plan["m_size"] -= 1
    assert launcher._cute_wrapper_plan_bakes_tensor_shapes(plan) is (not partial)
    assert launcher._cute_wrapper_plan_bakes_tensor_shapes(
        plan, native_metadata=True
    ) is (not partial)


def test_mixed_wrapper_denies_and_cache_key_policy_changes():
    kernel = SimpleNamespace(
        _helion_cute_wrapper_plans=[{"kind": "chained_rectangular_leaf_tma"}]
    )
    assert not launcher._cute_bake_tensor_shapes_guard(kernel)
    kernel._helion_cute_native_metadata_specialization = True
    assert launcher._cute_bake_tensor_shapes_guard(kernel)
    kernel._helion_cute_wrapper_plans.append({"kind": "unknown"})
    assert not launcher._cute_bake_tensor_shapes_guard(kernel)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_unrelated_initialized_root_original_body_exact(dtype):
    from test.test_cute_chained_accumulator import _cpu
    from test.test_cute_chained_accumulator import _initialized_args
    from test.test_cute_chained_accumulator import _initialized_config
    from test.test_cute_chained_accumulator import _initialized_pair

    with _cpu():
        config = _initialized_config(128)
        before = _initialized_pair._bind_isolated(
            _initialized_args(dtype=dtype, n=128)
        ).to_code(config)
        config.config[native.KEY] = True
        after = _initialized_pair._bind_isolated(
            _initialized_args(dtype=dtype, n=128)
        ).to_code(config)
    before_tree, after_tree = ast.parse(before), ast.parse(after)
    device_before = next(
        n
        for n in before_tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion")
    )
    device_after = next(
        n
        for n in after_tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion")
    )
    assert ast.dump(device_before) == ast.dump(device_after)
    assert "_helion_cute_native_metadata_specialization = True" in after
    assert "_helion_cute_disable_bake_tensor_shapes = False" in after


@pytest.fixture
def root_bound():
    from test.test_cute_chained_accumulator import _cpu
    from test.test_cute_chained_accumulator import _initialized_args
    from test.test_cute_chained_accumulator import _initialized_pair

    with _cpu():
        yield _initialized_pair._bind_isolated(_initialized_args(n=128))


@pytest.mark.parametrize("repair", (False, True))
@pytest.mark.parametrize("value", (None, 0, 1, 0.0, "True", [], {}))
def test_schema_strict_bool(root_bound, repair, value):
    config = helion.Config.from_dict({native.KEY: value})
    with pytest.raises(exc.InvalidConfig, match=native.KEY):
        root_bound.config_spec.normalize(config, _fix_invalid=repair)


def test_schema_default_flat_suffix_and_entire_old_seed_pool(root_bound):
    spec = root_bound.config_spec
    fields = spec._flat_fields()
    assert tuple(fields)[-3:] == (
        native.KEY,
        "cute_chained_drain_tile_columns",
        "cute_chained_fragment_epilogues",
    )
    assert fields[native.KEY].search_values() == [False, True]
    assert spec.flatten_missing_field_default(native.KEY, {}) == (True, False)
    from test.test_cute_chained_accumulator import _initialized_config

    absent = spec.normalized_config(_initialized_config(128))
    assert (
        spec.normalized_config(
            helion.Config.from_dict({**absent.config, native.KEY: False})
        )
        == absent
    )
    seeds = tuple(spec.compiler_seed_configs)
    old_contents = [dict(seed.config) for seed in seeds]
    generation = spec.create_config_generation()
    pairs = generation.seed_flat_config_pairs()
    with patch.object(
        spec,
        "_flat_fields",
        return_value={k: v for k, v in fields.items() if k != native.KEY},
    ):
        old_pairs = spec.create_config_generation().seed_flat_config_pairs()
    for (flat, config), (old_flat, old_config) in zip(pairs, old_pairs, strict=True):
        assert flat[:-3] + flat[-2:] == old_flat and flat[-3] is False
        assert config == old_config and native.KEY not in config.config
    assert all(a is b for a, b in zip(seeds, spec.compiler_seed_configs, strict=True))
    assert old_contents == [dict(seed.config) for seed in seeds]
    config = helion.Config.from_dict({**absent.config, native.KEY: True})
    normalized = spec.normalized_config(config)
    assert generation.unflatten(generation.flatten(normalized)) == normalized
    assert normalized.config == {**absent.config, native.KEY: True}


@pytest.mark.parametrize(
    "backend,discovered", (("triton", True), ("pallas", True), ("cute", False))
)
def test_schema_requires_discovery_and_backend_without_injection(backend, discovered):
    spec = SimpleNamespace(
        backend_name=backend, cute_chained_matmul_search_enabled=discovered
    )
    config: dict[str, object] = {native.KEY: True}
    with pytest.raises(exc.InvalidConfig):
        ConfigSpec._normalize_cute_native_matmul_metadata(
            cast("ConfigSpec", spec), config
        )
    assert config == {native.KEY: True}
    config = {native.KEY: False}
    ConfigSpec._normalize_cute_native_matmul_metadata(cast("ConfigSpec", spec), config)
    assert config == {}


def test_unrelated_loop_uses_same_committed_capability():
    from test._cute_aux import _cpu_codegen
    from test.test_cute_chained_loop_search import _loop_search

    args = (
        torch.empty((3, 128, 32), dtype=torch.bfloat16),
        torch.empty((3, 32, 128), dtype=torch.bfloat16),
        torch.empty((3, 32, 128), dtype=torch.bfloat16),
        torch.empty((128, 128), dtype=torch.float32),
    )
    with _cpu_codegen():
        config = helion.Config(
            block_sizes=[128], num_warps=4, cute_chained_mma_schedule="tcgen05_tmem"
        )
        before = _loop_search._bind_isolated(args).to_code(config)
        config.config[native.KEY] = True
        after = _loop_search._bind_isolated(args).to_code(config)
    assert "_helion_cute_native_metadata_specialization = True" in after
    before_device = next(
        n
        for n in ast.parse(before).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion")
    )
    after_device = next(
        n
        for n in ast.parse(after).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion")
    )
    assert ast.dump(before_device) == ast.dump(after_device)


def test_cache_and_last_launch_guard_include_effective_metadata_choice():
    def kernel(value):
        return value

    cast("Any", kernel)._helion_cute_wrapper_plans = [
        {"kind": "chained_rectangular_leaf_tma"}
    ]
    args, grid = (0.25,), (1, 1, 1)
    before = launcher._cute_launch_arg_cache_key(kernel, args, grid)
    guard = launcher._cute_last_launch_arg_guard(kernel, args, grid)
    assert guard.matches(kernel, args, grid)
    cast("Any", kernel)._helion_cute_native_metadata_specialization = True
    assert launcher._cute_launch_arg_cache_key(kernel, args, grid) != before
    assert not guard.matches(kernel, args, grid)
    cast("Any", kernel)._helion_cute_disable_bake_tensor_shapes = True
    assert launcher._cute_launch_arg_cache_key(kernel, args, grid) == before
    assert guard.matches(kernel, args, grid)
