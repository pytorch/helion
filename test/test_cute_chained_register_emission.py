from __future__ import annotations

import ast
from unittest.mock import patch

import pytest

from ._cute_aux import _cpu_codegen
from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_pipeline import _config
from helion import exc
from helion._compiler.cute import chained_register_islands as islands
from helion.autotuner.config_spec import CUTE_CHAINED_COLLECTIVE_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_LEAF_COUNT_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_REGISTER_ISLANDS_KEY


def _register_config(cohorts=1, **kwargs):
    config = _config(16, pipeline=True)
    config.config.update(
        block_sizes=[128],
        cute_chained_group_contractions=True,
        cute_chained_scratch_layout="xor",
        cute_chained_pointwise_vectorize=True,
        cute_chained_scan_schedule="warp",
        cute_chained_pointwise_cache_bytes=4096,
        cute_chained_pointwise_unroll=8,
        cute_chained_preparation_cohorts=cohorts,
        cute_chained_preparation_unroll=1,
    )
    config.config.update(kwargs)
    return config


@pytest.mark.parametrize("cohorts", [1, 3])
def test_register_island_emits_original_typed_graph_and_role_barriers_cpu(cohorts):
    kernel, args = _kda_fixture()
    source = _source(
        kernel, args, _register_config(cohorts, cute_chained_register_islands=True)
    )
    tree = ast.parse(source)
    regions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "chain_prep_thread < 64"
        and "chain_register_island" in ast.unparse(node)
    ]
    assert len(regions) == 1
    body = ast.unparse(regions[0])
    assert body.count("cute.gemm(") == 6
    assert "chain_prep_barrier.arrive_and_wait()" not in body
    assert "cute.arch.sync_threads()" not in body
    assert "movmatrix_b16" in body
    assert "cute.arch.shuffle_sync(" in body
    assert "cutlass.Float16" in body and "cutlass.Float32" in body
    assert "chain_2_a_ptr" not in source
    for stage in (2, 3, 6):
        assert not any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Subscript)
                and ast.unparse(target.value) == f"chain_{stage}_c"
                for target in node.targets
            )
            for node in ast.walk(tree)
        )
    for stage in (4, 5, 7):
        assert f"chain_{stage}_c[" in body
    parent = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.For, ast.If)) and regions[0] in node.body
    )
    index = parent.body.index(regions[0])
    assert ast.unparse(parent.body[index - 1]) == "chain_prep_barrier.arrive_and_wait()"
    assert ast.unparse(parent.body[index + 1]) == "chain_prep_barrier.arrive_and_wait()"


def test_register_disabled_and_strict_policy_preserve_source_cpu():
    kernel, args = _kda_fixture()
    with patch.object(
        islands, "plan_register_islands", side_effect=AssertionError("discovery ran")
    ):
        assert _source(kernel, args, _register_config()) == _source(
            kernel, args, _register_config(cute_chained_register_islands=False)
        )
        with patch.object(kernel.settings, "fast_math", False):
            assert _source(kernel, args, _register_config()) == _source(
                kernel, args, _register_config(cute_chained_register_islands=True)
            )


@pytest.mark.parametrize("value", [None, 0, 1, "true", 1.0])
def test_register_option_requires_exact_bool_cpu(value):
    kernel, args = _kda_fixture()
    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        with pytest.raises(exc.InvalidConfig, match="register_islands must be bool"):
            bound.config_spec.normalize(
                _register_config(cute_chained_register_islands=value)
            )


def test_register_field_appends_and_default_prefix_roundtrips_cpu():
    kernel, args = _kda_fixture()
    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
            spec = bound.config_spec
            fields = spec._flat_fields()
            layout = spec.flat_key_layout()
            field_index = next(
                i
                for i, item in enumerate(layout)
                if item[0] == CUTE_CHAINED_REGISTER_ISLANDS_KEY
            )
            # Later options append after this original three-field suffix.
            assert tuple(fields)[field_index : field_index + 3] == (
                CUTE_CHAINED_REGISTER_ISLANDS_KEY,
                CUTE_CHAINED_LEAF_COUNT_KEY,
                CUTE_CHAINED_COLLECTIVE_RETENTION_KEY,
            )
            index = sum(size for _, size, _ in layout[:field_index])
            generation = spec.create_config_generation()
            default = spec.default_config()
            assert CUTE_CHAINED_REGISTER_ISLANDS_KEY not in default.config
            config = spec.normalized_config(
                _register_config(cute_chained_register_islands=True)
            )
            assert generation.unflatten(generation.flatten(config)) == config
            with patch.object(
                spec,
                "_flat_fields",
                return_value={
                    key: field
                    for key, field in fields.items()
                    if key != CUTE_CHAINED_REGISTER_ISLANDS_KEY
                },
            ):
                old = spec.create_config_generation()
                flat = generation.flatten(default)
                assert flat[:index] + flat[index + 1 :] == old.flatten(default)
                assert flat[index] is False
