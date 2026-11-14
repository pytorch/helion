from __future__ import annotations

import ast
from copy import deepcopy
import json
import random
from unittest.mock import patch

import pytest
import torch

from test.cute_population_contracts import tcgen05_mma_support
from test.test_cute_pure_lane_packets_codegen import _capture
from test.test_cute_pure_lane_packets_codegen import _execute

import helion
from helion._compiler.cute.autotune.heuristics.cute import CutePointwiseVecHeuristic
from helion._compiler.cute.full_tile_bounds import FullTilePlan
from helion._compiler.cute.full_tile_bounds import specialize_full_tile_bounds
from helion._compiler.cute.tensor_layout_relations import TensorLayoutRelation
from helion._compiler.rng_utils import philox_rand_ref
from helion._testing import cpu_render_target
from helion._testing import import_example
from helion._testing.cute_codegen import cpu_bind
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch
import helion.language as hl

pytestmark = pytest.mark.cpu_only

_examples_low_mem_dropout = import_example("low_mem_dropout")
low_mem_dropout = _examples_low_mem_dropout.low_mem_dropout
low_mem_dropout_bwd = _examples_low_mem_dropout.low_mem_dropout_bwd


@pytest.fixture(autouse=True)
def _cpu_only():
    with cpu_render_target(
        mma_support=tcgen05_mma_support(), reduction_smem_budget_bytes=232448
    ):
        yield


def _config(width=8, threads=128, cluster=1, enabled=False):
    return helion.Config(
        block_sizes=[width * threads],
        num_threads=[threads],
        cute_vector_widths=[width],
        cute_lane_layouts=["blocked"],
        cute_cluster_n=cluster,
        cute_proven_bounds=enabled,
    )


def _functions(source):
    return [
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
    ]


def _body_dump(body):
    return ast.dump(ast.Module(body=body, type_ignores=[]))


def _bound(
    function=low_mem_dropout, *, dtype=torch.float32, tensor=None, effort="none"
):
    kernel = helion.kernel(
        function.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort=effort,
        cute_rng_stream="word0",
    )
    return cpu_bind(
        kernel,
        (0.25, torch.empty(33554432, dtype=dtype) if tensor is None else tensor, 123),
    )


@pytest.mark.parametrize(
    "function,cluster", [(low_mem_dropout, 1), (low_mem_dropout_bwd, 2)]
)
@pytest.mark.parametrize("width,threads", [(4, 128), (8, 128), (8, 256)])
def test_original_dropout_host_abi_and_fallback_are_unchanged(
    function, cluster, width, threads
):
    bound = _bound(function)
    control = bound.to_code(_config(width, threads, cluster, False))
    enabled = bound.to_code(_config(width, threads, cluster, True))
    control_helper, control_device, control_host = _functions(control)
    helper, device, host = _functions(enabled)
    assert ast.dump(helper) == ast.dump(control_helper)
    assert ast.dump(host) == ast.dump(control_host)
    assert ast.dump(device.args) == ast.dump(control_device.args)
    assert len(device.body) == 1 and isinstance(device.body[0], ast.If)
    branch = device.body[0]
    assert _body_dump(branch.orelse) == _body_dump(control_device.body)
    assert "cute.is_static(" in ast.unparse(branch.test)
    fast = ast.unparse(ast.Module(body=branch.body, type_ignores=[]))
    assert "n = cutlass.Int64(cute.size(" in fast
    assert "_helion_affine_load" in fast and "_cute_store_u32_vec(" in fast
    comparisons = [
        node
        for statement in branch.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Compare)
    ]
    assert len(comparisons) == 1
    # The control has packet and scalar fallbacks with the same RNG predicate.
    # Derive its exact input from those bodies: temporary names are not an ABI.
    original_uniform = {
        ast.dump(node): node
        for node in ast.walk(control_device)
        if isinstance(node, ast.Compare)
        and len(node.comparators) == 1
        and ast.unparse(node.comparators[0]) == "_helion_uniform_span"
    }
    assert len(original_uniform) == 1
    uniform = next(iter(original_uniform.values()))
    assert isinstance(uniform.left, ast.Call)
    difference = uniform.left.args[0]
    assert isinstance(difference, ast.BinOp)
    assert isinstance(difference.left, ast.Call)
    random_word = difference.left.args[0]
    expected_uniform = ast.parse(
        f"cutlass.Uint32(cutlass.Uint32({ast.unparse(random_word)})"
        " - _helion_uniform_first) < _helion_uniform_span",
        mode="eval",
    ).body
    assert ast.dump(uniform) == ast.dump(expected_uniform)
    assert ast.dump(comparisons[0]) == ast.dump(uniform)
    # The only new scalar expression is lossless widening of the proved
    # nonnegative index. All Philox rounds and floating-point expressions,
    # including the full runtime seed/probability/scale, remain unchanged.
    original_values = {
        ast.dump(node)
        for node in ast.walk(control_device)
        if isinstance(node, ast.Assign)
    }
    changed = [
        node
        for statement in branch.body[1:]
        for node in ast.walk(statement)
        if isinstance(node, ast.Assign) and ast.dump(node) not in original_values
    ]
    parameters = {arg.arg for arg in control_device.args.args}
    original_index_casts = {
        ast.dump(node): node
        for node in ast.walk(control_device)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "cutlass.Int64"
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id not in parameters
    }
    assert len(original_index_casts) == 1
    widened = deepcopy(next(iter(original_index_casts.values())))
    assert isinstance(widened.value, ast.Call)
    widened.value.args[0] = ast.parse(
        f"cutlass.Uint32({ast.unparse(widened.value.args[0])})", mode="eval"
    ).body
    assert sum(ast.dump(node) == ast.dump(widened) for node in changed) == 1
    assert all(
        isinstance(node.value, ast.Constant)
        and node.value.value is True
        or ast.dump(node) == ast.dump(widened)
        for node in changed
    )
    assert "seed" in fast and "_uniform_cutoff(p)" in fast and " * scale" in fast


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_typed_loads_and_output_conversions_are_preserved(dtype):
    bound = _bound(dtype=dtype)
    control = _functions(bound.to_code(_config(enabled=False)))[1]
    enabled = _functions(bound.to_code(_config(enabled=True)))[1]
    assert isinstance(enabled.body[0], ast.If)
    assert _body_dump(enabled.body[0].orelse) == _body_dump(control.body)
    original_calls = {
        ast.dump(node)
        for node in ast.walk(control)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "cutlass"
        and node.func.attr in {"Float16", "BFloat16", "Float32", "Int64"}
    }
    kept_calls = {
        ast.dump(node)
        for statement in enabled.body[0].body[1:]
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "cutlass"
        and node.func.attr in {"Float16", "BFloat16", "Float32", "Int64"}
    }
    widened = ast.dump(
        ast.parse("cutlass.Int64(cutlass.Uint32(indices_0))", mode="eval").body
    )
    assert kept_calls - original_calls == {widened}


@pytest.mark.parametrize("stride", [2, 3])
def test_normal_strided_binding_keeps_scalar_memory_path(stride):
    bound = _bound(tensor=torch.empty(1027 * stride)[::stride])
    source = bound.to_code(_config(enabled=True))
    device = _functions(source)[1]
    # Ordinary binding/admission must decline contiguous packet operations.
    # The runtime guard is additional protection, not a dispatch around an
    # otherwise unsafe stride-specialized kernel.
    assert "_helion_affine_load" not in source
    assert "_cute_store_u32_vec(" not in source
    if len(device.body) == 1 and isinstance(device.body[0], ast.If):
        assert "layout.stride[0] == 1" in ast.unparse(device.body[0].test)


@helion.kernel(backend="cute", static_shapes=False)
def _independent_length(x: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(n):
        out[tile] = x[tile] * 2.0
    return out


@helion.kernel(backend="cute", static_shapes=False)
def _metadata_length(values: torch.Tensor) -> torch.Tensor:
    count = values.size(0)
    result = torch.empty_like(values)
    for positions in hl.tile(count):
        result[positions] = values[positions] * 2.0
    return result


def test_general_pointwise_metadata_extent_admits_without_rng_or_flatten():
    bound = cpu_bind(_metadata_length, (torch.empty(1024),))
    source = bound.to_code(_config(enabled=True))
    assert "cute.is_static(" in source
    assert "cutlass.Int64(cute.size(" in source


def test_equal_example_length_is_not_a_metadata_relation():
    bound = cpu_bind(_independent_length, (torch.empty(1024), 1024))
    source = bound.to_code(_config(enabled=True))
    assert "cute.is_static(" not in source


def _population(bound, *, count=100, explicit=None):
    with bound.env:
        kwargs = LFBOTreeSearch.get_kwargs_from_profile(
            get_effort_profile("full"), bound.settings
        )
        kwargs["initial_population"] = count
        with patch.object(bound.settings, "autotune_seed_configs", explicit):
            search = LFBOTreeSearch(
                bound, tuple(bound.host_function.params.arguments.values()), **kwargs
            )
            assert (
                search.initial_population_strategy
                == InitialPopulationStrategy.FROM_RANDOM
            )
            random.seed(2026091523)
            flats = search._generate_initial_population_flat()
            configs = [search.config_gen.unflatten(flat) for flat in flats]
            assert len(configs) == len(set(configs)) == count
            assert [search.config_gen.flatten(config) for config in configs] == flats
            assert configs[0] == search.config_gen.unflatten(
                search.config_gen.default_flat()
            )
            seeds = [config for _, config in search.config_gen.seed_flat_config_pairs()]
    return configs, seeds


@pytest.mark.parametrize("function", [low_mem_dropout, low_mem_dropout_bwd])
def test_full_initial_lfbo_population_contains_every_existing_and_true_seed(function):
    bound = _bound(function, effort="full")
    population, seeds = _population(bound)
    assert all(seed in population for seed in seeds)
    assert not population[0]["cute_proven_bounds"]
    with bound.env:
        with patch.object(bound.config_spec, "cute_packet_prefetch_enabled", False):
            legacy = CutePointwiseVecHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
        generation = ConfigGeneration(bound.config_spec)
        prefix = list(
            dict.fromkeys(
                generation.canonicalize_flat(generation.flatten(config))[1]
                for config in legacy
            )
        )
    assert seeds[: len(prefix)] == prefix
    assert all(seed in population for seed in seeds[len(prefix) :])
    old = [seed for seed in prefix if not seed["cute_proven_bounds"]]
    new = [seed for seed in prefix if seed["cute_proven_bounds"]]
    assert len(old) == len(new) >= 4
    assert prefix == old + new
    for control, enabled in zip(old, new, strict=True):
        expected = deepcopy(control.config)
        expected["cute_proven_bounds"] = True
        assert enabled.config == expected
    for threads in (128, 256):
        for width in (4, 8):
            for enabled in (False, True):
                assert any(
                    config.get("num_threads") == [threads]
                    and config.get("cute_vector_widths") == [width]
                    and config.block_sizes == [threads * width]
                    and config["cute_proven_bounds"] is enabled
                    for config in population
                )


@pytest.mark.parametrize("count", [2, 7, 30, 100])
def test_user_seed_priority_and_budget_are_preserved(count):
    bound = _bound(effort="full")
    explicit = _config(width=8, threads=512, enabled=True)
    population, seeds = _population(bound, count=count, explicit=[explicit])
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        expected = generation.canonicalize_flat(generation.flatten(explicit))[1]
    assert population[1] == expected
    retained = min(len(seeds), count - 2)
    assert population[2 : 2 + retained] == seeds[:retained]


def test_true_seed_lists_do_not_alias_prior_or_sibling_configs():
    bound = _bound()
    with bound.env:
        seeds = CutePointwiseVecHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    assert seeds
    halfway = len(seeds) // 2
    before = json.dumps([seed.config for seed in seeds[:halfway]], sort_keys=True)
    seeds[halfway].config["block_sizes"][0] = 1
    seeds[halfway].config["cute_vector_widths"][0] = 1
    assert (
        json.dumps([seed.config for seed in seeds[:halfway]], sort_keys=True) == before
    )


@pytest.mark.parametrize("width", [4, 8])
@pytest.mark.parametrize("length", [0, 7, 512, 1024, 1027])
@pytest.mark.parametrize("seed", [123, -1, 0x1234567812345678])
def test_actual_lane_ast_full_tiles_and_tails_preserve_original_rng_stream(
    width, length, seed
):
    before, packet, helpers = _capture(width)
    plan = FullTilePlan(
        TensorLayoutRelation("n", "x_flat", "size", 0),
        128 * width,
        (128, 1, 1),
        ("x_flat", "out_flat"),
    )
    specialized = specialize_full_tile_bounds(
        packet,
        plan,
        argument_names=("x_flat", "out_flat", "n", "seed", "p", "scale"),
        constexpr_values={"_BLOCK_SIZE_0": 128 * width},
        rename_groups={},
    )
    assert specialized is not packet and specialized[0].orelse is packet
    data = torch.arange(length, dtype=torch.float32) / 37 - 2
    probability = 0.25
    expected = torch.where(
        philox_rand_ref(seed, torch.arange(length, dtype=torch.int64)) > probability,
        data * (1 / (1 - probability)),
        0.0,
    )
    original = _execute(before, width, data, seed, probability, helpers)
    actual = _execute(specialized, width, data, seed, probability, helpers)
    assert torch.equal(original.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(actual.view(torch.int32), original.view(torch.int32))


@pytest.mark.parametrize("width", [4, 8])
def test_full_tile_preserves_runtime_probability_and_ieee_values(width):
    before, packet, helpers = _capture(width)
    specialized = specialize_full_tile_bounds(
        packet,
        FullTilePlan(
            TensorLayoutRelation("n", "x_flat", "size", 0),
            128 * width,
            (128, 1, 1),
            ("x_flat", "out_flat"),
        ),
        argument_names=("x_flat", "out_flat", "n", "seed", "p", "scale"),
        constexpr_values={"_BLOCK_SIZE_0": 128 * width},
        rename_groups={},
    )
    data = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan"), -1.0, 1.0, 2.0]
        * (16 * width)
    )
    for probability in (0.0, 0.6, -0.25, float("nan")):
        expected = _execute(before, width, data, -1, probability, helpers)
        actual = _execute(specialized, width, data, -1, probability, helpers)
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_compile_options_preserve_full_tile_specialization():
    bound = _bound(tensor=torch.empty(1024))
    default = ast.parse(bound.to_code(_config(enabled=True)))
    config = _config(enabled=True)
    config.config["advanced_controls_file"] = "compile-controls.txt"
    probe = ast.parse(bound.to_code(config))
    device = next(
        node
        for node in default.body
        if isinstance(node, ast.FunctionDef) and node.name == "_helion_low_mem_dropout"
    )
    assert len(device.body) == 1 and isinstance(device.body[0], ast.If)
    options = []
    for node in ast.walk(probe):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_launcher"
        ):
            for keyword in list(node.keywords):
                if keyword.arg == "cute_compile_options":
                    options.append(ast.literal_eval(keyword.value))
                    node.keywords.remove(keyword)
    assert options == ["--ptxas-options='--apply-controls compile-controls.txt'"]
    assert ast.dump(probe) == ast.dump(default)
