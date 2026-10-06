from __future__ import annotations

from copy import deepcopy
import random
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_compiler_coverage import make_search

import helion
from helion import exc
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.pattern_search import InitialPopulationStrategy
import helion.language as hl

KEY = "cute_fragment_scan"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _computed_scan(x: torch.Tensor, axis: hl.constexpr, reverse: hl.constexpr):
    out = torch.empty_like(x)
    reused = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        values = x[row, :, :] + 1
        out[row, :, :] = hl.cumsum(values, dim=axis, reverse=reverse)
        reused[row, :, :] = values * 2
    return out, reused


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _direct_scan(x: torch.Tensor):
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        out[row, :] = hl.cumsum(x[row, :], dim=-1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _computed_product(x: torch.Tensor):
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        out[row, :] = hl.cumprod(x[row, :] + 1, dim=-1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _barrier_scan(x: torch.Tensor):
    scratch = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        scratch[row, :] = hl.cumsum(x[row, :] + 1, dim=-1)
    hl.barrier()
    for row in hl.tile(x.size(0)):
        out[row, :] = hl.cumsum(scratch[row, :] * 2, dim=-1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _tiled_scan(x: torch.Tensor):
    out = torch.empty(x.shape, dtype=torch.int32, device=x.device)
    for row, col in hl.tile(x.shape, block_size=[2, 32]):
        flags = (x[row, col].view(torch.int32) < 0).to(torch.int32)
        out[row, col] = hl.cumsum(flags, dim=-1, reverse=True)
    return out


@pytest.fixture(autouse=True)
def _cpu_only():
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
    ):
        yield


def _bind():
    return _cpu_bind(_computed_scan, (torch.ones(3, 5, 65), 2, False))


def _config(bound, mode):
    return helion.Config.from_dict(
        dict(bound.config_spec.default_config()) | {KEY: mode}
    )


def test_strict_scan_enum_and_legacy_serial_config():
    bound = _bind()
    spec = bound.config_spec
    field = spec._flat_fields()[KEY]
    assert isinstance(field, EnumFragment)
    assert field.choices == ("serial", "cooperative")
    assert "input_tensor_metadata" in bound.env.compiler_fact_specialization_facts
    default = spec.default_config()
    assert KEY not in default
    serial = _config(bound, "serial")
    spec.normalize(serial.config)
    assert serial == default
    generation = spec.create_config_generation()
    flat, effective = generation.strict_config_pair(_config(bound, "cooperative"))
    assert effective[KEY] == "cooperative"
    assert generation.unflatten(flat)[KEY] == "cooperative"
    assert helion.Config.from_json(effective.to_json()) == effective
    old_flat = generation.flatten(default)
    index = generation._key_to_flat_indices[KEY][0][0]
    assert old_flat[index] == "serial"
    for invalid in ("unknown", "COOPERATIVE", True, 1, None):
        with pytest.raises(exc.InvalidConfig, match="computed additive scan"):
            spec.normalize(_config(bound, invalid).config)


@pytest.mark.parametrize("kernel", [_direct_scan, _computed_product])
def test_ineligible_scan_never_exposes_cooperative(kernel):
    bound = _cpu_bind(kernel, (torch.ones(3, 65),))
    assert not bound.config_spec.cute_fragment_scan_root_ids
    assert KEY not in bound.config_spec._flat_fields()
    with pytest.raises(exc.InvalidConfig, match="computed additive scan"):
        bound.config_spec.normalize(_config(bound, "cooperative").config)
    assert "input_tensor_metadata" in bound.env.compiler_fact_specialization_facts


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int8,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ],
)
@pytest.mark.parametrize("axis,reverse", [(1, False), (2, True)])
def test_both_scan_strategies_codegen_numeric_axes(dtype, axis, reverse):
    bound = _cpu_bind(
        _computed_scan, (torch.ones(3, 5, 65, dtype=dtype), axis, reverse)
    )
    serial = bound.to_code(_config(bound, "serial"))
    cooperative = bound.to_code(_config(bound, "cooperative"))
    assert "fragment_scan_initialized" in serial
    assert "fragment_scan_initialized" not in cooperative
    assert serial != cooperative
    assert bound.to_code(bound.config_spec.default_config()) == serial


def test_partial_tile_bitcast_capability_is_independent_of_scan_strategy():
    bound = _cpu_bind(_tiled_scan, (torch.randn(5, 65),))
    for mode in ("serial", "cooperative"):
        code = bound.to_code(_config(bound, mode))
        assert "fragment_smem" in code
        assert "32" in code


def test_phase_bundle_inherits_scan_strategy():
    bound = _cpu_bind(_barrier_scan, (torch.ones(3, 65),))
    assert len(bound.config_spec.cute_fragment_scan_root_ids) == 2
    serial = bound.to_code(_config(bound, "serial"))
    cooperative = bound.to_code(_config(bound, "cooperative"))
    assert "fragment_scan_initialized" in serial
    assert "fragment_scan_initialized" not in cooperative
    assert "__region_0" in cooperative and "__region_1" in cooperative


def test_scan_codegen_rechecks_lost_structural_admission():
    bound = _bind()
    with (
        patch(
            "helion._compiler.cute.computed_fragment.computed_fragment_supported",
            return_value=False,
        ),
        pytest.raises(exc.InvalidConfig, match="computed fragment root"),
    ):
        bound.to_code(_config(bound, "cooperative"))


def test_cooperative_scan_oversized_shared_memory_rejected():
    bound = _bind()
    with (
        patch(
            "helion._compiler.cute.tcgen05_config.CuteTcgen05Config.per_cta_smem_capacity_bytes",
            return_value=128,
        ),
        pytest.raises(exc.InvalidConfig, match="shared bytes"),
    ):
        bound.to_code(_config(bound, "cooperative"))


@pytest.mark.usefixtures("_without_later_fragment_coverage")
@pytest.mark.parametrize(
    "strategy",
    [
        InitialPopulationStrategy.FROM_RANDOM,
        InitialPopulationStrategy.FROM_BEST_AVAILABLE,
    ],
)
@pytest.mark.usefixtures("_without_thread_coverage")
@pytest.mark.usefixtures("_without_warp_scan_coverage")
def test_scan_coverage_preserves_original_population_and_rng(strategy):
    with patch("helion._compiler.autotuner_heuristics.register_fragment_scan_coverage"):
        old_bound = _bind()
    bound = _bind()
    old = make_search(old_bound.config_spec, count=20, strategy=strategy)
    full = make_search(bound.config_spec, count=20, strategy=strategy)
    seeds = deepcopy(bound.config_spec.compiler_seed_configs)
    random.seed(32189)
    expected = old._generate_initial_population_flat()
    state = random.getstate()
    random.seed(32189)
    actual = full._generate_initial_population_flat()
    assert random.getstate() == state
    indices = [
        index
        for key, (positions, _) in full.config_gen._key_to_flat_indices.items()
        if key != KEY
        for index in positions
    ]
    assert [[row[index] for index in indices] for row in actual[:20]] == expected
    assert len(actual) == len(expected) + 1
    assert bound.config_spec.compiler_seed_configs == seeds
    group = next(
        group
        for group in bound.config_spec.compiler_coverage_groups
        if group.key == KEY
    )
    assert group.domain == ("serial", "cooperative") and group.legacy == "serial"
    assert len(group.witnesses) == 1
    assert full.config_gen.unflatten(actual[-1])[KEY] == "cooperative"


def test_scan_helper_alpha_rejected_by_shared_proof_and_codegen():
    from helion._compiler.cute.computed_fragment import computed_fragment_supported
    from helion.language import scan_ops

    bound = _bind()
    ir = bound.host_function.device_ir
    scan = next(
        node
        for info in ir.graphs
        for node in info.graph.nodes
        if node.target is scan_ops._associative_scan
    )
    helper = ir.graphs[scan.args[0]]
    add = next(node for node in helper.graph.nodes if node.op == "call_function")
    original = add.kwargs
    try:
        add.kwargs = {"alpha": 2}
        with bound.env, bound.host_function:
            assert not computed_fragment_supported(bound.env, ir.graphs)
        with pytest.raises(exc.InvalidConfig, match="computed fragment root"):
            bound.to_code(_config(bound, "cooperative"))
    finally:
        add.kwargs = original


def test_cooperative_scan_cannot_be_intercepted_by_another_root_emitter():
    from helion._compiler.generate_ast import GenerateAST

    bound = _bind()
    with patch.object(
        GenerateAST,
        "_try_codegen_block_scaled_root",
        side_effect=AssertionError("requested fragment owner must run first"),
    ):
        assert "fragment_smem" in bound.to_code(_config(bound, "cooperative"))


def test_scan_search_full_neighbors_and_random_mutation_reach_both_modes():
    generation = _bind().config_spec.create_config_generation()
    projections = generation.coordinate_neighbor_projections(generation.default_flat())
    assert any(item.key == KEY and item.outcome == "candidate" for item in projections)
    random.seed(67129)
    assert {generation.random_config().get(KEY, "serial") for _ in range(24)} == {
        "serial",
        "cooperative",
    }


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.usefixtures("_without_thread_coverage")
@pytest.mark.usefixtures("_without_warp_scan_coverage")
def test_scan_coverage_honors_explicit_legacy_override_and_disabled_heuristics(
    disabled,
):
    bound = _bind()
    search = make_search(
        bound.config_spec,
        count=20,
        disabled=disabled,
        overrides={} if disabled else {KEY: "serial"},
    )
    rows = search._generate_initial_population_flat()
    assert len(rows) == 20
    assert all(
        search.config_gen.unflatten(row).get(KEY, "serial") == "serial" for row in rows
    )


def test_scan_fact_registration_remains_available_when_heuristics_disabled():
    kernel = helion.kernel(
        _computed_scan.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        disable_autotuner_heuristics=True,
    )
    bound = _cpu_bind(kernel, (torch.ones(3, 5, 65), 2, False))
    assert KEY in bound.config_spec._flat_fields()
    assert any(group.key == KEY for group in bound.config_spec.compiler_coverage_groups)
    assert bound.config_spec.compiler_seed_configs == []
    assert "fragment_scan_initialized" not in bound.to_code(
        _config(bound, "cooperative")
    )


REDUCTION_KEY = "cute_fragment_reduction"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _factored_scalar_reduction(x: torch.Tensor, kind: hl.constexpr):
    columns = hl.specialize(x.size(1))
    out = torch.empty((x.size(0), 4), device=x.device, dtype=x.dtype)
    for row in hl.tile(x.size(0)):
        value = x[row, :] + 1
        factored = value.reshape(value.size(0), 4, columns // 4)
        if kind == "sum":
            reduced = factored.sum(-1)
        elif kind == "min":
            reduced = factored.amin(-1)
        elif kind == "max":
            reduced = factored.amax(-1)
        elif kind == "prod":
            reduced = factored.prod(-1)
        else:
            reduced = factored.argmax(-1)
        out[row, :] = reduced
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _ordinary_scalar_reduction(x: torch.Tensor):
    out = torch.empty((x.size(0),), device=x.device)
    for row in hl.tile(x.size(0)):
        out[row] = x[row, :].sum(-1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _phase_fragment_reductions(x: torch.Tensor):
    scratch = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        value = x[row, :]
        factored = value.reshape(value.size(0), 4, x.size(1) // 4)
        scratch[row, :] = (factored + factored.sum(-1, keepdim=True)).reshape(
            value.shape
        )
    hl.barrier()
    for row in hl.tile(x.size(0)):
        value = scratch[row, :]
        factored = value.reshape(value.size(0), 4, x.size(1) // 4)
        out[row, :] = (factored + factored.amax(-1, keepdim=True)).reshape(value.shape)
    return out


def _bind_reduction(kind="sum", dtype=torch.float32):
    return _cpu_bind(
        _factored_scalar_reduction, (torch.ones(3, 128, dtype=dtype), kind)
    )


def _reduction_config(bound, mode):
    return helion.Config.from_dict(
        dict(bound.config_spec.default_config()) | {REDUCTION_KEY: mode}
    )


@pytest.mark.parametrize("kind", ["sum", "min", "max", "prod"])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
    ],
)
def test_fragment_reduction_algorithm_fact_matches_supported_combines(kind, dtype):
    bound = _bind_reduction(kind, dtype)
    spec = bound.config_spec
    assert spec.cute_fragment_reduction_root_ids
    assert "input_tensor_metadata" in bound.env.compiler_fact_specialization_facts
    field = spec._flat_fields()[REDUCTION_KEY]
    assert isinstance(field, EnumFragment)
    assert field.choices == ("serial", "warp")
    assert REDUCTION_KEY not in spec.default_config()
    generation = spec.create_config_generation()
    for mode in ("serial", "warp"):
        flat, config = generation.strict_config_pair(_reduction_config(bound, mode))
        assert config.get(REDUCTION_KEY, "serial") == mode
        assert generation.unflatten(flat).get(REDUCTION_KEY, "serial") == mode
        assert helion.Config.from_json(config.to_json()) == config


def test_fragment_reduction_strict_normalization_and_owner_conflicts():
    bound = _bind_reduction()
    spec = bound.config_spec
    serial = _reduction_config(bound, "serial")
    spec.normalize(serial.config)
    assert serial == spec.default_config()
    for invalid in ("unknown", "WARP", 1, True, None):
        with pytest.raises(exc.InvalidConfig, match="computed scalar reduction"):
            spec.normalize(_reduction_config(bound, invalid).config)
    for owner in ("cute_collective_mma", "cute_register_chain"):
        requested = {REDUCTION_KEY: "warp", owner: True}
        with pytest.raises(exc.InvalidConfig, match="computed scalar reduction"):
            spec._normalize_cute_fragment_reduction(requested, fix_invalid=False)
        spec._normalize_cute_fragment_reduction(requested, fix_invalid=True)
        assert REDUCTION_KEY not in requested


@pytest.mark.parametrize(
    "kind,dtype", [("argmax", torch.float32), ("max", torch.int16)]
)
def test_fragment_reduction_unsupported_combine_or_dtype_has_no_warp(kind, dtype):
    bound = _bind_reduction(kind, dtype)
    assert not bound.config_spec.cute_fragment_reduction_root_ids
    assert REDUCTION_KEY not in bound.config_spec._flat_fields()
    with pytest.raises(exc.InvalidConfig, match="computed scalar reduction"):
        bound.config_spec.normalize(_reduction_config(bound, "warp").config)


def test_fragment_reduction_ordinary_native_root_has_no_warp():
    bound = _cpu_bind(_ordinary_scalar_reduction, (torch.ones(3, 128),))
    assert not bound.config_spec.cute_fragment_reduction_root_ids
    assert REDUCTION_KEY not in bound.config_spec._flat_fields()
    with pytest.raises(exc.InvalidConfig, match="computed scalar reduction"):
        bound.config_spec.normalize(_reduction_config(bound, "warp").config)


def test_fragment_reduction_explicit_phase_roots_share_capability():
    bound = _cpu_bind(_phase_fragment_reductions, (torch.ones(3, 128),))
    assert len(bound.config_spec.cute_fragment_reduction_root_ids) == 2
    assert REDUCTION_KEY in bound.config_spec._flat_fields()


@pytest.mark.usefixtures("_without_later_fragment_coverage")
@pytest.mark.parametrize(
    "strategy",
    [
        InitialPopulationStrategy.FROM_RANDOM,
        InitialPopulationStrategy.FROM_BEST_AVAILABLE,
    ],
)
@pytest.mark.usefixtures("_without_thread_coverage")
def test_fragment_reduction_coverage_preserves_original_population_and_rng(strategy):
    with patch(
        "helion._compiler.autotuner_heuristics.register_fragment_reduction_coverage"
    ):
        old_bound = _bind_reduction()
    bound = _bind_reduction()
    old = make_search(old_bound.config_spec, count=20, strategy=strategy)
    full = make_search(bound.config_spec, count=20, strategy=strategy)
    seeds = deepcopy(bound.config_spec.compiler_seed_configs)
    random.seed(16317)
    expected = old._generate_initial_population_flat()
    state = random.getstate()
    random.seed(16317)
    actual = full._generate_initial_population_flat()
    assert random.getstate() == state
    indices = [
        index
        for key, (positions, _) in full.config_gen._key_to_flat_indices.items()
        if key != REDUCTION_KEY
        for index in positions
    ]
    assert [[row[index] for index in indices] for row in actual[:20]] == expected
    assert len(actual) == len(expected) + 1
    assert bound.config_spec.compiler_seed_configs == seeds
    group = next(
        group
        for group in bound.config_spec.compiler_coverage_groups
        if group.key == REDUCTION_KEY
    )
    assert group.domain == ("serial", "warp") and group.legacy == "serial"
    assert len(group.witnesses) == 1
    assert full.config_gen.unflatten(actual[-1])[REDUCTION_KEY] == "warp"


@pytest.mark.usefixtures("_without_thread_coverage")
def test_fragment_reduction_neighbors_random_mutation_and_overrides():
    bound = _bind_reduction()
    generation = bound.config_spec.create_config_generation()
    projections = generation.coordinate_neighbor_projections(generation.default_flat())
    assert any(
        item.key == REDUCTION_KEY and item.outcome == "candidate"
        for item in projections
    )
    random.seed(4182)
    assert {
        generation.random_config().get(REDUCTION_KEY, "serial") for _ in range(24)
    } == {"serial", "warp"}
    for disabled in (False, True):
        search = make_search(
            bound.config_spec,
            count=20,
            disabled=disabled,
            overrides={} if disabled else {REDUCTION_KEY: "serial"},
        )
        population = search._generate_initial_population_flat()
        assert len(population) == 20
        assert all(
            search.config_gen.unflatten(row).get(REDUCTION_KEY, "serial") == "serial"
            for row in population
        )


def test_fragment_reduction_facts_survive_disabled_heuristics():
    kernel = helion.kernel(
        _factored_scalar_reduction.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        disable_autotuner_heuristics=True,
    )
    bound = _cpu_bind(kernel, (torch.ones(3, 128), "sum"))
    assert bound.config_spec.cute_fragment_reduction_root_ids
    assert REDUCTION_KEY in bound.config_spec._flat_fields()
    assert any(
        group.key == REDUCTION_KEY
        for group in bound.config_spec.compiler_coverage_groups
    )
    assert bound.config_spec.compiler_seed_configs == []
    assert "input_tensor_metadata" in bound.env.compiler_fact_specialization_facts


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scan_and_fragment_reduction(x: torch.Tensor):
    out = torch.empty((x.size(0), 4), device=x.device)
    for row in hl.tile(x.size(0)):
        value = x[row, :]
        factored = value.reshape(value.size(0), 4, x.size(1) // 4)
        out[row, :] = hl.cumsum(factored.sum(-1) + 1, dim=-1)
    return out


@pytest.mark.usefixtures("_without_thread_coverage")
@pytest.mark.usefixtures("_without_warp_scan_coverage")
def test_fragment_reduction_coverage_composes_with_existing_scan_coverage():
    with patch(
        "helion._compiler.autotuner_heuristics.register_fragment_reduction_coverage"
    ):
        old_bound = _cpu_bind(_scan_and_fragment_reduction, (torch.ones(3, 128),))
    bound = _cpu_bind(_scan_and_fragment_reduction, (torch.ones(3, 128),))
    assert {group.key for group in bound.config_spec.compiler_coverage_groups} >= {
        KEY,
        REDUCTION_KEY,
    }
    old = make_search(old_bound.config_spec, count=20)
    full = make_search(bound.config_spec, count=20)
    random.seed(94513)
    expected = old._generate_initial_population_flat()
    state = random.getstate()
    random.seed(94513)
    actual = full._generate_initial_population_flat()
    assert random.getstate() == state
    indices = [
        index
        for key, (positions, _) in full.config_gen._key_to_flat_indices.items()
        if key != REDUCTION_KEY
        for index in positions
    ]
    assert [
        [row[index] for index in indices] for row in actual[: len(expected)]
    ] == expected
    assert len(expected) == 21 and len(actual) == 22
    assert full.config_gen.unflatten(actual[-1])[REDUCTION_KEY] == "warp"


def test_fragment_reduction_mixed_supported_and_indexed_nodes_use_shared_predicate():
    from helion._compiler.autotuner_heuristics.cute_fragment_reduction import (
        fragment_warp_reduction_supported,
    )
    from helion._compiler.inductor_lowering import ReductionLowering

    supported = _bind_reduction("max")
    indexed = _bind_reduction("argmax")
    supported_nodes = [
        node
        for info in supported.host_function.device_ir.graphs
        for node in info.graph.nodes
        if isinstance(node.meta.get("lowering"), ReductionLowering)
    ]
    indexed_nodes = [
        node
        for info in indexed.host_function.device_ir.graphs
        for node in info.graph.nodes
        if isinstance(node.meta.get("lowering"), ReductionLowering)
    ]
    assert supported_nodes and indexed_nodes
    assert all(fragment_warp_reduction_supported(node) for node in supported_nodes)
    assert not any(fragment_warp_reduction_supported(node) for node in indexed_nodes)


def test_fragment_reduction_default_codegen_retains_explicit_serial_path():
    bound = _bind_reduction()
    assert bound.to_code(bound.config_spec.default_config()) == bound.to_code(
        _reduction_config(bound, "serial")
    )


def test_fragment_reduction_key_is_specific_to_cute_backend():
    from helion._compiler.cute.backend import CuteBackend
    from helion._compiler.triton.backend import TritonBackend

    assert CuteBackend().supports_config_key(REDUCTION_KEY)
    assert not TritonBackend().supports_config_key(REDUCTION_KEY)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _resource_fullrow_scan(x: torch.Tensor):
    out = torch.empty((x.size(0),), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.size(0)):
        values = x[row, :]
        maximum = values.amax(-1)
        prefix = hl.cumsum(values - maximum[:, None], dim=-1)
        out[row] = prefix.sum(-1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _resource_repeated_scan_dot(x: torch.Tensor):
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    for row in hl.tile(x.size(0)):
        first = hl.cumsum(x[row, :, :] + 1, dim=-1)
        second = hl.cumsum(first, dim=-1)
        out[row, :, :] = hl.dot(first, second.transpose(-1, -2))
    return out


def _resource_bound_and_actual(bound, config):
    import math

    from helion._compiler.autotuner_heuristics.cute_fragment_resources import (
        FragmentResourceCatalog,
    )
    from helion._compiler.cute.computed_fragment import FragmentCompiler

    allocations = []
    original = FragmentCompiler.allocate

    def allocate(compiler, value):
        result = original(compiler, value)
        allocations.append((compiler, math.prod(value.shape) * value.dtype.itemsize))
        return result

    with bound.env, bound.host_function:
        catalog = FragmentResourceCatalog.create(
            bound.env, bound.host_function.device_ir, config
        )
        assert catalog is not None
        estimate = catalog.shared_bytes(bound.env, config)
    with (
        patch.object(FragmentCompiler, "allocate", allocate),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        source = bound.to_code(config)
    compilers = {compiler for compiler, _ in allocations}
    assert compilers
    for compiler in compilers:
        request_sum = sum(
            ((size + 15) // 16) * 16 for owner, size in allocations if owner is compiler
        )
        assert estimate >= request_sum >= compiler.smem_bytes
    return source


@pytest.mark.usefixtures("_without_later_fragment_coverage")
@pytest.mark.parametrize("width", [4097, 8193])
@pytest.mark.usefixtures("_without_thread_coverage")
@pytest.mark.usefixtures("_without_warp_scan_coverage")
def test_fragment_resource_supplements_preserve_full_cold_prefix_and_rng(width):
    inputs = (torch.ones(3, width),)
    with patch(
        "helion._compiler.autotuner_heuristics.fragment_resource_carrier",
        return_value=None,
    ):
        old_bound = _cpu_bind(_resource_fullrow_scan, inputs)
    kernel = helion.kernel(
        _resource_fullrow_scan.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    bound = _cpu_bind(kernel, inputs)
    old = make_search(old_bound.config_spec, count=20)
    full = make_search(bound.config_spec, count=20)
    seeds = deepcopy(bound.config_spec.compiler_seed_configs)
    default = bound.config_spec.default_config()
    random.seed(151)
    expected = old._generate_initial_population_flat()
    expected_state = random.getstate()
    random.seed(151)
    actual = full._generate_initial_population_flat()
    assert random.getstate() == expected_state
    assert actual[: len(expected)] == expected
    assert len(actual) == len(expected) + 2
    assert bound.config_spec.compiler_seed_configs == seeds
    assert bound.config_spec.default_config() == default
    assert len(full._pinned_finalist_configs) == len(old._pinned_finalist_configs) + 2
    for row in actual[len(expected) :]:
        config = full.config_gen.unflatten(row)
        assert config in full._pinned_finalist_configs
        _resource_bound_and_actual(bound, config)
    random.seed(155)
    modes = {
        full.config_gen.random_config().get(REDUCTION_KEY, "serial") for _ in range(32)
    }
    assert modes == {"serial", "warp"}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [5, 17])
def test_fragment_resource_bound_covers_scan_aliases_dot_and_tails(dtype, width):
    bound = _cpu_bind(
        _resource_repeated_scan_dot, (torch.ones(3, width, width, dtype=dtype),)
    )
    for mode in ("serial", "cooperative"):
        config = bound.config_spec.default_config()
        config.config[KEY] = mode
        _resource_bound_and_actual(bound, config)


def test_fragment_resource_catalog_declines_nested_and_unproved_runtime_extents():
    from test.test_cute_computed_fragment import _fragment_captured_reduction

    from helion._compiler.autotuner_heuristics.cute_fragment_resources import (
        FragmentResourceCatalog,
    )
    from helion._compiler.autotuner_heuristics.cute_fragment_resources import (
        FragmentStorageTerm,
    )

    bound = _cpu_bind(_fragment_captured_reduction, (torch.ones(3, 17), 3))
    with bound.env, bound.host_function:
        config = bound.config_spec.default_config()
        assert (
            FragmentResourceCatalog.create(
                bound.env, bound.host_function.device_ir, config
            )
            is None
        )
        unknown = bound.env.shape_env.create_unbacked_symint()
        with pytest.raises(
            exc.InvalidConfig, match="unproved fragment resource extent"
        ):
            FragmentStorageTerm((unknown,), torch.float32).shared_bytes(
                bound.env, config
            )


@pytest.mark.usefixtures("_without_thread_coverage")
@pytest.mark.usefixtures("_without_warp_scan_coverage")
def test_fragment_resource_carrier_builds_catalog_only_once():
    from helion._compiler.autotuner_heuristics.cute_fragment_resources import (
        FragmentResourceCatalog,
    )

    with patch.object(
        FragmentResourceCatalog, "create", wraps=FragmentResourceCatalog.create
    ) as create:
        kernel = helion.kernel(
            _resource_fullrow_scan.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
        )
        bound = _cpu_bind(kernel, (torch.ones(3, 8193),))
    assert create.call_count == 1
    assert all(
        group.supplemental_witnesses
        for group in bound.config_spec.compiler_coverage_groups
    )


@pytest.mark.usefixtures("_without_later_fragment_coverage")
@pytest.mark.parametrize("rows", [65536, 131072])
@pytest.mark.usefixtures("_without_thread_coverage")
@pytest.mark.usefixtures("_without_warp_scan_coverage")
def test_fragment_resource_hard_minima_preserve_prefix_and_rng(rows):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from helion._compiler.autotuner_heuristics.cute_fragment_resources import (
        FragmentResourceCatalog,
    )

    with FakeTensorMode():
        inputs = (torch.empty(rows, 4097),)
    # Without the supplemental tier, these large-grid floors exclude every
    # storage-feasible row tile. Ordinary seeds and coverage remain unchanged.
    with patch(
        "helion._compiler.autotuner_heuristics.fragment_resource_carrier",
        return_value=None,
    ):
        old_bound = _cpu_bind(_resource_fullrow_scan, inputs)
    kernel = helion.kernel(
        _resource_fullrow_scan.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    bound = _cpu_bind(kernel, inputs)
    spec = bound.config_spec
    block = spec.block_sizes[0]
    assert block.min_size < block.autotuner_min
    assert spec.default_config() == old_bound.config_spec.default_config()
    assert spec.compiler_seed_configs == old_bound.config_spec.compiler_seed_configs
    old = make_search(old_bound.config_spec, count=20)
    full = make_search(spec, count=20)
    random.seed(421)
    expected = old._generate_initial_population_flat()
    expected_state = random.getstate()
    random.seed(421)
    actual = full._generate_initial_population_flat()
    assert random.getstate() == expected_state
    assert actual[: len(expected)] == expected
    assert len(actual) == len(expected) + 2
    for flat in actual[len(expected) :]:
        config = full.config_gen.unflatten(flat)
        assert config in full._pinned_finalist_configs
        assert config["block_sizes"] == [block.min_size]
        assert full.config_gen.strict_config_pair(config)[1] == config
        with bound.env, bound.host_function:
            catalog = FragmentResourceCatalog.create(
                bound.env, bound.host_function.device_ir, config
            )
            assert catalog is not None
            floor = deepcopy(config)
            floor["block_sizes"][0] = block.autotuner_min
            assert catalog.shared_bytes(bound.env, floor) > 232448
        _resource_bound_and_actual(bound, config)


def test_fragment_resource_hard_minima_keep_legal_bounds_and_scheduling():
    from torch._subclasses.fake_tensor import FakeTensorMode

    from helion._compiler.autotuner_heuristics.cute_fragment_resources import (
        FragmentResourceCatalog,
    )
    from helion._compiler.autotuner_heuristics.cute_fragment_resources import (
        fragment_resource_carrier,
    )

    with FakeTensorMode():
        inputs = (torch.empty(131072, 4097),)
    bound = _cpu_bind(_resource_fullrow_scan, inputs)
    spec = bound.config_spec
    original_default = spec.default_config()
    # A non-unit hard minimum is legal but must never be crossed. Raising it
    # further makes the conservative model infeasible, rather than waiving it.
    with bound.env, bound.host_function:
        spec.block_sizes[0].min_size = 2
        carrier = fragment_resource_carrier(bound.env, bound.host_function.device_ir)
        assert carrier is not None and carrier["block_sizes"] == [2]
        assert {
            key: value for key, value in carrier.items() if key != "block_sizes"
        } == {
            key: value
            for key, value in original_default.items()
            if key != "block_sizes"
        }
        catalog = FragmentResourceCatalog.create(
            bound.env, bound.host_function.device_ir, carrier
        )
        assert catalog is not None
        assert catalog.shared_bytes(bound.env, carrier) <= 232448
        spec.block_sizes[0].min_size = 4
        assert (
            fragment_resource_carrier(bound.env, bound.host_function.device_ir) is None
        )


def test_fragment_resource_hard_minima_never_codegen_neighbors():
    from torch._subclasses.fake_tensor import FakeTensorMode

    from helion._compiler.autotuner_heuristics.cute_fragment_resources import (
        FragmentResourceCatalog,
    )
    from helion._compiler.autotuner_heuristics.cute_fragment_resources import (
        fragment_resource_carrier,
    )
    from helion.runtime.kernel import BoundKernel

    with FakeTensorMode():
        inputs = (torch.empty(65536, 4097),)
    bound = _cpu_bind(_resource_fullrow_scan, inputs)
    with (
        bound.env,
        bound.host_function,
        patch.object(
            FragmentResourceCatalog, "create", wraps=FragmentResourceCatalog.create
        ) as create,
        patch.object(
            BoundKernel,
            "to_code",
            side_effect=AssertionError("no codegen in resource scoring"),
        ),
    ):
        carrier = fragment_resource_carrier(bound.env, bound.host_function.device_ir)
    assert carrier is not None and carrier["block_sizes"] == [1]
    assert create.call_count == 1


@pytest.fixture
def _without_thread_coverage():
    # Keep each earlier coordinate's population-size tests isolated. Combined
    # old/new coverage and the full prior prefix are tested below.
    with patch(
        "helion._compiler.autotuner_heuristics.register_fragment_threads_coverage"
    ):
        yield


THREAD_KEY = "cute_fragment_threads"


def _thread_config(bound, count):
    return helion.Config.from_dict(
        dict(bound.config_spec.default_config()) | {THREAD_KEY: count}
    )


def test_fragment_threads_strict_domain_and_roundtrip():
    from helion._compiler.autotuner_heuristics.cute_fragment_threads import THREADS

    bound = _bind()
    spec = bound.config_spec
    assert spec._flat_fields()[THREAD_KEY].choices == THREADS
    default = spec.default_config()
    assert THREAD_KEY not in default
    generation = spec.create_config_generation()
    for count in THREADS:
        flat, config = generation.strict_config_pair(_thread_config(bound, count))
        assert config.get(THREAD_KEY, 128) == count
        assert generation.unflatten(flat) == config
        assert helion.Config.from_json(config.to_json()) == config
    assert generation.strict_config_pair(_thread_config(bound, 128))[1] == default
    for value in (True, False, 0, 16, 96, 1023, 1056, 2048, "1024", 128.0, None):
        with pytest.raises(exc.InvalidConfig, match="computed fragment root"):
            spec.normalize(_thread_config(bound, value).config)


@pytest.mark.parametrize("kernel", [_direct_scan, _computed_product])
def test_fragment_threads_unsupported_root_rejected(kernel):
    bound = _cpu_bind(kernel, (torch.ones(3, 65),))
    assert not bound.config_spec.cute_fragment_thread_root_ids
    assert THREAD_KEY not in bound.config_spec._flat_fields()
    with pytest.raises(exc.InvalidConfig, match="computed fragment root"):
        bound.config_spec.normalize(_thread_config(bound, 512).config)


@pytest.mark.parametrize("count", [32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("mode", ["serial", "cooperative"])
def test_fragment_threads_scan_loop_and_launch_ownership(count, mode):
    import ast

    from test.test_cute_computed_fragment import _simulate_independent_fragment

    x = torch.arange(3 * 5 * 65, dtype=torch.float32).reshape(3, 5, 65) % 8
    bound = _cpu_bind(_computed_scan, (x, 2, False))
    config = _thread_config(bound, count)
    config.config[KEY] = mode
    code = bound.to_code(config)
    assert f"block=({count}, 1, 1)" in code
    loops = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("fragment_index")
    ]
    assert loops
    for node in loops:
        assert ast.literal_eval(node.iter.args[2]) == count
        extent = ast.literal_eval(node.iter.args[1])
        owned = [i for thread in range(count) for i in range(thread, extent, count)]
        assert sorted(owned) == list(range(extent))
    actual = torch.empty_like(x)
    reused = torch.empty_like(x)
    _simulate_independent_fragment(code, {"x": x}, {"out": actual, "reused": reused}, 1)
    torch.testing.assert_close(actual, (x + 1).cumsum(2), rtol=0, atol=0)
    torch.testing.assert_close(reused, (x + 1) * 2, rtol=0, atol=0)


def test_fragment_threads_codegen_rechecks_owner_and_preserves_default():
    bound = _bind()
    assert bound.to_code(_thread_config(bound, 128)) == bound.to_code(
        bound.config_spec.default_config()
    )
    assert "block=(512, 1, 1)" in bound.to_code(_thread_config(bound, 512))
    with (
        patch(
            "helion._compiler.cute.computed_fragment.computed_fragment_supported",
            return_value=False,
        ),
        pytest.raises(exc.InvalidConfig, match="computed fragment root"),
    ):
        bound.to_code(_thread_config(bound, 512))


@pytest.mark.usefixtures("_without_later_fragment_coverage")
@pytest.mark.parametrize(
    "strategy",
    [
        InitialPopulationStrategy.FROM_RANDOM,
        InitialPopulationStrategy.FROM_BEST_AVAILABLE,
    ],
)
@pytest.mark.parametrize("resource", [False, True])
@pytest.mark.usefixtures("_without_warp_scan_coverage")
def test_fragment_threads_complete_prior_population_and_rng(strategy, resource):
    kernel = _resource_fullrow_scan if resource else _scan_and_fragment_reduction
    x = torch.ones(3, 8193) if resource else torch.ones(3, 128)
    with patch(
        "helion._compiler.autotuner_heuristics.register_fragment_threads_coverage"
    ):
        old_bound = _cpu_bind(kernel, (x,))
    bound = _cpu_bind(kernel, (x,))
    old = make_search(old_bound.config_spec, count=20, strategy=strategy)
    full = make_search(bound.config_spec, count=20, strategy=strategy)
    random.seed(91234)
    expected = old._generate_initial_population_flat()
    rng = random.getstate()
    random.seed(91234)
    actual = full._generate_initial_population_flat()
    assert random.getstate() == rng
    old_configs = [old.config_gen.unflatten(row) for row in expected]
    new_configs = [full.config_gen.unflatten(row) for row in actual]
    assert new_configs[: len(old_configs)] == old_configs
    assert len(new_configs) == len(old_configs) + 4
    assert [config[THREAD_KEY] for config in new_configs[len(old_configs) :]] == [
        32,
        64,
        256,
        512,
    ]
    assert (
        bound.config_spec.compiler_seed_configs
        == old_bound.config_spec.compiler_seed_configs
    )
    assert bound.config_spec.default_config() == old_bound.config_spec.default_config()


def test_fragment_threads_disabled_seeds_neighbors_and_override():
    kernel = helion.kernel(
        _computed_scan.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        disable_autotuner_heuristics=True,
    )
    bound = _cpu_bind(kernel, (torch.ones(3, 5, 65), 2, False))
    assert bound.config_spec.cute_fragment_thread_root_ids
    assert "input_tensor_metadata" in bound.env.compiler_fact_specialization_facts
    generation = bound.config_spec.create_config_generation()
    assert any(
        item.key == THREAD_KEY and item.outcome == "candidate"
        for item in generation.coordinate_neighbor_projections(
            generation.default_flat()
        )
    )
    for overrides, disabled in (({THREAD_KEY: 128}, False), ({}, True)):
        search = make_search(
            bound.config_spec, count=20, overrides=overrides, disabled=disabled
        )
        rows = search._generate_initial_population_flat()
        assert all(
            search.config_gen.unflatten(row).get(THREAD_KEY, 128) == 128 for row in rows
        )


def test_fragment_threads_both_ordered_phases_and_dynamic_rebind():
    for shape in ((3, 65), (5, 129)):
        bound = _cpu_bind(_barrier_scan, (torch.ones(shape),))
        assert len(bound.config_spec.cute_fragment_thread_root_ids) == 2
        for threads in (512, 1024):
            code = bound.to_code(_thread_config(bound, threads))
            assert code.count(f"block=({threads}, 1, 1)") == 2


def test_fragment_1024_threads_late_search_reachability():
    bound = _bind()
    generation = bound.config_spec.create_config_generation()
    default = generation.default_flat()
    projections = [
        item
        for item in generation.coordinate_neighbor_projections(default)
        if item.key == THREAD_KEY and item.outcome == "candidate"
    ]
    candidate = next(item for item in projections if item.config[THREAD_KEY] == 1024)
    _, canonical = generation.strict_config_pair(candidate.config)
    assert canonical[THREAD_KEY] == 1024
    assert "block=(1024, 1, 1)" in bound.to_code(canonical)

    # The initial coverage view pins this entire coordinate to legacy128,
    # then appends the same four old witnesses. Later neighbors can explore
    # the appended enum value without changing that initial-population policy.
    group = next(
        g for g in bound.config_spec.compiler_coverage_groups if g.key == THREAD_KEY
    )
    assert [w.value for w in group.witnesses] == [32, 64, 256, 512]
    current = generation.flatten(canonical)
    previous = generation.flatten(_thread_config(bound, 512))
    values = set()
    for seed in (3, 11, 29, 41, 59, 71, 83, 97):
        random.seed(seed)
        mutated = generation.differential_mutation(
            default, default, current, previous, crossover_rate=1.0
        )
        _, config = generation.canonicalize_flat(mutated)
        values.add(config.get(THREAD_KEY, 128))
    assert 1024 in values


def test_fragment_threads_backend_specific():
    from helion._compiler.cute.backend import CuteBackend
    from helion._compiler.triton.backend import TritonBackend

    assert CuteBackend().supports_config_key(THREAD_KEY)
    assert not TritonBackend().supports_config_key(THREAD_KEY)


REGISTER_LOADS_KEY = "cute_fragment_register_loads"


def _register_load_bound():
    from test.test_atomic_ops import _register_load_histogram

    return _cpu_bind(_register_load_histogram, (torch.ones(2, 65), 3, 128))


def test_register_loads_strict_bool_roundtrip_and_neighbors():
    from helion.autotuner.config_fragment import BooleanFragment

    bound = _register_load_bound()
    spec = bound.config_spec
    assert isinstance(spec._flat_fields()[REGISTER_LOADS_KEY], BooleanFragment)
    assert REGISTER_LOADS_KEY not in spec.default_config()
    assert "input_tensor_metadata" in bound.env.compiler_fact_specialization_facts
    config = spec.default_config()
    config.config[REGISTER_LOADS_KEY] = False
    spec.normalize(config.config)
    assert config == spec.default_config()
    for value in (0, 1, "true", None):
        config = spec.default_config()
        config.config[REGISTER_LOADS_KEY] = value
        with pytest.raises(exc.InvalidConfig, match="lane-private"):
            spec.normalize(config.config)
    config = spec.default_config()
    config.config[REGISTER_LOADS_KEY] = True
    generation = spec.create_config_generation()
    flat, effective = generation.strict_config_pair(config)
    assert effective[REGISTER_LOADS_KEY] is True
    assert generation.unflatten(flat)[REGISTER_LOADS_KEY] is True
    assert any(
        item.key == REGISTER_LOADS_KEY and item.outcome == "candidate"
        for item in generation.coordinate_neighbor_projections(
            generation.default_flat()
        )
    )
    random.seed(147)
    assert {
        generation.random_config().get(REGISTER_LOADS_KEY, False) for _ in range(32)
    } == {False, True}


@pytest.mark.usefixtures("_without_later_fragment_coverage")
@pytest.mark.parametrize(
    "strategy",
    [
        InitialPopulationStrategy.FROM_RANDOM,
        InitialPopulationStrategy.FROM_BEST_AVAILABLE,
    ],
)
def test_register_loads_complete_existing_prefix_and_rng(strategy):
    with patch(
        "helion._compiler.autotuner_heuristics.register_fragment_register_loads_coverage"
    ):
        old_bound = _register_load_bound()
    bound = _register_load_bound()
    old = make_search(old_bound.config_spec, count=20, strategy=strategy)
    new = make_search(bound.config_spec, count=20, strategy=strategy)
    random.seed(741)
    prior = old._generate_initial_population_flat()
    rng = random.getstate()
    random.seed(741)
    actual = new._generate_initial_population_flat()
    assert random.getstate() == rng
    expected_configs = [old.config_gen.unflatten(row) for row in prior]
    actual_configs = [new.config_gen.unflatten(row) for row in actual]
    assert actual_configs[: len(expected_configs)] == expected_configs
    assert len(actual_configs) == len(expected_configs) + 1
    assert all(
        config[REGISTER_LOADS_KEY] is True
        for config in actual_configs[len(expected_configs) :]
    )
    assert (
        bound.config_spec.compiler_seed_configs
        == old_bound.config_spec.compiler_seed_configs
    )
    assert bound.config_spec.default_config() == old_bound.config_spec.default_config()
    for overrides, disabled in (({REGISTER_LOADS_KEY: False}, False), ({}, True)):
        search = make_search(
            bound.config_spec, count=20, overrides=overrides, disabled=disabled
        )
        rows = search._generate_initial_population_flat()
        assert all(
            not search.config_gen.unflatten(row).get(REGISTER_LOADS_KEY, False)
            for row in rows
        )


def test_register_loads_ineligible_scan_and_backend():
    from helion._compiler.cute.backend import CuteBackend
    from helion._compiler.triton.backend import TritonBackend

    bound = _cpu_bind(_direct_scan, (torch.ones(2, 65),))
    assert not bound.config_spec.cute_fragment_register_load_root_ids
    assert REGISTER_LOADS_KEY not in bound.config_spec._flat_fields()
    config = bound.config_spec.default_config()
    config.config[REGISTER_LOADS_KEY] = True
    with pytest.raises(exc.InvalidConfig, match="lane-private"):
        bound.config_spec.normalize(config.config)
    assert CuteBackend().supports_config_key(REGISTER_LOADS_KEY)
    assert not TritonBackend().supports_config_key(REGISTER_LOADS_KEY)


def test_register_loads_proof_rejects_remapping_reduction_carry_branch_and_mutation():
    import operator
    from types import SimpleNamespace

    from torch.fx import Graph

    from helion._compiler.cute.register_loads import lane_private_load
    from helion.language import _tracing_ops
    from helion.language import atomic_ops
    from helion.language import memory_ops

    env = SimpleNamespace(known_equal=operator.eq)
    for mode in (
        "pointwise",
        "same_shape_transpose",
        "scan",
        "reduce",
        "carry",
        "branch",
        "mutable",
        "shape_change",
        "atomic_return",
    ):
        graph = Graph()
        host = graph.call_function(_tracing_ops._host_tensor, ("input",))
        host.meta["val"] = torch.empty(4, 4)
        load = graph.call_function(
            memory_ops.load, (host, [slice(None), slice(None)], None, None)
        )
        load.meta["val"] = torch.empty(4, 4)
        target = graph.call_function(_tracing_ops._host_tensor, ("output",))
        target.meta["val"] = torch.empty(4, 4)
        op = {
            "pointwise": torch.ops.aten.add.Tensor,
            "same_shape_transpose": torch.ops.aten.transpose.int,
            "scan": torch.ops.aten.cumsum.default,
            "reduce": torch.ops.aten.sum.dim_IntList,
            "carry": _tracing_ops._phi,
            "branch": _tracing_ops._if,
            "mutable": torch.ops.aten.add_.Tensor,
            "shape_change": torch.ops.aten.reshape.default,
            "atomic_return": torch.ops.aten.add.Tensor,
        }[mode]
        user = graph.call_function(op, (load, 1))
        user.meta["val"] = (
            torch.empty(16) if mode == "shape_change" else torch.empty(4, 4)
        )
        atomic = graph.call_function(
            atomic_ops.atomic_add, (target, [0, 0], user, "relaxed")
        )
        atomic.meta["val"] = torch.empty(4, 4)
        graph.output(atomic if mode == "atomic_return" else None)
        assert lane_private_load(load, env) == (mode == "pointwise")


def test_register_loads_preloop_capture_remains_shared():
    from test.test_atomic_ops import _local_atomic_histogram

    bound = _cpu_bind(_local_atomic_histogram, (torch.ones(2, 32), 17, 3, "plain"))
    assert not bound.config_spec.cute_fragment_register_load_root_ids
    config = bound.config_spec.default_config()
    config.config[REGISTER_LOADS_KEY] = True
    with pytest.raises(exc.InvalidConfig, match="lane-private"):
        bound.to_code(config)


WARP_SCAN_KEY = "cute_fragment_warp_scan"


@pytest.fixture
def _without_warp_scan_coverage():
    # The original groups retain their isolated historical assertions. The
    # complete prefix including every previous group is checked separately.
    with patch(
        "helion._compiler.autotuner_heuristics.register_fragment_warp_scan_coverage"
    ):
        yield


@pytest.fixture
def _without_later_fragment_coverage():
    """Keep historical prefix tests in their original registration universe.

    Complete staging populations, including these additions, are compared
    separately; these tests retain their original exact extension counts.
    """
    from contextlib import ExitStack

    with ExitStack() as stack:
        for registration in (
            "register_fragment_register_snapshots_coverage",
        ):
            stack.enter_context(
                patch("helion._compiler.autotuner_heuristics." + registration)
            )
        yield


def _warp_scan_config(bound, value=True):
    return helion.Config.from_dict(
        dict(bound.config_spec.default_config()) | {WARP_SCAN_KEY: value}
    )


def test_warp_scan_boolean_strict_admission_and_legacy_emitters():
    bound = _bind()
    spec = bound.config_spec
    generation = spec.create_config_generation()
    assert WARP_SCAN_KEY not in spec.default_config()
    assert spec._flat_fields()[KEY].choices == ("serial", "cooperative")
    assert (
        generation.strict_config_pair(_warp_scan_config(bound, False))[1]
        == spec.default_config()
    )
    flat, config = generation.strict_config_pair(_warp_scan_config(bound))
    assert config[WARP_SCAN_KEY] is True
    assert generation.unflatten(flat) == config
    for invalid in (1, 0, None, "true"):
        with pytest.raises(exc.InvalidConfig, match="bounded warp-prefix"):
            generation.strict_config_pair(_warp_scan_config(bound, invalid))
    for mode in ("serial", "cooperative"):
        legacy = _config(bound, mode)
        explicit = helion.Config.from_dict(dict(legacy) | {WARP_SCAN_KEY: False})
        assert bound.to_code(legacy) == bound.to_code(explicit)
    code = bound.to_code(_warp_scan_config(bound))
    assert "shuffle_sync_up" in code and "fragment_scan_carry" in code
    assert any(
        p.key == WARP_SCAN_KEY and p.outcome == "candidate"
        for p in generation.coordinate_neighbor_projections(generation.default_flat())
    )
    random.seed(67312)
    assert {
        generation.random_config().get(WARP_SCAN_KEY, False) for _ in range(24)
    } == {False, True}


@pytest.mark.usefixtures("_without_later_fragment_coverage")
@pytest.mark.parametrize(
    "strategy",
    [
        InitialPopulationStrategy.FROM_RANDOM,
        InitialPopulationStrategy.FROM_BEST_AVAILABLE,
    ],
)
def test_warp_scan_complete_prior_prefix_rng_seeds_and_overrides(strategy):
    with patch(
        "helion._compiler.autotuner_heuristics.register_fragment_warp_scan_coverage"
    ):
        old_bound = _bind()
    bound = _bind()
    old, new = (
        make_search(b.config_spec, count=20, strategy=strategy)
        for b in (old_bound, bound)
    )
    random.seed(72391)
    before = [
        old.config_gen.unflatten(row) for row in old._generate_initial_population_flat()
    ]
    rng = random.getstate()
    random.seed(72391)
    after = [
        new.config_gen.unflatten(row) for row in new._generate_initial_population_flat()
    ]
    assert random.getstate() == rng
    assert after[: len(before)] == before
    assert len(after) == len(before) + 1
    assert after[-1][WARP_SCAN_KEY] is True
    assert bound.config_spec.default_config() == old_bound.config_spec.default_config()
    assert (
        bound.config_spec.compiler_seed_configs
        == old_bound.config_spec.compiler_seed_configs
    )
    for disabled, overrides in ((True, {}), (False, {WARP_SCAN_KEY: False})):
        search = make_search(
            bound.config_spec, count=20, disabled=disabled, overrides=overrides
        )
        assert all(
            not search.config_gen.unflatten(row).get(WARP_SCAN_KEY, False)
            for row in search._generate_initial_population_flat()
        )


@pytest.mark.parametrize("kernel", [_direct_scan, _computed_product])
def test_warp_scan_unsupported_owner_rejects(kernel):
    bound = _cpu_bind(kernel, (torch.ones(3, 65),))
    assert not bound.config_spec.cute_fragment_warp_scan_root_ids
    with pytest.raises(exc.InvalidConfig, match="bounded warp-prefix"):
        bound.to_code(_warp_scan_config(bound))


@pytest.mark.parametrize(
    "dtype,columns", [(torch.float16, 65), (torch.bfloat16, 65), (torch.float32, 1025)]
)
def test_warp_scan_dtype_and_capacity_rejection(dtype, columns):
    bound = _cpu_bind(
        _computed_scan, (torch.ones(1, 2, columns, dtype=dtype), 2, False)
    )
    with pytest.raises(exc.InvalidConfig, match="(warp-prefix|axis capacity)"):
        bound.to_code(_warp_scan_config(bound))


def test_warp_scan_lost_proof_and_specialized_owner_do_not_fallback():
    from helion._compiler.generate_ast import GenerateAST

    bound = _bind()
    with patch.object(
        GenerateAST,
        "_try_codegen_block_scaled_root",
        side_effect=AssertionError("wrong owner"),
    ):
        assert "shuffle_sync_up" in bound.to_code(_warp_scan_config(bound))
    with (
        patch(
            "helion._compiler.cute.computed_fragment.computed_fragment_supported",
            return_value=False,
        ),
        pytest.raises(exc.InvalidConfig, match="computed fragment root"),
    ):
        bound.to_code(_warp_scan_config(bound))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mixed_dtype_phase_scans(x: torch.Tensor, y: torch.Tensor):
    a = torch.empty_like(x)
    b = torch.empty_like(y)
    for row in hl.tile(x.size(0)):
        a[row, :] = hl.cumsum(x[row, :] + 1, -1)
    hl.barrier()
    for row in hl.tile(y.size(0)):
        b[row, :] = hl.cumsum(y[row, :] + 1, -1)
    return a, b


def test_warp_scan_mixed_phase_roots_keep_ineligible_serial_emitter():
    bound = _cpu_bind(
        _mixed_dtype_phase_scans,
        (torch.ones(3, 33, dtype=torch.float16), torch.ones(3, 65)),
    )
    assert len(bound.config_spec.cute_fragment_warp_scan_root_ids) == 1
    code = bound.to_code(_warp_scan_config(bound))
    assert "fragment_scan_initialized" in code
    assert "shuffle_sync_up" in code


@pytest.mark.parametrize("reverse", [False, True])
def test_warp_scan_zero_physical_capacity_rejects(reverse):
    bound = _cpu_bind(_computed_scan, (torch.empty(3, 5, 0), 2, reverse))
    with pytest.raises(exc.InvalidConfig, match="axis capacity"):
        bound.to_code(_warp_scan_config(bound))


@pytest.mark.usefixtures("_without_later_fragment_coverage")
@pytest.mark.parametrize("family", ["private", "producer", "scan"])
@pytest.mark.parametrize(
    "strategy",
    [
        InitialPopulationStrategy.FROM_RANDOM,
        InitialPopulationStrategy.FROM_BEST_AVAILABLE,
    ],
)
def test_fragment_extensions_complete_legacy_prefix_rng_and_final_coverage(
    family, strategy
):
    from contextlib import ExitStack

    from test.test_atomic_ops import _fragment_terminal_loop_finalizer
    from test.test_atomic_ops import _terminal_loop_args
    from test.test_cute_computed_fragment import _fragment_cached_transcendental

    keys = (
        "cute_fragment_private_scalar_loops",
        "cute_fragment_producer_cache",
        "cute_fragment_warp_scan",
        "cute_fragment_atomic_aggregation",
    )
    registrations = (
        "register_fragment_private_scalar_loops_coverage",
        "register_fragment_producer_cache_coverage",
        "register_fragment_warp_scan_coverage",
        "register_fragment_atomic_aggregation_coverage",
    )
    if family == "private":
        fn = _fragment_terminal_loop_finalizer.fn
        args = _terminal_loop_args(3, 17, torch.int32, 0, 2, 1)
        eligible = {keys[0], keys[3]}
    elif family == "producer":
        fn = _fragment_cached_transcendental.fn
        args = (torch.ones(3, 65),)
        eligible = {keys[1], keys[2]}
    else:
        fn = _computed_scan.fn
        args = (torch.ones(3, 5, 65), 2, False)
        eligible = {keys[2]}

    specs = []
    for enabled in range(len(registrations) + 1):
        with ExitStack() as stack:
            for registration in registrations[enabled:]:
                stack.enter_context(
                    patch("helion._compiler.autotuner_heuristics." + registration)
                )
            specs.append(
                _cpu_bind(
                    helion.kernel(fn, backend="cute", static_shapes=True), args
                ).config_spec
            )
    for spec in specs[1:]:
        assert spec.default_config() == specs[0].default_config()
        assert spec.compiler_seed_configs == specs[0].compiler_seed_configs

    for seed in (73, 741, 2031):
        populations, states = [], []
        for spec in specs:
            search = make_search(spec, count=20, strategy=strategy)
            random.seed(seed)
            populations.append(
                [
                    search.config_gen.unflatten(row)
                    for row in search._generate_initial_population_flat()
                ]
            )
            states.append(random.getstate())
        assert all(state == states[0] for state in states)
        # The whole pre-extension population remains an exact prefix.
        assert populations[-1][: len(populations[0])] == populations[0]
        for index, key in enumerate(keys):
            before, after = populations[index : index + 2]
            assert after[: len(before)] == before
            assert len(after) == len(before) + (key in eligible)
            if key in eligible:
                assert after[-1][key] is True
        observed = {
            key
            for key in keys
            if any(config.get(key, False) for config in populations[-1])
        }
        assert observed == eligible


ATOMIC_AGGREGATION_KEY = "cute_fragment_atomic_aggregation"


def _atomic_aggregation_bind():
    from test.test_atomic_ops import _fragment_aggregated_histogram

    return _cpu_bind(
        _fragment_aggregated_histogram,
        (
            torch.ones((2, 65), dtype=torch.int32),
            torch.zeros((2, 65), dtype=torch.int32),
            17,
            0,
            1,
        ),
    )


def test_atomic_aggregation_strict_scope_and_ordinary_mutation():
    bound = _atomic_aggregation_bind()
    generation = bound.config_spec.create_config_generation()
    default = bound.config_spec.default_config()
    assert ATOMIC_AGGREGATION_KEY not in default
    for invalid in (1, 0, "true", None):
        with pytest.raises(exc.InvalidConfig, match="CTA-private Int32"):
            generation.strict_config_pair(
                helion.Config.from_dict(
                    dict(default) | {ATOMIC_AGGREGATION_KEY: invalid}
                )
            )
    _, config = generation.strict_config_pair(
        helion.Config.from_dict(dict(default) | {ATOMIC_AGGREGATION_KEY: True})
    )
    assert "match_sync" in bound.to_code(config)
    assert "match_sync" not in bound.to_code(default)
    random.seed(12831)
    assert {
        generation.random_config().get(ATOMIC_AGGREGATION_KEY, False) for _ in range(24)
    } == {False, True}
    plain = _bind()
    with pytest.raises(exc.InvalidConfig, match="CTA-private Int32"):
        plain.to_code(
            helion.Config.from_dict(
                dict(plain.config_spec.default_config())
                | {ATOMIC_AGGREGATION_KEY: True}
            )
        )


@pytest.mark.parametrize(
    "strategy",
    [
        InitialPopulationStrategy.FROM_RANDOM,
        InitialPopulationStrategy.FROM_BEST_AVAILABLE,
    ],
)
def test_atomic_aggregation_complete_prior_prefix_rng_seeds_and_overrides(strategy):
    with patch(
        "helion._compiler.autotuner_heuristics.register_fragment_atomic_aggregation_coverage"
    ):
        old_bound = _atomic_aggregation_bind()
    bound = _atomic_aggregation_bind()
    old, new = (
        make_search(b.config_spec, count=20, strategy=strategy)
        for b in (old_bound, bound)
    )
    random.seed(72391)
    before = [
        old.config_gen.unflatten(row) for row in old._generate_initial_population_flat()
    ]
    rng = random.getstate()
    random.seed(72391)
    after = [
        new.config_gen.unflatten(row) for row in new._generate_initial_population_flat()
    ]
    assert random.getstate() == rng
    assert after[: len(before)] == before and len(after) == len(before) + 1
    assert after[-1][ATOMIC_AGGREGATION_KEY] is True
    assert bound.config_spec.default_config() == old_bound.config_spec.default_config()
    assert (
        bound.config_spec.compiler_seed_configs
        == old_bound.config_spec.compiler_seed_configs
    )
    for disabled, overrides in ((True, {}), (False, {ATOMIC_AGGREGATION_KEY: False})):
        search = make_search(
            bound.config_spec, count=20, disabled=disabled, overrides=overrides
        )
        assert all(
            not search.config_gen.unflatten(row).get(ATOMIC_AGGREGATION_KEY, False)
            for row in search._generate_initial_population_flat()
        )


@pytest.mark.parametrize("mode", ["serial", "cooperative"])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_warp_scan_phase_capacity_and_default_parity(mode, reverse):
    args = (torch.ones(3, 1057), torch.ones(3, 65))
    if reverse:
        args = args[::-1]
    bound = _cpu_bind(_mixed_dtype_phase_scans, args)
    default = _config(bound, mode)
    assert bound.to_code(default) == bound.to_code(
        helion.Config.from_dict(dict(default) | {WARP_SCAN_KEY: False})
    )
    code = bound.to_code(helion.Config.from_dict(dict(default) | {WARP_SCAN_KEY: True}))
    assert "shuffle_sync_up" in code
    assert ("fragment_scan_initialized" in code) == (mode == "serial")


def test_mixed_warp_scan_all_ineligible_phases_reject():
    bound = _cpu_bind(
        _mixed_dtype_phase_scans, (torch.ones(3, 1057), torch.ones(3, 2049))
    )
    with pytest.raises(exc.InvalidConfig, match="at least one supported operation"):
        bound.to_code(_warp_scan_config(bound))


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _mixed_warp_scan_dynamic(x: torch.Tensor):
    out = torch.empty_like(x)
    for row, column in hl.tile(x.shape):
        out[row, column] = hl.cumsum(x[row, column] + 1, dim=-1)
    return out


def test_mixed_warp_scan_configured_capacity_changes_on_same_binding():
    bound = _cpu_bind(_mixed_warp_scan_dynamic, (torch.ones(3, 2053),))
    for width in (32, 1024, 2048, 256):
        config = helion.Config.from_dict(
            dict(bound.config_spec.default_config())
            | {"block_sizes": [2, width], WARP_SCAN_KEY: True}
        )
        if width > 1024:
            with pytest.raises(
                exc.InvalidConfig, match="at least one supported operation"
            ):
                bound.to_code(config)
        else:
            assert "shuffle_sync_up" in bound.to_code(config)
