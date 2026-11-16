from __future__ import annotations

import ast
from copy import deepcopy
import inspect
import sys
import textwrap
from types import ModuleType
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from test.cute_population_contracts import checked_initial_population
from test.test_cute_split_k_workspace import _embedded_sources
from test.test_cute_split_k_workspace import _fp32_bias
from test.test_cute_split_k_workspace import _nonlinear_partial
from test.test_cute_split_k_workspace import _nonzero_output
from test.test_cute_split_k_workspace import _output_alias
from test.test_cute_split_k_workspace import _output_argument

import helion
from helion._compiler.cute import target_facts
from helion._compiler.cute.autotune.heuristics.cute_split_k_cluster import (
    cluster_carrier,
)
from helion._compiler.cute.split_k_cluster import CLUSTER_K_SCHEDULE
from helion._compiler.cute.split_k_cluster_config import CLUSTER8_K4
from helion._compiler.cute.split_k_cluster_config import SCHEDULE_KEY
from helion._compiler.cute.state import cute_env_state
from helion._testing import default_cute_mma_support
from helion._testing import import_example
from helion._testing.cute_codegen import check_config_compile_options
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch

matmul_split_k = import_example("matmul_split_k").matmul_split_k

# ``tcgen05_cpu_target``: the B200 model with the default (tcgen05-on) MMA
# support record and the full 227 KiB opt-in SMEM as the AB budget.
pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.cpu_render(
        mma_support=default_cute_mma_support(), smem_budget_bytes=232448
    ),
]

if TYPE_CHECKING:
    from pathlib import Path


def _bind(shape=(32, 4096, 64), *, bias=False, disable_heuristics=False, **settings):
    # The cluster-K schedule is validated on fp32 (promoted) destinations; the
    # float16 output accumulates in place by default now.
    settings.setdefault("cute_promote_half_atomic_outputs", True)
    m, k, n = shape
    a = torch.empty((m, k), dtype=torch.float16)
    b = torch.empty((k, n), dtype=torch.float16)
    bias_tensor = torch.empty((n,), dtype=torch.float16)
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        autotune_random_seed=917,
        disable_autotuner_heuristics=disable_heuristics,
        **settings,
    )
    args = (a, b, lambda acc, tile: acc + bias_tensor[tile[1]]) if bias else (a, b)
    return kernel._bind_isolated(args), args


def _carrier(bound):
    with bound.env, bound.host_function:
        return cluster_carrier(bound.env, bound.host_function.device_ir)


@pytest.mark.parametrize("shape", ((32, 4096, 64), (64, 16384, 128), (128, 32768, 256)))
@pytest.mark.parametrize("bias", (False, True))
def test_original_family_normal_set_config_preserves_allocating_host(shape, bias):
    bound, args = _bind(shape, bias=bias)
    config = _carrier(bound)
    assert config is not None and config["split_k"] == 8
    bound.set_config(config)
    assert bound._run is not None
    host = cast("Any", bound._run)
    metadata = host._helion_cute_split_k_schedule
    assert metadata["carrier_tile"] == (16, 16, 128)
    assert metadata["tile"] == (16, 8, 128)
    assert metadata["chunk_k"] == shape[1] // 8
    assert metadata["waves"] == shape[1] // 4096
    assert metadata["bias_partition"] == (0 if bias else None)
    calls = []

    def capture(kernel, grid, *values, **kwargs):
        calls.append((kernel, grid, values, kwargs))

    outputs = [host(*args, _launcher=capture) for _ in range(2)]
    assert outputs[0].data_ptr() != outputs[1].data_ptr()
    assert all(output.shape == (shape[0], shape[2]) for output in outputs)
    assert all(output.dtype is torch.float16 for output in outputs)
    assert len(calls) == 2
    for index, (kernel, grid, values, kwargs) in enumerate(calls):
        assert kernel._helion_cute_cluster_shape == (8, 1, 1)
        assert grid == ((shape[0] // 16) * (shape[2] // 8) * 8,)
        assert kwargs == {"block": (32, 4, 1)}
        assert values[0] is args[0] and values[1] is args[1]
        assert values[2] is outputs[index]
        assert len(values) == (4 if bias else 3)
    source = _embedded_sources(bound.to_code(config))[0]
    assert "atomic" not in source and "torch.zeros" not in source
    if shape == (32, 4096, 64) and not bias:
        check_config_compile_options(bound, config)
    assert "cluster_arrive_relaxed()" in source
    assert source.count("cluster_wait()") == 2
    if bias:
        assert source.index("value + cutlass.Float32((bias.iterator") < source.index(
            "cute.arch.cluster_arrive()"
        )
    tree = ast.parse(source)
    kernel_ast = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    waves = [
        node
        for node in kernel_ast.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "wave"
    ]
    assert len(waves) == int(shape[1] > 4096)
    if waves:
        assert ast.unparse(waves[0].body[-1]) == "cute.arch.sync_threads()"


@pytest.mark.parametrize("bias", (False, True))
@pytest.mark.parametrize("disabled", (False, True))
def test_real_full_population_keeps_explicit_cluster_witness(bias, disabled):
    bound, args = _bind(bias=bias, disable_heuristics=disabled)
    profile = get_effort_profile("full").pattern_search
    assert profile is not None
    with bound.env:
        search = PatternSearch(
            bound,
            args,
            initial_population=profile.initial_population,
            initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
        )
        flat = checked_initial_population(search)
        configs = [search.config_gen.unflatten(row) for row in flat]
    if disabled:
        assert not bound.config_spec.compiler_seed_configs
        assert "compiler_coverage_outcomes" not in vars(search)
        assert any(
            group.key == SCHEDULE_KEY
            for group in bound.config_spec.compiler_coverage_groups
        )
        requested = _carrier(bound)
        assert requested is not None
        with bound.env:
            flat, effective = search.config_gen.strict_config_pair(requested)
        assert effective[SCHEDULE_KEY] == CLUSTER8_K4
        assert effective.block_sizes == [16, 16, 128]
        return
    choices = [config for config in configs if config.get(SCHEDULE_KEY) == CLUSTER8_K4]
    assert choices
    assert any(
        row.outcome in ("added", "already_present")
        and row.mechanism == "cute.split_k_cluster"
        for row in search.compiler_coverage_outcomes
    )
    for config in choices:
        assert config.block_sizes == [16, 16, 128]
        assert config["split_k"] == 8
        assert "tile" in bound.to_code(config)


@pytest.mark.parametrize(
    "change",
    (
        {"split_k": 32},
        {"block_sizes": [16, 8, 128]},
        {"block_sizes": [32, 16, 128]},
        {"num_threads": [4, 8, 1]},
        {"cute_split_k_workspace": True},
        {"cute_collective_mma": True},
        {SCHEDULE_KEY: True},
    ),
)
def test_explicit_incompatible_request_rejects(change):
    bound, args = _bind()
    config = deepcopy(_carrier(bound))
    assert config is not None
    config.config.update(change)
    with pytest.raises((helion.exc.InvalidConfig, helion.exc.BackendUnsupported)):
        bound.to_code(config)


@pytest.mark.parametrize(
    "shape", ((31, 4096, 64), (32, 4096, 63), (32, 4095, 64), (32, 2048, 64))
)
def test_partial_tiles_and_incomplete_waves_keep_legacy(shape):
    bound, args = _bind(shape)
    assert bound.config_spec.cute_split_k_cluster_facts is None
    assert _carrier(bound) is None


@pytest.mark.parametrize(
    "kernel", (_nonlinear_partial, _nonzero_output, _output_alias, _output_argument)
)
def test_observable_or_non_affine_reductions_not_admitted(kernel):
    args = (
        torch.empty((32, 4096), dtype=torch.float16),
        torch.empty((4096, 64), dtype=torch.float16),
    )
    if kernel is _output_argument:
        args = (*args, torch.empty((32, 64), dtype=torch.float32))
    bound = kernel._bind_isolated(args)
    assert bound.config_spec.cute_split_k_cluster_facts is None


def test_shared_capacity_covers_all_aligned_allocations_and_reserved_bytes():
    with patch.object(
        target_facts,
        "per_cta_smem_capacity_bytes",
        return_value=CLUSTER_K_SCHEDULE.shared_upper_bound - 1,
    ):
        bound, args = _bind()
    assert bound.config_spec.cute_split_k_cluster_facts is None


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_permuted_root_bias_and_fp32_output(dtype):
    args = (
        torch.empty((48, 8192), dtype=dtype),
        torch.empty((8192, 80), dtype=dtype),
        torch.empty((80,), dtype=dtype),
    )
    bound = _fp32_bias._bind_isolated(args)
    config = _carrier(bound)
    assert config is not None and config["partitions"] == 8
    source = _embedded_sources(bound.to_code(config))[0]
    assert ".store(cutlass.Float32(value))" in source
    assert "(tile_n + n) * 1" in source


@pytest.mark.parametrize("shift", (1, 2, 4))
def test_unaligned_input_rejects_before_default_launcher(shift):
    a = torch.empty(32 * 4096 + shift, dtype=torch.float16)[shift:].view(32, 4096)
    b = torch.empty((4096, 64), dtype=torch.float16)
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_promote_half_atomic_outputs=True,
    )
    bound = kernel._bind_isolated((a, b))
    with pytest.raises(helion.exc.BackendUnsupported, match="aligned inputs"):
        bound.to_code(_carrier(bound))


def test_positive_padded_input_strides_and_non_original_extents():
    a = torch.empty_strided((48, 8192), (8200, 1), dtype=torch.float16)
    b = torch.empty_strided((8192, 80), (88, 1), dtype=torch.float16)
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_promote_half_atomic_outputs=True,
    )
    bound = kernel._bind_isolated((a, b))
    config = _carrier(bound)
    assert config is not None
    bound.set_config(config)
    calls = []
    assert bound._run is not None
    result = bound._run(
        a, b, _launcher=lambda *args, **kwargs: calls.append((args, kwargs))
    )
    assert result.shape == (48, 80)
    assert calls[0][0][1] == (240,)
    assert calls[0][0][2] is a and calls[0][0][3] is b
    assert calls[0][0][4] is result


def test_legacy_override_suppresses_cluster_coverage_and_seed():
    bound, args = _bind(autotune_config_overrides={SCHEDULE_KEY: "legacy"})
    profile = get_effort_profile("full").pattern_search
    assert profile is not None
    with bound.env:
        search = PatternSearch(
            bound,
            args,
            initial_population=profile.initial_population,
            initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
        )
        rows = checked_initial_population(search)
        configs = [search.config_gen.unflatten(row) for row in rows]
    assert all(config.get(SCHEDULE_KEY, "legacy") == "legacy" for config in configs)


def test_legacy_explicit_value_keeps_original_source_exact():
    bound, args = _bind()
    config = helion.Config(
        block_sizes=[32, 32, 128], num_threads=[4, 32, 1], split_k=32
    )
    source = bound.to_code(config)
    explicit = deepcopy(config)
    explicit.config[SCHEDULE_KEY] = "legacy"
    assert bound.to_code(explicit) == source
    assert "_helion_cluster_module" not in source


def test_external_helper_and_physical_metadata_participate_in_cache_identity():
    bound, args = _bind()
    config = _carrier(bound)
    source = bound.to_code(config)
    target = "helion._compiler.cute.split_k_cluster_codegen.wrapper_source_dependencies"
    with patch(target, return_value=(("cluster_helpers.py", "changed"),)):
        changed = bound.to_code(config)
    assert source != changed
    assert _embedded_sources(source) == _embedded_sources(changed)
    with patch(target, return_value=None):
        missing = bound.to_code(config)
    assert "._helion_cute_source_hash = None" in missing


@pytest.mark.parametrize("value", (None, 0, False, -8, 8.0))
def test_invalid_or_missing_partition_tunable_is_a_config_rejection(value):
    bound, args = _bind()
    config = _carrier(bound)
    assert config is not None
    if value is None:
        config.config.pop("split_k")
    else:
        config.config["split_k"] = value
    with pytest.raises(helion.exc.InvalidConfig, match="eight full partitions"):
        bound.to_code(config)


@pytest.mark.parametrize("operand", ("lhs", "rhs", "bias"))
@pytest.mark.parametrize("overflow", (False, True))
def test_padded_element_offset_int32_boundary(operand, overflow):
    # Fake storage avoids allocating the multi-GiB address spans. The ordinary
    # binder still sees the actual strides; no typed proof values are patched.
    m, k, n = 32, 4096, 64
    shape = (m, k) if operand == "lhs" else (k, n) if operand == "rhs" else (n,)
    steps = shape[0] - 1
    tail = shape[1] - 1 if len(shape) == 2 else 0
    alignment = 8 if len(shape) == 2 else 1
    last_valid_stride = (((1 << 31) - 1 - tail) // steps // alignment) * alignment
    stride = last_valid_stride + (alignment if overflow else 0)
    maximum_offset = steps * stride + tail
    assert (maximum_offset >= 1 << 31) is overflow
    with FakeTensorMode():
        a = torch.empty_strided(
            (m, k), (stride if operand == "lhs" else k, 1), dtype=torch.float16
        )
        b = torch.empty_strided(
            (k, n), (stride if operand == "rhs" else n, 1), dtype=torch.float16
        )
        bias = torch.empty_strided(
            (n,), (stride if operand == "bias" else 1,), dtype=torch.float16
        )
        kernel = helion.kernel(
            matmul_split_k.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
            cute_promote_half_atomic_outputs=True,
        )
        args = (a, b, lambda acc, tile: acc + bias[tile[1]])
        bound = kernel._bind_isolated(args)
        facts = bound.config_spec.cute_split_k_cluster_facts
        carrier = _carrier(bound)
        if overflow:
            assert facts is None and carrier is None
            assert all(
                seed.get(SCHEDULE_KEY) != CLUSTER8_K4
                for seed in bound.config_spec.compiler_seed_configs
            )
            explicit = helion.Config(
                block_sizes=[16, 16, 128],
                num_threads=[4, 8, 4],
                split_k=8,
            )
            explicit.config[SCHEDULE_KEY] = CLUSTER8_K4
            with pytest.raises(helion.exc.InvalidConfig, match="proved domain"):
                bound.to_code(explicit)
        else:
            assert facts is not None and carrier is not None
            assert facts.valid_config(bound.config_spec, carrier)


@pytest.mark.parametrize("collision", ("posonly", "posonly_twice", "function"))
def test_generated_module_name_does_not_shadow_host_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, collision
):
    tree = ast.parse(textwrap.dedent(inspect.getsource(matmul_split_k.fn)))
    (function,) = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    function.decorator_list = []
    if collision == "function":
        function.name = "_helion_cluster_module"
    else:
        names = ["_helion_cluster_module"]
        if collision == "posonly_twice":
            names.append("_helion_cluster_module_")
        function.args.posonlyargs = [
            *function.args.args,
            *[ast.arg(arg=name) for name in names],
        ]
        function.args.args = []
        function.args.defaults.extend(ast.Constant(None) for name in names)
    ast.fix_missing_locations(tree)
    path = tmp_path / "namespace_fixture.py"
    path.write_text("from __future__ import annotations\n" + ast.unparse(tree) + "\n")
    module = ModuleType(f"_cluster_host_fixture_{collision}")
    module.__dict__.update(matmul_split_k.fn.__globals__)
    module.__name__ = f"_cluster_host_fixture_{collision}"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = module.__dict__
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    kernel = helion.kernel(
        namespace[function.name],
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_promote_half_atomic_outputs=True,
    )
    args = (
        torch.empty((32, 4096), dtype=torch.float16),
        torch.empty((4096, 64), dtype=torch.float16),
    )
    bound = kernel._bind_isolated(args)
    config = _carrier(bound)
    assert config is not None
    source = bound.to_code(config)
    normal, _ = _bind()
    assert _embedded_sources(source) == _embedded_sources(
        normal.to_code(_carrier(normal))
    )
    bound.set_config(config)
    assert bound._run is not None
    calls = []
    outputs = [
        bound._run(
            *args, _launcher=lambda *values, **kwargs: calls.append((values, kwargs))
        )
        for _ in range(2)
    ]
    assert outputs[0].data_ptr() != outputs[1].data_ptr()
    assert len(calls) == 2
    for index, (values, kwargs) in enumerate(calls):
        assert values[0]._helion_cute_cluster_shape == (8, 1, 1)
        assert values[1] == (128,)
        assert (
            values[2] is args[0]
            and values[3] is args[1]
            and values[4] is outputs[index]
        )
        assert kwargs == {"block": (32, 4, 1)}


@pytest.mark.parametrize("promote", (False, True))
def test_half_output_cluster_facts_follow_the_promotion_setting(promote: bool) -> None:
    """Cluster-K is validated on fp32 destinations: with the default
    ``cute_promote_half_atomic_outputs=False`` matmul_split_k's float16
    output accumulates in place and the schedule declines; with the
    promotion on (the fp32 twin) the facts and the carrier exist."""
    bound, _args = _bind(cute_promote_half_atomic_outputs=promote)
    assert bool(cute_env_state(bound.env).half_atomic_output_promotions) == promote
    assert (bound.config_spec.cute_split_k_cluster_facts is not None) == promote
    assert (_carrier(bound) is not None) == promote


# GPU numerics of the cluster split-K schedule: the finalizer warps execute
# and replay under a CUDA graph (sm_90+).


CUDA_DEVICE = "cuda"


def _check_cluster(
    shape: tuple[int, int, int],
    *,
    bias: bool,
    finalizer_warps: int,
    fp32_output: bool = False,
) -> None:
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Cluster shared memory requires SM90 or newer")
    m, k, n = shape
    dtype = torch.bfloat16 if fp32_output else torch.float16
    device = torch.device(CUDA_DEVICE)
    generator = torch.Generator(device=device).manual_seed(7021)
    # Offset by a complete vector to exercise aligned, padded input row strides.
    padding = 8 if k > 4096 or fp32_output else 0
    a = torch.empty((m, k + padding), dtype=dtype, device=device)[:, padding:]
    b = torch.empty((k, n + padding), dtype=dtype, device=device)[:, padding:]
    bias_tensor = torch.empty((n * 3,), dtype=dtype, device=device)[::3]

    def refresh() -> None:
        a.normal_(std=k**-0.5, generator=generator)
        b.normal_(generator=generator)
        bias_tensor.normal_(generator=generator)

    refresh()
    kernel = helion.kernel(
        _fp32_bias.fn if fp32_output else matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_promote_half_atomic_outputs=True,
        cute_full_slice_matmul_tiling=False,
        cute_region_fission=False,
    )
    args: tuple[object, ...]
    if fp32_output:
        args = (a, b, bias_tensor)
    elif bias:
        args = (a, b, lambda acc, tile: acc + bias_tensor[tile[1]])
    else:
        args = (a, b)
    bound = kernel.bind(args)
    with bound.env, bound.host_function:
        config = cluster_carrier(bound.env, bound.host_function.device_ir)
    assert config is not None
    config.config["cute_split_k_finalizer_warps"] = finalizer_warps
    bound.set_config(config)
    assert bound._run is not None
    compiled = cast("Any", bound._run)
    metadata = compiled._helion_cute_split_k_schedule
    assert metadata["cluster"] == (8, 1, 1)
    assert metadata.get("finalizer_warps", 1) == finalizer_warps

    def check(output: torch.Tensor) -> None:
        expected = a.double() @ b.double()
        if bias:
            expected = expected + bias_tensor.double()
        tolerance = 2e-4 if fp32_output else 2e-3
        torch.testing.assert_close(
            output,
            expected.to(torch.float32 if fp32_output else dtype),
            rtol=tolerance,
            atol=tolerance,
        )

    first = compiled(*args)
    second = compiled(*args)
    assert first.data_ptr() != second.data_ptr()
    check(first)
    check(second)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_output = compiled(*args)
    for _ in range(2):
        refresh()
        replay_output.fill_(float("nan"))
        graph.replay()
        check(replay_output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
@pytest.mark.parametrize("shape", ((32, 4096, 64), (48, 8192, 80)))
@pytest.mark.parametrize("bias", (False, True))
@pytest.mark.parametrize("finalizer_warps", (1, 4))
def test_cluster_finalizers_execute_and_replay(
    shape: tuple[int, int, int], bias: bool, finalizer_warps: int
) -> None:
    _check_cluster(shape, bias=bias, finalizer_warps=finalizer_warps)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
@pytest.mark.parametrize("finalizer_warps", (1, 4))
def test_cluster_bfloat16_fp32_output_and_strided_bias(finalizer_warps: int) -> None:
    _check_cluster(
        (48, 8192, 80), bias=True, finalizer_warps=finalizer_warps, fp32_output=True
    )
