from __future__ import annotations

import ast
import hashlib
import inspect
import sys
import textwrap
from types import ModuleType
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

from examples.matmul_split_k import matmul_split_k
import numpy as np
import pytest
import torch
from torch._inductor.codecache import PyCodeCache

from test.cute_population_contracts import checked_initial_population
from test.test_cute_collective_chunk_seeds import _direct_chunk_matmul
from test.test_cute_collective_chunk_seeds import _matmul_with_unrelated_knob
from test.test_cute_collective_chunk_seeds import _partitioned_matmul
from test.test_cute_fuse_mm_accumulation import _cpu_target

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.split_k_workspace import analyze_split_k_workspace
from helion._compiler.cute.split_k_workspace import workspace_schedule
from helion._compiler.cute.split_k_workspace_codegen import _reducer_source
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion.autotuner.config_fragment import PowerOfTwoFragment
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        _cpu_target(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        yield


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fp32_bias(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    partitions = hl.register_tunable("partitions", PowerOfTwoFragment(1, 128))
    chunk = helion.next_power_of_2(helion.cdiv(k, partitions))
    out = torch.zeros((m, n), dtype=torch.float32, device=a.device)
    for outer, row, column in hl.tile((k, m, n), block_size=(chunk, None, None)):
        accumulator = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(outer.begin, outer.end):
            accumulator = torch.addmm(accumulator, a[row, inner], b[inner, column])
        if outer.begin == 0:
            accumulator = accumulator + bias[column]
        hl.atomic_add(out, [row, column], accumulator)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nonzero_output(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.ones((m, n), dtype=torch.float32, device=a.device)
    for row, column, outer in hl.tile((m, n, k), block_size=(None, None, 64)):
        accumulator = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(outer.begin, outer.end):
            accumulator = torch.addmm(accumulator, a[row, inner], b[inner, column])
        hl.atomic_add(out, [row, column], accumulator)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nonzero_seed(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.zeros((m, n), dtype=torch.float32, device=a.device)
    for row, column, outer in hl.tile((m, n, k), block_size=(None, None, 64)):
        accumulator = hl.full([row, column], 1.0, dtype=torch.float32)
        for inner in hl.tile(outer.begin, outer.end):
            accumulator = torch.addmm(accumulator, a[row, inner], b[inner, column])
        hl.atomic_add(out, [row, column], accumulator)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _output_argument(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    for row, column, outer in hl.tile((m, n, k), block_size=(None, None, 64)):
        accumulator = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(outer.begin, outer.end):
            accumulator = torch.addmm(accumulator, a[row, inner], b[inner, column])
        hl.atomic_add(out, [row, column], accumulator)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _output_alias(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.zeros((m, n), dtype=torch.float32, device=a.device)
    alias = out
    for row, column, outer in hl.tile((m, n, k), block_size=(None, None, 64)):
        accumulator = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(outer.begin, outer.end):
            accumulator = torch.addmm(accumulator, a[row, inner], b[inner, column])
        hl.atomic_add(alias, [row, column], accumulator)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nonlinear_partial(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.zeros((m, n), dtype=torch.float32, device=a.device)
    for row, column, outer in hl.tile((m, n, k), block_size=(None, None, 64)):
        accumulator = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(outer.begin, outer.end):
            accumulator = torch.addmm(accumulator, a[row, inner], b[inner, column])
        if outer.begin == 0:
            accumulator = torch.relu(accumulator)
        hl.atomic_add(out, [row, column], accumulator)
    return out


def _args(
    shape: tuple[int, int, int] = (128, 2048, 128),
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k, n = shape
    return torch.empty((m, k), dtype=dtype), torch.empty((k, n), dtype=dtype)


def _bind(kernel: Kernel[Any], args: tuple[object, ...]) -> BoundKernel:
    return kernel._bind_isolated(args)


def _config(**overrides: object) -> helion.Config:
    return helion.Config.from_dict(
        {
            "block_sizes": [128, 128, 64],
            "partitions": 4,
            "cute_split_k_workspace": True,
            "cute_split_k_stages": 2,
        }
        | overrides
    )


def _proof(bound: BoundKernel):
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        return analyze_split_k_workspace(bound.env, bound.host_function.device_ir)


def _embedded_sources(code: str) -> list[str]:
    return [
        node.value.args[0].value
        for node in ast.parse(code).body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "PyCodeCache.load"
        and isinstance(node.value.args[0], ast.Constant)
        and isinstance(node.value.args[0].value, str)
    ]


@pytest.mark.parametrize(
    "kernel,parameter,value,expected",
    (
        (_partitioned_matmul, "partitions", 4, (4, 512)),
        (_direct_chunk_matmul, "tile_extent", 128, (16, 128)),
    ),
)
def test_partition_count_follows_the_original_config_expression(
    kernel, parameter, value, expected
) -> None:
    args = _args()
    bound = _bind(kernel, args)
    proof = _proof(bound)
    assert proof is not None
    config = helion.Config(block_sizes=[128, 128, 64], **{parameter: value})
    with bound.env:
        schedule = workspace_schedule(proof, bound.env, config, capacity_bytes=232448)
    assert schedule is not None
    assert (schedule.partitions, schedule.chunk) == expected


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_fp32_destination_with_permuted_root_and_bias_is_proved(
    dtype: torch.dtype,
) -> None:
    args = (*_args(dtype=dtype), torch.empty((128,), dtype=dtype))
    bound = _bind(_fp32_bias, args)
    proof = _proof(bound)
    assert proof is not None and proof.output_dtype is torch.float32
    assert proof.partition_block_id == 0 and proof.bias is not None
    assert proof.bias.dimensions == (1,) and proof.bias.strides == (1,)
    source = bound.to_code(_config())
    producer, reducer = _embedded_sources(source)
    assert "CtaGroup.TWO" in producer
    assert "num_stages=2" in producer
    assert "cutlass.Float32" in reducer and "cutlass.Float16" not in reducer
    assert "if partition == cutlass.Int32(0):" in reducer
    assert reducer.index(
        "loaded[element] = loaded[element] + bias_value"
    ) < reducer.index("values.store(values.load() + loaded.load())")


@pytest.mark.parametrize("with_bias", (False, True))
def test_original_example_emits_only_two_source_hashed_launches(
    with_bias: bool,
) -> None:
    args = _args((128, 32768, 256))
    bias = torch.empty((256,), dtype=torch.float16)
    epilogue = (
        (lambda acc, tile: acc + bias[tile[1]])
        if with_bias
        else (lambda acc, tile: acc)
    )
    kernel = helion.kernel(
        matmul_split_k.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = _bind(kernel, (*args, epilogue))
    source = bound.to_code(
        helion.Config(
            block_sizes=[128, 128, 64],
            split_k=32,
            cute_split_k_workspace=True,
            cute_split_k_stages=12,
        )
    )
    producer, reducer = _embedded_sources(source)
    assert "CtaGroup.TWO" in producer and "num_stages=12" in producer
    assert "atomic_add" not in producer + reducer
    host = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "matmul_split_k"
    )
    assert "torch.zeros" not in ast.unparse(host)
    assert "out.to(" not in ast.unparse(host)
    assert ast.unparse(host).count("torch.empty(") == 2
    assert "(32, 128, 1024)" in ast.unparse(host)
    assert "(1024, 32768, 1)" in ast.unparse(host)
    assert "(262144, 256, 1)" in ast.unparse(host)
    for stage in (producer, reducer):
        assert hashlib.sha256(stage.encode()).hexdigest() in source
    assert "_helion_cute_kernels" in source


@pytest.mark.parametrize("helper", ("_helion_split_k_native", "_helion_split_k_reduce"))
@pytest.mark.parametrize("collision", ("posonly", "posonly_twice", "function"))
def test_generated_module_name_does_not_shadow_host_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, helper: str, collision: str
) -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(_partitioned_matmul.fn)))
    (function,) = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    function.decorator_list = []
    if collision == "function":
        function.name = helper
    else:
        names = [helper, helper + "_"] if collision == "posonly_twice" else [helper]
        function.args.posonlyargs = [
            *function.args.args,
            *[ast.arg(arg=name) for name in names],
        ]
        function.args.args = []
        function.args.defaults.extend(ast.Constant(None) for name in names)
    ast.fix_missing_locations(tree)
    path = tmp_path / "namespace_fixture.py"
    path.write_text("from __future__ import annotations\n" + ast.unparse(tree) + "\n")
    fixture = ModuleType(f"_workspace_host_fixture_{helper}_{collision}")
    fixture.__dict__.update(_partitioned_matmul.fn.__globals__)
    fixture.__name__ = f"_workspace_host_fixture_{helper}_{collision}"
    monkeypatch.setitem(sys.modules, fixture.__name__, fixture)
    exec(compile(path.read_text(), str(path), "exec"), fixture.__dict__)
    kernel = helion.kernel(
        fixture.__dict__[function.name],
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    args = _args()
    bound = _bind(kernel, args)
    source = bound.to_code(_config())
    assert _embedded_sources(source) == _embedded_sources(
        _bind(_partitioned_matmul, args).to_code(_config())
    )
    module = PyCodeCache.load(source)
    host = module.__dict__[function.name]
    calls = []
    outputs = [
        host(*args, _launcher=lambda *values, **kwargs: calls.append((values, kwargs)))
        for _ in range(2)
    ]
    assert len(calls) == 4
    assert len(host._helion_cute_kernels) == 2
    assert outputs[0].data_ptr() != outputs[1].data_ptr()
    assert outputs[0].dtype is outputs[1].dtype is torch.float16
    assert calls[0][0][4] is calls[1][0][2]
    assert calls[2][0][4] is calls[3][0][2]
    assert calls[1][0][3] is outputs[0] and calls[3][0][3] is outputs[1]


def test_host_launches_allocate_fresh_fp32_partials_and_half_results() -> None:
    args = _args()
    bound = _bind(_partitioned_matmul, args)
    module = PyCodeCache.load(bound.to_code(_config()))
    calls = []

    def capture(kernel, grid, *arguments, **kwargs):
        calls.append((kernel, grid, arguments, kwargs))

    outputs = [
        module._partitioned_matmul(*args, _launcher=capture) for index in range(2)
    ]
    assert len(calls) == 4
    assert len(module._partitioned_matmul._helion_cute_kernels) == 2
    assert outputs[0].dtype is outputs[1].dtype is torch.float16
    first, second = calls[0][2][2], calls[2][2][2]
    assert first.shape == second.shape == (4, 128, 128)
    assert first.dtype is second.dtype is torch.float32
    assert first.data_ptr() != second.data_ptr()
    assert calls[1][2][0] is first and calls[3][2][0] is second
    assert calls[1][2][1] is outputs[0] and calls[3][2][1] is outputs[1]
    for call in (calls[0], calls[2]):
        assert call[3]["block"] == (32, 6, 1)
    for call in (calls[1], calls[3]):
        assert call[3]["block"] == (128, 1, 1)


@pytest.mark.parametrize(
    "shape", ((127, 2048, 128), (128, 2056, 128), (128, 2048, 127), (128, 0, 128))
)
def test_tails_and_empty_k_retain_the_atomic_fallback(
    shape: tuple[int, int, int],
) -> None:
    args = _args(shape)
    bound = _bind(_partitioned_matmul, args)
    assert _proof(bound) is None
    assert not bound.config_spec.cute_split_k_workspace_available
    source = bound.to_code(helion.Config(block_sizes=[32, 32, 32], partitions=4))
    assert "PyCodeCache.load" not in source
    assert "torch.float32" in source and "to(torch.float16)" in source


@pytest.mark.parametrize(
    "kernel",
    (
        _nonzero_output,
        _nonzero_seed,
        _output_alias,
        _nonlinear_partial,
        _matmul_with_unrelated_knob,
    ),
)
def test_unproved_effects_and_non_partitioned_matmul_keep_the_original_path(
    kernel,
) -> None:
    args = _args()
    bound = _bind(kernel, args)
    assert _proof(bound) is None
    assert not bound.config_spec.cute_split_k_workspace_available
    assert not any(
        seed.get("cute_split_k_workspace")
        for seed in bound.config_spec.compiler_seed_configs
    )


def test_user_output_atomic_semantics_are_not_replaced() -> None:
    args = (*_args(), torch.empty((128, 128), dtype=torch.float32))
    bound = _bind(_output_argument, args)
    assert _proof(bound) is None
    with pytest.raises(helion.exc.InvalidConfig, match="private FP32"):
        bound.to_code(
            helion.Config(block_sizes=[128, 128, 64], cute_split_k_workspace=True)
        )


def test_bf16_requested_atomics_are_not_promoted_by_this_optimization() -> None:
    args = _args(dtype=torch.bfloat16)
    bound = _bind(_partitioned_matmul, args)
    assert not bound.env.cute_half_atomic_output_promotions
    assert _proof(bound) is None


@pytest.mark.parametrize("shape", ((128, 192, 128), (256, 4096, 256)))
def test_invalid_partition_view_and_smem_fail_closed(
    shape: tuple[int, int, int],
) -> None:
    args = _args(shape)
    bound = _bind(_partitioned_matmul, args)
    proof = _proof(bound)
    assert proof is not None
    # 192 split in two gives a 128-element block and an incomplete final view.
    config = _config(partitions=2, cute_split_k_stages=16)
    with bound.env:
        assert (
            workspace_schedule(proof, bound.env, config, capacity_bytes=232448) is None
        )
    with pytest.raises(helion.exc.BackendUnsupported, match="full aligned partitions"):
        bound.to_code(config)


@pytest.mark.parametrize("shift", (1, 2, 4))
def test_unaligned_base_cannot_borrow_the_internal_producer_alignment(
    shift: int,
) -> None:
    a, b = _args()
    shifted = torch.empty(a.numel() + shift, dtype=a.dtype)[shift:].view(a.shape)
    bound = _bind(_partitioned_matmul, (shifted, b))
    assert _proof(bound) is not None
    with pytest.raises(helion.exc.BackendUnsupported, match="TensorMap alignment"):
        bound.to_code(_config())
    assert not bound.env.cute_proven_tma_inputs


def test_child_compilation_restores_the_parent_environment_on_failure() -> None:
    args = _args()
    bound = _bind(_partitioned_matmul, args)
    with (
        patch(
            "helion._compiler.cute.split_k_workspace_codegen.analyze_pipeline_smem_facts",
            return_value=None,
        ),
        pytest.raises(helion.exc.BackendUnsupported, match="SMEM proof"),
    ):
        bound.to_code(_config())
    assert not CompileEnvironment.has_current()
    assert not bound.env.cute_proven_tma_inputs
    assert _proof(bound) is not None
    assert len(_embedded_sources(bound.to_code(_config()))) == 2


def test_partition_zero_bias_rounds_before_any_partial_reduction() -> None:
    partials = np.array([1e8, 1.0, 0.0, 0.0], dtype=np.float32)
    bias = np.float32(-1e8)
    expected = partials.copy()
    expected[0] = np.float32(expected[0] + bias)
    assert expected.sum(dtype=np.float32) == 1.0
    assert np.float32(partials.sum(dtype=np.float32) + bias) == 0.0
    args = (*_args(), torch.empty((128,), dtype=torch.float16))
    bound = _bind(_fp32_bias, args)
    proof = _proof(bound)
    assert proof is not None
    with bound.env:
        schedule = workspace_schedule(
            proof, bound.env, _config(), capacity_bytes=232448
        )
    assert schedule is not None
    source = _reducer_source(proof, schedule)
    assert source.index("if partition == cutlass.Int32(0)") < source.index(
        "values.store(values.load() + loaded.load())"
    )
    assert source.count(".to(cutlass.Float32)") == 1


def test_real_full_initial_population_reaches_deep_workspace_and_atomic_choices() -> (
    None
):
    args = _args((128, 32768, 256))
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        autotune_random_seed=2026091347,
    )

    def epilogue(acc, tile):
        return acc

    arguments = (*args, epilogue)
    bound = _bind(kernel, arguments)
    profile = get_effort_profile("full").pattern_search
    assert profile is not None
    with bound.env:
        search = PatternSearch(
            bound,
            arguments,
            initial_population=profile.initial_population,
            initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
        )
        flat = checked_initial_population(search)
        configs = [search.config_gen.unflatten(row) for row in flat]
        assert [search.config_gen.flatten(config) for config in configs] == flat
    # The 100-row base and the separate coverage suffix were checked above.
    assert not configs[0].get("cute_split_k_workspace", False)
    assert any(
        config.get("cute_collective_mma") and not config.get("cute_split_k_workspace")
        for config in configs
    )
    choices = [config for config in configs if config.get("cute_split_k_workspace")]
    assert choices
    assert any(
        config.block_sizes == [128, 128, 64]
        and config["split_k"] == 32
        and config["cute_split_k_stages"] == 12
        for config in choices
    )
    assert {config["cute_split_k_stages"] for config in choices} >= {2, 12, 16}
    for config in choices:
        assert len(_embedded_sources(bound.to_code(config))) == 2
