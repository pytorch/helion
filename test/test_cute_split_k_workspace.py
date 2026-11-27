from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import inspect
import sys
import textwrap
from types import ModuleType
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch._inductor.codecache import PyCodeCache

from test.cute_population_contracts import checked_initial_population
from test.test_cute_collective_chunk_seeds import _direct_chunk_matmul
from test.test_cute_collective_chunk_seeds import _matmul_with_unrelated_knob
from test.test_cute_collective_chunk_seeds import _partitioned_matmul

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.autotune.heuristics.cute_split_k_workspace import (
    CuteSplitKWorkspaceHeuristic,
)
from helion._compiler.cute.split_k_workspace import analyze_split_k_workspace
from helion._compiler.cute.split_k_workspace import atomic_epilogue_geometry
from helion._compiler.cute.split_k_workspace import plain_store_chunk
from helion._compiler.cute.split_k_workspace import workspace_schedule
from helion._compiler.cute.split_k_workspace_codegen import REDUCER_GROUP
from helion._compiler.cute.split_k_workspace_codegen import REDUCER_THREADS
from helion._compiler.cute.split_k_workspace_codegen import _reducer_source
from helion._compiler.cute.split_k_workspace_codegen import (
    _rewrite_producer_for_atomic_epilogue,
)
from helion._compiler.cute.state import cute_env_state
from helion._testing import DEVICE
from helion._testing import default_cute_mma_support
from helion._testing import import_example
from helion._testing import patch_target_facts
from helion.autotuner.benchmarking import _launches_several_kernels
from helion.autotuner.benchmarking import _make_cudagraph_replay
from helion.autotuner.config_fragment import PowerOfTwoFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl

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

    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


@pytest.fixture(autouse=True)
def _promotion_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests exercise the schedule on the fp32-promoted float16 output;
    pin the promotion on for the module's kernels (the Setting defaults to
    accumulating in the half storage)."""
    for kernel in (
        _partitioned_matmul,
        _fp32_bias,
        _nonzero_output,
        _nonzero_seed,
        _output_argument,
        _output_alias,
        _nonlinear_partial,
        _plain_store_matmul,
        _plain_store_bias_matmul,
        _plain_store_relu_matmul,
        _plain_store_into_argument,
        _plain_store_into_out0,
        _plain_store_reading_out0,
        matmul_split_k,
    ):
        monkeypatch.setattr(
            kernel,
            "settings",
            kernel.settings.copy(cute_promote_half_atomic_outputs=True),
        )


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


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _matrix_bias(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    rows, inner_size = a.shape
    columns = b.size(1)
    chunk = hl.register_tunable("tile_extent", PowerOfTwoFragment(1, 256))
    output = torch.zeros_like(bias, dtype=torch.float32)
    for row, column, partition in hl.tile(
        (rows, columns, inner_size), block_size=(None, None, chunk)
    ):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(partition.begin, partition.end):
            acc = torch.addmm(acc, a[row, inner], b[inner, column])
        if partition.begin == 0:
            acc = acc + bias[row, column]
        hl.atomic_add(output, [row, column], acc)
    return output


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _plain_store_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, column in hl.tile((m, n)):
        accumulator = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            accumulator = torch.addmm(accumulator, a[row, inner], b[inner, column])
        out[row, column] = accumulator.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _plain_store_bias_matmul(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, column in hl.tile((m, n)):
        accumulator = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            accumulator = torch.addmm(accumulator, a[row, inner], b[inner, column])
        out[row, column] = (accumulator + bias[column]).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _plain_store_relu_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, column in hl.tile((m, n)):
        accumulator = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            accumulator = torch.addmm(accumulator, a[row, inner], b[inner, column])
        out[row, column] = torch.relu(accumulator).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _plain_store_into_argument(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    for row, column in hl.tile((m, n)):
        accumulator = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            accumulator = torch.addmm(accumulator, a[row, inner], b[inner, column])
        out[row, column] = accumulator.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _plain_store_into_out0(
    in0: torch.Tensor, in1: torch.Tensor, out0: torch.Tensor
) -> None:
    # helion.inductor's mm kernel: the output is an argument, nothing returned.
    for tile_y, tile_x in hl.tile(out0.shape):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(in0.size(1)):
            acc = torch.addmm(acc, in0[tile_y, tile_r], in1[tile_r, tile_x])
        out0[tile_y, tile_x] = acc.to(out0.dtype)


@helion.kernel(
    backend="cute",
    static_shapes=True,
    autotune_effort="none",
    split_k_reduction="output",
)
def _plain_store_into_out0_output_partials(
    in0: torch.Tensor, in1: torch.Tensor, out0: torch.Tensor
) -> None:
    # helion.inductor's mm kernel under torch's default reduced-precision
    # split-K flags (split_k_reduction="output").
    for tile_y, tile_x in hl.tile(out0.shape):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(in0.size(1)):
            acc = torch.addmm(acc, in0[tile_y, tile_r], in1[tile_r, tile_x])
        out0[tile_y, tile_x] = acc.to(out0.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _plain_store_column_bias_into_out0(
    in0: torch.Tensor, in1: torch.Tensor, bias: torch.Tensor, out0: torch.Tensor
) -> None:
    for tile_y, tile_x in hl.tile(out0.shape):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(in0.size(1)):
            acc = torch.addmm(acc, in0[tile_y, tile_r], in1[tile_r, tile_x])
        out0[tile_y, tile_x] = (acc + bias[tile_x]).to(out0.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _plain_store_reading_out0(
    in0: torch.Tensor, in1: torch.Tensor, out0: torch.Tensor
) -> None:
    for tile_y, tile_x in hl.tile(out0.shape):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(in0.size(1)):
            acc = torch.addmm(acc, in0[tile_y, tile_r], in1[tile_r, tile_x])
        out0[tile_y, tile_x] = (acc + out0[tile_y, tile_x]).to(out0.dtype)


def _major_args(
    m: int, k: int, n: int, a_major: str, b_major: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """bf16 operands of an m x k x n product: A K-major (row-major) or
    M-major (column-major), B N-major (row-major) or K-major (column-major),
    and a row-major output."""
    dtype = torch.bfloat16
    a = (
        torch.empty((m, k), dtype=dtype)
        if a_major == "k"
        else torch.empty((k, m), dtype=dtype).T
    )
    b = (
        torch.empty((k, n), dtype=dtype)
        if b_major == "n"
        else torch.empty((n, k), dtype=dtype).T
    )
    return a, b, torch.empty((m, n), dtype=dtype)


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
    # The bias joins partition zero before the partitions are summed.
    assert reducer.index(
        "loaded_0[element] = loaded_0[element] + cutlass.Float32(bias_scalar[0])"
    ) < reducer.index("converted.store((loaded_0.load()")


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
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_promote_half_atomic_outputs=True,
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
    # both kernels launch on every call: the benchmark timers read the count
    assert host._helion_cute_launches == 2
    assert _launches_several_kernels(host)
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
        # 128 x 128 outputs: 4096 four-float vectors, one a reducer thread,
        # 16 CTAs of 256 threads.
        assert call[1] == (16,)
        assert call[3]["block"] == (256, 1, 1)


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
    with pytest.raises(helion.exc.InvalidConfig, match="proved split-K reduction"):
        bound.to_code(
            helion.Config(block_sizes=[128, 128, 64], cute_split_k_workspace=True)
        )


def test_fp16_output_is_proved_without_promotion() -> None:
    # ``cute_promote_half_atomic_outputs=False`` keeps a float16 destination
    # in its requested dtype like bfloat16; the workspace schedule proves the
    # split and its reducer casts the FP32 partial sum to float16 once.
    kernel = helion.kernel(
        _partitioned_matmul.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_promote_half_atomic_outputs=False,
    )
    bound = _bind(kernel, _args())
    assert not cute_env_state(bound.env).half_atomic_output_promotions
    proof = _proof(bound)
    assert proof is not None
    assert proof.output_dtype is torch.float16
    source = bound.to_code(_config())
    producer, reducer = _embedded_sources(source)
    assert "atomic_add" not in producer + reducer
    assert "cutlass.Float16" in reducer
    host = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_partitioned_matmul"
    )
    assert "dtype=torch.float16" in ast.unparse(host)
    assert "torch.float32" not in ast.unparse(host).split("_helion_split_k_partials")[0]
    assert "out.to(" not in ast.unparse(host)


def test_bf16_output_is_proved_without_promotion() -> None:
    # The atomic lowering keeps a requested bfloat16 destination (no FP32
    # promotion); the workspace schedule still proves the split and its
    # reducer casts the FP32 partial sum to bfloat16 once.
    args = _args(dtype=torch.bfloat16)
    bound = _bind(_partitioned_matmul, args)
    assert not cute_env_state(bound.env).half_atomic_output_promotions
    proof = _proof(bound)
    assert proof is not None
    assert proof.output_dtype is torch.bfloat16
    source = bound.to_code(_config())
    producer, reducer = _embedded_sources(source)
    assert "atomic_add" not in producer + reducer
    assert "cutlass.BFloat16" in reducer
    host = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_partitioned_matmul"
    )
    assert "dtype=torch.bfloat16" in ast.unparse(host)
    assert "out.to(" not in ast.unparse(host)


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
    assert not cute_env_state(bound.env).proven_tma_inputs


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
    assert not cute_env_state(bound.env).proven_tma_inputs
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
    assert source.index("bias_scalar = ") < source.index(
        "converted.store((loaded_0.load()"
    )
    # One cast: the fp32 sum is rounded once.
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
        cute_promote_half_atomic_outputs=True,
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


def test_plain_store_matmul_is_proved_with_its_partition_knob() -> None:
    bound = _bind(_plain_store_matmul, _args((128, 16384, 128)))
    proof = _proof(bound)
    assert proof is not None
    assert proof.store and proof.bias is None
    assert proof.output_dtype is torch.float16
    spec = bound.config_spec
    assert spec.cute_split_k_workspace_available
    # 16384 = 256 K tiles of 64: every search choice keeps at least eight
    # whole tiles per partition (three partitions of 86, the third running
    # past K: ``plain_store_chunk``).
    assert spec.cute_split_k_partition_choices == (1, 2, 3, 4, 8, 16, 32)
    assert any(
        seed.get("cute_split_k_workspace")
        and seed.get("cute_split_k_partitions", 1) > 1
        for seed in spec.compiler_seed_configs
    )
    config = helion.Config(
        block_sizes=[128, 128, 64],
        cute_split_k_workspace=True,
        cute_split_k_partitions=4,
        cute_split_k_stages=2,
    )
    source = bound.to_code(config)
    producer, reducer = _embedded_sources(source)
    assert "atomic_add" not in producer + reducer
    assert "cutlass.Float16" in reducer
    host = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_plain_store_matmul"
    )
    unparsed = ast.unparse(host)
    # The host keeps a fresh fp16 allocation the reducer fills; partials are
    # (partitions, m, n) fp32.
    assert unparsed.count("torch.empty(") == 2
    assert "dtype=torch.float16" in unparsed
    assert "out.to(" not in unparsed
    assert "(4, 128, 128)" in unparsed
    # One partition is the plain kernel: strict normalization rejects the
    # schedule and the search's repair pass drops it.
    plain = helion.Config(
        block_sizes=[128, 128, 64],
        cute_split_k_workspace=True,
        cute_split_k_partitions=1,
    )
    with bound.env:
        with pytest.raises(helion.exc.InvalidConfig, match="partitions > 1"):
            spec.normalize(plain)
        plain_normalized = helion.Config(**plain.config)
        spec.normalize(plain_normalized, _fix_invalid=True)
    assert plain_normalized.config["cute_split_k_workspace"] is False
    # The workspace launch supersedes the in-kernel split-K knob.
    both = helion.Config(
        block_sizes=[128, 128, 64],
        cute_split_k_workspace=True,
        cute_split_k_partitions=4,
        tcgen05_split_k=4,
    )
    with bound.env:
        spec.normalize(both, _fix_invalid=True)
    assert both.config["cute_split_k_workspace"] is True
    assert both.config["tcgen05_split_k"] == 1


def test_plain_store_bias_is_reduced_on_partition_zero() -> None:
    args = (*_args((128, 16384, 128)), torch.empty((128,), dtype=torch.float16))
    bound = _bind(_plain_store_bias_matmul, args)
    proof = _proof(bound)
    assert proof is not None and proof.store
    assert proof.bias is not None and proof.bias.dimensions == (1,)
    source = bound.to_code(
        helion.Config(
            block_sizes=[128, 128, 64],
            cute_split_k_workspace=True,
            cute_split_k_partitions=8,
            cute_split_k_stages=2,
        )
    )
    _producer, reducer = _embedded_sources(source)
    assert (
        "loaded_0[element] = loaded_0[element] + cutlass.Float32(bias_scalar[0])"
        in reducer
    )


@pytest.mark.parametrize(
    "kernel,extra",
    (
        (_plain_store_relu_matmul, ()),
        # The device loop reads the argument it writes: every tile must see
        # the argument's old value, which the reducer would have overwritten.
        (_plain_store_reading_out0, (torch.empty((128, 128), dtype=torch.float16),)),
    ),
)
def test_plain_store_variants_keep_the_original_path(kernel, extra) -> None:
    bound = _bind(kernel, (*_args((128, 16384, 128)), *extra))
    assert _proof(bound) is None
    assert not bound.config_spec.cute_split_k_workspace_available
    assert bound.config_spec.cute_split_k_partition_choices == ()


@pytest.mark.parametrize(
    "a_major,b_major", (("k", "n"), ("m", "n"), ("k", "k"), ("m", "k"))
)
def test_plain_store_into_an_argument_is_proved_for_either_operand_major(
    a_major: str, b_major: str
) -> None:
    # helion.inductor's kernel for a weight gradient, grad_out.T @ x: out0 is
    # an argument and A is M-major. 256 x 256 is one CTA pair of 148 SMs, so
    # the 8192-deep K loop is worth splitting.
    a, b, out = _major_args(256, 8192, 256, a_major, b_major)
    bound = _bind(_plain_store_into_out0, (a, b, out))
    proof = _proof(bound)
    assert proof is not None
    assert proof.store and proof.argument_output
    assert bound.config_spec.cute_split_k_partition_choices == (1, 2, 3, 4, 8, 16)
    assert any(
        seed.get("cute_split_k_workspace")
        for seed in bound.config_spec.compiler_seed_configs
    )
    source = bound.to_code(
        helion.Config(
            block_sizes=[128, 128, 64],
            cute_split_k_workspace=True,
            cute_split_k_partitions=4,
            cute_split_k_stages=4,
        )
    )
    producer, _reducer = _embedded_sources(source)
    host = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_plain_store_into_out0"
    )
    unparsed = ast.unparse(host)
    # The partials are the only allocation and the reducer writes out0;
    # the host still returns nothing.
    assert unparsed.count("torch.empty(") == 1
    assert "(4, 256, 256), dtype=torch.float32" in unparsed
    assert not any(isinstance(node, ast.Return) for node in ast.walk(host))
    (reduce_launch,) = [
        line for line in unparsed.splitlines() if "_reduce_partitions" in line
    ]
    assert "_helion_split_k_partials, out0" in reduce_launch
    # Each partition views the operands with their own strides: an M-major A
    # [256, 8192] with strides (1, 256) is [4, 256, 2048] with strides
    # (524288, 1, 256).
    a_view = (2048 * a.stride(1), *a.stride())
    b_view = (2048 * b.stride(0), *b.stride())
    assert f"in0.as_strided((4, 256, 2048), {a_view!r})" in unparsed
    assert f"in1.as_strided((4, 2048, 256), {b_view!r})" in unparsed
    a_mode = "K" if a_major == "k" else "MN"
    b_mode = "K" if b_major == "k" else "MN"
    assert (
        f"cute.nvgpu.OperandMajorMode.{a_mode}, cute.nvgpu.OperandMajorMode.{b_mode}"
        in producer
    )


@pytest.mark.parametrize("b_major", ("n", "k"))
def test_plain_store_into_a_column_major_argument_runs_the_swapped_product(
    b_major: str,
) -> None:
    # helion.inductor's transposed product of bf16 256 x 512 x 8192 (out.T =
    # b.T @ a.T): in0 = b.T [512, 8192], in1 = a.T [8192, 256] (a row-major,
    # so K-major) and out0 = out.T, [512, 256] with strides (1, 512). The
    # producer multiplies a by b into [P, 256, 512] row-major partials,
    # which is out0's storage order, so the reducer stays one flat loop.
    a = torch.empty((256, 8192), dtype=torch.bfloat16)
    b = (
        torch.empty((8192, 512), dtype=torch.bfloat16)
        if b_major == "n"
        else torch.empty((512, 8192), dtype=torch.bfloat16).T
    )
    out = torch.empty((256, 512), dtype=torch.bfloat16)
    bound = _bind(_plain_store_into_out0, (b.T, a.T, out.T))
    proof = _proof(bound)
    assert proof is not None
    assert proof.store and proof.argument_output
    # The swapped product: its rows are out0's columns, its tile comes from
    # out0's column block (tile_x) and row block (tile_y) in that order.
    assert (proof.m, proof.n, proof.k) == (256, 512, 8192)
    assert (proof.lhs_expression, proof.rhs_expression) == ("in1", "in0")
    assert proof.lhs_strides == a.stride() and proof.rhs_strides == b.stride()
    assert any(
        seed.get("cute_split_k_workspace")
        for seed in bound.config_spec.compiler_seed_configs
    )
    # block_sizes [tile_y, tile_x, tile_r]: the producer's 128-row tile is
    # out0's 128-column block, its 256-column tile out0's 256-row block.
    source = bound.to_code(
        helion.Config(
            block_sizes=[256, 128, 64],
            cute_split_k_workspace=True,
            cute_split_k_partitions=4,
            cute_split_k_stages=4,
        )
    )
    producer, _reducer = _embedded_sources(source)
    host = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_plain_store_into_out0"
    )
    unparsed = ast.unparse(host)
    assert "(4, 256, 512), dtype=torch.float32" in unparsed
    a_view = (2048 * a.stride(1), *a.stride())
    b_view = (2048 * b.stride(0), *b.stride())
    assert f"in1.as_strided((4, 256, 2048), {a_view!r})" in unparsed
    assert f"in0.as_strided((4, 2048, 512), {b_view!r})" in unparsed
    (reduce_launch,) = [
        line for line in unparsed.splitlines() if "_reduce_partitions" in line
    ]
    assert "_helion_split_k_partials, out0" in reduce_launch
    b_mode = "K" if b_major == "k" else "MN"
    assert (
        f"cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.{b_mode}"
        in producer
    )
    assert "'tile': (128, 256, 64)" in source


def test_column_major_output_bias_indexes_the_swapped_product() -> None:
    # A bias over out0's columns (tile_x) is a bias over the swapped
    # product's rows: the reducer's flat index i of the swapped [256, 512]
    # product (m = 256, n = 512) is row i // 512.
    a = torch.empty((256, 8192), dtype=torch.bfloat16)
    b = torch.empty((8192, 512), dtype=torch.bfloat16)
    out = torch.empty((256, 512), dtype=torch.bfloat16)
    bias = torch.empty((256,), dtype=torch.bfloat16)
    bound = _bind(_plain_store_column_bias_into_out0, (b.T, a.T, bias, out.T))
    proof = _proof(bound)
    assert proof is not None and proof.bias is not None
    assert proof.bias.dimensions == (0,)
    source = bound.to_code(
        helion.Config(
            block_sizes=[256, 128, 64],
            cute_split_k_workspace=True,
            cute_split_k_partitions=4,
            cute_split_k_stages=4,
        )
    )
    _producer, reducer = _embedded_sources(source)
    assert "(output_begin + element) // 512" in reducer


def test_sixteen_bit_partials_follow_split_k_reduction() -> None:
    # bf16 256 x 256 x 8192 into out0: under split_k_reduction="output" the
    # schedule may keep its partials in bf16 (cute_split_k_partials="output"),
    # and every workspace seed does; under the default "accumulator" the knob
    # is not offered and a config asking for it is invalid.
    args = _major_args(256, 8192, 256, "k", "n")
    output = _bind(_plain_store_into_out0_output_partials, args)
    assert output.config_spec.cute_split_k_partials_available
    seeds = [
        seed
        for seed in output.config_spec.compiler_seed_configs
        if seed.get("cute_split_k_workspace")
    ]
    assert seeds and all(s.get("cute_split_k_partials") == "output" for s in seeds)
    # ... and keep it through the search's seed transfer (flatten and
    # canonicalize over the tcgen05 search fields).
    transferred = [
        config
        for _, config in ConfigGeneration(output.config_spec).seed_flat_config_pairs()
        if config.get("cute_split_k_workspace")
    ]
    assert len(transferred) == len(seeds)
    assert all(c.get("cute_split_k_partials") == "output" for c in transferred)
    # Inert without the workspace schedule.
    with pytest.raises(InvalidConfig, match="requires cute_split_k_workspace"):
        output.config_spec.normalize(
            helion.Config(
                block_sizes=[128, 128, 64],
                cute_split_k_workspace=False,
                cute_split_k_partials="output",
            )
        )
    # A seed's mutant to one partition is repaired to the plain kernel, knob
    # dropped, and that config normalizes strictly.
    repaired = helion.Config(
        block_sizes=[128, 128, 64],
        cute_split_k_workspace=True,
        cute_split_k_partitions=1,
        cute_split_k_stages=4,
        cute_split_k_partials="output",
    ).config
    output.config_spec.normalize(repaired, _fix_invalid=True)
    assert repaired["cute_split_k_workspace"] is False
    assert "cute_split_k_partials" not in repaired
    output.config_spec.normalize(dict(repaired), _fix_invalid=False)
    accumulator = _bind(_plain_store_into_out0, args)
    assert not accumulator.config_spec.cute_split_k_partials_available
    with pytest.raises(InvalidConfig, match="cute_split_k_partials"):
        accumulator.config_spec.normalize(
            helion.Config(
                block_sizes=[128, 128, 64],
                cute_split_k_workspace=True,
                cute_split_k_partitions=4,
                cute_split_k_stages=4,
                cute_split_k_partials="output",
            )
        )


def test_sixteen_bit_partials_round_in_the_producer_and_widen_in_the_reducer() -> None:
    # The producer stores each partition's fp32 sum rounded to bf16 into
    # [4, 256, 256] bf16 partials; the reducer loads four of them at a time
    # (64-bit copies) and adds them in fp32.
    bound = _bind(
        _plain_store_into_out0_output_partials, _major_args(256, 8192, 256, "k", "n")
    )
    source = bound.to_code(
        helion.Config(
            block_sizes=[128, 128, 64],
            cute_split_k_workspace=True,
            cute_split_k_partitions=4,
            cute_split_k_stages=4,
            cute_split_k_partials="output",
        )
    )
    _producer, reducer = _embedded_sources(source)
    assert "(4, 256, 256), dtype=torch.bfloat16" in source
    assert "_native_partition_gemm_rounded(" in source
    assert "'partial_dtype': 'torch.bfloat16'" in source
    assert (
        "cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, "
        "num_bits_per_copy=64)"
    ) in reducer
    assert "loaded_3.store(raw_3.load().to(cutlass.Float32))" in reducer
    # fp32 partials (the knob's default) render as before
    fp32 = bound.to_code(
        helion.Config(
            block_sizes=[128, 128, 64],
            cute_split_k_workspace=True,
            cute_split_k_partitions=4,
            cute_split_k_stages=4,
        )
    )
    assert "(4, 256, 256), dtype=torch.float32" in fp32
    assert "_native_partition_gemm_rounded" not in fp32
    assert "raw_0" not in _embedded_sources(fp32)[1]


@pytest.mark.parametrize(
    "block_sizes,stages,expected",
    (
        # A tile the partition producer takes keeps its schedule, a ring
        # deeper than the tile fits is cut to the deepest that fits (12 for
        # 128 x 128 x 64 two-CTA tiles, 6 for 128 x 128 x 128).
        ([128, 128, 64], 12, (True, 12, 4)),
        ([128, 128, 64], 16, (True, 12, 4)),
        ([128, 128, 128], 7, (True, 6, 4)),
        # A tile it does not take (16-deep K steps) runs the plain kernel
        # instead of failing to compile.
        ([128, 128, 16], 12, (False, 2, 1)),
    ),
)
def test_search_repair_keeps_workspace_mutants_on_producer_geometries(
    block_sizes: list[int], stages: int, expected: tuple[bool, int, int]
) -> None:
    bound = _bind(_plain_store_into_out0, _major_args(256, 8192, 256, "m", "n"))
    spec = bound.config_spec
    config = helion.Config(
        block_sizes=block_sizes,
        cute_split_k_workspace=True,
        cute_split_k_partitions=4,
        cute_split_k_stages=stages,
    )
    with bound.env:
        spec.normalize(config, _fix_invalid=True)
    assert config.config["block_sizes"] == block_sizes
    assert (
        config.config["cute_split_k_workspace"],
        config.config["cute_split_k_stages"],
        config.config["cute_split_k_partitions"],
    ) == expected
    source = bound.to_code(config)
    assert ("_helion_cute_split_k_workspace" in source) is expected[0]


def test_workspace_seeds_keep_their_256_row_partition_tiles() -> None:
    # The seeds carry the one-CTA flat surface (pid_type "flat", cluster_m 1;
    # the schedule's producer is a CTA pair of its own), whose repair keeps a
    # one-CTA M tile within 128 rows. A 256-row partition tile must reach the
    # search as written: projected to 128 rows, bf16 2048 x 768 x 20008 ran
    # its three partitions on 128x256 instead of 256x256 tiles, 54.7 against
    # 45.7 us per call (GB300, in one process).
    bound = _bind(_plain_store_into_out0, _major_args(512, 8192, 256, "k", "n"))
    spec = bound.config_spec
    with bound.env:
        generation = ConfigGeneration(spec)
        pairs = generation.seed_flat_config_pairs()
    written = {
        (tuple(seed.config["block_sizes"]), seed.config["cute_split_k_partitions"])
        for seed in spec.compiler_seed_configs
        if seed.get("cute_split_k_workspace")
    }
    normalized = {
        (tuple(config.config["block_sizes"]), config.config["cute_split_k_partitions"])
        for _flat, config in pairs
        if config.config.get("cute_split_k_workspace")
    }
    assert any(tile[0] == 256 for tile, _partitions in written)
    assert normalized == written
    # The plain kernel's flat one-CTA sample still keeps 128 rows.
    with bound.env:
        _flat, plain = generation.canonicalize_flat(
            generation.flatten(
                helion.Config(
                    block_sizes=[256, 256, 64],
                    pid_type="flat",
                    tcgen05_cluster_m=1,
                    cute_split_k_workspace=False,
                )
            )
        )
    assert plain.config["block_sizes"][0] == 128


def test_search_repair_renders_every_sampled_workspace_mutant() -> None:
    # The mutants a full-effort search of a bf16 384 x 768 x 16384 weight
    # gradient failed to compile, before the repair, all render after it.
    bound = _bind(_plain_store_into_out0, _major_args(384, 16384, 768, "m", "n"))
    spec = bound.config_spec
    for block_sizes, partitions, stages in (
        ([64, 128, 128], 4, 6),
        ([128, 256, 128], 4, 6),
        ([64, 64, 128], 4, 6),
        ([128, 64, 128], 2, 16),
        ([128, 256, 16], 8, 8),
        ([128, 128, 64], 2, 16),
        ([128, 64, 32], 4, 16),
        ([64, 128, 64], 32, 16),
    ):
        config = helion.Config(
            block_sizes=block_sizes,
            cute_split_k_workspace=True,
            cute_split_k_partitions=partitions,
            cute_split_k_stages=stages,
        )
        with bound.env:
            spec.normalize(config, _fix_invalid=True)
        bound.to_code(config)


@pytest.mark.parametrize(
    "block_sizes,stages,expected",
    (
        ([128, 128, 64], 16, (True, 12)),
        ([128, 128, 16], 12, (False, 2)),
    ),
)
def test_search_repair_keeps_atomic_proof_mutants_on_producer_geometries(
    block_sizes: list[int], stages: int, expected: tuple[bool, int]
) -> None:
    # An atomic split-K proof partitions through the kernel's own tunable
    # (no ``cute_split_k_partitions``); its workspace mutants are repaired
    # like the plain store's.
    bound = _bind(_partitioned_matmul, _args((128, 2048, 128)))
    config = helion.Config(
        block_sizes=block_sizes,
        partitions=4,
        cute_split_k_workspace=True,
        cute_split_k_stages=stages,
    )
    with bound.env:
        bound.config_spec.normalize(config, _fix_invalid=True)
    assert config.config["block_sizes"] == block_sizes
    assert (
        config.config["cute_split_k_workspace"],
        config.config["cute_split_k_stages"],
    ) == expected
    source = bound.to_code(config)
    assert ("_helion_cute_split_k_workspace" in source) is expected[0]


@pytest.mark.parametrize(
    "shape,expected",
    (
        # 384 rows take no 256-row partition tile: the plain kernel.
        ((384, 16384, 768), (False, 2, 1)),
        # 512 x 256 takes it, with the six stages its ring fits.
        ((512, 8192, 256), (True, 6, 4)),
    ),
)
def test_search_repair_reads_the_tile_the_projections_leave(
    shape: tuple[int, int, int], expected: tuple[bool, int, int]
) -> None:
    # A workspace seed's mutant to ``tcgen05_cluster_m=2`` is projected onto
    # the 256x256 CTA pair tile after it was sampled
    # (``_fix_cluster_m2_search_config``); the repair judges that tile.
    bound = _bind(_plain_store_into_out0, _major_args(*shape, "m", "n"))
    spec = bound.config_spec
    seed = next(
        seed
        for seed in spec.compiler_seed_configs
        if seed.get("cute_split_k_workspace")
    )
    config = helion.Config.from_dict(
        seed.config
        | {
            "block_sizes": [128, 128, 64],
            "cute_split_k_partitions": 4,
            "cute_split_k_stages": 12,
            "tcgen05_cluster_m": 2,
        }
    )
    with bound.env:
        spec.normalize(config, _fix_invalid=True)
    assert config.config["block_sizes"] == [256, 256, 64]
    assert (
        config.config["cute_split_k_workspace"],
        config.config["cute_split_k_stages"],
        config.config["cute_split_k_partitions"],
    ) == expected
    source = bound.to_code(config)
    assert ("_helion_cute_split_k_workspace" in source) is expected[0]


def test_plain_store_into_a_returned_argument_keeps_its_return() -> None:
    bound = _bind(
        _plain_store_into_argument,
        (*_args((128, 16384, 128)), torch.empty((128, 128), dtype=torch.float16)),
    )
    proof = _proof(bound)
    assert proof is not None and proof.argument_output
    source = bound.to_code(
        helion.Config(
            block_sizes=[128, 128, 64],
            cute_split_k_workspace=True,
            cute_split_k_partitions=4,
            cute_split_k_stages=2,
        )
    )
    host = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_plain_store_into_argument"
    )
    (ret,) = [node for node in ast.walk(host) if isinstance(node, ast.Return)]
    assert ast.unparse(ret) == "return out"
    assert ast.unparse(host).count("torch.empty(") == 1


def test_misaligned_argument_output_is_not_given_the_reducer() -> None:
    # An argument output whose base sits 2 bytes past the 8 the reducer's
    # four-element bf16 stores need: the proof may admit it before the binding
    # records alignments, and declines it once they are bound (code
    # generation proves the schedule again).
    a, b, _out = _major_args(256, 8192, 256, "m", "n")
    misaligned = torch.empty((256 * 256 + 1,), dtype=torch.bfloat16)[1:].view(256, 256)
    bound = _bind(_plain_store_into_out0, (a, b, misaligned))
    proof = _proof(bound)
    assert proof is None
    aligned_offset = torch.empty((256 * 256 + 4,), dtype=torch.bfloat16)[4:].view(
        256, 256
    )
    assert _proof(_bind(_plain_store_into_out0, (a, b, aligned_offset))) is not None
    config = helion.Config(
        block_sizes=[128, 128, 64],
        cute_split_k_workspace=True,
        cute_split_k_partitions=4,
        cute_split_k_stages=4,
    )
    # the search was offered the schedule (its heuristic proved it before
    # the binding recorded the output's alignment); code generation declines
    assert bound.config_spec.cute_split_k_workspace_available
    with pytest.raises(BackendUnsupported, match="proof declined"):
        bound.to_code(config)


@pytest.mark.parametrize(
    "k,choices",
    (
        # 32 K tiles of 64: three partitions keep eleven tiles (the third
        # ten, running past K), four keep eight, eight would keep four.
        (2048, (1, 2, 3, 4)),
        # Eight tiles: no split keeps eight per partition, so the schedule
        # is not offered at all (a 128^3 .. 512^3 GEMM has a small grid but
        # nothing to split).
        (512, ()),
    ),
)
def test_plain_store_partitions_keep_eight_k_tiles(k, choices) -> None:
    bound = _bind(_plain_store_matmul, _args((128, k, 128)))
    spec = bound.config_spec
    assert spec.cute_split_k_partition_choices == choices
    assert spec.cute_split_k_workspace_available is bool(choices)
    if not choices:
        assert _proof(bound) is None
        assert not any(
            seed.get("cute_split_k_workspace") for seed in spec.compiler_seed_configs
        )


@pytest.mark.parametrize(
    "k,partitions,bk,expected",
    (
        # Whole tiles cut exactly: the batch views' chunk.
        (16384, 4, 64, 4096),
        # bf16 2048 x 768 x 20008: 312 tiles of 64 and 40 columns; three
        # partitions of 105 tiles, the third 152 columns past K.
        (20008, 3, 64, 6720),
        (20008, 2, 128, 10112),
        # Whole 64-deep tiles leave nothing for the last of eight
        # partitions of 520 (7 x 128 >= 520).
        (520, 8, 64, None),
    ),
)
def test_plain_store_chunk(
    k: int, partitions: int, bk: int, expected: int | None
) -> None:
    assert plain_store_chunk(k, partitions, bk) == expected


def test_plain_store_partitions_offset_k_past_a_tail() -> None:
    # helion.inductor's kernel over a K that no [P, M, K / P] batch view cuts
    # into whole tiles: 8008 = 125 tiles of 64 and 8 columns (an LM head's
    # gradient over an odd vocabulary has such a K).
    a, b, out = _major_args(256, 8008, 256, "k", "n")
    bound = _bind(_plain_store_into_out0, (a, b, out))
    assert _proof(bound) is not None
    assert 4 in bound.config_spec.cute_split_k_partition_choices
    source = bound.to_code(
        helion.Config(
            block_sizes=[128, 128, 64],
            cute_split_k_workspace=True,
            cute_split_k_partitions=4,
            cute_split_k_stages=4,
        )
    )
    producer, _reducer = _embedded_sources(source)
    # Four partitions of 2048: each partition shifts its loads' K coordinate,
    # and the TensorMap zero-fills the last one's 184 columns past K.
    for tensor in ("tma_tensor_a", "tma_tensor_b"):
        assert (
            f"cute.domain_offset((0, cutlass.Int32(indices_0) * 2048, 0), {tensor})"
            in producer
        )
    host = ast.unparse(
        next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_plain_store_into_out0"
        )
    )
    # The operands go in whole, one batch of all of K.
    a_view = (2048 * a.stride(1), *a.stride())
    b_view = (2048 * b.stride(0), *b.stride())
    assert f"in0.as_strided((1, 256, 8008), {a_view!r})" in host
    assert f"in1.as_strided((1, 8008, 256), {b_view!r})" in host
    assert "(4, 256, 256), dtype=torch.float32" in host
    assert "'k_offset': True" in source


@pytest.mark.parametrize(
    "m,n,k,choices",
    (
        # 16 x 6 = 96 tiles of 256x256 fill 63% of two waves of a GB300's 76
        # SM pairs: a K deep enough for 128 tiles per partition splits.
        (4096, 1536, 16384, (1, 2)),
        # 19 x 8 = 152 tiles fill both waves: nothing to win.
        (4864, 2048, 16384, ()),
        # The same 96 tiles over a K of 128 tiles: two partitions of 64 would
        # pay the partials' traffic on a short K loop.
        (4096, 1536, 8192, ()),
    ),
)
def test_plain_store_splits_a_fuller_grid_whose_waves_run_short(
    m: int, n: int, k: int, choices: tuple[int, ...]
) -> None:
    with patch_target_facts(num_sm=152):
        bound = _bind(_plain_store_into_out0, _major_args(m, k, n, "k", "n"))
        assert bound.config_spec.cute_split_k_partition_choices == choices


def test_fuller_grid_seeds_fill_the_waves_on_256_row_tiles() -> None:
    # 96 tiles of 256x256 (bf16 4096 x 1536 x K, b stored [K, N]): three
    # partitions run 288 units, 95% of four waves of 76 pairs, two 84% of
    # three. The 256-row tile reaches the search: the seeds carry the
    # one-CTA flat surface (pid_type "flat", cluster_m 1), whose repair
    # would otherwise project the producer's tile to 128 rows.
    with patch_target_facts(num_sm=152):
        bound = _bind(_plain_store_into_out0, _major_args(4096, 24576, 1536, "k", "n"))
        spec = bound.config_spec
        with bound.env:
            generation = ConfigGeneration(spec)
            seeds = [
                config
                for _flat, config in generation.seed_flat_config_pairs()
                if config.config.get("cute_split_k_workspace")
            ]
    pair_tiles = [
        config.config["cute_split_k_partitions"]
        for config in seeds
        if config.config["block_sizes"] == [256, 256, 64]
    ]
    assert pair_tiles[:2] == [3, 3]
    assert 3 in pair_tiles and 2 in pair_tiles


def test_own_split_k_seeds_keep_one_cluster_per_pair_on_a_fuller_grid() -> None:
    # A kernel that splits K through its own tunable has no wave-fill gate:
    # on a fuller grid (bf16 4096 x 1536 x 16384, 96 tiles of 256x256 on 76
    # SM pairs) its seeds keep about one native cluster per SM pair, not the
    # finest split the wave fill ranks first (256 partitions of one 64-deep K
    # tile each, 6.4 GB of fp32 partials).
    with patch_target_facts(num_sm=152):
        bound = _bind(
            _partitioned_matmul, _args((4096, 16384, 1536), dtype=torch.bfloat16)
        )
        assert bound.host_function is not None
        with bound.env:
            seeds = CuteSplitKWorkspaceHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
    splits = {seed["partitions"] for seed in seeds}
    assert splits and max(splits) <= 2


def test_multi_wave_producer_hands_out_units_partition_by_partition() -> None:
    # 16 x 6 tiles of 256x256 in three partitions: 288 units on 76 SM pairs.
    with patch_target_facts(num_sm=152):
        bound = _bind(_plain_store_into_out0, _major_args(4096, 24576, 1536, "k", "n"))
        source = bound.to_code(
            helion.Config(
                block_sizes=[256, 256, 64],
                cute_split_k_workspace=True,
                cute_split_k_partitions=3,
                cute_split_k_stages=6,
            )
        )
    producer, _reducer = _embedded_sources(source)
    remap = "virtual_pid = virtual_pid // 96 + 3 * (virtual_pid // 6 % 16 + 16 * (virtual_pid % 6))"
    # Once per role whose loop decodes a unit (the TMA producer, the MMA and
    # the epilogue), right after its linear index.
    assert producer.count(remap) == 3
    # One wave (2 x 1 tiles in four partitions): each unit its own pair.
    with patch_target_facts(num_sm=152):
        small = _bind(_plain_store_into_out0, _major_args(512, 8192, 256, "k", "n"))
        source = small.to_code(
            helion.Config(
                block_sizes=[256, 256, 64],
                cute_split_k_workspace=True,
                cute_split_k_partitions=4,
                cute_split_k_stages=6,
            )
        )
    producer, _reducer = _embedded_sources(source)
    assert "virtual_pid = virtual_pid //" not in producer


def test_plain_store_default_config_carries_no_workspace_knobs() -> None:
    # The schedule is a seeded search alternative; the public default is the
    # plain kernel, for a K-major and an M-major A alike (both admit the
    # proof: the partition producer keeps either major).
    a, b = _args((128, 16384, 128))
    row_major = _bind(_plain_store_matmul, (a, b))
    m_major = _bind(_plain_store_matmul, (a.T.contiguous().T, b))
    assert _proof(row_major) is not None
    assert _proof(m_major) is not None
    for bound in (row_major, m_major):
        default = bound.config_spec.default_config().config
        assert not any(key.startswith("cute_split_k_") for key in default)
    assert (
        row_major.config_spec.default_config() == m_major.config_spec.default_config()
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _plain_store_full_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.full((m, n), fill_value=1.0, dtype=a.dtype, device=a.device)
    for row, column in hl.tile((m, n)):
        accumulator = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            accumulator = torch.addmm(accumulator, a[row, inner], b[inner, column])
        out[row, column] = accumulator.to(out.dtype)
    return out


def test_plain_store_full_factory_keyword_fill_is_dropped() -> None:
    bound = _bind(_plain_store_full_matmul, _args((128, 16384, 128)))
    assert _proof(bound) is not None
    source = bound.to_code(
        helion.Config(
            block_sizes=[128, 128, 64],
            cute_split_k_workspace=True,
            cute_split_k_partitions=4,
            cute_split_k_stages=2,
        )
    )
    host = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_plain_store_full_matmul"
    )
    unparsed = ast.unparse(host)
    # The grid writes every output element, so the fill is dropped with the
    # size kept: ``torch.full((m, n), fill_value=1.0)`` -> ``torch.empty((m, n))``.
    assert "fill_value" not in unparsed
    assert "torch.empty((m, n)" in unparsed


def test_workspace_knobs_read_their_defaults_when_the_schedule_is_off() -> None:
    # With the schedule off its stage and partition knobs are inert: samples
    # read their defaults so the search does not compile one plain kernel per
    # inert value, and the public default config carries none of them.
    bound = _bind(_plain_store_matmul, _args((128, 16384, 128)))
    spec = bound.config_spec
    assert spec.cute_split_k_workspace_available
    gen = spec.create_config_generation()
    with bound.env:
        for _ in range(100):
            config = gen.unflatten(gen.random_flat()).config
            if not config.get("cute_split_k_workspace"):
                assert config.get("cute_split_k_stages", 2) == 2
                assert config.get("cute_split_k_partitions", 1) == 1
    default = spec.default_config().config
    assert not any(key.startswith("cute_split_k_") for key in default)


@pytest.mark.parametrize("partitions", (2, 4, 16))
def test_reducer_issues_every_partial_load_before_the_first_sum(
    partitions: int,
) -> None:
    args = _args((128, 64 * partitions, 128))
    bound = _bind(_direct_chunk_matmul, args)
    proof = _proof(bound)
    assert proof is not None
    config = helion.Config(block_sizes=[128, 128, 64], tile_extent=64)
    with bound.env:
        schedule = workspace_schedule(proof, bound.env, config, capacity_bytes=232448)
    assert schedule is not None and schedule.partitions == partitions
    source = _reducer_source(proof, schedule)
    lines = source.splitlines()
    # The next kernel may launch once every reducer CTA is resident; the
    # reducer itself waits for the producer before it loads a partial.
    assert lines[5].strip() == "cute.arch.griddepcontrol_launch_dependents()"
    assert lines[6].strip() == "cute.arch.griddepcontrol_wait()"
    # One 128 x 128 output: 4096 vectors, one a thread of 16 CTAs; a thread
    # loads each partition's four floats of its vector before the first sum.
    assert f"vector = block * cutlass.Int32({REDUCER_THREADS}) + thread" in source
    loads = [index for index, line in enumerate(lines) if "cute.copy(load_atom" in line]
    assert len(loads) == partitions
    (store,) = [index for index, line in enumerate(lines) if "converted.store(" in line]
    if partitions <= REDUCER_GROUP:
        assert max(loads) < store
        assert " + ".join(f"loaded_{q}.load()" for q in range(partitions)) in source
    else:
        # Deeper schedules load and fold REDUCER_GROUP partitions at a time,
        # in partition order: at most that many loads in flight a thread.
        folds = [index for index, line in enumerate(lines) if "total = " in line]
        assert len(folds) == -(-partitions // REDUCER_GROUP)
        for group, fold in enumerate(folds):
            members = range(
                group * REDUCER_GROUP, min((group + 1) * REDUCER_GROUP, partitions)
            )
            expected = " + ".join(f"loaded_{q}.load()" for q in members)
            assert lines[fold].strip() == (
                f"total = {expected}" if group == 0 else f"total = total + {expected}"
            )
            assert all(
                loads[q] < fold and (group == 0 or loads[q] > folds[group - 1])
                for q in members
            )
        assert "converted.store((total).to(" in lines[store]
    # 4096 vectors fill whole CTAs: no bound check.
    assert "if vector <" not in source


def test_reducer_launches_as_a_programmatic_dependent() -> None:
    bound = _bind(_partitioned_matmul, _args())
    source = bound.to_code(_config())
    assert (
        "_helion_split_k_reduce._reduce_partitions._helion_cute_use_pdl = True"
        in source
    )
    _producer, reducer = _embedded_sources(source)
    assert reducer.index("griddepcontrol_wait()") < reducer.index("cute.copy(load_atom")


@pytest.mark.parametrize(
    "arguments,expected",
    (
        # The s2 winner (64x64 CTA tiles, (2, 2) warps): one 64x64 subtile,
        # 16 lanes per row, 8 instructions per warp.
        (
            {"rows": 64, "cta_m": 64, "cta_n": 64, "use_2cta_instrs": True},
            (64, 64, 1, 4, 16, 8, 32, 32),
        ),
        # 64x128 CTA tiles: 64-column subtiles made of two interleaved
        # 32-column runs 64 columns apart.
        (
            {"rows": 64, "cta_m": 64, "cta_n": 128, "use_2cta_instrs": True},
            (64, 64, 1, 4, 16, 8, 32, 64),
        ),
        # fp32 output: two values (8 bytes) per lane.
        (
            {
                "rows": 64,
                "cta_m": 64,
                "cta_n": 64,
                "use_2cta_instrs": True,
                "itemsize": 4,
            },
            (64, 64, 1, 2, 32, 16, 32, 32),
        ),
        # The lowering's 256-row pair form: a 256x128 CTA shape with a
        # source-C description, (4, 1) warps, 128x16 subtiles (4 lanes per
        # row, 8 rows per warp instruction).
        (
            {
                "rows": 128,
                "cta_m": 256,
                "cta_n": 128,
                "use_2cta_instrs": False,
                "elem_width_c": 32,
                "c_is_m_major": False,
            },
            (128, 16, 1, 4, 4, 4, 16, 128),
        ),
    ),
)
def test_atomic_epilogue_geometry(arguments, expected) -> None:
    geometry = atomic_epilogue_geometry(
        **{
            "elem_width_c": None,
            "d_is_m_major": False,
            "c_is_m_major": True,
            "itemsize": 2,
        }
        | arguments
    )
    assert geometry is not None
    assert (
        geometry.tile_m,
        geometry.tile_n,
        geometry.m_subtiles,
        geometry.width,
        geometry.chunks_per_row,
        geometry.iterations,
        geometry.run,
        geometry.run_stride,
    ) == expected
    assert geometry.lane_column_fixed
    cta_n = arguments["cta_n"]
    columns = [
        eval(geometry.column_expression(str(subtile), str(column)))
        for subtile in range(cta_n // geometry.tile_n)
        for column in range(geometry.tile_n)
    ]
    # Every CTA tile column exactly once over the N subtiles.
    assert sorted(columns) == list(range(cta_n))
    assert geometry.subtile_expressions("s") == ("0", "s")
    assert geometry.row_expression("0", "r") == "r"


def test_atomic_epilogue_geometry_with_several_m_subtiles() -> None:
    geometry = atomic_epilogue_geometry(
        rows=256,
        cta_m=256,
        cta_n=128,
        use_2cta_instrs=False,
        elem_width_c=32,
        d_is_m_major=False,
        c_is_m_major=False,
        itemsize=2,
    )
    assert geometry is not None and geometry.m_subtiles == 2
    assert geometry.subtile_expressions("s") == ("s % 2", "s // 2")
    assert geometry.row_expression("s % 2", "r") == "s % 2 * 128 + r"
    rows = [
        eval(geometry.row_expression(str(subtile % 2), str(row)))
        for subtile in range(4)
        for row in range(geometry.tile_m)
    ]
    assert sorted(rows) == sorted(list(range(256)) * 2)
    # A packed width that does not tile the run is declined.
    assert (
        atomic_epilogue_geometry(
            rows=64,
            cta_m=64,
            cta_n=64,
            use_2cta_instrs=True,
            elem_width_c=None,
            d_is_m_major=False,
            c_is_m_major=True,
            itemsize=16,
        )
        is None
    )


def test_atomic_epilogue_is_offered_for_outputs_kept_in_their_dtype() -> None:
    # The module fixture pins the fp32 promotion on: the promoted float16
    # output keeps fp32 storage and a host cast, which the reducer form serves.
    promoted = _bind(_partitioned_matmul, _args())
    assert promoted.config_spec.cute_split_k_workspace_available
    assert not promoted.config_spec.cute_split_k_atomic_epilogue_available
    with pytest.raises(InvalidConfig, match="cute_split_k_atomic_epilogue"):
        promoted.config_spec.normalize(
            _config(cute_split_k_atomic_epilogue=True).config, _fix_invalid=False
        )
    relaxed = _config(cute_split_k_atomic_epilogue=True).config
    promoted.config_spec.normalize(relaxed, _fix_invalid=True)
    assert "cute_split_k_atomic_epilogue" not in relaxed
    assert not any(
        seed.get("cute_split_k_atomic_epilogue")
        for seed in promoted.config_spec.compiler_seed_configs
    )
    # A bfloat16 output is kept in its dtype: the knob is offered and seeded.
    kept = _bind(_partitioned_matmul, _args(dtype=torch.bfloat16))
    assert kept.config_spec.cute_split_k_atomic_epilogue_available
    normalized = _config(cute_split_k_atomic_epilogue=True).config
    kept.config_spec.normalize(normalized, _fix_invalid=False)
    assert normalized["cute_split_k_atomic_epilogue"] is True
    # Inert without the workspace schedule.
    with pytest.raises(InvalidConfig, match="requires cute_split_k_workspace"):
        kept.config_spec.normalize(
            _config(
                cute_split_k_workspace=False, cute_split_k_atomic_epilogue=True
            ).config,
            _fix_invalid=False,
        )
    assert kept.host_function is not None
    with kept.env:
        seeds = CuteSplitKWorkspaceHeuristic.get_seed_configs(
            kept.env, kept.host_function.device_ir
        )
    twins = [seed for seed in seeds if seed.get("cute_split_k_atomic_epilogue")]
    assert twins and len(twins) < len(seeds)
    for twin in twins:
        assert twin["cute_split_k_workspace"] is True
    # The twins reach the search's population as twins.
    transferred = [
        config
        for _, config in ConfigGeneration(kept.config_spec).seed_flat_config_pairs()
        if config.get("cute_split_k_atomic_epilogue")
    ]
    assert len(transferred) == len(twins)
    # The plain store has no zero-filled destination to add into.
    store = _bind(_plain_store_matmul, _args((128, 16384, 128)))
    assert store.config_spec.cute_split_k_workspace_available
    assert not store.config_spec.cute_split_k_atomic_epilogue_available


def _kernel_source(module_source: str, name: str) -> str:
    return ast.unparse(
        next(
            node
            for node in ast.parse(module_source).body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )


@pytest.mark.parametrize("with_bias", (False, True))
def test_atomic_epilogue_emits_one_producer_and_a_fill_kernel(with_bias: bool) -> None:
    args = _args((128, 32768, 256))
    bias = torch.empty((256,), dtype=torch.float16)
    epilogue = (
        (lambda acc, tile: acc + bias[tile[1]])
        if with_bias
        else (lambda acc, tile: acc)
    )
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_promote_half_atomic_outputs=False,
    )
    bound = _bind(kernel, (*args, epilogue))
    assert bound.config_spec.cute_split_k_atomic_epilogue_available
    source = bound.to_code(
        helion.Config(
            block_sizes=[128, 64, 128],
            split_k=16,
            cute_split_k_workspace=True,
            cute_split_k_stages=6,
            cute_split_k_atomic_epilogue=True,
        )
    )
    (producer,) = _embedded_sources(source)
    host = _kernel_source(source, "matmul_split_k")
    # One launch of the producer behind the fill kernel; no partials, no reducer.
    assert "_reduce_partitions" not in source and "_helion_split_k_partials" not in host
    assert host.count("torch.empty(") == 1 and "torch.zeros" not in host
    assert "_launcher(_helion_zero_fill_split_k, " in host
    assert "_helion_zero_fill_split_k._helion_cute_use_pdl = True" in source
    fill = _kernel_source(source, "_helion_zero_fill_split_k")
    assert fill.index("griddepcontrol_wait()") < fill.index("launch_dependents()")
    assert "'epilogue': 'atomic'" in source
    call = next(line for line in host.splitlines() if "_native_partition_gemm(" in line)
    positional = call.split("(", 1)[1].rsplit(", _launcher=", 1)[0].split(", ")
    assert positional[:3] == ["_helion_split_k_a", "_helion_split_k_b", "out"]
    assert len(positional) == (4 if with_bias else 3)
    # The producer: packed half reductions from the staged tile, no TMA store.
    kernel_def = _kernel_source(producer, "_helion__native_partition_gemm")
    parameters = next(
        line for line in kernel_def.splitlines() if line.startswith("def ")
    )
    assert parameters.startswith(
        "def _helion__native_partition_gemm(a, b, out, bias, tma_atom_a, "
        if with_bias
        else "def _helion__native_partition_gemm(a, b, out, tma_atom_a, "
    )
    assert "tcgen05_tma_store" not in parameters
    assert "cute.copy(tcgen05_tma_store_atom" not in producer
    assert "prefetch_descriptor(tcgen05_tma_store_atom)" not in producer
    assert "'tcgen05_d_tma'" not in producer
    assert producer.count("_cute_red_add_f16_vec(") == 1
    assert "red_add_f16_vec as _cute_red_add_f16_vec" in producer
    assert (
        "tcgen05_sD[tcgen05_red_row, tcgen05_red_col + cutlass.Int32(3), tcgen05_c_buffer]"
        in producer
    )
    assert ("tcgen05_red_bias_3" in producer) == with_bias
    assert ("if tile_offset_0 == cutlass.Int32(0):" in producer) == with_bias
    # The TMA warp issues its loads under the fill; the epilogue warps wait.
    assert kernel_def.count("cute.arch.griddepcontrol_wait()") == 1
    tma_warp = kernel_def.index("== cutlass.Int32(5):")
    epilogue_warps = kernel_def.index("< cutlass.Int32(4):")
    wait = kernel_def.index("cute.arch.griddepcontrol_wait()")
    assert tma_warp < epilogue_warps < wait
    assert hashlib.sha256(producer.encode()).hexdigest() in source
    wrapper = _kernel_source(producer, "_native_partition_gemm")
    assert ("a, b, out, bias, block=" in wrapper) == with_bias
    assert ("a, b, out, block=" in wrapper) == (not with_bias)
    assert "return out" in wrapper


def _atomic_rewrite_inputs(
    with_bias: bool = True,
) -> tuple[ast.Module, Any, Any, list[dict[str, object]]]:
    """The TMA-store producer module the reducer form emits for the example
    at s2 (the input of the atomic rewrite), with its proof, schedule and
    wrapper plans."""
    args = _args((128, 32768, 256))
    bias = torch.empty((256,), dtype=torch.float16)
    epilogue = (
        (lambda acc, tile: acc + bias[tile[1]])
        if with_bias
        else (lambda acc, tile: acc)
    )
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_promote_half_atomic_outputs=False,
    )
    bound = _bind(kernel, (*args, epilogue))
    config = helion.Config(
        block_sizes=[128, 64, 128],
        split_k=16,
        cute_split_k_workspace=True,
        cute_split_k_stages=6,
    )
    (producer, _reducer) = _embedded_sources(bound.to_code(config))
    module = ast.parse(producer)
    plans = next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "_helion_cute_wrapper_plans"
    )
    proof = _proof(bound)
    assert proof is not None
    with bound.env:
        schedule = workspace_schedule(
            proof,
            bound.env,
            helion.Config.from_dict(
                config.config | {"cute_split_k_atomic_epilogue": True}
            ),
            capacity_bytes=232448,
        )
    assert schedule is not None and schedule.atomic_epilogue
    return module, proof, schedule, plans


def _kernel_def(module: ast.Module) -> ast.FunctionDef:
    return next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_helion__native_partition_gemm"
    )


def _drop_statements(node: ast.AST, text: str) -> int:
    """Delete every statement unparsing to ``text`` under ``node``."""
    removed = 0
    for child in ast.walk(node):
        for field in ("body", "orelse"):
            statements = getattr(child, field, None)
            if isinstance(statements, list):
                kept = [s for s in statements if ast.unparse(s) != text]
                removed += len(statements) - len(kept)
                statements[:] = kept or [ast.Pass()]
    return removed


def test_atomic_rewrite_checks_the_emitted_tiler_in_the_kernel() -> None:
    module, proof, schedule, plans = _atomic_rewrite_inputs()
    _rewrite_producer_for_atomic_epilogue(module, proof, schedule, plans)
    kernel = ast.unparse(_kernel_def(module))
    check = (
        "_helion_split_k_atomic.check_epilogue_tiler(tcgen05_store_epi_tile, "
        "rows=64, tile_m=64, tile_n=64, run=32, run_stride=32)"
    )
    assert kernel.count(check) == 1
    assert (
        kernel.index("tcgen05_store_epi_tile = ")
        < kernel.index(check)
        < kernel.index("_cute_red_add_f16_vec(")
    )
    assert (
        "import helion._compiler.cute._split_k_atomic_runtime as _helion_split_k_atomic"
        in ast.unparse(module)
    )


def test_atomic_rewrite_refuses_a_producer_without_the_epilogue_wait() -> None:
    module, proof, schedule, plans = _atomic_rewrite_inputs()
    kernel = _kernel_def(module)
    # The epilogue role (warps 0-3) waits right after its entry; the TMA warp's
    # wait is the one the rewrite drops. Remove the epilogue one.
    epilogue = next(
        statement
        for statement in kernel.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test)
        == "cute.arch.make_warp_uniform(cute.arch.warp_idx()) < cutlass.Int32(4)"
    )
    assert _drop_statements(epilogue, "cute.arch.griddepcontrol_wait()") == 1
    with pytest.raises(BackendUnsupported, match="wait ahead of the first reduction"):
        _rewrite_producer_for_atomic_epilogue(module, proof, schedule, plans)


def test_atomic_rewrite_refuses_a_producer_without_any_tma_wait() -> None:
    module, proof, schedule, plans = _atomic_rewrite_inputs()
    kernel = _kernel_def(module)
    tma = next(
        statement
        for statement in kernel.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test)
        == "cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(5)"
    )
    assert _drop_statements(tma, "cute.arch.griddepcontrol_wait()") == 1
    with pytest.raises(BackendUnsupported, match="TMA warp's dependent-launch wait"):
        _rewrite_producer_for_atomic_epilogue(module, proof, schedule, plans)


@pytest.mark.parametrize(
    "mutation,message",
    (
        (
            ("cute.nvgpu.tcgen05.CtaGroup.TWO", "cute.nvgpu.tcgen05.CtaGroup.ONE"),
            "two-CTA tcgen05 MMA",
        ),
        (
            (
                "(128, 64), cute.nvgpu.tcgen05.OperandSource",
                "(256, 64), cute.nvgpu.tcgen05.OperandSource",
            ),
            "two-CTA tcgen05 MMA",
        ),
        (
            (
                "_helion_tc_pro.mod_nonneg(cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()), 2)",
                "cutlass.Int32(0)",
            ),
            "cluster rank MMA slice",
        ),
    ),
)
def test_atomic_rewrite_refuses_other_producer_pairings(mutation, message) -> None:
    module, proof, schedule, plans = _atomic_rewrite_inputs()
    source = ast.unparse(module)
    before, after = mutation
    assert source.count(before) >= 1
    module = ast.parse(source.replace(before, after))
    with pytest.raises(BackendUnsupported, match=message):
        _rewrite_producer_for_atomic_epilogue(module, proof, schedule, plans)


# GPU numerics of the split-K workspace schedule: the partials, the atomic
# epilogue and the fp32 fallback under poisoned CUDA-graph replays (sm_100).


@dataclass(frozen=True)
class WorkspaceCase:
    name: str
    shape: tuple[int, int, int]
    tile: tuple[int, int, int]
    parameter: str
    value: int
    stages: int
    kind: str = "original"
    bias: bool = False
    dtype: torch.dtype = torch.float16
    padding: int = 0
    # A is K-major ("k", row-major) or M-major ("m"); B is N-major ("n",
    # row-major) or K-major ("k").
    a_major: str = "k"
    b_major: str = "n"


WORKSPACE_CASES = (
    WorkspaceCase(
        "original-plain", (128, 32768, 256), (128, 128, 64), "split_k", 32, 12
    ),
    WorkspaceCase(
        "original-bias",
        (128, 32768, 256),
        (128, 128, 64),
        "split_k",
        32,
        12,
        bias=True,
    ),
    WorkspaceCase("n64-deep", (128, 4096, 256), (128, 64, 64), "split_k", 16, 16),
    WorkspaceCase(
        "original-bf16",
        (128, 32768, 256),
        (128, 128, 64),
        "split_k",
        32,
        12,
        dtype=torch.bfloat16,
    ),
    WorkspaceCase(
        "persistent-multiwave",
        (512, 2048, 512),
        (128, 128, 64),
        "split_k",
        16,
        2,
        padding=8,
    ),
    WorkspaceCase(
        "bf16-fp32-padded",
        (256, 4096, 256),
        (256, 128, 128),
        "partitions",
        8,
        2,
        kind="fp32",
        bias=True,
        dtype=torch.bfloat16,
        padding=8,
    ),
    WorkspaceCase(
        "plain-store-bf16",
        (256, 4096, 256),
        (128, 128, 64),
        "cute_split_k_partitions",
        8,
        4,
        kind="store",
        dtype=torch.bfloat16,
    ),
    WorkspaceCase(
        "plain-store-bias-fp16",
        (128, 8192, 512),
        (128, 128, 64),
        "cute_split_k_partitions",
        16,
        6,
        kind="store",
        bias=True,
    ),
    # helion.inductor's kernel: out0 is an argument, operands of either major
    # (a weight gradient's A is M-major).
    *(
        WorkspaceCase(
            f"argument-{a_major}{b_major}-bf16",
            (256, 8192, 256),
            (128, 128, 64),
            "cute_split_k_partitions",
            4,
            4,
            kind="argument",
            dtype=torch.bfloat16,
            padding=8,
            a_major=a_major,
            b_major=b_major,
        )
        for a_major, b_major in (("m", "n"), ("k", "k"), ("m", "k"))
    ),
    # 8 x 8 tiles in four partitions: 256 units on at most 76 CTA pairs, which
    # the producer hands out partition by partition
    # (``_order_partition_units_output_major``).
    WorkspaceCase(
        "argument-multi-wave-bf16",
        (1024, 16384, 1024),
        (128, 128, 64),
        "cute_split_k_partitions",
        4,
        4,
        kind="argument",
        dtype=torch.bfloat16,
    ),
    # helion.inductor's transposed product, out.T = b.T @ a.T into the
    # column-major out.T (``_swap_output_major``): the producer multiplies
    # a by b into [4, 256, 512] partials in out's storage order.
    *(
        WorkspaceCase(
            f"transposed-{b_major}-bf16",
            (256, 8192, 512),
            (128, 128, 64),
            "cute_split_k_partitions",
            4,
            4,
            kind="transposed",
            dtype=torch.bfloat16,
            b_major=b_major,
        )
        for b_major in ("n", "k")
    ),
    # K = 125 tiles of 64 + 8: four partitions of 32 tiles (2048), the last
    # one running 184 columns past K into the loads' zero fill.
    WorkspaceCase(
        "argument-k-tail-bf16",
        (256, 8008, 256),
        (128, 128, 64),
        "cute_split_k_partitions",
        4,
        4,
        kind="argument",
        dtype=torch.bfloat16,
        padding=8,
    ),
    WorkspaceCase(
        "direct-three-partitions",
        (128, 192, 128),
        (128, 128, 64),
        "tile_extent",
        64,
        2,
        kind="direct",
    ),
    WorkspaceCase(
        "matrix-bias-three-partitions",
        (128, 192, 128),
        (128, 128, 64),
        "tile_extent",
        64,
        2,
        kind="matrix",
        bias=True,
        padding=8,
    ),
)


def workspace_case(case: WorkspaceCase, device: str):
    """Shared CPU-preflight/GPU input construction, including aligned views."""
    m, k, n = case.shape
    offset = case.padding
    a = (
        torch.empty((m, k + offset), dtype=case.dtype, device=device)[:, offset:]
        if case.a_major == "k"
        else torch.empty((k, m + offset), dtype=case.dtype, device=device)[:, offset:].T
    )
    b = (
        torch.empty((k, n + offset), dtype=case.dtype, device=device)[:, offset:]
        if case.b_major == "n"
        else torch.empty((n, k + offset), dtype=case.dtype, device=device)[:, offset:].T
    )
    if case.kind == "matrix":
        bias = torch.empty((m, n * 2), dtype=case.dtype, device=device)[:, ::2]
        kernel, arguments = _matrix_bias, (a, b, bias)
    else:
        bias = torch.empty((n * 3,), dtype=case.dtype, device=device)[::3]
        if case.kind == "fp32":
            kernel, arguments = _fp32_bias, (a, b, bias)
        elif case.kind == "store":
            kernel, arguments = (
                (_plain_store_bias_matmul, (a, b, bias))
                if case.bias
                else (_plain_store_matmul, (a, b))
            )
        elif case.kind == "argument":
            out = torch.empty((m, n), dtype=case.dtype, device=device)
            kernel, arguments = _plain_store_into_out0, (a, b, out)
        elif case.kind == "transposed":
            out = torch.empty((m, n), dtype=case.dtype, device=device)
            kernel, arguments = _plain_store_into_out0, (b.T, a.T, out.T)
        elif case.kind == "direct":
            kernel, arguments = _direct_chunk_matmul, (a, b)
        else:
            kernel = helion.kernel(
                matmul_split_k.fn,
                backend="cute",
                static_shapes=True,
                autotune_effort="none",
            )
            arguments = (
                (a, b, lambda acc, tile: acc + bias[tile[1]]) if case.bias else (a, b)
            )
    config = helion.Config.from_dict(
        {
            "block_sizes": list(case.tile),
            case.parameter: case.value,
            "cute_split_k_workspace": True,
            "cute_split_k_stages": case.stages,
        }
    )
    return kernel._bind_isolated(arguments), config, arguments, (a, b, bias)


def ordered_reduction(
    partials: torch.Tensor, bias: torch.Tensor | None, dtype: torch.dtype
) -> torch.Tensor:
    """The emitted sum of the partitions in order, the bias added to partition
    zero first."""
    total = partials[0]
    if bias is not None:
        total = total + bias.float()
    for partition in range(1, partials.size(0)):
        total = total + partials[partition]
    return total.to(dtype)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("case", WORKSPACE_CASES, ids=lambda case: case.name)
def test_native_split_k_workspace_partials_and_poisoned_graph(case: WorkspaceCase):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(2026091371)
    bound, config, arguments, (a, b, bias) = workspace_case(case, "cuda")
    code = bound.to_code(config)
    assert "_helion_cute_split_k_workspace" in code
    bound.set_config(config)
    partial_buffers: list[torch.Tensor] = []
    original_empty = torch.empty

    def capture_empty(*args: Any, **kwargs: Any):
        result = original_empty(*args, **kwargs)
        if result.ndim == 3 and result.dtype == torch.float32:
            partial_buffers.append(result)
        return result

    def run():
        before = len(partial_buffers)
        result = bound(*arguments)
        assert len(partial_buffers) == before + 1
        if case.kind in ("argument", "transposed"):
            # The kernel returns nothing; the reducer wrote its out0 (out.T
            # for the transposed product).
            assert result is None
            result = arguments[2] if case.kind == "argument" else arguments[2].T
        return result, partial_buffers[-1]

    def check(result, partials, *, integer_inputs=False):
        p = partials.size(0)
        k = a.size(1)
        # A plain store's partitions run whole K tiles, the last one past K
        # where they do not cut it exactly (``plain_store_chunk``).
        chunk = (
            plain_store_chunk(k, p, case.tile[2])
            if case.kind in ("store", "argument", "transposed")
            else k // p
        )
        assert chunk is not None
        expected_partials = torch.stack(
            [
                a[:, i * chunk : (i + 1) * chunk].float()
                @ b[i * chunk : (i + 1) * chunk].float()
                for i in range(p)
            ]
        )
        torch.testing.assert_close(
            partials,
            expected_partials,
            atol=0 if integer_inputs else 1e-3,
            rtol=0 if integer_inputs else 1e-3,
        )
        torch.testing.assert_close(
            result,
            ordered_reduction(partials, bias if case.bias else None, result.dtype),
            atol=0,
            rtol=0,
        )
        reference = a.float() @ b.float()
        if case.bias:
            reference = reference + bias.float()
        # The archived example's unchanged numerical contract.
        torch.testing.assert_close(
            result, reference.to(result.dtype), atol=1, rtol=1e-2
        )

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with patch("torch.empty", side_effect=capture_empty):
            # Integer products and sums are exactly representable in FP32;
            # checking every partial detects lost/duplicated partition ownership.
            a.random_(-2, 3)
            b.random_(-2, 3)
            bias.random_(-2, 3)
            eager, partials = run()
            check(eager, partials, integer_inputs=True)
            replay = _make_cudagraph_replay(run)
            for _iteration in range(4):
                a.normal_()
                b.normal_()
                bias.normal_()
                eager, partials = run()
                check(eager, partials)
                # Sixteen replays per mutation: 64 poisoned graph executions.
                for _repeat in range(16):
                    result, partials = replay()
                    check(result, partials)
                    result.fill_(float("nan"))
                    partials.fill_(float("nan"))
            a.zero_()
            b.zero_()
            bias.zero_()
            result, partials = replay()
            torch.testing.assert_close(result, torch.zeros_like(result), atol=0, rtol=0)
            torch.testing.assert_close(
                partials, torch.zeros_like(partials), atol=0, rtol=0
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("transposed", (False, True))
def test_sixteen_bit_partials_reduce_in_fp32(transposed: bool) -> None:
    # bf16 256 x 8192 x 512 in four partitions with bf16 partials, as written
    # and as the transposed product (out.T = b.T @ a.T into the column-major
    # out.T): each partial is its fp32 chunk product rounded to bf16, and
    # the output is the fp32 sum of those partials, in partition order,
    # rounded once.
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(2026112400)
    m, k, n = 256, 8192, 512
    a = torch.randn((m, k), dtype=torch.bfloat16, device=DEVICE)
    b = torch.randn((k, n), dtype=torch.bfloat16, device=DEVICE)
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=DEVICE)
    arguments = (b.T, a.T, out.T) if transposed else (a, b, out)
    bound = _plain_store_into_out0_output_partials._bind_isolated(arguments)
    config = helion.Config.from_dict(
        {
            "block_sizes": [128, 128, 64],
            "cute_split_k_workspace": True,
            "cute_split_k_partitions": 4,
            "cute_split_k_stages": 4,
            "cute_split_k_partials": "output",
        }
    )
    assert "dtype=torch.bfloat16, device=" in bound.to_code(config)
    bound.set_config(config)
    captured: list[torch.Tensor] = []
    original_empty = torch.empty

    def capture_empty(*args: Any, **kwargs: Any):
        result = original_empty(*args, **kwargs)
        if result.ndim == 3:
            captured.append(result)
        return result

    with patch("torch.empty", side_effect=capture_empty):
        bound(*arguments)
    (partials,) = captured
    assert partials.dtype == torch.bfloat16
    chunk = k // 4
    expected = torch.stack(
        [
            (
                a[:, i * chunk : (i + 1) * chunk].float()
                @ b[i * chunk : (i + 1) * chunk].float()
            )
            for i in range(4)
        ]
    )
    torch.testing.assert_close(partials.float(), expected, atol=2e-1, rtol=1e-2)
    torch.testing.assert_close(
        out, ordered_reduction(partials.float(), None, out.dtype), atol=0, rtol=0
    )
    torch.testing.assert_close(
        out, (a.float() @ b.float()).to(out.dtype), atol=1, rtol=1e-2
    )


ATOMIC_EPILOGUE_CASES = (
    # The s2 catalogue winner's geometry: one 64x64 epilogue subtile per CTA.
    WorkspaceCase(
        "atomic-fp16-bias",
        (128, 32768, 256),
        (128, 64, 128),
        "split_k",
        16,
        6,
        bias=True,
    ),
    WorkspaceCase(
        "atomic-fp16-plain", (128, 32768, 256), (128, 64, 128), "split_k", 16, 6
    ),
    # 64x128 CTA tiles: two subtiles whose columns interleave (the (32, 2)
    # epilogue tiler), the exactness case below checks every output once.
    WorkspaceCase(
        "atomic-fp16-n128-bias",
        (128, 32768, 256),
        (128, 128, 64),
        "split_k",
        16,
        12,
        bias=True,
    ),
    WorkspaceCase(
        "atomic-fp16-exact",
        (128, 2048, 256),
        (128, 128, 64),
        "split_k",
        4,
        12,
        bias=True,
    ),
    WorkspaceCase(
        "atomic-bf16-plain",
        (128, 4096, 256),
        (128, 64, 64),
        "split_k",
        16,
        16,
        dtype=torch.bfloat16,
    ),
    # fp32 output (``red.global.add.v2.f32``), 128-row CTA tiles, padded bias.
    WorkspaceCase(
        "atomic-fp32-bias",
        (256, 4096, 256),
        (256, 128, 128),
        "partitions",
        8,
        2,
        kind="fp32",
        bias=True,
        padding=8,
    ),
    # A persistent grid with several tiles per CTA pair (the staging buffers
    # alternate under the atomics).
    WorkspaceCase(
        "atomic-fp16-persistent",
        (512, 2048, 512),
        (128, 128, 64),
        "split_k",
        16,
        2,
        bias=True,
    ),
)


def _rounded_partition_reference(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None, partitions: int
) -> torch.Tensor:
    """What the source's ``hl.atomic_add`` computes when every partial is
    rounded to the output dtype on its way in (exact for small integers)."""
    m, k = a.shape
    chunk = k // partitions
    total = torch.zeros((m, b.size(1)), dtype=torch.float32, device=a.device)
    for partition in range(partitions):
        partial = (
            a[:, partition * chunk : (partition + 1) * chunk].float()
            @ b[partition * chunk : (partition + 1) * chunk].float()
        )
        if partition == 0 and bias is not None:
            partial = partial + bias.float()
        total = total + partial.to(a.dtype).float()
    return total.to(a.dtype)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("case", ATOMIC_EPILOGUE_CASES, ids=lambda case: case.name)
def test_native_split_k_atomic_epilogue_reduces_into_the_filled_output(
    case: WorkspaceCase,
):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(2026103101)
    bound, config, arguments, (a, b, bias) = workspace_case(case, "cuda")
    config = helion.Config.from_dict(
        config.config | {"cute_split_k_atomic_epilogue": True}
    )
    code = bound.to_code(config)
    assert "'epilogue': 'atomic'" in code
    assert "_reduce_partitions" not in code
    assert code.count("PyCodeCache.load(") == 1
    assert (
        "_helion_zero_fill_split_k" in code
        and "torch.zeros" not in code.split("def ")[-1].split("return")[0]
    )
    bound.set_config(config)
    output_dtype = torch.float32 if case.kind == "fp32" else case.dtype
    partitions = case.value

    def run():
        return bound(*arguments)

    def check(result, *, exact: bool):
        assert result.dtype is output_dtype
        reference = a.float() @ b.float()
        if case.bias:
            reference = reference + bias.float()
        if exact:
            torch.testing.assert_close(
                result,
                _rounded_partition_reference(
                    a, b, bias if case.bias else None, partitions
                ).to(output_dtype),
                atol=0,
                rtol=0,
            )
        # The example's numerical contract (atol=1) for the half outputs, a
        # per-partition rounding error budget for bfloat16.
        tolerance = {
            torch.float16: {"atol": 1, "rtol": 1e-2},
            torch.bfloat16: {"atol": 4, "rtol": 5e-2},
            torch.float32: {"atol": 1e-2, "rtol": 1e-3},
        }[output_dtype]
        torch.testing.assert_close(result, reference.to(output_dtype), **tolerance)

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        # Small integers: every partial and every partial sum is exact in the
        # output dtype (|values| <= chunk <= 2048 and sums <= K <= 2048 for the
        # exact case), so the result is order independent.
        a.random_(-1, 2)
        b.random_(-1, 2)
        bias.random_(-1, 2)
        check(run(), exact=case.name == "atomic-fp16-exact")
        replay = _make_cudagraph_replay(run)
        for _iteration in range(3):
            a.normal_()
            b.normal_()
            bias.normal_()
            check(run(), exact=False)
            for _repeat in range(8):
                result = replay()
                check(result, exact=False)
                # The fill kernel must zero the output on every replay.
                result.fill_(float("nan"))
        a.zero_()
        b.zero_()
        bias.zero_()
        result = replay()
        torch.testing.assert_close(result, torch.zeros_like(result), atol=0, rtol=0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


FALLBACK_SHAPES = ((32, 4096, 64), (64, 16384, 128), (65, 530, 71), (128, 0, 128))


def fallback_case(shape, device):
    m, k, n = shape
    a = torch.empty((m, k), dtype=torch.float16, device=device)
    b = torch.empty((k, n), dtype=torch.float16, device=device)
    # The fallback under test is the fp32-promoted form of the float16 atomic
    # output (the Setting defaults to accumulating in float16 in place).
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_promote_half_atomic_outputs=True,
    )
    bound = kernel._bind_isolated((a, b))
    assert not bound.config_spec.cute_split_k_workspace_available
    config = helion.Config(
        block_sizes=[32, 64, 32],
        split_k=4,
        num_threads=[2, 64, 1],
        cute_collective_mma=k != 0,
        cute_collective_compute="warp",
        cute_collective_copy="async_cached",
        cute_collective_stages=4 if k else 1,
    )
    return bound, config, (a, b)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shape", FALLBACK_SHAPES)
def test_split_k_workspace_fallback_keeps_fp32_and_resets_graph(shape):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(2026091372)
    bound, config, (a, b) = fallback_case(shape, "cuda")
    source = bound.to_code(config)
    assert "_helion_cute_split_k_workspace" not in source
    assert "dtype=torch.float32" in source and "return out.to(torch.float16)" in source
    bound.set_config(config)
    a.normal_()
    b.normal_()
    replay = _make_cudagraph_replay(lambda: bound(a, b))
    for _iteration in range(4):
        a.normal_()
        b.normal_()
        expected = (a.float() @ b.float()).half()
        result = replay()
        torch.testing.assert_close(result, expected, atol=1, rtol=1e-2)
        result.fill_(float("nan"))
    a.zero_()
    b.zero_()
    result = replay()
    torch.testing.assert_close(result, torch.zeros_like(result), atol=0, rtol=0)


@pytest.mark.parametrize("atomic", [False, True])
def test_workspace_launches_preserve_compile_options(atomic: bool) -> None:
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_promote_half_atomic_outputs=False,
    )
    bound = _bind(kernel, (*_args((128, 32768, 256)), lambda acc, tile: acc))
    source = bound.to_code(
        helion.Config(
            block_sizes=[128, 64, 128],
            split_k=16,
            cute_split_k_workspace=True,
            cute_split_k_stages=6,
            cute_split_k_atomic_epilogue=atomic,
            advanced_controls_file="compile-controls.txt",
        )
    )
    trees = [ast.parse(code) for code in [source, *_embedded_sources(source)]]
    launches = [
        node
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    ]
    assert len(launches) == 2
    for launch in launches:
        options = [
            ast.literal_eval(keyword.value)
            for keyword in launch.keywords
            if keyword.arg == "cute_compile_options"
        ]
        assert options == ["--ptxas-options='--apply-controls compile-controls.txt'"]
