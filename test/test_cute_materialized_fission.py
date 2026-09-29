from __future__ import annotations

import ast
from contextlib import ExitStack
import dataclasses
import hashlib
import inspect
import operator
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import Mock
from unittest.mock import patch

from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_fwd
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import with_flat_min_blocks_default

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.mma_support import CuteMmaSupport
from helion._compiler.cute.promote_output_axis import _access_tensor
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.precompile_future import SerializedCompiledFunction
from helion.autotuner.precompile_future import _load_compiled_fn
from helion.autotuner.precompile_future import _serialize_compiled_fn
from helion.autotuner.precompile_future import _unload_compiled_fn
import helion.language as hl
from helion.language import memory_ops
from helion.runtime.ref_mode import RefMode
from helion.runtime.ref_mode import RefModeContext
from helion.runtime.ref_mode import RefModeTorchFunctionMode
from helion.runtime.settings import Settings

CPU_DEVICE = "cpu"

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path
    from typing import Any

    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


_allocation_calls: list[int] = []


def _opaque_allocation(x: torch.Tensor) -> torch.Tensor:
    _allocation_calls.append(1)
    return torch.empty_like(x)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _opaque_host_callback(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    middle = _opaque_allocation(x)
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        for column in hl.tile(columns):
            middle[row, column] = x[row, column]
        for column in hl.tile(columns):
            out[row, column] = middle[row, column]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _pointwise_pair(
    x: torch.Tensor, scale: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = x.shape
    middle = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        for column in hl.tile(columns):
            middle[row, column] = x[row, column] + 1
        for column in hl.tile(columns):
            out[row, column] = middle[row, column] * scale
    return out, middle


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _reload_feeds_dot(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    middle = torch.empty_like(x)
    materialized = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        for column in hl.tile(columns):
            middle[row, column] = x[row, column]
        for column in hl.tile(columns):
            materialized[row, column] = middle[row, column] / middle[row, column]
            out[row, column] = materialized[row, column] @ w[column, column]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _reload_feeds_dot_after_pointwise(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    middle = torch.empty_like(x)
    materialized = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        for column in hl.tile(columns):
            middle[row, column] = x[row, column]
        for column in hl.tile(columns):
            materialized[row, column] = middle[row, column] / middle[row, column]
            out[row, column] = (
                materialized[row, column].float() @ w[column, column].float()
            )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _matmul_fanout(
    x: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, reduction = x.shape
    columns = w.size(1)
    first = torch.empty((rows, columns), dtype=x.dtype, device=x.device)
    second = torch.empty_like(first)
    for row, column in hl.tile((rows, columns)):
        acc = hl.zeros((row, column), dtype=torch.float32)
        for k in hl.tile(reduction):
            acc = torch.addmm(acc, x[row, k], w[k, column])
        first[row, column] = acc
        second[row, column] = acc * 2
    return first, second


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rebound_scalar_input(x: torch.Tensor, scale: int = 2) -> torch.Tensor:
    scale = scale + 1
    rows, columns = x.shape
    middle = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        for column in hl.tile(columns):
            middle[row, column] = x[row, column]
        for column in hl.tile(columns):
            out[row, column] = middle[row, column] * scale
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rebound_destructured_input(x: torch.Tensor, scale: int = 2) -> torch.Tensor:
    scale, rows, columns = scale + 1, x.size(0), x.size(1)
    middle = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        for column in hl.tile(columns):
            middle[row, column] = x[row, column]
        for column in hl.tile(columns):
            out[row, column] = middle[row, column] * scale
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rebound_capture(x: torch.Tensor, scale: int = 2) -> torch.Tensor:
    _dead = torch.empty_like(x)
    _dead = scale + 1
    rows, columns = x.shape
    middle = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        for column in hl.tile(columns):
            middle[row, column] = x[row, column]
        for column in hl.tile(columns):
            out[row, column] = middle[row, column] * scale
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rebound_destructured_capture(x: torch.Tensor, scale: int = 2) -> torch.Tensor:
    _dead = torch.empty_like(x)
    _dead, rows, columns = scale + 1, x.size(0), x.size(1)
    middle = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        for column in hl.tile(columns):
            middle[row, column] = x[row, column]
        for column in hl.tile(columns):
            out[row, column] = middle[row, column] * scale
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sequence_row_binding(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    middle = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile([rows]):
        for column in hl.tile(columns):
            middle[row[0], column] = x[row[0], column]
        for column in hl.tile(columns):
            out[row[0], column] = middle[row[0], column]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sequence_column_binding(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    middle = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        for column in hl.tile([columns]):
            middle[row, column[0]] = x[row, column[0]]
        for column in hl.tile([columns]):
            out[row, column[0]] = middle[row, column[0]]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _cross_row(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    middle = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        for column in hl.tile(columns):
            middle[row, column] = x[row, column]
        for column in hl.tile(columns):
            out[row, column] = middle[(row.index + 1) % rows, column]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _partial_producer(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    middle = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        for column in hl.tile(columns // 2):
            middle[row, column] = x[row, column]
        for column in hl.tile(columns):
            out[row, column] = middle[row, column]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _alias_view(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    middle = torch.empty_like(x)
    view = torch.as_strided(middle, x.shape, [0, 1])
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        for column in hl.tile(columns):
            middle[row, column] = x[row, column]
        for column in hl.tile(columns):
            out[row, column] = view[row, column]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _write_input(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        for column in hl.tile(columns):
            x[row, column] = x[row, column] + 1
        for column in hl.tile(columns):
            out[row, column] = x[row, column]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _same_region_full_row(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    middle = torch.empty_like(x)
    second = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        for column in hl.tile(columns):
            middle[row, column] = x[row, column]
        for column in hl.tile(columns):
            second[row, column] = middle[row, column] + 1
            out[row, column] = second[row, :].sum(dim=-1)[:, None]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _read_before_producer(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    middle = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        for column in hl.tile(columns):
            out[row, column] = middle[row, column]
        for column in hl.tile(columns):
            middle[row, column] = x[row, column]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _escaped_register(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    middle = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        value = hl.zeros([row], dtype=x.dtype)
        for column in hl.tile(columns):
            middle[row, column] = x[row, column]
            value += x[row, column].sum(dim=-1)
        for column in hl.tile(columns):
            out[row, column] = middle[row, column] + value[:, None]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _conditional_write(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    middle = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        for column in hl.tile(columns):
            if row.begin > 0:
                middle[row, column] = x[row, column]
        for column in hl.tile(columns):
            out[row, column] = middle[row, column]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_flip(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    middle = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        for column in hl.tile(columns):
            middle[row, column] = torch.flip(x[row, column], [0])
        for column in hl.tile(columns):
            out[row, column] = middle[row, column]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _tile_boundary_value(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    middle = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        for column in hl.tile(columns):
            middle[row, column] = x[row, column] + row.begin
        for column in hl.tile(columns):
            out[row, column] = middle[row, column]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _masked_producer(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    middle = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        for column in hl.tile(columns):
            hl.store(
                middle,
                [row, column],
                x[row, column],
                extra_mask=column.index < columns // 2,
            )
        for column in hl.tile(columns):
            out[row, column] = middle[row, column]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _transposed_row_read(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    middle = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        for column in hl.tile(columns):
            middle[row, column] = x[column, row]
        for column in hl.tile(columns):
            out[row, column] = middle[row, column]
    return out


def _cpu_target() -> ExitStack:
    stack = ExitStack()
    for name, value in (
        ("helion.runtime.kernel.target_device_capability", (10, 0)),
        ("helion._compiler.compile_environment.target_device_capability", (10, 0)),
        ("helion.runtime.get_num_sm", 148),
        ("helion._compat._is_hip", False),
    ):
        stack.enter_context(patch(name, return_value=value))
    return stack


@pytest.fixture(autouse=True)
def _cpu_only() -> Generator[None, None, None]:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    # Hardware specialization also runs for CPU arguments. Keep these target
    # mocks independent of whether this pytest worker can see a real GPU.
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU use forbidden")),
    ):
        yield
    torch.set_num_threads(previous_threads)


def _bind(
    kernel: Kernel[Any], *args: object, enabled: bool = True, static_shapes: bool = True
) -> BoundKernel[Any]:
    kernel = helion.kernel(
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
        cute_region_fission=enabled,
        # This fixture isolates fission's original axis and wrapper contracts.
        # test_cute_full_slice_matmul covers composition with contraction tiling.
        cute_full_slice_matmul_tiling=False,
    )(kernel.fn)
    with _cpu_target():
        return kernel._bind_isolated(tuple(args))


def _se_args(
    shape: tuple[int, int, int] = (32, 64, 16), dtype: torch.dtype = torch.bfloat16
) -> tuple[torch.Tensor, ...]:
    m, n, k = shape
    generator = torch.Generator(device=CPU_DEVICE).manual_seed(27)
    return tuple(
        torch.randn(dimensions, generator=generator, dtype=dtype) * 0.25
        for dimensions in ((m, n), (n, k), (k, n))
    )


def _sources(source: str) -> list[str]:
    module = ast.parse(source)
    return [
        cast("str", node.value)
        for stmt in module.body
        if isinstance(stmt, ast.Assign)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "load"
        for node in stmt.value.args
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


class _HostModules:
    """Load actual generated host code, replacing device functions with tokens."""

    def __init__(self) -> None:
        self.sources: list[str] = []
        self.markers: list[SimpleNamespace] = []

    def load(self, source: str, extra: str = "") -> SimpleNamespace:
        self.sources.append(source)
        module = ast.parse(source)
        namespace: dict[str, object] = {
            "torch": torch,
            "operator": operator,
            "_default_cute_launcher": None,
            "PyCodeCache": self,
            "__file__": "<cpu-fission-test>",
        }
        stage_hosts = [
            stmt
            for stmt in module.body
            if isinstance(stmt, ast.FunctionDef)
            and "__region_" in stmt.name
            and not stmt.name.startswith("_helion_")
        ]
        if stage_hosts:
            assert len(stage_hosts) == 1
            host = stage_hosts[0]
            marker = SimpleNamespace(index=len(self.markers))
            self.markers.append(marker)
            namespace[f"_helion_{host.name}"] = marker
            statements = [host]
        else:
            statements = [
                stmt
                for stmt in module.body
                if not isinstance(stmt, (ast.Import, ast.ImportFrom))
            ]
        statements.insert(
            0,
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])),
                "<cpu-fission-test>",
                "exec",
            ),
            namespace,
        )
        return SimpleNamespace(**namespace)


class _CpuRefMode(RefModeTorchFunctionMode):
    def _handle_mm_with_bias(
        self,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        mm_func: object,
        op_name: str,
    ) -> torch.Tensor:
        # CPU torch.mm does not implement out_dtype; preserve the FP32
        # accumulator explicitly for the reference path only.
        assert op_name == "addmm"
        bias, lhs, rhs = (cast("torch.Tensor", value) for value in args[:3])
        assert bias.dtype is torch.float32
        assert bias.device.type == lhs.device.type == rhs.device.type == "cpu"
        return (
            cast("float", kwargs.get("alpha", 1)) * (lhs.float() @ rhs.float())
            + cast("float", kwargs.get("beta", 1)) * bias
        )


def _run_reference(
    bound: BoundKernel[Any], args: tuple[object, ...], transformed: bool
) -> object:
    if transformed:
        with bound.env, bound.host_function:
            definition = bound.host_function.codegen_function_def(
                bound.host_function.body
            )
        definition = cast(
            "ast.FunctionDef",
            ast.parse(ast.unparse(ast.fix_missing_locations(definition))).body[0],
        )
        # Use different row and column tile sizes in the two launch domains.
        roots = [stmt for stmt in definition.body if isinstance(stmt, ast.For)]
        for root, blocks in zip(roots, ((8, 8), (16, 16)), strict=True):
            assert isinstance(root.iter, ast.Call)
            root.iter.keywords.append(
                ast.keyword(
                    arg="block_size",
                    value=ast.List(
                        elts=[ast.Constant(value=value) for value in blocks],
                        ctx=ast.Load(),
                    ),
                )
            )
        namespace = dict(bound.kernel.fn.__globals__)
        namespace["_default_cute_launcher"] = None
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[definition], type_ignores=[])
                ),
                "<fission-reference>",
                "exec",
            ),
            namespace,
        )
        fn = namespace[definition.name]
    else:
        fn = bound.kernel.fn
    with _cpu_target():
        env = CompileEnvironment(
            torch.device("cpu"),
            dataclasses.replace(bound.settings, ref_mode=RefMode.EAGER),
        )
    context = RefModeContext(env, None)
    context.func_mode = _CpuRefMode()
    with context:
        return fn(*args)


@pytest.mark.parametrize("static_shapes", [False, True])
@skipUnlessBackends(["cute"])
def test_fission_gives_each_stage_independent_axis_and_config_slots(
    static_shapes: bool,
) -> None:
    bound = _bind(
        squeeze_and_excitation_net_fwd, *_se_args(), static_shapes=static_shapes
    )
    assert bound.env.cute_fission_plan is not None
    ir = bound.host_function.device_ir
    assert ir.grid_block_ids == [[0, 1], [3, 4]]
    assert [
        graph.block_ids for graph in ir.graphs if isinstance(graph, ForLoopGraphInfo)
    ] == [[5]]
    assert ir.tile_dependency_graph is None
    assert {"flat", "persistent_blocked", "persistent_interleaved", "xyz"} <= set(
        bound.config_spec.allowed_pid_types
    )
    first = bound._normalized_config_copy(
        helion.Config(block_sizes=[32, 16, 16, 32, 16], loop_orders=[[0, 1], [1, 0]])
    )
    second = bound._normalized_config_copy(
        helion.Config(block_sizes=[16, 16, 16, 32, 16], loop_orders=[[1, 0], [1, 0]])
    )
    assert first.block_sizes[2:] == second.block_sizes[2:]
    assert first.block_sizes[:2] != second.block_sizes[:2]
    with bound.env, bound.host_function:
        generation = ConfigGeneration(bound.config_spec)
        first = helion.Config(
            **(bound.config_spec.default_config().config | first.config)
        )
        assert generation.unflatten(
            generation.flatten(first)
        ) == with_flat_min_blocks_default(bound.config_spec, first)
    assert tuple(bound.kernel.signature.parameters) == ("x", "a", "b")
    assert "cute_region_fission=True" in bound.format_kernel_decorator(
        first, bound.settings
    )
    original = _bind(
        squeeze_and_excitation_net_fwd,
        *_se_args(),
        enabled=False,
        static_shapes=static_shapes,
    )
    assert (
        original.config_spec.cache_fingerprint_hash()
        != bound.config_spec.cache_fingerprint_hash()
    )


@pytest.mark.parametrize(
    "pid_type", ["flat", "persistent_blocked", "persistent_interleaved", "xyz"]
)
@skipUnlessBackends(["cute"])
def test_codegen_isolates_stages_and_inactive_reductions(pid_type: str) -> None:
    bound = _bind(squeeze_and_excitation_net_fwd, *_se_args())
    with _cpu_target():
        source = bound.to_code(
            helion.Config(block_sizes=[16, 16, 16, 32, 16], pid_type=pid_type)
        )
    stages = _sources(source)
    assert len(stages) == 2
    assert "_cute_grouped_reduce_shared_two_stage" in stages[0]
    assert "_cute_grouped_reduce_shared_two_stage" not in stages[1]
    assert "_BLOCK_SIZE_3" not in stages[0]
    assert "_BLOCK_SIZE_0" not in stages[1]
    for stage_source in stages:
        compile(stage_source, "<generated-stage>", "exec")
        assert "torch.empty(" not in stage_source
    assert "torch.empty(" in source


@skipUnlessBackends(["cute"])
def test_generated_wrapper_preserves_shared_allocations_order_and_compile_cache() -> (
    None
):
    args = _se_args()
    bound = _bind(squeeze_and_excitation_net_fwd, *args)
    modules = _HostModules()
    calls: list[tuple[SimpleNamespace, tuple[int, ...], tuple[object, ...]]] = []

    def launch(
        marker: SimpleNamespace,
        grid: tuple[int, ...],
        *values: object,
        **kwargs: object,
    ) -> None:
        calls.append((marker, grid, values))

    config = helion.Config(block_sizes=[16, 16, 16, 32, 16])
    with (
        _cpu_target(),
        patch("torch._inductor.codecache.PyCodeCache.load", side_effect=modules.load),
        patch.object(bound.env.backend, "setup_compile_cache_dir"),
    ):
        compiled = bound.compile_config(config, allow_print=False)
        assert bound.compile_config(config, allow_print=False) is compiled
        assert len(modules.sources) == 3
        out, c, d = compiled(*args, _launcher=launch)
        assert [call[0].index for call in calls] == [0, 1]
        assert [call[1] for call in calls] == [(2,), (4,)]
        assert calls[0][2] == (args[0], args[1], c)
        assert all(
            actual is expected
            for actual, expected in zip(
                calls[1][2], (c, args[2], d, args[0], out), strict=True
            )
        )
        calls.clear()
        next_out, next_c, next_d = compiled(*args, _launcher=launch)
        assert next_out is not out and next_c is not c and next_d is not d
        assert calls[0][2][-1] is next_c and calls[1][2][0] is next_c
        assert (
            bound.env.backend.generated_source_hash(compiled)
            == hashlib.sha256(modules.sources[0].encode()).hexdigest()
        )
        persisted = []
        for index, marker in enumerate(modules.markers):
            marker._helion_cute_compiled_launchers = {
                "launch": SimpleNamespace(
                    _cache_key=f"stage-{index}",
                    persist_compiled=Mock(
                        side_effect=lambda index=index: persisted.append(index)
                    ),
                )
            }
        key = bound.env.backend.compiled_cache_key(bound, compiled)
        assert isinstance(key, str)
        bound.env.backend.finalize_ephemeral_cache(
            bound, bound._normalized_config_copy(config)
        )
        assert persisted == [0, 1]
        modules.markers[1]._helion_cute_compiled_launchers[
            "launch"
        ]._cache_key = "stage-changed"
        assert bound.env.backend.compiled_cache_key(bound, compiled) != key


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(32, 64, 16), (13, 37, 19)])
@skipUnlessBackends(["cute"])
def test_cpu_reference_preserves_materialization_with_independent_tiles(
    dtype: torch.dtype, shape: tuple[int, int, int]
) -> None:
    args = _se_args(shape, dtype)
    bound = _bind(squeeze_and_excitation_net_fwd, *args)
    assert bound.env.cute_fission_plan is not None
    expected = _run_reference(bound, args, False)
    actual = _run_reference(bound, args, True)
    for value, reference in zip(actual, expected, strict=True):
        # Different CPU GEMM row/column tiles may change FP32 accumulation by
        # a few ulps. BF16 materialized values remain bit-identical here.
        tolerance = {} if dtype is torch.float32 else {"atol": 0, "rtol": 0}
        torch.testing.assert_close(value, reference, **tolerance)
    assert actual[0] is not actual[1] and actual[1] is not actual[2]


@skipUnlessBackends(["cute"])
def test_general_pointwise_program_and_default_arguments() -> None:
    x = torch.randn(13, 37)
    bound = _bind(_pointwise_pair, x, 2)
    assert bound.env.cute_fission_plan is not None
    out, middle = _run_reference(bound, (x, 2), True)
    torch.testing.assert_close(middle, x + 1)
    torch.testing.assert_close(out, (x + 1) * 2)
    with _cpu_target():
        source = bound.to_code(bound.config_spec.default_config())
    modules = _HostModules()
    module = modules.load(source)
    compiled = getattr(module, bound.kernel.name)
    assert inspect.signature(compiled).parameters["scale"].default == 2


@pytest.mark.parametrize(
    "kernel",
    [
        _rebound_scalar_input,
        _rebound_destructured_input,
        _rebound_capture,
        _rebound_destructured_capture,
        _sequence_row_binding,
        _sequence_column_binding,
        _cross_row,
        _partial_producer,
        _alias_view,
        _write_input,
        _same_region_full_row,
        _read_before_producer,
        _escaped_register,
        _conditional_write,
        _row_flip,
        _tile_boundary_value,
        _masked_producer,
    ],
)
@skipUnlessBackends(["cute"])
def test_rejects_unsafe_fission_and_leaves_original_structure(
    kernel: Kernel[Any],
) -> None:
    x = torch.empty(13, 37)
    original = _bind(kernel, x, enabled=False)
    candidate = _bind(kernel, x, enabled=True)
    assert candidate.env.cute_fission_plan is None
    assert (
        candidate.host_function.device_ir.grid_block_ids
        == original.host_function.device_ir.grid_block_ids
    )
    assert ast.dump(ast.Module(candidate.host_function.body, [])) == ast.dump(
        ast.Module(original.host_function.body, [])
    )


@skipUnlessBackends(["cute"])
def test_opt_in_and_reference_mode_do_not_run_analysis_by_default() -> None:
    assert not Settings(cute_region_fission=False).cute_region_fission
    with patch(
        "helion._compiler.cute.materialized_fission.plan_materialized_fission",
        side_effect=AssertionError("unexpected analysis"),
    ):
        bound = _bind(_pointwise_pair, torch.empty(13, 37), 2, enabled=False)
        assert bound.env.cute_fission_plan is None
        reference = helion.kernel(
            backend="cute", cute_region_fission=True, ref_mode=RefMode.EAGER
        )(_pointwise_pair.fn)
        with _cpu_target():
            reference._bind_isolated((torch.empty(13, 37), 2))


@pytest.mark.parametrize("region_enabled", [False, True])
@pytest.mark.parametrize("operand_enabled", [False, True])
@skipUnlessBackends(["cute"])
def test_planners_respect_independent_opt_outs(
    region_enabled: bool, operand_enabled: bool
) -> None:
    kernel = helion.kernel(
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_region_fission=region_enabled,
        cute_materialize_transformed_operands=operand_enabled,
    )(_pointwise_pair.fn)
    with (
        _cpu_target(),
        patch(
            "helion._compiler.cute.materialize_operand.plan_operand_materialization",
            return_value=None,
        ) as operand_planner,
        patch(
            "helion._compiler.cute.materialized_fission.plan_materialized_fission",
            return_value=None,
        ) as region_planner,
    ):
        bound = kernel._bind_isolated((torch.empty(13, 37), 2))
    assert bound.env.cute_fission_plan is None
    assert operand_planner.call_count == int(operand_enabled)
    assert region_planner.call_count == int(region_enabled)


@skipUnlessBackends(["cute"])
def test_second_gemm_exposes_tcgen05_search_after_fission() -> None:
    support = CuteMmaSupport(
        universal=True, warp_f16bf16=True, warpgroup_f16bf16=True, tcgen05_f16bf16=True
    )
    with patch(
        "helion._compiler.cute.mma_support.get_cute_mma_support", return_value=support
    ):
        bound = _bind(squeeze_and_excitation_net_fwd, *_se_args((128, 256, 64)))
    assert bound.env.cute_fission_plan is not None
    assert bound.config_spec.cute_tcgen05_search_enabled


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("shape", "pid_type"),
    [
        ((128, 256, 64), "flat"),
        ((73, 88, 56), "flat"),
        ((2048, 512, 64), "persistent_interleaved"),
    ],
)
@skipUnlessBackends(["cute"])
def test_tcgen05_codegen_preserves_second_stage_wrapper_metadata(
    shape: tuple[int, int, int], pid_type: str, dtype: torch.dtype
) -> None:
    support = CuteMmaSupport(
        universal=True, warp_f16bf16=True, warpgroup_f16bf16=True, tcgen05_f16bf16=True
    )
    with (
        _cpu_target(),
        patch(
            "helion._compiler.cute.mma_support.get_cute_mma_support",
            return_value=support,
        ),
        patch(
            "helion._compiler.cute.cute_mma.get_cute_mma_support", return_value=support
        ),
        patch(
            "helion._compiler.cute.tcgen05_config.CuteTcgen05Config.per_cta_smem_budget_bytes",
            return_value=232448,
        ),
    ):
        inputs = _se_args(shape, dtype)
        bound = _bind(squeeze_and_excitation_net_fwd, *inputs)
        config = bound.config_spec.default_config()
        if pid_type != "flat":
            config = helion.Config(
                **(
                    config.config
                    | {
                        "pid_type": pid_type,
                        "tcgen05_persistence_model": "static_persistent",
                    }
                )
            )
        code = bound.to_code(config)
    stages = _sources(code)
    assert "cute.gemm(" not in stages[0]
    assert "cute.gemm(" in stages[1]
    assert "tcgen05_ab_tma" in stages[1]
    plans = bound.env.cute_resolved_wrapper_plans
    assert (plans[0]["lhs_idx"], plans[0]["rhs_idx"]) == (0, 1)
    if shape != (73, 88, 56):
        assert [plan["kind"] for plan in plans] == [
            "tcgen05_ab_tma",
            "tcgen05_d_tma",
            "tcgen05_d_tma",
        ]
        assert (plans[1]["d_idx"], plans[2]["d_idx"]) == (2, 3)
        assert plans[1]["kernel_args"] != plans[2]["kernel_args"]
    else:
        assert [plan["kind"] for plan in plans] == ["tcgen05_ab_tma"]
        assert "tcgen05_tma_store_atom" not in stages[1]
    assert stages[1].count("tcgen05_acc_pipeline.consumer_wait(") == 1
    assert stages[1].count("tcgen05_acc_pipeline.consumer_release(") == 1
    assert stages[1].count("tcgen05_acc_consumer_state.advance()") == 1
    assert stages[1].count(".free(tcgen05_epi_acc_tmem_ptr)") == 1
    assert (
        stages[1].index("tcgen05_acc_pipeline.consumer_wait(")
        < stages[1].index("cute.copy(tcgen05_tiled_copy_t2r,")
        < stages[1].index("cute.copy(tcgen05_tiled_copy_t2r_1,")
        < stages[1].index("tcgen05_acc_pipeline.consumer_release(")
        < stages[1].index("tcgen05_acc_consumer_state.advance()")
    )
    if pid_type != "flat":
        assert "while tcgen05_role_local_0_work_tile.is_valid_tile:" in stages[1]
        assert "while tcgen05_role_local_2_work_tile.is_valid_tile:" in stages[1]
        assert stages[1].count("tcgen05_c_pipeline.producer_tail()") == 1
        assert stages[1].count("tcgen05_c_pipeline_1.producer_tail()") == 1
    assert "tcgen05_acc_stage_index" not in stages[1]
    dtype_name = "BFloat16" if dtype is torch.bfloat16 else "Float16"
    assert f".to(cutlass.{dtype_name}).to(cutlass.Float32)" in stages[1]
    assert "tcgen05_aux_loaded_0 * tcgen05_chain_step" in stages[1]
    assert "indices_3" not in stages[1] and "indices_4" not in stages[1]


@skipUnlessBackends(["cute"])
def test_forwarding_keeps_the_materialized_cast_and_store() -> None:
    bound = _bind(squeeze_and_excitation_net_fwd, *_se_args())
    ir = bound.host_function.device_ir
    root = ir.graphs[ir.root_ids[1]].graph
    stores = [node for node in root.nodes if node.target is memory_ops.store]
    assert len(stores) == 2
    materialized, output = stores
    rounded = materialized.args[2]
    assert isinstance(rounded, torch.fx.Node)
    assert rounded.target is torch.ops.prims.convert_element_type.default
    assert rounded.args[1] is torch.bfloat16
    product = output.args[2]
    assert isinstance(product, torch.fx.Node)
    assert rounded in product.all_input_nodes
    stored_tensor = _access_tensor(materialized)
    assert not any(
        node.target is memory_ops.load and _access_tensor(node) is stored_tensor
        for node in root.nodes
    )


@skipUnlessBackends(["cute"])
def test_paired_m_subtiles_reject_multioutput_lifecycle() -> None:
    support = CuteMmaSupport(
        universal=True, warp_f16bf16=True, warpgroup_f16bf16=True, tcgen05_f16bf16=True
    )
    inputs = (
        torch.empty((1024, 128), dtype=torch.bfloat16),
        torch.empty((128, 256), dtype=torch.bfloat16),
    )
    with (
        _cpu_target(),
        patch(
            "helion._compiler.cute.mma_support.get_cute_mma_support",
            return_value=support,
        ),
        patch(
            "helion._compiler.cute.cute_mma.get_cute_mma_support", return_value=support
        ),
        patch(
            "helion._compiler.cute.tcgen05_config.CuteTcgen05Config.per_cta_smem_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = _bind(_matmul_fanout, *inputs)
        config = helion.Config(
            **(
                bound.config_spec.default_config().config
                | {
                    "block_sizes": [512, 256, 128],
                    "pid_type": "persistent_interleaved",
                    "tcgen05_persistence_model": "static_persistent",
                    "tcgen05_cluster_m": 2,
                    "tcgen05_acc_stages": 2,
                }
            )
        )
        with pytest.raises(helion.exc.BackendUnsupported, match="paired M subtiles"):
            bound.to_code(config)


@pytest.mark.parametrize(
    "kernel", [_reload_feeds_dot, _reload_feeds_dot_after_pointwise]
)
@skipUnlessBackends(["cute"])
def test_forwarding_retains_zero_padded_reload_before_dot(kernel: Kernel[Any]) -> None:
    args = (
        torch.ones((13, 19), dtype=torch.float16),
        torch.ones((19, 19), dtype=torch.float16),
    )
    bound = _bind(kernel, *args)
    assert bound.env.cute_fission_plan is not None
    ir = bound.host_function.device_ir
    root = ir.graphs[ir.root_ids[1]].graph
    stores = [node for node in root.nodes if node.target is memory_ops.store]
    assert len(stores) == 2
    materialized = _access_tensor(stores[0])
    assert materialized is not None
    assert any(
        node.target is memory_ops.load and _access_tensor(node) is materialized
        for node in root.nodes
    )
    # The producer's invalid lanes evaluate 0/0. A fresh memory load supplies
    # zero instead, which is observable when a later dot reduces those lanes.
    padded = torch.tensor([1, 1, 1, 0], dtype=torch.float16)
    forwarded = padded / padded
    reloaded = torch.where(padded != 0, forwarded, 0)
    assert torch.isnan(forwarded @ padded)
    assert torch.isfinite(reloaded @ padded)


@skipUnlessBackends(["cute"])
def test_load_hints_keep_global_positions_across_stages() -> None:
    bound = _bind(squeeze_and_excitation_net_fwd, *_se_args(dtype=torch.float32))
    config = helion.Config(
        **(
            bound.config_spec.default_config().config
            | {"load_eviction_policies": ["streaming", "", "last", "", ""]}
        )
    )
    with _cpu_target():
        stages = _sources(bound.to_code(config))
    assert "cop='cs'" in stages[0]
    assert "evict_last" not in stages[0]
    assert "cop='cs'" not in stages[1]
    assert "evict_last" in stages[1]


@skipUnlessBackends(["cute"])
def test_subprocess_roundtrip_and_cleanup_cover_every_stage(tmp_path: Path) -> None:
    bound = _bind(_pointwise_pair, torch.empty(13, 37), 2)
    with _cpu_target():
        source = bound.to_code(bound.config_spec.default_config())
    filename = tmp_path / "fission_wrapper.py"
    filename.write_text(source)
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    spec = SerializedCompiledFunction(
        bound.kernel.name, source, str(filename), None, source_hash
    )
    modules = _HostModules()
    with patch("torch._inductor.codecache.PyCodeCache.load", side_effect=modules.load):
        compiled = _load_compiled_fn(spec)
    module_name = compiled.__module__
    assert bound.env.backend.generated_source_hash(compiled) == source_hash
    saved = _serialize_compiled_fn(compiled)
    assert saved.source_code == source and saved.source_hash == source_hash
    assert tuple(
        marker._helion_cute_source_hash for marker in modules.markers
    ) == tuple(hashlib.sha256(stage.encode()).hexdigest() for stage in _sources(source))
    for marker in modules.markers:
        marker.clear_fast_path_caches = Mock()
        marker._helion_cute_compiled_launchers = {0: SimpleNamespace()}
        marker._helion_cute_launch_arg_cache = {0: object()}
    _unload_compiled_fn(compiled)
    assert module_name not in sys.modules
    for marker in modules.markers:
        marker.clear_fast_path_caches.assert_called_once_with()
        assert not marker._helion_cute_compiled_launchers
        assert not marker._helion_cute_launch_arg_cache


@skipUnlessBackends(["cute"])
def test_analysis_does_not_repeat_opaque_host_callbacks() -> None:
    x = torch.empty(13, 37)
    _allocation_calls.clear()
    _bind(_opaque_host_callback, x, enabled=False)
    original_count = len(_allocation_calls)
    assert original_count > 0
    _allocation_calls.clear()
    candidate = _bind(_opaque_host_callback, x, enabled=True)
    assert candidate.env.cute_fission_plan is None
    assert len(_allocation_calls) == original_count


@skipUnlessBackends(["cute"])
def test_rejects_input_rows_broadcast_on_a_different_axis() -> None:
    bound = _bind(_transposed_row_read, torch.empty(32, 32))
    assert bound.env.cute_fission_plan is None


def test_single_kernel_cache_helpers_keep_legacy_behavior() -> None:
    backend = CuteBackend()
    kernel = SimpleNamespace(
        _helion_cute_compiled_launchers={0: SimpleNamespace(_cache_key="legacy-key")},
        _helion_cute_source_hash="legacy-source",
    )
    compiled = SimpleNamespace(
        __name__="sample", __globals__={"_helion_sample": kernel}
    )
    bound = SimpleNamespace(kernel=SimpleNamespace(name="sample"))
    assert backend.compiled_cache_key(bound, compiled) == "legacy-key"
    assert backend.generated_source_hash(compiled) == "legacy-source"
