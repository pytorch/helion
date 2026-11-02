from __future__ import annotations

import ast
from contextlib import ExitStack
import dataclasses
import sys
from types import FunctionType
from types import ModuleType
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_fwd
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.runtime.ref_mode import RefMode
from helion.runtime.ref_mode import RefModeContext
from helion.runtime.ref_mode import RefModeTorchFunctionMode

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator
    from typing import Any

    from helion._compiler.host_function import HostFunction
    from helion.runtime.kernel import BoundKernel


def _matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=torch.float32, device=a.device)
    for row, column in hl.tile([a.size(0), b.size(1)]):
        product = a[row, :] @ b[:, column]
        out[row, column] = product.float() * 1.003
    return out


def _partial(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=a.dtype, device=a.device)
    for row, column in hl.tile([a.size(0), b.size(1)]):
        product = a[row, :-1] @ b[:-1, column]
        out[row, column] = product
    return out


def _computed(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=a.dtype, device=a.device)
    for row, column in hl.tile([a.size(0), b.size(1)]):
        product = (a[row, :] + 1) @ b[:, column]
        out[row, column] = product
    return out


def _rebound(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a[:, :]
    out = torch.empty([a.size(0), b.size(1)], dtype=a.dtype, device=a.device)
    for row, column in hl.tile([a.size(0), b.size(1)]):
        product = a[row, :] @ b[:, column]
        out[row, column] = product
    return out


def _destructured_rebound(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, _unused = (a[:, :], 0)
    out = torch.empty([a.size(0), b.size(1)], dtype=a.dtype, device=a.device)
    for row, column in hl.tile([a.size(0), b.size(1)]):
        product = a[row, :] @ b[:, column]
        out[row, column] = product
    return out


def _alias(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    alias = a
    out = torch.empty([a.size(0), b.size(1)], dtype=a.dtype, device=a.device)
    for row, column in hl.tile([a.size(0), b.size(1)]):
        product = alias[row, :] @ b[:, column]
        out[row, column] = product
    return out


def _one_axis(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=a.dtype, device=a.device)
    for row in hl.tile(a.size(0)):
        for column in hl.tile(b.size(1)):
            product = a[row, :] @ b[:, column]
            out[row, column] = product
    return out


def _nested_consumer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=a.dtype, device=a.device)
    for row, column in hl.tile([a.size(0), b.size(1)]):
        out[row, column] = torch.relu(a[row, :] @ b[:, column])
    return out


def _row_alias(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=a.dtype, device=a.device)
    for row, column in hl.tile([a.size(0), b.size(1)]):
        x_tile = a[row, :]
        out[row, column] = x_tile @ b[:, column]
    return out


def _row_alias_reused(
    a: torch.Tensor, b: torch.Tensor, aux: torch.Tensor
) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=a.dtype, device=a.device)
    for row, column in hl.tile([a.size(0), b.size(1)]):
        x_tile = a[row, :]
        out[row, column] = x_tile @ b[:, column]
        aux[row, :] = x_tile * 2
    return out


def _single_axis(a: torch.Tensor, b: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=a.dtype, device=a.device)
    for row in hl.tile(a.size(0)):
        x_tile = a[row, :]
        activation = torch.sigmoid(x_tile @ b[:, :])
        gate[row, :] = activation
        out[row, :] = (2.0 * x_tile.to(torch.float32) * activation).to(a.dtype)
    return out


def _single_axis_direct(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n = a.size(0), b.size(1)
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for row in hl.tile(m):
        out[row, :] = torch.relu(a[row, :] @ b[:, :])
    return out


def _single_axis_reduction(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=torch.float32, device=a.device)
    for row in hl.tile(a.size(0)):
        product = a[row, :] @ b[:, :]
        out[row, :] = product.float() - product.float().amax(-1)[:, None]
    return out


def _single_axis_tile_index(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=a.dtype, device=a.device)
    for row in hl.tile(a.size(0)):
        x_tile = a[row, :]
        out[row, :] = (x_tile @ b[:, :]) * row.index[:, None]
    return out


def _single_axis_two_matmuls(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=a.dtype, device=a.device)
    for row in hl.tile(a.size(0)):
        out[row, :] = (a[row, :] @ b[:, :]) + (a[row, :] @ b[:, :])
    return out


def _single_axis_in_place(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    for row in hl.tile(a.size(0)):
        a[row, :] = torch.relu(a[row, :] @ b[:, :])
    return a


def _single_axis_rebound_alias(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=a.dtype, device=a.device)
    for row in hl.tile(a.size(0)):
        x_tile = a[row, :]
        x_tile = x_tile * 1
        out[row, :] = x_tile @ b[:, :]
    return out


def _single_axis_leading_store(
    a: torch.Tensor, b: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=a.dtype, device=a.device)
    for row in hl.tile(a.size(0)):
        x_tile = a[row, :]
        gate[row, :] = x_tile * 2
        out[row, :] = torch.relu(x_tile @ b[:, :])
    return out


def _single_axis_narrow_output(
    a: torch.Tensor, b: torch.Tensor, narrow: torch.Tensor
) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=a.dtype, device=a.device)
    for row in hl.tile(a.size(0)):
        product = a[row, :] @ b[:, :]
        out[row, :] = product
        narrow[row, :] = a[row, :]
    return out


def _single_axis_rebound_dimension(
    a: torch.Tensor, b: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # ``aux``/``aux2`` are [m, K]; ``n`` is rebound to N only afterwards.
    m, n = a.size()
    aux = torch.empty([m, n], dtype=a.dtype, device=a.device)
    aux2 = torch.zeros([m, n], dtype=a.dtype, device=a.device)
    n = gate.size(1)
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for row in hl.tile(m):
        out[row, :] = torch.relu(a[row, :] @ b[:, :])
        aux[row, :] = aux2[row, :] * 2
    return out, aux


def _single_axis_dimension_rebound_first(
    a: torch.Tensor, b: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    # ``n`` reads N by the time ``out`` and ``aux`` are allocated.
    m, n = a.size()
    n = b.size(1)
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    aux = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for row in hl.tile(m):
        out[row, :] = torch.relu(a[row, :] @ b[:, :])
        aux[row, :] = gate[row, :] * 2
    return out + aux


def _masked_consumer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=torch.float32, device=a.device)
    for row, column in hl.tile([a.size(0), b.size(1)]):
        out[row, column] = 9
        product = a[row, :] @ b[:, column]
        hl.store(
            out,
            [row, column],
            product.float() * 1.003,
            extra_mask=column.index[None, :] < b.size(1) // 2,
        )
    return out


def _two_matmuls(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=torch.float32, device=a.device)
    for row, column in hl.tile([a.size(0), b.size(1)]):
        first = a[row, :] @ b[:, column]
        out[row, column] = first.float()
        second = a[row, :] @ b[:, column]
        out[row, column] = out[row, column] + second.float()
    return out


array_lib = torch
blocks = hl


def _alternate_aliases(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = array_lib.empty([a.size(0), b.size(1)], dtype=a.dtype, device=a.device)
    for row, column in blocks.tile([a.size(0), b.size(1)]):
        _helion_fullslice_acc = 2
        _helion_fullslice_k = 3
        product = array_lib.mm(a[row, :], b[:, column])
        out[row, column] = product + _helion_fullslice_acc + _helion_fullslice_k
    return out


tile = hl.tile
empty = torch.empty
matrix = torch.mm


def _direct_imports(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = empty([a.size(0), b.size(1)], dtype=a.dtype, device=a.device)
    for row, column in tile([a.size(0), b.size(1)]):
        product = matrix(a[row, :], b[:, column])
        out[row, column] = product
    return out


def _cpu_target() -> ExitStack:
    stack = ExitStack()
    for name, value in (
        ("helion.runtime.kernel.target_device_capability", (10, 0)),
        ("helion._compiler.compile_environment.target_device_capability", (10, 0)),
        ("helion.runtime.get_num_sm", 148),
        ("helion._compat._is_hip", False),
        (
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            128 * 1024,
        ),
    ):
        stack.enter_context(patch(name, return_value=value))
    return stack


@pytest.fixture(autouse=True)
def _cpu_only() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    # Hardware specialization also runs for CPU arguments. Keep these target
    # mocks independent of whether this pytest worker can see a real GPU.
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield
    torch.set_num_threads(previous)


def _bind(
    function: Callable[..., object],
    args: tuple[torch.Tensor, ...],
    *,
    enabled: bool = True,
    fission: bool = False,
    static_shapes: bool = True,
    backend: str = "cute",
) -> BoundKernel[Any]:
    kernel = helion.kernel(
        backend=backend,
        static_shapes=static_shapes,
        autotune_effort="none",
        cute_full_slice_matmul_tiling=enabled,
        cute_region_fission=fission,
    )(function)
    with _cpu_target():
        return kernel._bind_isolated(args)


def _host(bound: BoundKernel[Any]) -> HostFunction:
    assert bound.host_function is not None
    return bound.host_function


def _inputs(
    shape: tuple[int, int, int] = (13, 37, 19),
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n, k = shape
    generator = torch.Generator().manual_seed(17)
    # Dyadic values make the reference sum exact in FP32 and isolate the
    # original output cast from changes in GEMM accumulation order.
    a, b = (
        torch.randint(-7, 8, size, generator=generator).to(dtype) / 8
        for size in ((m, k), (k, n))
    )
    return a, b


class _CpuRefMode(RefModeTorchFunctionMode):
    def _handle_mm_with_bias(
        self,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        mm_func: object,
        op_name: str,
    ) -> torch.Tensor:
        assert op_name == "addmm"
        accumulator, lhs, rhs = (cast("torch.Tensor", arg) for arg in args[:3])
        assert accumulator.dtype is torch.float32
        assert accumulator.device.type == lhs.device.type == rhs.device.type == "cpu"
        assert not kwargs
        return accumulator + lhs.float() @ rhs.float()


def _run_transformed_reference(
    bound: BoundKernel[Any], args: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    host = _host(bound)
    with bound.env, host:
        definition = host.codegen_function_def(list(host.body))
    definition = ast.parse(ast.unparse(ast.fix_missing_locations(definition))).body[0]
    assert isinstance(definition, ast.FunctionDef)
    for node in ast.walk(definition):
        if not isinstance(node, ast.For):
            continue
        assert isinstance(node.iter, ast.Call)
        if isinstance(node.target, (ast.Tuple, ast.List)):
            value = ast.List(elts=[ast.Constant(8), ast.Constant(16)], ctx=ast.Load())
        else:
            value = ast.Constant(8)
        node.iter.keywords.append(ast.keyword(arg="block_size", value=value))
    namespace = dict(bound.kernel.fn.__globals__)
    namespace["_default_cute_launcher"] = None
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])),
            "<stripmine-reference>",
            "exec",
        ),
        namespace,
    )
    with _cpu_target():
        env = CompileEnvironment(
            torch.device("cpu"),
            dataclasses.replace(bound.settings, ref_mode=RefMode.EAGER),
        )
    context = RefModeContext(env, None)
    context.func_mode = _CpuRefMode()
    with context:
        result = namespace[definition.name](*args)
    assert isinstance(result, torch.Tensor)
    return result


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_default_settings_preserve_declared_config_axes(
    dtype: torch.dtype, static_shapes: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "REGION_FISSION",
        "MATERIALIZE_TRANSFORMED_OPERANDS",
        "FULL_SLICE_MATMUL_TILING",
        "SEGMENTED_MATMUL_TILING",
        "FLATTEN_NESTED_REDUCTIONS",
    ):
        monkeypatch.delenv(f"HELION_CUTE_{name}", raising=False)
    kernel = helion.kernel(
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
        config=helion.Config(block_sizes=[16, 32]),
    )(_matmul)
    with _cpu_target():
        bound = kernel._bind_isolated(_inputs((32, 16, 64), dtype))
        assert len(bound.config_spec.block_sizes) == 2
        ast.parse(bound.to_code())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_explicit_k_has_normal_config_slots(dtype: torch.dtype) -> None:
    bound = _bind(_matmul, _inputs(dtype=dtype))
    ir = _host(bound).device_ir
    assert ir.grid_block_ids == [[0, 1]]
    assert [
        graph.block_ids for graph in ir.graphs if type(graph) is ForLoopGraphInfo
    ] == [[2]]
    assert [spec.block_ids for spec in bound.config_spec.block_sizes] == [[0], [1], [2]]
    assert bound.config_spec.matmul_facts[0].k_block_id == 2
    for bk in (16, 32, 64):
        with _cpu_target():
            code = bound.to_code(helion.Config(block_sizes=[16, 32, bk]))
        ast.parse(code)
        assert f"_BLOCK_SIZE_2 = {bk}" in code
        assert "for tile_offset_2 in range" in code
    assert "cute_full_slice_matmul_tiling=True" in bound.format_kernel_decorator(
        bound.config_spec.default_config(), bound.settings
    )


def test_unproved_dynamic_k_equality_keeps_original_ast() -> None:
    args = _inputs()
    before = _bind(_matmul, args, enabled=False, static_shapes=False)
    after = _bind(_matmul, args, static_shapes=False)
    assert ast.dump(ast.Module(_host(before).body, [])) == ast.dump(
        ast.Module(_host(after).body, [])
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(16, 32, 16), (13, 37, 19), (4, 6, 0)])
def test_reference_preserves_cast_epilogue_tails_and_empty_k(
    dtype: torch.dtype, shape: tuple[int, int, int]
) -> None:
    args = _inputs(shape, dtype)
    bound = _bind(_matmul, args)
    actual = _run_transformed_reference(bound, args)
    expected = (args[0].float() @ args[1].float()).to(dtype).float() * 1.003
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("aliased", [False, True])
def test_strided_and_aliased_input_views(aliased: bool) -> None:
    a, b = _inputs((17, 17, 17), torch.bfloat16)
    if aliased:
        b = a.t()
    else:
        a = a.t()
        b = b.t()
    args = (a, b)
    actual = _run_transformed_reference(_bind(_matmul, args), args)
    expected = (a.float() @ b.float()).bfloat16().float() * 1.003
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _with_aux(columns: int) -> Callable[[], tuple[torch.Tensor, ...]]:
    return lambda: (*_inputs(), torch.zeros(13, columns, dtype=torch.float16))


_UNPROVED: list[
    tuple[Callable[..., object], Callable[[], tuple[torch.Tensor, ...]]]
] = [
    (_partial, _inputs),
    (_computed, _inputs),
    (_rebound, _inputs),
    (_destructured_rebound, _inputs),
    (_alias, _inputs),
    (_one_axis, _inputs),
    # A row alias that stays live beside the new K loop would keep a
    # whole-row axis in the thread plan that no default config fits.
    (_row_alias_reused, lambda: (*_inputs((8, 8, 8)), torch.zeros(8, 8).half())),
    # Single-axis roots: the sliced axis is reduced, indexed, contracted
    # twice, stored through (an N tile would race on the operand rows), read
    # through a rebound alias, preceded by a non-alias statement, or stored
    # into a tensor whose dim-1 extent differs from N -- including one sized
    # by a dimension name that was rebound to N only after the allocation.
    (_single_axis_reduction, _inputs),
    (_single_axis_tile_index, _inputs),
    (_single_axis_two_matmuls, _inputs),
    (_single_axis_in_place, lambda: _inputs((16, 16, 16))),
    (_single_axis_rebound_alias, _inputs),
    (_single_axis_leading_store, _with_aux(19)),
    (_single_axis_narrow_output, _with_aux(19)),
    (_single_axis_rebound_dimension, _with_aux(37)),
]


@pytest.mark.parametrize(
    ("function", "make_args"), _UNPROVED, ids=[item[0].__name__ for item in _UNPROVED]
)
def test_unproved_patterns_keep_the_original_ast(
    function: Callable[..., object],
    make_args: Callable[[], tuple[torch.Tensor, ...]],
) -> None:
    args = make_args()
    original = _bind(function, args, enabled=False)
    candidate = _bind(function, args)
    assert ast.dump(ast.Module(_host(candidate).body, [])) == ast.dump(
        ast.Module(_host(original).body, [])
    )


def _source(bound: BoundKernel[Any]) -> str:
    return ast.unparse(ast.Module(_host(bound).body, []))


def test_nested_consumer_hoists_the_contraction() -> None:
    args = _inputs()
    bound = _bind(_nested_consumer, args)
    assert bound.config_spec.matmul_facts[0].k_block_id == 2
    text = _source(bound)
    assert "out[row, column] = torch.relu(_helion_fullslice_mm)" in text
    assert text.index("for _helion_fullslice_k in") < text.index("torch.relu(")
    actual = _run_transformed_reference(bound, args)
    expected = torch.relu((args[0].float() @ args[1].float()).half())
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_row_alias_operand_is_followed_and_consumed() -> None:
    args = _inputs()
    bound = _bind(_row_alias, args)
    assert bound.config_spec.matmul_facts[0].k_block_id == 2
    text = _source(bound)
    assert "a[row, _helion_fullslice_k]" in text
    assert "x_tile" not in text
    actual = _run_transformed_reference(bound, args)
    expected = (args[0].float() @ args[1].float()).half()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_single_axis_root_gains_an_n_tile() -> None:
    a, b = _inputs((16, 32, 32))
    gate = torch.zeros(16, 32, dtype=torch.float16)
    bound = _bind(_single_axis, (a, b, gate))
    ir = _host(bound).device_ir
    assert ir.grid_block_ids == [[0, 1]]
    assert [
        graph.block_ids for graph in ir.graphs if type(graph) is ForLoopGraphInfo
    ] == [[2]]
    assert bound.config_spec.matmul_facts[0].k_block_id == 2
    text = _source(bound)
    assert "for row, _helion_fullslice_n in hl.tile([a.size(0), b.size(1)])" in text
    assert "gate[row, _helion_fullslice_n] = activation" in text
    assert "out[row, _helion_fullslice_n] = " in text
    assert "activation = torch.sigmoid(_helion_fullslice_mm)" in text
    # The surviving alias load is re-sliced and re-issued after the
    # contraction loop (the tcgen05 K-loop family refuses it ahead of it).
    assert text.index("for _helion_fullslice_k in") < text.index(
        "x_tile = a[row, _helion_fullslice_n]"
    )
    reference_gate = torch.zeros_like(gate)
    actual = _run_transformed_reference(bound, (a, b, reference_gate))
    sigmoid = torch.sigmoid((a.float() @ b.float()).half())
    torch.testing.assert_close(reference_gate, sigmoid, atol=0, rtol=0)
    torch.testing.assert_close(
        actual, (2.0 * a.float() * sigmoid).half(), atol=0, rtol=0
    )
    with _cpu_target():
        code = bound.to_code(helion.Config(block_sizes=[16, 16, 16]))
    ast.parse(code)


def test_single_axis_allocation_extents_resolve_at_the_allocation() -> None:
    # The same rebinding that declines above is harmless when every
    # allocation reads the name after it was rebound to N.
    a, b = _inputs((16, 32, 24))
    gate = torch.zeros(16, 32, dtype=torch.float16)
    bound = _bind(_single_axis_dimension_rebound_first, (a, b, gate))
    assert _host(bound).device_ir.grid_block_ids == [[0, 1]]
    assert bound.config_spec.matmul_facts[0].k_block_id == 2
    text = _source(bound)
    assert "aux[row, _helion_fullslice_n] = gate[row, _helion_fullslice_n] * 2" in text
    actual = _run_transformed_reference(bound, (a, b, gate))
    expected = torch.relu((a.float() @ b.float()).half())
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(16, 32, 16), (13, 37, 19), (4, 6, 0)])
def test_single_axis_direct_store_consumer(shape: tuple[int, int, int]) -> None:
    args = _inputs(shape)
    bound = _bind(_single_axis_direct, args)
    assert _host(bound).device_ir.grid_block_ids == [[0, 1]]
    assert bound.config_spec.matmul_facts[0].k_block_id == 2
    actual = _run_transformed_reference(bound, args)
    expected = torch.relu((args[0].float() @ args[1].float()).half())
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_single_axis_dynamic_n_equality_keeps_original_ast() -> None:
    args = _inputs((16, 32, 32))
    before = _bind(_single_axis_direct, args, enabled=False, static_shapes=False)
    after = _bind(_single_axis_direct, args, static_shapes=False)
    assert ast.dump(ast.Module(_host(before).body, [])) == ast.dump(
        ast.Module(_host(after).body, [])
    )


def _se_block_full_slice_fwd(
    x: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # examples/se_block.py's forward before it tiled the output columns and
    # the reduction (the policy's original subject): a whole-row matmul with
    # a sigmoid epilogue and two stores in a single-axis tile.
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    s = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :]
        sigmoid_result = torch.sigmoid(x_tile @ w[:, :])
        s[tile_m, :] = sigmoid_result
        acc = 2.0 * x_tile.to(torch.float32) * sigmoid_result
        out[tile_m, :] = acc.to(x.dtype)
    return out, s


def test_se_block_forward_is_stripmined() -> None:
    x, w = _inputs((32, 32, 32))
    bound = _bind(_se_block_full_slice_fwd, (x, w))
    ir = _host(bound).device_ir
    assert ir.grid_block_ids == [[0, 1]]
    assert [
        graph.block_ids for graph in ir.graphs if type(graph) is ForLoopGraphInfo
    ] == [[2]]
    assert bound.config_spec.matmul_facts[0].k_block_id == 2
    text = _source(bound)
    assert "for tile_m, _helion_fullslice_n in hl.tile([m, w.size(1)])" in text
    assert "x_tile = x[tile_m, _helion_fullslice_n]" in text
    assert "s[tile_m, _helion_fullslice_n] = sigmoid_result" in text
    with _cpu_target():
        code = bound.to_code(helion.Config(block_sizes=[16, 16, 16]))
    ast.parse(code)


def test_existing_masked_stores_keep_their_order() -> None:
    args = _inputs()
    bound = _bind(_masked_consumer, args)
    assert bound.config_spec.matmul_facts[0].k_block_id == 2
    actual = _run_transformed_reference(bound, args)
    expected = (args[0].float() @ args[1].float()).half().float() * 1.003
    expected[:, args[1].size(1) // 2 :] = 9
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_multiple_matmuls_keep_independent_accumulators_and_k_tiles() -> None:
    args = _inputs()
    bound = _bind(_two_matmuls, args)
    assert [fact.k_block_id for fact in bound.config_spec.matmul_facts] == [2, 3]
    actual = _run_transformed_reference(bound, args)
    expected = (args[0].float() @ args[1].float()).half().float() * 2
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with _cpu_target():
        code = bound.to_code(helion.Config(block_sizes=[16, 32, 16, 32]))
    ast.parse(code)


def test_other_dtype_and_backend_keep_original_ast() -> None:
    for dtype, backend in ((torch.float32, "cute"), (torch.float16, "triton")):
        args = _inputs(dtype=dtype)
        before = _bind(_matmul, args, enabled=False, backend=backend)
        after = _bind(_matmul, args, backend=backend)
        assert ast.dump(ast.Module(_host(before).body, [])) == ast.dump(
            ast.Module(_host(after).body, [])
        )


@pytest.mark.parametrize("direct", [False, True])
def test_helpers_resolve_actual_global_aliases_without_mutating_them(
    monkeypatch: pytest.MonkeyPatch, direct: bool
) -> None:
    source = _direct_imports if direct else _alternate_aliases
    module = ModuleType(f"_helion_stripmine_aliases_{direct}")
    if direct:
        bindings = {
            "tile": hl.tile,
            "empty": torch.empty,
            "matrix": torch.mm,
            "zero_tiles": hl.zeros,
            "matrix_accumulate": torch.addmm,
            "single": torch.float32,
        }
    else:
        bindings = {"array_lib": torch, "blocks": hl}
    module.__dict__.update(bindings)
    function = FunctionType(source.__code__, module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    before = dict(module.__dict__)
    bound = _bind(function, _inputs())
    assert module.__dict__ == before
    text = ast.unparse(ast.Module(_host(bound).body, []))
    assert "_helion_fullslice_k" in text
    if direct:
        assert "matrix_accumulate(" in text
        assert "zero_tiles(" in text
    else:
        assert "_helion_fullslice_acc_ =" in text
        assert "for _helion_fullslice_k_ in" in text
        assert "array_lib.addmm(" in text
        assert "blocks.zeros(" in text


def test_missing_helper_bindings_leave_source_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("_helion_stripmine_no_helpers")
    module.__dict__.update(tile=hl.tile, empty=torch.empty, matrix=torch.mm)
    function = FunctionType(_direct_imports.__code__, module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    args = _inputs()
    before = _bind(function, args, enabled=False)
    after = _bind(function, args)
    assert ast.dump(ast.Module(_host(before).body, [])) == ast.dump(
        ast.Module(_host(after).body, [])
    )


def test_fission_first_stage_gets_its_own_k_loop() -> None:
    x, a = _inputs((32, 16, 64), torch.bfloat16)
    b = torch.empty((16, 64), dtype=torch.bfloat16)
    bound = _bind(
        cast("FunctionType", squeeze_and_excitation_net_fwd.fn), (x, a, b), fission=True
    )
    assert bound.env.cute_fission_plan is not None
    ir = _host(bound).device_ir
    assert ir.grid_block_ids == [[0, 1], [3, 4]]
    assert [
        graph.block_ids for graph in ir.graphs if type(graph) is ForLoopGraphInfo
    ] == [[2], [5]]
    assert [fact.k_block_id for fact in bound.config_spec.matmul_facts] == [2, 5]
    with _cpu_target():
        code = bound.to_code(helion.Config(block_sizes=[16, 16, 16, 16, 32, 16]))
    ast.parse(code)
