from __future__ import annotations

import ast
import functools
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from .test_triton_scalar_codegen import cpu_codegen as cpu_codegen
import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.roll_reduction import ReductionRoller
from helion._testing import DEVICE
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
from helion._utils import triton_is_available
import helion.language as hl
from helion.runtime.settings import _get_backend

pytestmark = pytest.mark.skipif(
    not triton_is_available(), reason="Triton is required for source generation"
)


@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def _row_topk(x: torch.Tensor, k: int, largest: hl.constexpr):
    k = hl.specialize(k)
    values = torch.empty((x.size(0), k), dtype=x.dtype, device=x.device)
    indices = torch.empty((x.size(0), k), dtype=torch.int64, device=x.device)
    for row in hl.tile(x.size(0)):
        selected, order = torch.topk(x[row, :], k, dim=-1, largest=largest)
        values[row, :] = selected
        indices[row, :] = order
    return values, indices


@functools.cache
def _native_topk(backend: str):
    return helion.kernel(
        _row_topk.fn, backend=backend, static_shapes=True, autotune_effort="none"
    )


def _input(case: str, dtype: torch.dtype, device=DEVICE) -> torch.Tensor:
    if case == "negative_tail":
        row = -torch.arange(1, 18).float()
    elif case == "positive_tail":
        row = torch.arange(1, 18).float()
    elif case == "all_nan":
        row = torch.full((16,), float("nan"))
        row[:4] = torch.tensor(
            [0x7FC00001, 0x7FC12345, 0xFFC00001, 0xFFC54321], dtype=torch.uint32
        ).view(torch.float32)
    elif case == "mixed_nan":
        row = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 2, 3, 4, 5, 6] * 2
        )
    elif case == "signed_zero":
        row = torch.tensor([0.0, -0.0] * 8)
    else:
        assert case == "equal_finite"
        row = torch.ones(16)
    return row.to(dtype=dtype, device=device)[None, :].repeat(3, 1)


def _assert_topk(x, k, largest, values, indices):
    assert values.dtype == x.dtype
    assert indices.dtype == torch.int64
    assert values.shape == indices.shape == (*x.shape[:-1], k)
    assert bool(((indices >= 0) & (indices < x.size(-1))).all())
    assert all(row.unique().numel() == k for row in indices.reshape(-1, k))
    expected = torch.topk(x, k, dim=-1, largest=largest).values
    torch.testing.assert_close(values, expected, rtol=0, atol=0, equal_nan=True)
    # Tied indices are unspecified by torch.topk. Whichever valid indices are
    # chosen, their original NaN/zero payloads must match the returned values.
    bits = torch.int32 if x.dtype == torch.float32 else torch.int16
    assert torch.equal(values.view(bits), x.view(bits).gather(-1, indices))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skipUnlessBackends(["triton", "tileir"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "negative_tail",
        "positive_tail",
        "all_nan",
        "mixed_nan",
        "signed_zero",
        "equal_finite",
    ],
)
def test_float_topk_preserves_values_and_valid_indices(case, largest, dtype):
    x = _input(case, dtype)
    before = x.view(torch.uint8).clone()
    values, indices = _native_topk(_get_backend())(x, 4, largest)
    _assert_topk(x, 4, largest, values, indices)
    assert torch.equal(x.view(torch.uint8), before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skipUnlessBackends(["triton", "tileir"])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("algorithm", ["auto", "topk", "sort"])
@skipIfRefEager("uses an explicitly compiled dynamic-shape configuration")
def test_float_topk_dynamic_tails(largest, algorithm):
    if _get_backend() != "triton" and algorithm != "auto":
        pytest.skip("selection algorithms are a native Triton option")
    kernel = helion.kernel(
        _row_topk.fn,
        backend=_get_backend(),
        static_shapes=False,
        autotune_effort="none",
    )
    initial = torch.ones((3, 17), device=DEVICE)
    bound = kernel._bind_isolated((initial, 4, largest))
    config = bound.config_spec.default_config()
    if _get_backend() == "triton":
        config.config["triton_topk_algorithm"] = algorithm
    compiled = bound.compile_config(config)
    for width in (17, 19, 65):
        x = torch.arange(1, width + 1, device=DEVICE).float()[None, :].repeat(3, 1)
        if largest:
            x = -x
        values, indices = compiled(x, 4, largest)
        _assert_topk(x, 4, largest, values, indices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skipUnlessBackends(["triton"])
@pytest.mark.parametrize("algorithm", ["topk", "sort"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("k", [3, 17])
@skipIfRefEager("uses explicit selection and multi-row tile configurations")
def test_float_topk_algorithms_preserve_payloads(algorithm, dtype, largest, k):
    x = _input("mixed_nan", dtype)
    x = torch.cat((x, torch.zeros((3, 1), dtype=dtype, device=DEVICE)), dim=1)
    x[1, :16] = _input("all_nan", dtype)[0]
    x[2, :] = 0.0
    x[2, 1::2] = -0.0
    bound = _row_topk._bind_isolated((x, k, largest))
    config = bound.config_spec.default_config()
    config.config.update(block_sizes=[4], triton_topk_algorithm=algorithm)
    code = bound.to_code(config)
    assert ("tl.sort(" in code) == (algorithm == "sort")
    assert ("tl.topk(" in code) == (algorithm == "topk")
    values, indices = bound.compile_config(config)(x, k, largest)
    _assert_topk(x, k, largest, values, indices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skipUnlessBackends(["triton", "tileir"])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    ("width", "warps", "rows_per_program"), [(1024, 4, 1), (1024, 4, 2), (2048, 16, 1)]
)
@skipIfRefEager("uses small row blocks to keep large-axis compilation bounded")
def test_float_topk_selects_across_large_rows(largest, width, warps, rows_per_program):
    x = torch.arange(width, device=DEVICE).float()[None, :].repeat(3, 1)
    if largest:
        x = -x
    # Bound native compilation cost; the CPU test below checks 8192-element
    # selection axes that would otherwise be split into reduction loops.
    bound = _native_topk(_get_backend())._bind_isolated((x, 64, largest))
    config = bound.config_spec.default_config()
    config.config["block_sizes"] = [rows_per_program]
    config.config["num_warps"] = warps
    values, indices = bound.compile_config(config)(x, 64, largest)
    _assert_topk(x, 64, largest, values, indices)


@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def _row_sort(x: torch.Tensor):
    values = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        values[row, :] = torch.sort(x[row, :], dim=-1).values
    return values


@pytest.mark.parametrize("operation", ["topk", "sort"])
@skipIfRefEager("inspects reduction-loop choices and generated source")
def test_selection_axes_require_complete_input(cpu_codegen, operation):
    kernel = _row_topk if operation == "topk" else _row_sort
    x = torch.ones(3, 8192)
    inputs = (x, 64, True) if operation == "topk" else (x,)
    bound = kernel._bind_isolated(inputs)
    assert not bound.config_spec.reduction_loops.valid_block_ids()
    code = bound.to_code(bound.config_spec.default_config())
    assert not any(isinstance(node, ast.For) for node in ast.walk(ast.parse(code)))


@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def _topk_and_independent_sum(x: torch.Tensor, y: torch.Tensor):
    values = torch.empty((x.size(0), 8), device=x.device, dtype=x.dtype)
    indices = torch.empty((x.size(0), 8), device=x.device, dtype=torch.int64)
    sums = torch.empty((x.size(0),), device=x.device, dtype=x.dtype)
    for row in hl.tile(x.size(0)):
        selected, order = torch.topk(x[row, :], 8, dim=-1)
        values[row, :] = selected
        indices[row, :] = order
        sums[row] = y[row, :].sum(-1)
    return values, indices, sums


@skipIfRefEager("inspects reduction-loop choices and generated source")
def test_selection_keeps_independent_reductions_rollable(cpu_codegen):
    bound = _topk_and_independent_sum._bind_isolated(
        (torch.ones(3, 64), torch.ones(3, 8192))
    )
    blocks = bound.config_spec.reduction_loops.valid_block_ids()
    assert blocks
    with bound.env:
        assert all(bound.env.block_sizes[block].size_hint() == 8192 for block in blocks)
    config = bound.config_spec.default_config()
    config.config["reduction_loops"] = [4096] * len(blocks)
    code = bound.to_code(config)
    assert any(
        isinstance(node, ast.For) and "8192" in ast.unparse(node.iter)
        for node in ast.walk(ast.parse(code))
    )


@pytest.mark.parametrize("operation", ["topk", "sort"])
@pytest.mark.parametrize("dim", [-2, -1, 0, 1])
def test_selection_guard_checks_the_consumed_dimension(operation, dim):
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    source.meta["val"] = torch.empty(5, 9)
    target = (
        torch.ops.aten.topk.default
        if operation == "topk"
        else torch.ops.aten.sort.default
    )
    node = graph.call_function(
        target, (source, 3, dim) if operation == "topk" else (source, dim)
    )
    env = SimpleNamespace(get_block_id=lambda size: {5: 0, 9: 1}.get(size))
    roller = ReductionRoller(None, SimpleNamespace(block_id=dim % 2), {})
    with (
        patch.object(CompileEnvironment, "current", return_value=env),
        pytest.raises(
            NotImplementedError, match="selection axes require complete input"
        ),
    ):
        roller.should_go_in_inner_graph(node)


def _interpret_triton_selection(source, inputs, path, monkeypatch, selected_ops=None):
    """Execute the exact generated body through Triton's CPU interpreter."""
    import importlib.util
    import sys

    from triton.runtime.interpreter import InterpretedFunction
    from triton.runtime.interpreter import _patch_lang
    from triton.runtime.jit import JITFunction

    path.write_text(source)
    spec = importlib.util.spec_from_file_location("selection_interpreter_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    def launch(kernel, grid, *args, **kwargs):
        # The standard library was imported before interpreter mode. Dispatch
        # its existing JIT helpers through the same interpreter, without
        # replacing top-k/sort arithmetic with a reference implementation.
        def device_call(function, *values, **options):
            if selected_ops is not None and function.fn.__name__ in ("sort", "topk"):
                selected_ops.add(function.fn.__name__)
            interpreted = InterpretedFunction(function.fn)
            # InterpretedFunction.__call__ patches each helper's language
            # namespace without restoring it. Helpers may import tl.core,
            # which the outer grid's tl patch does not cover.
            scope = _patch_lang(function.fn)
            try:
                return interpreted.rewrite()(*values, **options)
            finally:
                scope.restore()

        with patch.object(JITFunction, "__call__", device_call):
            return InterpretedFunction(kernel.fn)[grid](*args, **kwargs)

    function = next(
        node.name
        for node in reversed(ast.parse(source).body)
        if isinstance(node, ast.FunctionDef)
    )
    return vars(module)[function](*inputs, _launcher=launch)


@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    ("k", "warps", "rows_per_program"),
    [
        (16, 4, 1),
        (16, 8, 1),
        (32, 4, 1),
        (32, 8, 1),
        (64, 4, 1),
        (64, 8, 1),
        (64, 4, 2),
        (64, 4, 4),
    ],
)
@skipIfRefEager("executes generated automatic dispatch through the CPU interpreter")
def test_float_topk_auto_dispatch_dynamic_widths(
    cpu_codegen, largest, warps, k, rows_per_program, tmp_path, monkeypatch
):
    kernel = helion.kernel(
        _row_topk.fn,
        backend="triton",
        static_shapes=False,
        autotune_effort="none",
    )
    initial = torch.ones((3, 256))
    bound = kernel._bind_isolated((initial, k, largest))
    monkeypatch.setattr(bound.env.config_spec, "target_device_capability", (10, 3))
    config = bound.config_spec.default_config()
    config.config.update(block_sizes=[rows_per_program], num_warps=warps)
    code = bound.to_code(config)
    for width in (256, 769, 1025):
        x = torch.arange(width).float()[None, :].repeat(3, 1)
        if largest:
            x = -x
        selected_ops = set()
        values, indices = _interpret_triton_selection(
            code,
            (x, k, largest),
            tmp_path / "automatic_topk.py",
            monkeypatch,
            selected_ops,
        )
        _assert_topk(x, k, largest, values, indices)
        expected = (
            "sort"
            if width == 769 and warps == 4 and k >= 32 and rows_per_program == 1
            else "topk"
        )
        assert selected_ops == {expected}


@pytest.mark.parametrize(
    ("capability", "hip"), [(None, False), ((9, 0), False), ((10, 3), True)]
)
@skipIfRefEager("inspects target-dependent automatic selection")
def test_float_topk_auto_other_targets(cpu_codegen, capability, hip, monkeypatch):
    bound = _row_topk._bind_isolated((torch.ones(3, 1024), 64, False))
    monkeypatch.setattr(bound.env.config_spec, "target_device_capability", capability)
    monkeypatch.setattr("helion._compiler.triton.aten_lowering.is_hip", lambda: hip)
    config = bound.config_spec.default_config()
    config.config.update(block_sizes=[1], num_warps=4)
    code = bound.to_code(config)
    assert "tl.sort(" not in code
    assert "tl.topk(" in code
    config.config["triton_topk_algorithm"] = "sort"
    assert "tl.sort(" in bound.to_code(config)


@pytest.mark.parametrize("largest", [False, True])
@skipIfRefEager("executes generated sort with two leading tile dimensions")
def test_float_topk_sort_multiple_leading_dimensions(
    cpu_codegen, largest, tmp_path, monkeypatch
):
    def batched_topk(x: torch.Tensor, k: hl.constexpr, largest: hl.constexpr):
        values = torch.empty((x.size(0), x.size(1), k), dtype=x.dtype, device=x.device)
        indices = torch.empty(
            (x.size(0), x.size(1), k), dtype=torch.int64, device=x.device
        )
        for batch, row in hl.tile(x.shape[:-1]):
            selected, order = torch.topk(x[batch, row, :], k, dim=-1, largest=largest)
            values[batch, row, :] = selected
            indices[batch, row, :] = order
        return values, indices

    kernel = helion.kernel(
        batched_topk, backend="triton", static_shapes=True, autotune_effort="none"
    )
    x = torch.arange(3 * 5 * 17).reshape(3, 5, 17).float()
    bound = kernel._bind_isolated((x, 3, largest))
    config = bound.config_spec.default_config()
    config.config.update(block_sizes=[2, 4], triton_topk_algorithm="sort")
    code = bound.to_code(config)
    values, indices = _interpret_triton_selection(
        code, (x, 3, largest), tmp_path / "batched_topk.py", monkeypatch
    )
    _assert_topk(x, 3, largest, values, indices)


@pytest.mark.parametrize("fail", [False, True])
def test_triton_selection_interpreter_restores_language(fail, tmp_path, monkeypatch):
    pytest.importorskip("triton")
    import triton.language as tl
    from triton.runtime.interpreter import InterpreterError

    source = f"""
import triton
import triton.language as tl

@triton.jit
def kernel(x, out):
    index = tl.arange(0, 4)
    value = tl.topk(tl.load(x + index), 2)
    tl.static_assert(not {fail!r}, "expected interpreter failure")
    tl.store(out + tl.arange(0, 2), value)

def wrapper(x, out, *, _launcher):
    _launcher(kernel, (1,), x, out)
    return out
"""
    namespaces = (tl, tl.core, tl.math, tl.tensor, tl.dtype)
    originals = [(obj, vars(obj).copy()) for obj in namespaces]
    x = torch.arange(4, dtype=torch.float32)
    out = torch.empty(2, dtype=x.dtype)
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")):
        if fail:
            with pytest.raises(InterpreterError, match="expected interpreter failure"):
                _interpret_triton_selection(
                    source, (x, out), tmp_path / "generated.py", monkeypatch
                )
        else:
            actual = _interpret_triton_selection(
                source, (x, out), tmp_path / "generated.py", monkeypatch
            )
            torch.testing.assert_close(actual, x.topk(2).values, rtol=0, atol=0)
    for obj, attributes in originals:
        for name, value in attributes.items():
            assert vars(obj)[name] is value, (obj.__name__, name)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("largest", [False, True])
@skipIfRefEager("executes generated TileIR source through Triton's CPU interpreter")
def test_tileir_topk_payload_selection(
    cpu_codegen, dtype, largest, tmp_path, monkeypatch
):
    x = _input("mixed_nan", dtype, "cpu")
    x = torch.cat((x, torch.zeros((3, 1), dtype=dtype)), dim=1)
    x[1, :16] = _input("all_nan", dtype, "cpu")[0]
    x[2, :] = 0.0
    x[2, 1::2] = -0.0
    with patch.dict(os.environ, {"ENABLE_TILE": "1"}):
        kernel = helion.kernel(
            _row_topk.fn, backend="tileir", static_shapes=True, autotune_effort="none"
        )
        bound = kernel._bind_isolated((x, 4, largest))
        code = bound.to_code(bound.config_spec.default_config())
        assert "tl.gather" not in code
        values, indices = _interpret_triton_selection(
            code, (x, 4, largest), tmp_path / "tileir_topk.py", monkeypatch
        )
    _assert_topk(x, 4, largest, values, indices)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@skipIfRefEager("executes generated TileIR source through Triton's CPU interpreter")
def test_tileir_topk_after_promoted_division(cpu_codegen, dtype, tmp_path, monkeypatch):
    def divided_topk(x: torch.Tensor, divisor: torch.Tensor):
        values = torch.empty((x.size(0), 3), device=x.device, dtype=x.dtype)
        indices = torch.empty((x.size(0), 3), device=x.device, dtype=torch.int64)
        for row in hl.tile(x.size(0)):
            source = x[row, :] / divisor[row, :]
            selected, order = torch.topk(source, 3, dim=-1)
            values[row, :] = selected
            indices[row, :] = order
        return values, indices

    # Exact quotients isolate the actual FP32 intermediate from interpreter
    # rounding differences in low-precision division.
    x = ((torch.arange(51).reshape(3, 17) - 25) * 0.75).to(dtype)
    divisor = torch.full_like(x, 0.75)
    with patch.dict(os.environ, {"ENABLE_TILE": "1"}):
        kernel = helion.kernel(
            divided_topk, backend="tileir", static_shapes=True, autotune_effort="none"
        )
        bound = kernel._bind_isolated((x, divisor))
        code = bound.to_code(bound.config_spec.default_config())
        values, indices = _interpret_triton_selection(
            code, (x, divisor), tmp_path / "tileir_promoted_topk.py", monkeypatch
        )
    expected = torch.topk(x.float() / divisor.float(), 3, dim=-1)
    torch.testing.assert_close(values, expected.values.to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(indices, expected.indices, rtol=0, atol=0)
