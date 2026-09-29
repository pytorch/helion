from __future__ import annotations

import ast
import linecache
import os
import subprocess
import sys
import tempfile
import types
from typing import Any
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

import helion
from helion._compiler.cute.fragment_epilogue import _is_literal_scalar
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_where(
    lhs: torch.Tensor, rhs: torch.Tensor, out: torch.Tensor, mode: hl.constexpr
) -> torch.Tensor:
    batches, m, k = lhs.shape
    n = rhs.shape[2]
    for bi, mi, ni in hl.tile([batches, m, n], block_size=[1, None, None]):
        acc = hl.zeros([bi, mi, ni], dtype=torch.float32)
        for ki in hl.tile(k):
            acc = torch.baddbmm(acc, lhs[bi, mi, ki], rhs[bi, ki, ni])
        if mode == "row":
            condition = mi.index[None, :, None] % 2 == 0
        elif mode == "column":
            condition = ni.index[None, None, :] % 2 == 0
        elif mode == "data":
            condition = acc >= 0.0
        elif mode == "cross_thread_condition":
            condition = acc.transpose(-1, -2) >= 0.0
        else:
            condition = mi.index[None, :, None] >= ni.index[None, None, :]
        if mode == "nan_false":
            value = torch.where(condition, acc, float("nan"))
        elif mode == "nan_true":
            value = torch.where(condition, float("nan"), acc)
        elif mode == "nested":
            other = torch.where(ni.index[None, None, :] % 2 == 0, 2.0, -3.0)
            value = torch.where(condition, acc, other)
        elif mode == "cross_thread_true":
            value = torch.where(condition, acc.transpose(-1, -2), acc)
        elif mode == "cross_thread_false":
            value = torch.where(condition, acc, acc.transpose(-1, -2))
        elif mode == "promoted":
            value = torch.where(condition, acc.to(torch.float16), 0.125)
        else:
            value = torch.where(condition, acc, 0.0)
        out[bi, mi, ni] = value.to(out.dtype)
    return out


def _args(
    device: Any, dtype: torch.dtype = torch.bfloat16, n: int = 128
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(912)
    return (
        torch.randn((2, 128, 64), device=device, dtype=dtype, generator=generator)
        * 0.1,
        torch.randn((2, 64, n), device=device, dtype=dtype, generator=generator) * 0.1,
        torch.full((2, 128, 2 * n), 17.0, device=device, dtype=dtype)[..., :n],
    )


def _config(n: int) -> helion.Config:
    return helion.Config(block_sizes=[128, n, 64], pid_type="flat")


def _code(args: tuple[torch.Tensor, ...], mode: str, n: int) -> str:
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        bound = _fragment_where._bind_isolated((*args, mode))
        assert bound.config_spec.cute_tcgen05_search_enabled
        return bound.to_code(_config(n))


@pytest.mark.parametrize("n", [64, 128])
@pytest.mark.parametrize(
    "mode",
    ["causal", "row", "column", "data", "nan_false", "nan_true", "nested", "promoted"],
)
def test_fragment_where_codegen(mode: str, n: int) -> None:
    code = _code(_args("cpu", n=n), mode, n)
    assert "cute.gemm(" in code
    assert "tcgen05_epi_where" in code
    selections = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id.startswith("tcgen05_epi_where")
            for target in node.targets
        )
    ]
    assert selections
    for selection in selections:
        assert isinstance(selection.value, ast.IfExp)
        assert isinstance(selection.value.body, ast.Call)
        assert isinstance(selection.value.orelse, ast.Call)
        assert ast.unparse(selection.value.body.func) == ast.unparse(
            selection.value.orelse.func
        )


@pytest.mark.parametrize(
    "mode", ["cross_thread_condition", "cross_thread_true", "cross_thread_false"]
)
def test_fragment_where_requires_all_local_accumulator_demands(mode: str) -> None:
    with patch_cute_mma_support(), patch("torch.cuda.is_available", return_value=False):
        bound = _fragment_where._bind_isolated((*_args("cpu"), mode))
        # Candidate extraction may succeed, but the ownership proof must not
        # commit a where whose predicate reads another thread's accumulator.
        with pytest.raises(helion.exc.BackendUnsupported, match="ownership proof"):
            bound.to_code(_config(128))


@pytest.mark.parametrize("value", [True, 3, -0.5])
def test_fragment_literal_terminals_require_literal_rank_zero(
    value: float,
) -> None:
    graph = torch.fx.Graph()
    literal = graph.call_function(torch.ops.aten.scalar_tensor.default, (value,))
    literal.meta["val"] = torch.tensor(value)
    assert _is_literal_scalar(literal)
    dynamic = graph.placeholder("dynamic")
    literal.args = (dynamic,)
    assert not _is_literal_scalar(literal)
    literal.args = (value,)
    literal.meta["val"] = torch.tensor([value])
    assert not _is_literal_scalar(literal)


def _compile_tma(code: str, args: tuple[torch.Tensor, ...], mode: str) -> str:
    from cuda.bindings.driver import CUstream
    import cutlass
    import cutlass.cute as cute

    from helion.runtime.cute.launcher import _create_cute_wrapper

    name = "_fragment_where_compile"
    module = types.ModuleType(name)
    path = f"<{name}>"
    module.__file__ = path
    sys.modules[name] = module
    linecache.cache[path] = (len(code), None, code.splitlines(keepends=True), path)
    exec(compile(code, path, "exec"), module.__dict__)
    compiled: list[str] = []

    def capture(kernel: Any, grid: Any, *tensors: torch.Tensor, **kwargs: Any) -> None:
        schema = tuple(
            (
                "tensor",
                tensor.dtype,
                tensor.ndim,
                tuple(tensor.shape),
                tuple(tensor.stride()),
            )
            for tensor in tensors
        )
        launch = _create_cute_wrapper(kernel, schema, (128, 1, 1), num_sm=148)
        source = "@cute.jit\ndef compile_tma(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor, stream: CUstream):\n    launch(a.iterator, b.iterator, out.iterator, 1, 1, 1, stream)\n"
        wrapper_path = "<fragment_where_tma_wrapper>"
        namespace = {"cute": cute, "CUstream": CUstream, "launch": launch}
        wrapper_lines: list[str] = list(source.splitlines(keepends=True))
        linecache.cache[wrapper_path] = (
            len(source),
            None,
            wrapper_lines,
            wrapper_path,
        )
        exec(compile(source, wrapper_path, "exec"), namespace)
        fake = tuple(
            cute.runtime.make_fake_tensor(
                cutlass.BFloat16 if tensor.dtype is torch.bfloat16 else cutlass.Float16,
                tuple(tensor.shape),
                tuple(tensor.stride()),
                assumed_align=16,
            )
            for tensor in tensors
        )
        with tempfile.TemporaryDirectory(prefix="fragment_where_") as directory:
            compiled.append(
                cute.compile(
                    namespace["compile_tma"],
                    *fake,
                    cute.runtime.make_fake_stream(),
                    options=f"--dump-dir {directory}",
                ).__ptx__
            )

    module.__dict__["_fragment_where"](*args, mode, _launcher=capture)
    assert not torch.cuda.is_initialized()
    return compiled[0]


@pytest.mark.parametrize("mode", ["causal", "nested", "nan_false", "promoted"])
def test_fragment_where_real_cpu_compile(mode: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", __name__, mode],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "CUTE_DSL_ARCH": "sm_103a",
            "CUTE_DSL_KEEP_PTX": "1",
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "mode",
    ["causal", "row", "column", "data", "nan_false", "nan_true", "nested", "promoted"],
)
def test_fragment_where_runtime(dtype: torch.dtype, mode: str) -> None:
    args = _args(DEVICE, dtype)
    lhs, rhs, out = args
    saved = tuple(value.clone() for value in args[:2])
    function = _fragment_where._bind_isolated((*args, mode)).compile_config(
        _config(128)
    )
    actual = function(*args, mode).clone()
    acc = lhs.double() @ rhs.double()
    row = torch.arange(128, device=DEVICE)[None, :, None]
    col = torch.arange(128, device=DEVICE)[None, None, :]
    condition = row >= col
    if mode == "row":
        condition = row % 2 == 0
    elif mode == "column":
        condition = col % 2 == 0
    elif mode == "data":
        condition = acc >= 0.0
    other: torch.Tensor | float = 0.0
    if mode == "nan_false":
        other = float("nan")
    elif mode == "nested":
        other = torch.where(col % 2 == 0, 2.0, -3.0)
    elif mode == "promoted":
        acc, other = acc.to(torch.float16), 0.125
    expected = (
        torch.where(condition, float("nan"), acc)
        if mode == "nan_true"
        else torch.where(condition, acc, other)
    )
    torch.testing.assert_close(
        actual, expected.to(dtype), atol=0.002, rtol=0.01, equal_nan=True
    )
    repeated = function(*args, mode)
    torch.testing.assert_close(actual, repeated, atol=0, rtol=0, equal_nan=True)
    for value, before in zip(args[:2], saved, strict=True):
        torch.testing.assert_close(value, before, atol=0, rtol=0)
    padding = out.as_strided((2, 128, 128), out.stride(), storage_offset=128)
    assert torch.all(padding == 17)


if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    values = _args("cpu")
    source = _code(values, sys.argv[1], 128)
    assert "tcgen05.mma" in _compile_tma(source, values, sys.argv[1])
    assert not torch.cuda.is_initialized()
