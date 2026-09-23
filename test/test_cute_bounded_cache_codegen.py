from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
from unittest.mock import patch

from examples.aot_example import row_softmax
import pytest
import torch
from torch._inductor.codecache import PyCodeCache

from test._cute_binding import _mock_cuda_unavailable

from .cute_population_contracts import _target
import helion
from helion._testing import skipUnlessBackends
from helion.autotuner.precompile_future import _load_compiled_fn
from helion.autotuner.precompile_future import _serialize_compiled_fn
from helion.autotuner.precompile_future import _unload_compiled_fn

pytestmark = skipUnlessBackends(["cute"])


@contextmanager
def _cpu_target():
    with (
        _target(),
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


def _bound(
    *, dtype=torch.bfloat16, static=False, shape=(5, 513), layout="dense", effort="none"
):
    if layout == "transpose":
        x = torch.empty(tuple(reversed(shape)), dtype=dtype).t()
    elif layout == "slice":
        x = torch.empty((shape[0], shape[1] * 2), dtype=dtype)[:, ::2]
    elif layout == "unaligned":
        x = torch.empty(shape[0] * shape[1] + 1, dtype=dtype).as_strided(
            shape, (shape[1], 1), 1
        )
    else:
        x = torch.empty(shape, dtype=dtype)
    kernel = helion.kernel(
        row_softmax.fn,
        backend="cute",
        static_shapes=static,
        autotune_effort=effort,
        cute_full_slice_matmul_tiling=True,
        cute_segmented_matmul_tiling=True,
        cute_flatten_nested_reductions=True,
    )
    return kernel._bind_isolated((x,))


def _config(mode="bounded_layout", *, width=4096, threads=128, rows=1, vector=8):
    return helion.Config(
        block_sizes=[rows, width],
        num_threads=[rows if rows > 1 else 0, threads],
        cute_vector_widths=[1, vector],
        cute_lane_layouts=["blocked", "strided"],
        cute_reduction_sequence=mode,
    )


def _functions(source):
    return {
        item.name: item
        for item in ast.parse(source).body
        if isinstance(item, ast.FunctionDef)
    }


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("static", [False, True])
def test_bounded_cache_preserves_fallback_and_vectorization(dtype, static):
    vector = 4 if dtype == torch.float32 else 8
    with _cpu_target():
        bound = _bound(dtype=dtype, static=static)
        original = bound.to_code(_config("scalar", vector=vector))
        source = bound.to_code(_config(vector=vector))
    functions = _functions(source)
    assert "_helion_row_softmax_bounded" in functions
    assert ast.dump(functions["_helion_row_softmax"]) == ast.dump(
        _functions(original)["_helion_row_softmax"]
    )
    fast = ast.unparse(functions["_helion_row_softmax_bounded"])
    assert "cute.make_rmem_tensor" in fast
    assert "cute.arch.load(" in fast
    assert "_cute_store_u" in fast
    collectives = [
        node
        for node in ast.walk(functions["_helion_row_softmax_bounded"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("_cute_grouped_reduce_")
    ]
    assert len(collectives) == 2
    if not static:
        assert "n = cutlass.Int64(cute.size(x, mode=[1]))" in fast
        assert "0 < n <= 4096" in source
    assert "_helion_cute_kernels = (" in source


@pytest.mark.parametrize("layout", ["transpose", "slice", "unaligned"])
def test_runtime_pointer_and_stride_admission_stays_scalar_where_required(layout):
    with _cpu_target():
        bound = _bound(layout=layout)
        source = bound.to_code(_config())
    functions = _functions(source)
    fast = ast.unparse(functions["_helion_row_softmax_bounded"])
    assert "n = cutlass.Int64(cute.size(x, mode=[1]))" in fast
    assert "cute.arch.load(" not in fast
    if layout == "transpose":
        assert "_cute_store_u16_vec(" not in fast


@pytest.mark.parametrize("rows,threads", [(1, 128), (2, 64), (4, 32)])
def test_variant_retains_actual_physical_launch_and_guard(rows, threads):
    with _cpu_target():
        bound = _bound()
        source = bound.to_code(_config(rows=rows, threads=threads))
        module = PyCodeCache.load(source)
        parameter_names = [
            argument.arg
            for argument in _functions(source)["_helion_row_softmax"].args.args
        ]
        calls = []

        def capture(kernel, grid, *args, **kwargs):
            calls.append((kernel, grid, args, kwargs))

        for columns in (0, 1, 513, 4096, 4097):
            x = torch.empty((5, columns), dtype=torch.bfloat16)
            result = module.row_softmax(x, _launcher=capture)
            kernel, grid, arguments, keywords = calls[-1]
            assert result.shape == x.shape
            assert arguments[0] is x and arguments[1] is result
            assert dict(zip(parameter_names, arguments, strict=True))["n"] == columns
            assert grid == ((5 + rows - 1) // rows,)
            assert keywords["block"] == (threads, rows, 1)
            expected = (
                module._helion_row_softmax_bounded
                if 0 < columns <= 4096
                else module._helion_row_softmax
            )
            assert kernel is expected


def test_bounded_and_layout_choices_have_distinct_source_and_keep_default():
    with _cpu_target():
        bound = _bound()
        assert (
            bound.config_spec.default_config().get("cute_reduction_sequence")
            == "scalar"
        )
        cached = bound.to_code(_config("bounded"))
        layout = bound.to_code(_config("bounded_layout"))
    assert "n = cutlass.Int64(cute.size(x, mode=[1]))" not in cached
    assert "n = cutlass.Int64(cute.size(x, mode=[1]))" in layout
    assert cached != layout


def test_conditional_bundle_roundtrip_retains_both_source_hashes():
    with _cpu_target():
        bound = _bound()
        source = bound.to_code(_config())
        module = PyCodeCache.load(source)
        backend = bound.env.backend
        backend.annotate_compiled_module(module, source, "row_softmax")
        function = module.row_softmax
        before = tuple(
            kernel._helion_cute_source_hash for kernel in function._helion_cute_kernels
        )
        assert len(before) == 2 and before[0] != before[1]
        digest = hashlib.sha256(source.encode()).hexdigest()
        assert backend.generated_source_hash(function) == digest
        serialized = _serialize_compiled_fn(function)
        loaded = _load_compiled_fn(serialized)
        try:
            assert backend.generated_source_hash(loaded) == digest
            assert (
                tuple(
                    kernel._helion_cute_source_hash
                    for kernel in loaded._helion_cute_kernels
                )
                == before
            )
        finally:
            _unload_compiled_fn(loaded)
