from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import helion
import helion._compiler.triton.narrow_dot_operands as rewrite
from helion._compiler.triton.narrow_dot_operands import narrow_dot_operands
from helion._testing import skipUnlessBackends
from helion.autotuner.config_spec import MatmulFact
import helion.language as hl
from helion.language._tracing_ops import _mask_to
from helion.language.matmul_ops import MATMUL_FACT_ID_META
from helion.language.matmul_ops import dot


def _graph(dtype: torch.dtype, *, shared_mask: bool = False):
    graph = torch.fx.Graph()
    values = []
    for name in ("lhs", "rhs"):
        src = graph.placeholder(name)
        src.meta["val"] = torch.empty((32, 32), dtype=dtype)
        cast = graph.call_function(
            torch.ops.prims.convert_element_type.default, (src, torch.float32)
        )
        cast.meta["val"] = torch.empty((32, 32), dtype=torch.float32)
        mask = graph.call_function(_mask_to, (cast, 0))
        mask.meta["val"] = cast.meta["val"]
        values.append((src, cast, mask))
    product = graph.call_function(dot, (values[0][2], values[1][2], None, None))
    product.meta["val"] = torch.empty((32, 32), dtype=torch.float32)
    product.meta[MATMUL_FACT_ID_META] = 0
    graph.output(
        (product, values[0][1], values[0][2])
        if shared_mask
        else (product, values[0][1])
    )
    facts = [
        MatmulFact(2, 2, None, None, None, 32, 32, 32, torch.float32, torch.float32)
    ]
    return graph, product, values, facts


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]
)
def test_masked_rewrite_updates_facts_and_preserves_shared_cast(dtype):
    graph, product, values, facts = _graph(dtype)
    assert narrow_dot_operands(graph, facts, allow_fp8=True) == 1
    assert values[0][1] in graph.nodes
    assert values[0][2].args[0] is values[0][0]
    assert values[0][2].meta["val"].dtype == dtype
    assert facts[0].lhs_dtype == facts[0].rhs_dtype == dtype
    assert product.meta["val"].dtype == torch.float32
    graph.lint()


@pytest.mark.cpu_only
def test_rewrite_can_disable_fp8():
    graph, _, _, facts = _graph(torch.float8_e4m3fn)
    assert narrow_dot_operands(graph, facts, allow_fp8=False) == 0
    assert facts[0].lhs_dtype == torch.float32


@pytest.mark.cpu_only
def test_shared_mask_and_mismatched_dtype_do_not_rewrite():
    graph, _, _, facts = _graph(torch.bfloat16, shared_mask=True)
    assert narrow_dot_operands(graph, facts, allow_fp8=True) == 0
    graph, _, values, facts = _graph(torch.bfloat16)
    values[1][0].meta["val"] = torch.empty((32, 32), dtype=torch.float16)
    assert narrow_dot_operands(graph, facts, allow_fp8=True) == 0


@pytest.mark.cpu_only
@pytest.mark.parametrize("case", ["native", "mask_value", "output_dtype", "acc_dtype"])
def test_other_dots_do_not_rewrite(case):
    graph, product, values, facts = _graph(torch.bfloat16)
    if case == "native":
        product.args = (values[0][0], values[1][0], None, None)
    elif case == "mask_value":
        values[0][2].args = (values[0][1], 1)
    elif case == "output_dtype":
        product.args = (*product.args[:3], torch.float16)
    else:
        with graph.inserting_before(product):
            acc = graph.placeholder("acc")
        acc.meta["val"] = torch.empty((32, 32), dtype=torch.float16)
        product.args = (*product.args[:2], acc, None)
    assert narrow_dot_operands(graph, facts, allow_fp8=True) == 0
    assert facts[0].lhs_dtype == torch.float32
    graph.lint()


def _matmul(a, b):
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), device=a.device, dtype=torch.float32)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(
                a[tm, tk].to(torch.float32), b[tk, tn].to(torch.float32), acc=acc
            )
        out[tm, tn] = acc
    return out


def _small_matmul(a, b):
    m, _ = a.shape
    _, n = b.shape
    out = torch.empty((m, n), device=a.device, dtype=torch.float32)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = hl.dot(a[tm, :].to(torch.float32), b[:, tn].to(torch.float32))
    return out


def _native(a, b):
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), device=a.device, dtype=torch.float32)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(a[tm, tk], b[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


CUDA_CAPABILITY = (
    torch.cuda.get_device_capability()
    if torch.cuda.is_available() and torch.version.hip is None
    else None
)
SUPPORTED = CUDA_CAPABILITY in ((8, 0), (8, 6), (9, 0), (10, 3), (12, 0))
FP8_SUPPORTED = CUDA_CAPABILITY in ((9, 0), (10, 3), (12, 0))


@skipUnlessBackends(["triton"])
@pytest.mark.skipif(not SUPPORTED, reason="requires a supported NVIDIA GPU")
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]
)
@pytest.mark.parametrize("static_shapes", [False, True])
def test_ragged_strided_upcast_dot(dtype, static_shapes):
    if dtype.itemsize == 1 and not FP8_SUPPORTED:
        pytest.skip("native FP8 dot is not enabled on Ampere")
    torch.manual_seed(11)
    a = torch.randn((37, 260), device="cuda").to(dtype)[:, ::2]
    b = torch.randn((130, 106), device="cuda").to(dtype)[:, ::2]
    kernel = helion.kernel(
        _matmul, static_shapes=static_shapes, fast_math=True, dot_precision="tf32"
    )
    cfg = helion.Config(block_sizes=[32, 32, 64], indexing="pointer", num_warps=4)
    bound = kernel._bind_isolated((a, b))
    fact = bound.config_spec.matmul_facts[0]
    assert fact.lhs_dtype == fact.rhs_dtype == dtype
    result = bound.compile_config(cfg)(a, b)
    torch.testing.assert_close(result, a.float() @ b.float(), rtol=0.01, atol=0.02)


@skipUnlessBackends(["triton"])
@pytest.mark.skipif(not SUPPORTED, reason="requires a supported NVIDIA GPU")
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]
)
def test_small_dot_padding(dtype):
    if dtype.itemsize == 1 and not FP8_SUPPORTED:
        pytest.skip("native FP8 dot is not enabled on Ampere")
    a = torch.randn((7, 8), device="cuda").to(dtype)
    b = torch.randn((8, 9), device="cuda").to(dtype)
    bound = helion.kernel(
        _small_matmul, fast_math=True, dot_precision="tf32"
    )._bind_isolated((a, b))
    cfg = helion.Config(block_sizes=[16, 16], indexing="pointer", num_warps=4)
    result = bound.compile_config(cfg)(a, b)
    torch.testing.assert_close(result, a.float() @ b.float(), rtol=0.01, atol=0.02)


@skipUnlessBackends(["triton"])
@pytest.mark.skipif(not SUPPORTED, reason="requires a supported NVIDIA GPU")
def test_default_precision_rewrites():
    a = torch.randn((32, 64), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((64, 32), device="cuda", dtype=torch.bfloat16)
    bound = helion.kernel(
        _matmul, fast_math=True, dot_precision="default"
    )._bind_isolated((a, b))
    assert bound.config_spec.matmul_facts[0].lhs_dtype == torch.bfloat16


@skipUnlessBackends(["triton"])
@pytest.mark.skipif(not SUPPORTED, reason="requires a supported NVIDIA GPU")
@pytest.mark.parametrize(
    ("fast_math", "precision"), [(False, "tf32"), (True, "ieee"), (True, "tf32x3")]
)
def test_strict_settings_do_not_rewrite(fast_math, precision):
    a = torch.randn((32, 64), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((64, 32), device="cuda", dtype=torch.bfloat16)
    with patch.object(
        rewrite, "narrow_dot_operands", wraps=rewrite.narrow_dot_operands
    ) as mocked:
        bound = helion.kernel(
            _matmul, fast_math=fast_math, dot_precision=precision
        )._bind_isolated((a, b))
    mocked.assert_not_called()
    assert bound.config_spec.matmul_facts[0].lhs_dtype == torch.float32


@skipUnlessBackends(["triton"])
@pytest.mark.skipif(not FP8_SUPPORTED, reason="requires Hopper or Blackwell FP8")
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("upcast", [False, True])
def test_fp8_accumulates_accurately(dtype, upcast):
    torch.manual_seed(123)
    a = torch.randn((64, 8192), device="cuda").to(dtype)
    b = torch.randn((8192, 32), device="cuda").to(dtype)
    kernel = (
        helion.kernel(_matmul, fast_math=True, dot_precision="tf32")
        if upcast
        else helion.kernel(_native)
    )
    bound = kernel._bind_isolated((a, b))
    cfg = helion.Config(block_sizes=[64, 32, 64], indexing="pointer", num_warps=4)
    assert ("max_num_imprecise_acc=0" in bound.to_code(cfg)) == (
        CUDA_CAPABILITY == (9, 0)
    )
    result = bound.compile_config(cfg)(a, b)
    torch.testing.assert_close(result, a.float() @ b.float(), rtol=0.001, atol=0.01)
