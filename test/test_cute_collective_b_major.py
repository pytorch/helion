from __future__ import annotations

import ast
from unittest.mock import patch

from examples.mamba2_chunk_scan import helion_mamba2_chunk_scan_kernel
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._testing import skipUnlessBackends
from helion.autotuner.accuracy import _chunked_assert_close
from helion.autotuner.benchmarking import _make_cudagraph_replay
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

CUDA_DEVICE = "cuda"


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _transposed_rhs(
    a: torch.Tensor, b: torch.Tensor, computed_b: hl.constexpr
) -> torch.Tensor:
    m, k = a.shape
    n = b.size(0)
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    for row, col in hl.tile([m, n]):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for red in hl.tile(k):
            left = a[row, red] * 0.5
            right = b[col, red].T
            if computed_b:
                right = right * 0.5
            acc = torch.addmm(acc, left, right)
        out[row, col] = acc.to(out.dtype)
    return out


def _inputs(
    dtype: torch.dtype, *, major: str = "k", stride_k: int = 80
) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.empty((130, 80), dtype=dtype)[:, :78]
    b = (
        torch.empty((70, stride_k), dtype=dtype)[:, :78]
        if major == "k"
        else torch.empty((78, 80), dtype=dtype)[:, :70].T
    )
    return a, b


def _source(
    inputs: tuple[torch.Tensor, torch.Tensor],
    *,
    static_shapes: bool = False,
    tile: tuple[int, int, int] = (32, 32, 64),
    copy: str = "async",
    compute: str = "warp",
    computed_b: bool = False,
) -> str:
    kernel = helion.kernel(
        _transposed_rhs.fn,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
    )
    bm, bn, bk = tile
    config = helion.Config(
        block_sizes=[bm, bn, bk],
        num_threads=[128 // bn, bn, 1],
        cute_vector_widths=[1, 1, 1],
        cute_collective_mma=True,
        cute_collective_copy=copy,
        cute_collective_compute=compute,
    )
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        return _cpu_bind(kernel, (*inputs, computed_b)).to_code(config)


def _b_copy(source: str) -> ast.For:
    return next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "collective_bi"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("static_shapes", [False, True])
@pytest.mark.parametrize("copy", ["async", "async_cached"])
@pytest.mark.parametrize("tile", [(32, 32, 16), (32, 32, 64), (64, 64, 128)])
def test_transposed_rhs_copies_along_proven_contiguous_k(
    dtype: torch.dtype, static_shapes: bool, copy: str, tile: tuple[int, int, int]
) -> None:
    source = _source(_inputs(dtype), static_shapes=static_shapes, copy=copy, tile=tile)
    bn, bk = tile[1:]
    body = ast.unparse(_b_copy(source))
    assert "cute.gemm(" in source
    assert f"stride=({bk}, 1)" in source
    assert "LdMatrix8x8x16bOp(transpose=True" not in source
    assert "cute.arch.cp_async_shared_global(" in body
    assert f"collective_n = collective_flat // {bk}" in body
    assert f"collective_k = collective_flat % {bk}" in body
    # The physical vector traversal spans the whole tile; its scalar fallback
    # still masks each element for the 78-element K and 70-element N tails.
    assert f"collective_flat < {bn * bk}" in body
    assert "copy_tail_lane" in body
    assert "else cutlass." in body
    assert "cute.arch.cp_async_wait_group(0)" in source


@pytest.mark.parametrize("copy", ["async", "async_cached"])
def test_n_contiguous_rhs_retains_existing_layout(copy: str) -> None:
    source = _source(_inputs(torch.float16, major="n"), copy=copy)
    body = ast.unparse(_b_copy(source))
    assert "stride=(1, 32)" in source
    assert "LdMatrix8x8x16bOp(transpose=True" in source
    assert "cute.arch.cp_async_shared_global(" in body
    assert "collective_n = collective_flat % 32" in body
    assert "collective_k = collective_flat // 32" in body


@pytest.mark.parametrize("reason", ["scalar", "computed", "stride", "offset"])
def test_k_major_requires_direct_aligned_vector_copy(reason: str) -> None:
    inputs = _inputs(torch.float16, stride_k=78 if reason == "stride" else 80)
    if reason == "offset":
        inputs = (inputs[0], torch.empty((70, 80), dtype=torch.float16)[:, 1:79])
    source = _source(
        inputs,
        copy="scalar" if reason == "scalar" else "async",
        computed_b=reason == "computed",
    )
    assert "stride=(1, 32)" in source
    assert "LdMatrix8x8x16bOp(transpose=True" in source
    assert "cute.arch.cp_async_shared_global(" not in ast.unparse(_b_copy(source))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("static_shapes", [False, True])
@pytest.mark.parametrize("copy", ["async", "async_cached"])
@pytest.mark.parametrize("tile", [(64, 32, 16), (128, 64, 128)])
def test_native_transposed_rhs_uses_proven_k_major_copy(
    dtype: torch.dtype, static_shapes: bool, copy: str, tile: tuple[int, int, int]
) -> None:
    source = _source(
        _inputs(dtype),
        static_shapes=static_shapes,
        tile=tile,
        copy=copy,
        compute="tcgen05",
    )
    bk = tile[2]
    body = ast.unparse(_b_copy(source))
    assert "tcgen05.CtaGroup.ONE" in source
    assert "LdMatrix8x8x16bOp" not in source
    assert "OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K" in source
    assert f"collective_n = collective_flat // {bk}" in body
    assert f"collective_k = collective_flat % {bk}" in body
    # Native descriptors carry a byte swizzle in the shared pointer. Keep the
    # typed copy so both complete vectors and scalar tails honor that swizzle.
    assert "cute.nvgpu.cpasync.CopyG2SOp(" in body
    assert "collective_b.iterator" in body
    assert "copy_tail_lane" in body


def test_native_n_contiguous_rhs_retains_mn_descriptor() -> None:
    source = _source(
        _inputs(torch.bfloat16, major="n"), tile=(64, 32, 64), compute="tcgen05"
    )
    assert "OperandMajorMode.K, cute.nvgpu.OperandMajorMode.MN" in source
    assert "SmemLayoutAtomKind.MN_SW64" in source
    assert "cute.nvgpu.cpasync.CopyG2SOp(" in ast.unparse(_b_copy(source))


@pytest.mark.parametrize("reason", ["scalar", "computed", "stride", "offset"])
def test_native_k_major_retains_full_copy_proof(reason: str) -> None:
    inputs = _inputs(torch.bfloat16, stride_k=78 if reason == "stride" else 80)
    if reason == "offset":
        inputs = (inputs[0], torch.empty((70, 80), dtype=torch.bfloat16)[:, 1:79])
    source = _source(
        inputs,
        tile=(64, 32, 64),
        copy="scalar" if reason == "scalar" else "async",
        compute="tcgen05",
        computed_b=reason == "computed",
    )
    assert "OperandMajorMode.K, cute.nvgpu.OperandMajorMode.MN" in source
    assert "cute.nvgpu.cpasync.CopyG2SOp(" not in ast.unparse(_b_copy(source))


@pytest.mark.parametrize("shape", [0, 1, 2])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_mamba_transposed_state_uses_k_copy(shape: int, static_shapes: bool) -> None:
    batch, heads, groups, sequence, chunk, head, state = (
        (1, 4, 1, 256, 64, 32, 32),
        (2, 8, 2, 1024, 64, 64, 64),
        (2, 16, 4, 2048, 128, 64, 128),
    )[shape]
    chunks = sequence // chunk
    inputs = tuple(
        torch.empty(size, dtype=torch.bfloat16)
        for size in (
            (batch, chunks, groups, chunk, chunk),
            (batch, sequence, heads, head),
            (batch, heads, chunks, chunk),
            (batch, heads, chunks, chunk),
            (batch, sequence, groups, state),
            (batch, chunks, heads, head, state),
            (heads,),
        )
    )
    kernel = helion.kernel(
        helion_mamba2_chunk_scan_kernel.fn,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
    )
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        bound = _cpu_bind(kernel, inputs)
        config = helion.Config.from_dict(
            bound.config_spec.default_config().config
            | {
                "block_sizes": [32, 32, 64],
                "num_threads": [4, 32, 1, 1],
                "cute_vector_widths": [1] * 7,
                "cute_lane_layouts": ["blocked"] * 7,
                "cute_collective_mma": True,
                "cute_collective_copy": "async",
            }
        )
        source = bound.to_code(config)
    assert source.count("cute.gemm(") == 2
    assert source.count("cute.arch.cp_async_shared_global(") == 3
    assert source.count("LdMatrix8x8x16bOp(transpose=True") == 1
    assert source.count("mul.rn.f32") == 5
    assert "prev_states.iterator" in ast.unparse(_b_copy(source))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("copy", ["async", "async_cached"])
@pytest.mark.parametrize(
    "tile,compute",
    [
        ((32, 32, 16), "warp"),
        ((32, 32, 64), "warp"),
        ((64, 64, 128), "warp"),
        ((64, 32, 16), "tcgen05"),
        ((128, 64, 128), "tcgen05"),
    ],
)
def test_k_major_copy_preserves_dynamic_tails_and_graph_replays(
    dtype: torch.dtype, copy: str, tile: tuple[int, int, int], compute: str
) -> None:
    if compute == "tcgen05" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    a = torch.randn((130, 80), device=CUDA_DEVICE, dtype=dtype)[:, :78]
    b = torch.randn((70, 80), device=CUDA_DEVICE, dtype=dtype)[:, :78]
    kernel = helion.kernel(
        _transposed_rhs.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated((a, b, False))
    bm, bn, bk = tile
    config = helion.Config(
        block_sizes=[bm, bn, bk],
        num_threads=[128 // bn, bn, 1],
        cute_vector_widths=[1, 1, 1],
        cute_collective_mma=True,
        cute_collective_copy=copy,
        cute_collective_compute=compute,
    )
    source = bound.to_code(config)
    if compute == "tcgen05":
        assert "OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K" in source
        assert "cute.nvgpu.cpasync.CopyG2SOp(" in ast.unparse(_b_copy(source))
    else:
        assert f"stride=({bk}, 1)" in source
        assert "cute.arch.cp_async_shared_global(" in ast.unparse(_b_copy(source))
    bound.set_config(config)

    def run() -> torch.Tensor:
        return bound(a, b, False)

    torch.testing.assert_close(run(), (a * 0.5) @ b.T, rtol=1e-2, atol=1e-2)
    replay = _make_cudagraph_replay(run)
    for _iteration in range(3):
        a.normal_()
        b.normal_()
        torch.testing.assert_close(replay(), (a * 0.5) @ b.T, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [1, 2])
def test_native_k_major_original_mamba_and_mutated_graphs(
    shape: int, dtype: torch.dtype
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    batch, heads, groups, sequence, chunk, head, state = (
        (2, 8, 2, 1024, 64, 64, 64),
        (2, 16, 4, 2048, 128, 64, 128),
    )[shape - 1]
    chunks = sequence // chunk
    torch.manual_seed(20260911 + shape)
    x = torch.randn(
        (batch, sequence, heads, head), device=CUDA_DEVICE, dtype=torch.bfloat16
    )
    dt = torch.rand((batch, heads, chunks, chunk), device=CUDA_DEVICE, dtype=x.dtype)
    decay = -torch.rand_like(dt).cumsum(-1)
    c = torch.randn((batch, sequence, groups, state), device=CUDA_DEVICE, dtype=x.dtype)
    cb = torch.randn(
        (batch, chunks, groups, chunk, chunk), device=CUDA_DEVICE, dtype=x.dtype
    )
    previous = torch.randn(
        (batch, chunks, heads, head, state), device=CUDA_DEVICE, dtype=x.dtype
    )
    residual = torch.randn((heads,), device=CUDA_DEVICE, dtype=x.dtype)
    arguments = tuple(
        value.to(dtype) for value in (cb, x, dt, decay, c, previous, residual)
    )
    kernel = helion.kernel(
        helion_mamba2_chunk_scan_kernel.fn,
        backend="cute",
        static_shapes=True,
        cute_region_fission=True,
        cute_full_slice_matmul_tiling=True,
        autotune_effort="none",
    )
    reference = kernel._bind_isolated(arguments)
    reference.set_config(reference.config_spec.autotune_reference_config())
    bound = kernel._bind_isolated(arguments)
    config = helion.Config(
        block_sizes=[64, 32, 64],
        num_threads=[4, 32, 1, 1],
        cute_vector_widths=[1] * 7,
        cute_lane_layouts=["blocked"] * 7,
        cute_collective_mma=True,
        cute_collective_compute="tcgen05",
        cute_collective_copy="async_cached",
    )
    source = bound.to_code(config)
    assert source.count("cute.gemm(") == 2
    assert "OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K" in source
    assert source.count("mul.rn.f32") == 5
    bound.set_config(config)

    def compare(actual: torch.Tensor, expected: torch.Tensor) -> None:
        _chunked_assert_close(
            actual, expected, atol=0.01, rtol=0.01, scale_atol_by_expected_rms=True
        )

    def run() -> torch.Tensor:
        return bound(*arguments)

    expected = reference(*arguments)
    for _iteration in range(3):
        compare(run(), expected)
    replay = _make_cudagraph_replay(run)
    for factor in (0.875, 1.125, 0.75):
        arguments[1].mul_(factor)
        arguments[5].mul_(factor)
        compare(replay(), reference(*arguments))
    replacement = tuple(value.clone() for value in arguments)
    compare(bound(*replacement), reference(*replacement))
