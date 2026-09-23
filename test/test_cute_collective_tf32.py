from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from examples.jagged_dense_bmm import jagged_dense_bmm
import pytest
import torch

from test._cute_binding import _cpu_bind

import helion
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._testing import skipUnlessBackends
from helion.autotuner.benchmarking import _make_cudagraph_replay
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

CUDA_DEVICE = "cuda"

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel


def _computed_batched_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    batches, rows, reduction = a.shape
    columns = b.size(2)
    out = torch.empty((batches, rows, columns), dtype=a.dtype, device=a.device)
    for batch, row, column in hl.tile([batches, rows, columns]):
        acc = hl.zeros([batch, row, column], dtype=torch.float32)
        for contraction in hl.tile(reduction):
            left = a[batch, row, contraction] * 0.75
            acc = torch.baddbmm(acc, left, b[batch, contraction, column])
        out[batch, row, column] = acc + 0.25
    return out


def _kernel(
    *, precision: str = "tf32", batched: bool = False, static_shapes: bool = True
) -> Kernel[torch.Tensor]:
    return helion.kernel(
        _computed_batched_matmul if batched else jagged_dense_bmm.fn,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
        dot_precision=precision,
    )


def _config(
    bm: int = 64,
    bn: int = 32,
    bk: int = 64,
    *,
    compute: str = "tcgen05",
    copy: str = "scalar",
) -> helion.Config:
    return helion.Config(
        block_sizes=[1, bm, bn, bk],
        num_threads=[0, 128 // bn, bn, 1],
        cute_vector_widths=[1] * 4,
        cute_lane_layouts=["blocked"] * 4,
        cute_collective_mma=True,
        cute_collective_compute=compute,
        cute_collective_copy=copy,
    )


def _jagged_inputs(
    *, batch: int, rows: int, k: int, n: int, device: str = "cpu"
) -> tuple[torch.Tensor, ...]:
    offsets = torch.zeros(batch + 1, dtype=torch.int64, device=device)
    offsets[-1] = rows
    return (
        offsets,
        torch.empty((rows, k), device=device),
        torch.empty((batch, k, n), device=device),
        torch.empty((batch, n), device=device),
    )


@pytest.mark.parametrize(
    "shape", [(16, 319, 32, 32), (64, 4674, 64, 64), (256, 30856, 128, 128)]
)
def test_original_jagged_shapes_have_native_and_warp_fp32_seeds(
    shape: tuple[int, int, int, int],
) -> None:
    batch, rows, k, n = shape
    arguments = _jagged_inputs(batch=batch, rows=rows, k=k, n=n)
    bound = _cpu_bind(_kernel(), arguments)
    assert bound.host_function is not None
    assert bound.host_function.device_ir.grid_block_ids == [[0, 2]]
    assert "input_tensor_metadata" in bound.env.compiler_fact_specialization_facts
    with bound.env:
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    assert seeds
    assert {seed.get("cute_collective_compute", "warp") for seed in seeds} == {
        "warp",
        "tcgen05",
    }
    for seed in seeds:
        source = bound.to_code(seed)
        ast.parse(source)
        if seed.get("cute_collective_compute", "warp") == "tcgen05":
            assert "tcgen05.CtaGroup.ONE" in source
        else:
            assert "warp.MmaTF32Op((16, 8, 8))" in source
            assert "TmemAllocator" not in source
        assert "cvt_f32_tf32" in source
        assert "cutlass.Float16" not in source and "cutlass.BFloat16" not in source
        assert "for tile_offset_1 in range" in source
        assert "for tile_offset_2 in range" not in source


@pytest.mark.parametrize("precision", ["ieee", "tf32x3"])
def test_exact_fp32_precision_keeps_scalar_fallback(precision: str) -> None:
    bound = _cpu_bind(
        _kernel(precision=precision), _jagged_inputs(batch=3, rows=137, k=78, n=128)
    )
    assert bound.host_function is not None
    with bound.env:
        assert not CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    source = bound.to_code(_config())
    assert "cvt_f32_tf32" not in source
    assert "cute.gemm(" not in source
    assert "cutlass.Float16" not in source and "cutlass.BFloat16" not in source


def test_fp32_does_not_fall_through_to_half_warp_mma() -> None:
    bound = _cpu_bind(_kernel(), _jagged_inputs(batch=3, rows=137, k=78, n=128))
    source = bound.to_code(_config(compute="warp"))
    assert "warp.MmaTF32Op((16, 8, 8))" in source
    assert "cute.gemm(" in source
    assert "MmaF16BF16Op" not in source
    assert "cutlass.Float16" not in source and "cutlass.BFloat16" not in source


@pytest.mark.parametrize("tile", [(64, 32, 16), (64, 64, 64), (128, 64, 128)])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_computed_batched_fp32_tail_uses_native_collective(
    tile: tuple[int, int, int], static_shapes: bool
) -> None:
    bound = _cpu_bind(
        _kernel(batched=True, static_shapes=static_shapes),
        (torch.empty((3, 130, 78)), torch.empty((3, 78, 70))),
    )
    source = bound.to_code(_config(*tile))
    assert "tcgen05.CtaGroup.ONE" in source
    assert "cvt_f32_tf32" in source
    assert "0.75" in source and "0.25" in source
    assert "mask_1" in source and "mask_2" in source and "mask_3" in source
    assert "SmemLayoutAtomKind.MN_SW128_32B" in source


def test_multiple_batch_elements_cannot_share_one_collective_tile() -> None:
    bound = _cpu_bind(
        _kernel(batched=True), (torch.empty((3, 130, 78)), torch.empty((3, 78, 70)))
    )
    config = _config()
    config.block_sizes[0] = 2
    source = bound.to_code(config)
    assert "cute.gemm(" not in source
    assert "TmemAllocator" not in source


def _round_tf32(value: torch.Tensor) -> torch.Tensor:
    # Independent finite-input oracle for cvt.rna.tf32.f32: add half a TF32
    # ULP to the IEEE magnitude then discard 13 significand bits.
    bits = value.contiguous().view(torch.int32)
    return ((bits + 0x1000) & -0x2000).view(torch.float32)


def _require_sm100() -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "shape", [(16, 319, 32, 32), (64, 4674, 64, 64), (256, 30856, 128, 128)]
)
@pytest.mark.parametrize(
    "copy,compute,recipe",
    [
        pytest.param("scalar", "tcgen05", "scalar", id="tcgen05-scalar"),
        pytest.param("async_cached", "tcgen05", "scalar", id="tcgen05-async_cached"),
        pytest.param("scalar", "warp", "scalar", id="warp-scalar"),
        pytest.param("async_cached", "warp", "scalar", id="warp-async_cached"),
        pytest.param("scalar", "tcgen05", "vector", id="tcgen05-register-vector"),
        pytest.param(
            "scalar", "tcgen05", "vector_unrolled", id="tcgen05-register-unrolled"
        ),
    ],
)
def test_original_jagged_fp32_and_mutating_metadata(
    shape: tuple[int, int, int, int], copy: str, compute: str, recipe: str
) -> None:
    _require_sm100()
    torch.manual_seed(20260912)
    batch, rows, k, n = shape
    lengths = torch.full((batch,), rows // batch, dtype=torch.int64)
    lengths[: rows % batch] += 1
    lengths[1] += lengths[0]
    lengths[0] = 0
    arguments = _jagged_inputs(batch=batch, rows=rows, k=k, n=n, device=CUDA_DEVICE)
    offsets, a, b, bias = arguments

    def mutate() -> None:
        nonlocal lengths
        lengths = lengths.roll(1)
        offsets[0] = 0
        offsets[1:].copy_(lengths.cumsum(0))
        a.normal_()
        b.normal_()
        bias.normal_()

    def expected() -> torch.Tensor:
        left, right = _round_tf32(a).double(), _round_tf32(b).double()
        pieces = []
        begin = 0
        for group, count in enumerate(lengths.tolist()):
            end = begin + count
            pieces.append((left[begin:end] @ right[group]).float() + bias[group])
            begin = end
        return torch.cat(pieces)

    mutate()
    kernel = _kernel()
    bound = kernel.bind(arguments)
    config = _config(
        128 if compute == "tcgen05" else 32,
        32 if n == 32 else 64,
        64,
        compute=compute,
        copy=copy,
    )
    config = helion.Config.from_dict(config.config | {"cute_collective_recipe": recipe})
    source = bound.to_code(config)
    mma_marker = "tcgen05.CtaGroup.ONE" if compute == "tcgen05" else "warp.MmaTF32Op"
    assert mma_marker in source
    assert ("collective_a_raw" in source) == (copy == "async_cached")
    bound.set_config(config)

    def run() -> torch.Tensor:
        return bound(*arguments)

    torch.testing.assert_close(run(), expected(), rtol=2e-4, atol=2e-4)
    replay = _make_cudagraph_replay(run)
    for _iteration in range(3):
        mutate()
        torch.testing.assert_close(replay(), expected(), rtol=2e-4, atol=2e-4)
    replacement = tuple(value.clone() for value in arguments)
    torch.testing.assert_close(bound(*replacement), expected(), rtol=2e-4, atol=2e-4)
    if copy == "async_cached":
        unaligned_a = torch.empty(a.numel() + 1, device=CUDA_DEVICE)[1:].view_as(a)
        unaligned_a.copy_(a)
        unaligned_args = (offsets, unaligned_a, b, bias)
        unaligned = kernel.bind(unaligned_args)
        assert unaligned is not bound
        unaligned_source = unaligned.to_code(config)
        assert mma_marker in unaligned_source
        assert "collective_a_raw" not in unaligned_source
        unaligned.set_config(config)
        torch.testing.assert_close(
            unaligned(*unaligned_args), expected(), rtol=2e-4, atol=2e-4
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "copy,compute",
    [
        pytest.param("scalar", "tcgen05", id="tcgen05-scalar"),
        pytest.param("async_cached", "tcgen05", id="tcgen05-async_cached"),
        pytest.param("scalar", "warp", id="warp-scalar"),
        pytest.param("async_cached", "warp", id="warp-async_cached"),
    ],
)
def test_jagged_fp32_reduction_tail_ignores_backing_storage(
    copy: str, compute: str
) -> None:
    _require_sm100()
    torch.manual_seed(20260912)
    batch, rows, k, n, bk = 16, 319, 32, 32, 64
    lengths = torch.tensor([19, 0, 40] + [20] * 13, dtype=torch.int64)
    offsets = torch.zeros(batch + 1, dtype=torch.int64, device=CUDA_DEVICE)
    offsets[1:].copy_(lengths.cumsum(0))
    # Keep the invalid K coordinates inside a live allocation, beyond the
    # logical A view. This controls the padding without allocator reuse.
    backing = torch.empty(rows * k + bk, device=CUDA_DEVICE)
    a = backing[: rows * k].view(rows, k)
    a.normal_()
    b = torch.randn((batch, k, n), device=CUDA_DEVICE)
    bias = torch.randn((batch, n), device=CUDA_DEVICE)
    arguments = (offsets, a, b, bias)
    original = tuple(value.clone() for value in arguments)
    assert a.is_contiguous() and a.stride() == (k, 1)
    assert a.data_ptr() == backing.data_ptr()

    left, right = _round_tf32(a).double(), _round_tf32(b).double()
    pieces = []
    begin = 0
    for group, count in enumerate(lengths.tolist()):
        end = begin + count
        pieces.append((left[begin:end] @ right[group]).float() + bias[group])
        begin = end
    expected = torch.cat(pieces)
    assert torch.isfinite(expected).all()

    bound = _kernel().bind(arguments)
    config = _config(
        128 if compute == "tcgen05" else 32,
        n,
        bk,
        compute=compute,
        copy=copy,
    )
    config = helion.Config.from_dict(
        config.config | {"cute_collective_recipe": "scalar"}
    )
    source = bound.to_code(config)
    mma_marker = "tcgen05.CtaGroup.ONE" if compute == "tcgen05" else "warp.MmaTF32Op"
    assert mma_marker in source
    assert ("collective_a_raw" in source) == (copy == "async_cached")
    bound.set_config(config)

    outputs = []
    for padding in (0.0, float("nan"), 0.0):
        backing[rows * k :].fill_(padding)
        output = bound(*arguments)
        outputs.append(output)
        torch.testing.assert_close(output, expected, rtol=2e-4, atol=2e-4)
        for before, after in zip(original, arguments, strict=True):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert len({output.data_ptr() for output in outputs}) == len(outputs)
    for output in outputs:
        torch.testing.assert_close(output, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("static_shapes", [False, True])
@pytest.mark.parametrize("layout", ["n_major", "k_major", "unaligned"])
def test_computed_fp32_mixed_copy_and_tails(static_shapes: bool, layout: str) -> None:
    _require_sm100()
    torch.manual_seed(1709)
    a = torch.randn((3, 130, 78), device=CUDA_DEVICE)
    if layout == "k_major":
        b = torch.randn((3, 70, 80), device=CUDA_DEVICE)[:, :, :78].transpose(1, 2)
    elif layout == "n_major":
        b = torch.randn((3, 78, 72), device=CUDA_DEVICE)[:, :, :70]
    else:
        b = torch.randn(3 * 78 * 70 + 1, device=CUDA_DEVICE)[1:].view(3, 78, 70)
    kernel = _kernel(batched=True, static_shapes=static_shapes)
    bound = kernel.bind((a, b))
    config = _config(128, 64, 128, copy="async_cached")
    source = bound.to_code(config)
    assert "tcgen05.CtaGroup.ONE" in source
    # The transformed left operand must keep recipe staging and TF32 rounding.
    assert "collective_a_raw" not in source
    if static_shapes and layout != "unaligned":
        assert "collective_b_raw" in source
    bound.set_config(config)

    def expected() -> torch.Tensor:
        left = _round_tf32(a * 0.75).double()
        right = _round_tf32(b).double()
        return torch.bmm(left, right).float() + 0.25

    def run() -> torch.Tensor:
        return bound(a, b)

    torch.testing.assert_close(run(), expected(), rtol=2e-4, atol=2e-4)
    replay = _make_cudagraph_replay(run)
    for _iteration in range(3):
        a.normal_()
        b.normal_()
        torch.testing.assert_close(replay(), expected(), rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("copy", ["scalar", "async_cached"])
def test_fp32_empty_reduction_preserves_epilogue(copy: str) -> None:
    _require_sm100()
    a = torch.empty((3, 130, 0), device=CUDA_DEVICE)
    b = torch.empty((3, 0, 70), device=CUDA_DEVICE)
    kernel = _kernel(batched=True)
    bound = kernel.bind((a, b))
    bound.set_config(_config(copy=copy))
    torch.testing.assert_close(
        bound(a, b), torch.full((3, 130, 70), 0.25, device=CUDA_DEVICE), rtol=0, atol=0
    )
