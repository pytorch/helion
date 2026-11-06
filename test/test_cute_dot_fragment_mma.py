from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import Tensor

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = skipUnlessBackends(["cute"])


def gemv(w: Tensor, x: Tensor) -> Tensor:
    """``w @ x`` as the scalar dot fallback: one row tile, a K loop, N = 1."""
    M, K = w.size()
    x2 = x.view(K, 1)
    out = torch.empty([M, 1], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(M):
        acc = hl.zeros([tile_m, 1], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = hl.dot(w[tile_m, tile_k], x2[tile_k, :], acc=acc)
        out[tile_m, :] = acc
    return out.view(M)


def gemv_scaled(w: Tensor, x: Tensor) -> Tensor:
    """A computed row operand: the element loop is not the plain fold."""
    M, K = w.size()
    x2 = x.view(K, 1)
    out = torch.empty([M, 1], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(M):
        acc = hl.zeros([tile_m, 1], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = hl.dot(w[tile_m, tile_k] * 2, x2[tile_k, :], acc=acc)
        out[tile_m, :] = acc
    return out.view(M)


def gather_gemv(w: Tensor, idx: Tensor, x: Tensor) -> Tensor:
    """examples/gather_gemv.py: a gathered row index and ``acc +=``."""
    B, S, _ = w.size()
    N = idx.size(0)
    w_view = w.view(B * S, S)
    x2 = x.view(S, 1)
    out = torch.empty([N * S, 1], dtype=x.dtype, device=x.device)
    for tile in hl.tile(N * S):
        acc = hl.zeros([tile, 1], dtype=torch.float32)
        idx_gather = idx[tile.index // S]
        for tile_k in hl.tile(S):
            gathered = w_view[idx_gather * S + tile.index % S, tile_k]
            acc += hl.dot(gathered, x2[tile_k, :])
        out[tile, :] = acc
    return out.view(N, S)


def _fragment_config(**extra: Any) -> helion.Config:
    return helion.Config(
        block_sizes=[16, 1024],
        num_threads=[8, 4],
        cute_vector_widths=[1, 16],
        cute_lane_layouts=["strided", "strided"],
        cute_dot_fragment_mma=True,
        **extra,
    )


def _cpu_code(
    fn: Callable[..., object], args: tuple[object, ...], config: helion.Config
) -> str:
    with (
        mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        kernel = helion.kernel(
            fn,
            backend="cute",
            static_shapes=True,
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )
        bound = cpu_bind(kernel, args)
        return bound.to_code(config)


@pytest.mark.parametrize(
    ("dtype", "helper"),
    [(torch.float16, "_cute_mma_f16"), (torch.bfloat16, "_cute_mma_bf16")],
)
def test_gemv_dot_runs_on_mma_fragments(dtype: torch.dtype, helper: str) -> None:
    code = _cpu_code(
        gemv,
        (torch.empty(256, 4096, dtype=dtype), torch.empty(4096, dtype=dtype)),
        _fragment_config(),
    )
    # Four m16n8k16 instructions per 32-byte packet, no shuffle reduction,
    # no per-element converts.
    assert code.count(f"{helper}(") == 4
    assert "_cute_grouped_reduce_warp(" not in code
    assert "warp_reduction" not in code
    assert "bitcast(cutlass.Float16)" not in code
    assert "bitcast(cutlass.BFloat16)" not in code
    # The row packets are read as 8 words; the vector chunk (1024 elements,
    # 2 KB) is staged in shared memory once per K chunk (one 32-byte packet
    # per lane and step: 2 steps, 2 shared stores each) and read back as two
    # 16-byte shared loads per packet.
    assert "ir.VectorType.get([8], cutlass.Uint32.mlir_type)" in code
    assert "cute.arch.alloc_smem(cutlass.Uint32, 512)" in code
    assert code.count("_cute_sts_v4_b32(") == 4
    assert code.count("_cute_lds_v4_b32(") == 2
    assert code.count("cute.arch.sync_warp()") == 2
    # The lanes are permuted into the fragment layout: row group
    # ``lane // 4``, K quarter ``lane % 4``.
    assert (
        "(cutlass.Int32(cute.arch.thread_idx()[0]) + "
        "cutlass.Int32(cute.arch.thread_idx()[1]) * 8) // 4" in code
    )
    assert (
        "(cutlass.Int32(cute.arch.thread_idx()[0]) + "
        "cutlass.Int32(cute.arch.thread_idx()[1]) * 8) % 4" in code
    )


def test_gathered_row_operand_takes_the_fragments() -> None:
    code = _cpu_code(
        gather_gemv,
        (
            torch.empty(8, 2048, 2048, dtype=torch.float16),
            torch.empty(2, dtype=torch.int32),
            torch.empty(2048, dtype=torch.float16),
        ),
        _fragment_config(),
    )
    assert code.count("_cute_mma_f16(") == 4
    assert "_cute_grouped_reduce_warp(" not in code
    # The row-bounds mask moves from the elements to the packet words.
    assert code.count("else _fmma_zero") == 8


def test_short_chunk_keeps_the_vector_in_registers() -> None:
    config = _fragment_config()
    config.config["block_sizes"] = [16, 256]
    code = _cpu_code(
        gemv,
        (
            torch.empty(256, 4096, dtype=torch.float16),
            torch.empty(4096, dtype=torch.float16),
        ),
        config,
    )
    # 256 elements are 16 packets: not a whole 32-lane fill step, so the
    # vector packet stays a register load and no shared memory is used.
    assert code.count("_cute_mma_f16(") == 4
    assert "alloc_smem" not in code
    assert "_cute_lds_v4_b32(" not in code
    assert code.count("ir.VectorType.get([8], cutlass.Uint32.mlir_type)") == 2


def test_k_tail_lifts_the_element_bound_to_the_packet() -> None:
    code = _cpu_code(
        gemv,
        (
            torch.empty(250, 4000, dtype=torch.float16),
            torch.empty(4000, dtype=torch.float16),
        ),
        _fragment_config(),
    )
    # 4000 is 250 whole packets: the element loop's ``mask_1 = index < 4000``
    # becomes the lane-base bound on the row words, and the staged vector
    # packets a tail chunk clamps are stored as zeros (8 words x 2 steps).
    assert code.count("_cute_mma_f16(") == 4
    assert "mask_1" not in code
    assert code.count("if mask_0 and lane_base_1 < 4000 else _fmma_zero") == 8
    assert code.count("else cutlass.Uint32(0)") == 16


def test_k_not_whole_packets_regenerates_knob_off() -> None:
    code = _cpu_code(
        gemv,
        (
            torch.empty(250, 4008, dtype=torch.float16),
            torch.empty(4008, dtype=torch.float16),
        ),
        _fragment_config(),
    )
    # 4008 elements end inside a packet: the element-level bound cannot be
    # lifted, so the kernel keeps the scalar fallback.
    assert "_cute_mma_f16(" not in code
    assert "_cute_grouped_reduce_warp(" in code


def test_computed_row_operand_regenerates_knob_off() -> None:
    code = _cpu_code(
        gemv_scaled,
        (
            torch.empty(256, 4096, dtype=torch.float16),
            torch.empty(4096, dtype=torch.float16),
        ),
        _fragment_config(),
    )
    # The element loop scales the operand first: not the plain fold, so the
    # kernel is generated again with the knob off and keeps the shuffle sum.
    assert "_cute_mma_f16(" not in code
    assert "_cute_grouped_reduce_warp(" in code


def test_knob_needs_the_fragment_thread_layout() -> None:
    with mock_cuda_unavailable():
        kernel = helion.kernel(gemv, backend="cute", static_shapes=True)
        bound = cpu_bind(
            kernel,
            (
                torch.empty(256, 4096, dtype=torch.float16),
                torch.empty(4096, dtype=torch.float16),
            ),
        )
        spec = bound.config_spec
    assert spec.normalized_config(_fragment_config()).config["cute_dot_fragment_mma"]
    # Other thread counts, 8-byte packets and fp32 operands cannot take it.
    off = _fragment_config()
    off.config["num_threads"] = [1, 32]
    assert not spec.normalized_config(off).config["cute_dot_fragment_mma"]
    off = _fragment_config()
    off.config["cute_vector_widths"] = [1, 4]
    assert not spec.normalized_config(off).config["cute_dot_fragment_mma"]


def test_fp32_dot_has_no_fragment_knob() -> None:
    with mock_cuda_unavailable():
        kernel = helion.kernel(gemv, backend="cute", static_shapes=True)
        bound = cpu_bind(
            kernel,
            (
                torch.empty(256, 4096, dtype=torch.float32),
                torch.empty(4096, dtype=torch.float32),
            ),
        )
        spec = bound.config_spec
    with pytest.raises(helion.exc.InvalidConfig):
        spec.normalized_config(_fragment_config())
    assert "cute_dot_fragment_mma" not in spec._flat_fields()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="mma.sync m16n8k16 needs compute capability 8.0",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("rows", "k", "chunk", "width", "unroll"),
    [
        (256, 4096, 1024, 16, 1),
        (256, 4096, 1024, 16, 4),
        (256, 4096, 256, 16, 1),
        (256, 4096, 128, 16, 1),
        (256, 4096, 128, 8, 1),
        (256, 4096, 64, 16, 1),
        (256, 4096, 4096, 16, 1),
        (256, 256, 256, 16, 1),
        (256, 512, 512, 16, 1),
        (250, 4000, 1024, 16, 1),
    ],
)
def test_gemv_fragments_match_torch(
    dtype: torch.dtype, rows: int, k: int, chunk: int, width: int, unroll: int
) -> None:
    # Chunks of 512 elements and up stage the vector in shared memory (whole
    # 32-packet fill steps); 256 and 128 (16- or 32-byte packets) and 64 keep
    # it in registers; 4096 is the 8 KB staging cap; 256/512 are whole-K
    # chunks of a small K; 250 x 4000 has a row tail and a K tail (the
    # element loop's K bound lifted to the packet).
    torch.manual_seed(0)
    w = torch.randn(rows, k, device=DEVICE, dtype=dtype)
    x = torch.randn(k, device=DEVICE, dtype=dtype)
    config = _fragment_config(cute_lane_unroll=unroll)
    config.config["block_sizes"] = [16, chunk]
    config.config["cute_vector_widths"] = [1, width]
    kernel = helion.kernel(gemv, backend="cute", static_shapes=True, config=config)
    code = kernel.bind((w, x)).to_triton_code(kernel.configs[0])
    assert "_cute_mma_" in code
    out = kernel(w, x)
    expected = (w.float() @ x.float()).to(dtype)
    torch.testing.assert_close(out.float(), expected.float(), rtol=1e-2, atol=1e-1)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="mma.sync m16n8k16 needs compute capability 8.0",
)
def test_gather_gemv_fragments_match_torch() -> None:
    torch.manual_seed(0)
    w = torch.randn(8, 2048, 2048, device=DEVICE, dtype=torch.float16)
    idx = torch.randint(0, 8, (2,), device=DEVICE, dtype=torch.int32)
    x = torch.randn(2048, device=DEVICE, dtype=torch.float16)
    kernel = helion.kernel(
        gather_gemv,
        backend="cute",
        static_shapes=True,
        config=helion.Config(
            block_sizes=[8, 512],
            num_threads=[8, 4],
            cute_vector_widths=[1, 16],
            cute_lane_layouts=["strided", "strided"],
            cute_dot_fragment_mma=True,
            cute_min_blocks_per_mp=1,
        ),
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    out = kernel(w, idx, x)
    expected = (w[idx].float() @ x.float()).to(torch.float16)
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-1)
