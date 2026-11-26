"""GPU numerical tests for CuTe fragment transfers and staged windows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion._testing.cute_fragment_kernels import anchored_gram
from helion._testing.cute_fragment_kernels import anchored_gram_reference
from helion._testing.cute_fragment_kernels import chunk_cumsum
from helion._testing.cute_fragment_kernels import fragment_config
from helion._testing.cute_fragment_kernels import singleton_tail_scan
from helion._testing.cute_fragment_kernels import varlen_anchored_gram
from helion._testing.cute_fragment_kernels import varlen_contiguous_dim_windows
from helion._testing.cute_fragment_kernels import varlen_contiguous_inputs
from helion._testing.cute_fragment_kernels import varlen_gram_inputs

if TYPE_CHECKING:
    from collections.abc import Callable

VEC = "cute_fragment_vector_io"
BATCH = "cute_fragment_batched_loads"
SPLIT = "cute_fragment_split_k"
SLAB = "cute_fragment_slab_loads"
DOT = "cute_fp32_dot"
PAD = "cute_fragment_padded_rows"
MMA = "_helion_fragment_mma.mma_tf32("
SYNC = "cute.arch.sync_threads()"
PACKET = "num_bits_per_copy=128"
pytestmark = skipUnlessBackends(["cute"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("vec", "batch", "split"),
    [(False, False, False), (True, True, False), (True, True, True)],
)
def test_anchored_gram_slabs_match_reference(
    vec: bool, batch: bool, split: bool
) -> None:
    torch.manual_seed(0)
    q = torch.randn(64, 32, 128, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(64, 32, 128, device=DEVICE, dtype=torch.bfloat16)
    gc = (-torch.rand(64, 32, 128, device=DEVICE) * 0.1).cumsum(1)
    scale = 0.125
    bound = anchored_gram.bind((q, k, gc, scale))
    knobs = {
        DOT: "tf32_mma",
        PAD: True,
        "cute_fragment_unrolled_loops": True,
        "cute_fragment_threads": 256,
        SLAB: True,
        VEC: vec,
        BATCH: batch,
    }
    if split:
        knobs[SPLIT] = True
    config = fragment_config(bound, **knobs)
    code = bound.to_code(config)
    assert code.count(".load()") + code.count("cute.copy(") // 2 <= 3 + 2
    out = bound.compile_config(config)(q, k, gc, scale)
    torch.cuda.synchronize()
    expected = anchored_gram_reference(q, k, gc, scale)
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
    if not split:
        plain = bound.compile_config(
            fragment_config(bound, **(knobs | {SLAB: False, VEC: False, BATCH: False}))
        )(q, k, gc, scale)
        torch.cuda.synchronize()
        assert torch.equal(out, plain)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_in_place_cast_operand_matches_reference() -> None:
    torch.manual_seed(0)
    g = torch.randn(8, 64, 128, device=DEVICE, dtype=torch.bfloat16)
    bound = chunk_cumsum.bind((g,))
    config = fragment_config(
        bound, **{DOT: "tf32_mma", PAD: True, VEC: True, "block_sizes": [64]}
    )
    code = bound.to_code(config)
    # One barrier after the packet load, one after the lane loop (which
    # stores the result itself: test_cute_fragment_direct_store).
    assert code.count(SYNC) == 2
    out = bound.compile_config(config)(g)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, g.float().cumsum(1), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("dot", "threads", "vec", "batch", "split"),
    [
        ("simt", 128, True, False, False),
        ("simt", 256, False, True, False),
        ("tf32_mma", 128, True, True, False),
        ("tf32_mma", 256, True, True, True),
        ("tf32_mma", 512, False, False, True),
    ],
)
def test_anchored_gram_transport_matches_reference(
    dot: str, threads: int, vec: bool, batch: bool, split: bool
) -> None:
    torch.manual_seed(0)
    q = torch.randn(64, 32, 128, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(64, 32, 128, device=DEVICE, dtype=torch.bfloat16)
    gc = (-torch.rand(64, 32, 128, device=DEVICE) * 0.1).cumsum(1)
    scale = 0.125
    bound = anchored_gram.bind((q, k, gc, scale))
    knobs = {
        DOT: dot,
        PAD: True,
        "cute_fragment_unrolled_loops": True,
        "cute_fragment_threads": threads,
        VEC: vec,
        BATCH: batch,
    }
    if split:
        knobs[SPLIT] = True
    config = fragment_config(bound, **knobs)
    code = bound.to_code(config)
    assert (PACKET in code) == vec
    assert ("fragment_mma_split" in code) == split
    out = bound.compile_config(config)(q, k, gc, scale)
    torch.cuda.synchronize()
    expected = anchored_gram_reference(q, k, gc, scale)
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
    # The transport knobs move the same scalars: the staged tiles, and so
    # the result, are bitwise those of the scalar walk.
    if not split:
        plain = bound.compile_config(
            fragment_config(bound, **(knobs | {VEC: False, BATCH: False}))
        )(q, k, gc, scale)
        torch.cuda.synchronize()
        assert torch.equal(out, plain)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("threads", [128, 256])
def test_cumsum_row_packets_match_reference(threads: int) -> None:
    torch.manual_seed(0)
    g = torch.randn(8, 64, 128, device=DEVICE, dtype=torch.bfloat16)
    bound = chunk_cumsum.bind((g,))
    config = fragment_config(
        bound, **{DOT: "tf32_mma", VEC: True, "cute_fragment_threads": threads}
    )
    code = bound.to_code(config)
    assert MMA in code and PACKET in code
    out = bound.compile_config(config)(g)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, g.float().cumsum(1), rtol=1e-5, atol=1e-5)
    plain = bound.compile_config(
        fragment_config(bound, **{DOT: "tf32_mma", "cute_fragment_threads": threads})
    )(g)
    torch.cuda.synchronize()
    assert torch.equal(out, plain)


# -- windows at a runtime row offset (the varlen anchored gram) ---------------


def _masked_token_rows(
    base: torch.Tensor, limit: torch.Tensor, tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # Preserve the kernel's 0-d table / nonscalar Int32 window promotion.
    rows = torch.arange(32, dtype=torch.int32, device=base.device)
    indices = base.to(torch.result_type(rows, base[0]))[:, None] + rows
    live = rows < limit.to(torch.result_type(rows, limit[0]))[:, None]
    return indices.long().clamp(0, tokens - 1), live & (indices >= 0) & (
        indices < tokens
    )


def _varlen_contiguous_dim_windows_reference(
    x: torch.Tensor, token_base: torch.Tensor, valid_len: torch.Tensor
) -> torch.Tensor:
    """Every chunk's 32 token columns gathered (zero past the sequence's
    end), then ``(2 a)^T b`` over its two 16-column halves."""
    h, d, t = x.shape
    nt = token_base.numel()
    index, valid = _masked_token_rows(token_base, valid_len, t)
    # [H, NT, D, 32], row h * NT + j
    chunks = x[:, :, index].permute(0, 2, 1, 3) * valid[None, :, None, :]
    chunks = chunks.reshape(h * nt, d, 32)
    a = chunks[:, :, :16] * 2.0
    b = chunks[:, :, 16:]
    return a.transpose(-2, -1) @ b


def _varlen_anchored_gram_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    gc: torch.Tensor,
    token_base: torch.Tensor,
    valid_len: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """The dense reference over every chunk's rows gathered (zero past the
    sequence's end)."""
    nt = token_base.numel()
    h = q.shape[1]
    index, valid = _masked_token_rows(token_base, valid_len, q.size(0))
    # [H * NT, 32, D], row h * NT + j
    qc = torch.cat([q[index, head] * valid[:, :, None] for head in range(h)])
    kc = torch.cat([k[index, head] * valid[:, :, None] for head in range(h)])
    assert qc.shape[0] == gc.shape[0] == h * nt
    return anchored_gram_reference(qc, kc, gc, scale)


def _check_changed_slab_bounds(
    args: tuple[object, ...],
    slab: Callable[..., torch.Tensor],
    plain: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    base: torch.Tensor,
    limit: torch.Tensor,
    tokens: int,
) -> None:
    def check() -> torch.Tensor:
        out, baseline = slab(*args), plain(*args)
        assert isinstance(args[0], torch.Tensor)
        assert out.dtype == baseline.dtype == args[0].dtype
        torch.testing.assert_close(out.float(), reference(*args), rtol=2e-2, atol=2e-2)
        assert torch.equal(
            out.contiguous().view(torch.uint8), baseline.contiguous().view(torch.uint8)
        )
        return out

    before = check()
    # Reuse the same table addresses and compiled kernels. In the source's
    # Int32 expressions the first base/limit become -3/26. A second chunk
    # crosses the physical tensor end before its logical limit.
    base[:2].copy_(base.new_tensor([2**32 - 3, tokens - 20]))
    limit[:2].copy_(limit.new_tensor([2**32 + 26, 32]))
    assert not torch.equal(check(), before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("vec", "batch", "threads"),
    [(False, False, 256), (True, True, 256), (True, True, 512)],
)
def test_varlen_anchored_gram_slabs_match_reference(
    vec: bool, batch: bool, threads: int
) -> None:
    # Sequences of 70, 32 and 45 tokens in chunks of 32: ragged last chunks
    # of 6 and 13 rows, and a last chunk whose rows past the end lie past
    # the tensor's end.
    args = varlen_gram_inputs((70, 32, 45), device=DEVICE, seed=0)
    bound = varlen_anchored_gram.bind(args)
    knobs = {
        DOT: "tf32_mma",
        PAD: True,
        "cute_fragment_unrolled_loops": True,
        "cute_fragment_threads": threads,
        SLAB: True,
        VEC: vec,
        BATCH: batch,
    }
    config = fragment_config(bound, **knobs)
    code = bound.to_code(config)
    assert code.count("fragment_slab_scalar = cutlass.Int32(0)") == 1
    plain = fragment_config(bound, **(knobs | {SLAB: False, VEC: False, BATCH: False}))
    _check_changed_slab_bounds(
        args,
        bound.compile_config(config),
        bound.compile_config(plain),
        _varlen_anchored_gram_reference,
        args[3],
        args[4],
        args[0].size(0),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("vec", "batch"), [(False, False), (True, True)])
def test_varlen_contiguous_dim_slab_matches_reference(vec: bool, batch: bool) -> None:
    # Ragged last chunks of 6 and 13 tokens, and a last chunk whose tokens
    # past the sequence's end lie past the tensor's end.
    args = varlen_contiguous_inputs((70, 32, 45), device=DEVICE, seed=0)
    bound = varlen_contiguous_dim_windows.bind(args)
    knobs = {SLAB: True, VEC: vec, BATCH: batch}
    config = fragment_config(bound, **knobs)
    code = bound.to_code(config)
    assert code.count("fragment_slab_scalar = cutlass.Int32(0)") == 1
    plain = fragment_config(bound, **(knobs | {SLAB: False, VEC: False, BATCH: False}))
    _check_changed_slab_bounds(
        args,
        bound.compile_config(config),
        bound.compile_config(plain),
        _varlen_contiguous_dim_windows_reference,
        args[1],
        args[2],
        args[0].size(-1),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("begin", "end", "expected"), [(-2, 0, [2.0]), (2**31 - 6, 2**31 - 1, [4.0, 1.0])]
)
def test_singleton_tail_excludes_padding(begin, end, expected) -> None:
    # Converting a negative Int32 origin directly to Uint64 would zero-extend
    # it in CuTe DSL. Exercise the signed lane comparison on an actual device.
    x = torch.ones(1, device=DEVICE)
    totals = torch.full((len(expected),), -73.0, device=DEVICE)
    scans = torch.full((len(expected), 4), -73.0, device=DEVICE)
    args = (x, totals, scans, begin, end)
    singleton_tail_scan.bind(args).compile_config(helion.Config(block_sizes=[]))(*args)
    assert totals.tolist() == expected
