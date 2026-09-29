from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
import helion.language as hl

_PUSH = "st.relaxed.gpu.global.u64"
_DYNAMIC = {
    "pid_type": "persistent_blocked",
    "cross_loop_pipeline": "dynamic",
    "num_sm_multiplier": 1,
    "num_warps": 4,
}


def gemv_chain(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Every consumer task reads the whole write-once intermediate."""
    hl.specialize(x.size())
    hl.specialize(w.size())
    n = x.size(0)
    m = w.size(0)
    scaled = torch.empty_like(x)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        scaled[tile_n] = x[tile_n] * 2.0
    for tile_m in hl.tile(m):
        row = scaled[:]
        out[tile_m] = torch.sum(w[tile_m, :] * row[None, :], dim=-1)
    return out


def dense_span_chain(x: torch.Tensor) -> torch.Tensor:
    """The producer writes ``tile.begin * 128 + arange(128)`` spans."""
    hl.specialize(x.size())
    n = x.size(0)
    partial = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile_g in hl.tile(n // 128, block_size=1):
        cols = tile_g.begin * 128 + hl.arange(128)
        partial[cols] = x[cols] + 1.0
    for tile_n in hl.tile(n):
        out[tile_n] = partial[tile_n] * 3.0
    return out


def strided_chain(x: torch.Tensor) -> torch.Tensor:
    """Interleaved lanes are not a provable partition, so counters remain."""
    hl.specialize(x.size())
    n = x.size(0)
    partial = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile_g, tile_s in hl.tile([n // 2, 2], block_size=[64, 1]):
        cols = tile_g.index * 2 + tile_s.begin
        partial[cols] = x[cols] + 1.0
    for tile_n in hl.tile(n):
        out[tile_n] = partial[tile_n] * 3.0
    return out


def spelled_span_chain(x: torch.Tensor) -> torch.Tensor:
    """The span extent is written with the tile's own block size."""
    hl.specialize(x.size())
    n = x.size(0)
    partial = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile_g in hl.tile(n // 2, block_size=64):
        cols = tile_g.begin * 2 + hl.arange(tile_g.block_size * 2)
        partial[cols] = x[cols] + 1.0
    for tile_n in hl.tile(n):
        out[tile_n] = partial[tile_n] * 3.0
    return out


def ragged_span_chain(x: torch.Tensor) -> torch.Tensor:
    """48 does not divide 512, so the last span's unmasked lanes overrun."""
    hl.specialize(x.size())
    n = x.size(0)
    partial = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile_g in hl.tile(n // 2, block_size=48):
        cols = tile_g.begin * 2 + hl.arange(96)
        partial[cols] = x[cols] + 1.0
    for tile_n in hl.tile(n):
        out[tile_n] = partial[tile_n] * 3.0
    return out


def two_buffer_chain(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Two local allocations, so the second polls at a nonzero mailbox offset."""
    hl.specialize(x.size())
    hl.specialize(w.size())
    n = x.size(0)
    m = w.size(0)
    doubled = torch.empty_like(x)
    shifted = torch.empty_like(x)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        doubled[tile_n] = x[tile_n] * 2.0
    for tile_n in hl.tile(n):
        shifted[tile_n] = x[tile_n] + 1.0
    for tile_m in hl.tile(m):
        row = doubled[:] + shifted[:]
        out[tile_m] = torch.sum(w[tile_m, :] * row[None, :], dim=-1)
    return out


def _kernel(fn: object, *, local_ll: bool) -> helion.Kernel:
    return helion.kernel(static_shapes=True, autotune_effort="none", local_ll=local_ll)(
        fn
    )


@onlyBackends(["triton"])
class TestLocalLL(TestCase):
    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_gemv_chain_polls_local_mailbox(self) -> None:
        x = torch.randn(512, device=DEVICE)
        w = torch.randn(256, 512, device=DEVICE)
        kernel = _kernel(gemv_chain, local_ll=True)
        code, out = code_and_output(kernel, (x, w), block_sizes=[128, 32], **_DYNAMIC)
        torch.testing.assert_close(out, w @ (x * 2.0), rtol=1e-4, atol=1e-3)
        self.assertIn(_PUSH, code)
        self.assertIn("tile_dependency_ll_local_mailbox", code)
        self.assertNotIn("sem='release'", code)
        # A later launch must wait for its own epoch, not reuse stale words.
        for _ in range(3):
            x2 = torch.randn(512, device=DEVICE)
            torch.testing.assert_close(
                kernel(x2, w), w @ (x2 * 2.0), rtol=1e-4, atol=1e-3
            )

        counter_code, counter_out = code_and_output(
            _kernel(gemv_chain, local_ll=False),
            (x, w),
            block_sizes=[128, 32],
            **_DYNAMIC,
        )
        torch.testing.assert_close(counter_out, out)
        self.assertNotIn(_PUSH, counter_code)
        self.assertIn("sem='release'", counter_code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_dense_span_store_under_cuda_graph(self) -> None:
        x = torch.randn(2048, device=DEVICE)
        kernel = _kernel(dense_span_chain, local_ll=True)
        code, out = code_and_output(kernel, (x,), block_sizes=[256], **_DYNAMIC)
        torch.testing.assert_close(out, (x + 1.0) * 3.0)
        self.assertIn(_PUSH, code)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            kernel(x)
            with torch.cuda.graph(graph, stream=stream):
                captured = kernel(x)
        torch.cuda.current_stream().wait_stream(stream)
        for _ in range(3):
            x.copy_(torch.randn_like(x))
            graph.replay()
            torch.testing.assert_close(captured, (x + 1.0) * 3.0)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_non_partition_store_keeps_counters(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        code, out = code_and_output(
            _kernel(strided_chain, local_ll=True), (x,), block_sizes=[128], **_DYNAMIC
        )
        torch.testing.assert_close(out, (x + 1.0) * 3.0)
        self.assertNotIn(_PUSH, code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_span_spelled_with_block_size(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        code, out = code_and_output(
            _kernel(spelled_span_chain, local_ll=True),
            (x,),
            block_sizes=[128],
            **_DYNAMIC,
        )
        torch.testing.assert_close(out, (x + 1.0) * 3.0)
        self.assertIn(_PUSH, code)

    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_ragged_span_keeps_counters(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        bound = _kernel(ragged_span_chain, local_ll=True).bind((x,))
        code = bound.to_triton_code(helion.Config(block_sizes=[128], **_DYNAMIC))
        self.assertNotIn(_PUSH, code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_two_allocations_share_the_mailbox(self) -> None:
        x = torch.randn(512, device=DEVICE)
        w = torch.randn(256, 512, device=DEVICE)
        kernel = _kernel(two_buffer_chain, local_ll=True)
        code, out = code_and_output(
            kernel, (x, w), block_sizes=[128, 128, 32], **_DYNAMIC
        )
        torch.testing.assert_close(out, w @ (3.0 * x + 1.0), rtol=1e-4, atol=1e-3)
        self.assertEqual(code.count(_PUSH), 2)
        for _ in range(3):
            x2 = torch.randn(512, device=DEVICE)
            torch.testing.assert_close(
                kernel(x2, w), w @ (3.0 * x2 + 1.0), rtol=1e-4, atol=1e-3
            )
