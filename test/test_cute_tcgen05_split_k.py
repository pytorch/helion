"""GPU tests of the flat one-CTA tcgen05 split-K pipeline (``tcgen05_split_k``).

Each output tile's K loop runs on ``split_k`` CTAs; every split stores an fp32
partial to the launcher-owned workspace and the last arriver reduces them in
its TMEM accumulator before the unchanged epilogue runs. These tests pin the
knob over full, ragged and K-tail shapes, uneven partitions, 64-row tiles,
fused epilogues and repeated launches (the arrival counters only grow; a
captured graph records its workspace's counter zeroing), and check that
configs the pipeline cannot split are rejected.
"""

from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
import helion.language as hl


@helion.kernel(static_shapes=True)
def _gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def _gemm_into(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def _bias_relu_gemm(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = torch.relu(acc + bias[tile_n]).to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def _gemm_fp8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def _gemm_with_counter(
    a: torch.Tensor, b: torch.Tensor, counter: torch.Tensor
) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
        hl.atomic_add(counter, [0], 1)
    return out


def _config(
    block_sizes: list[int], split_k: int, *, stages: int = 4, **overrides: object
) -> helion.Config:
    return helion.Config(
        **{
            "block_sizes": block_sizes,
            "indexing": ["tensor_descriptor"] * 3,
            "pid_type": "flat",
            "tcgen05_cluster_m": 1,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": stages,
            "l2_groupings": [1],
            "tcgen05_split_k": split_k,
            **overrides,
        }
    )


@onlyBackends(["cute"])
@skipIfNotCUDA()
class TestCuteTcgen05SplitK(RefEagerTestDisabled, TestCase):
    def _check(
        self,
        kernel: helion.Kernel,
        args: tuple[torch.Tensor, ...],
        expected: torch.Tensor,
        config: helion.Config,
        *,
        rtol: float = 1e-2,
        atol: float = 1e-1,
    ) -> str:
        bound = kernel.bind(args)
        code = bound.to_code(config)
        self.assertIn("tcgen05_sk_slab", code)
        self.assertIn("_cute_tmem_store_atom(", code)
        self.assertIn("helion_tcgen05_split_k", code)
        bound.set_config(config)
        # ``expected`` is the fp32 product rounded once to the output dtype,
        # like the kernel's fp32 partial sum; cuBLAS's own bf16 GEMM rounds
        # differently at small K.
        expected = expected.to(args[0].dtype)
        torch.testing.assert_close(bound(*args), expected, rtol=rtol, atol=atol)
        # The arrival counters derive each launch's base from the ticket
        # period, so a second launch over the cached workspace is as correct.
        torch.testing.assert_close(bound(*args), expected, rtol=rtol, atol=atol)
        return code

    def test_full_tiles_split_eight(self) -> None:
        a = torch.randn(512, 65536, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(65536, 512, device=DEVICE, dtype=torch.bfloat16)
        code = self._check(
            _gemm, (a, b), a.float() @ b.float(), _config([128, 128, 64], 8)
        )
        # Eight CTAs per output tile: the launch grid carries the multiplier.
        self.assertIn("* 8,)", code)
        self.assertIn("cute.arch.block_idx()[0] // 8", code)

    def test_uneven_partition_and_k_tail(self) -> None:
        # 1100 = 17 K tiles of 64 plus a 12-element tail (18 tiles): four
        # splits own 5, 5, 4 and 4 tiles, eight splits 3, 3, 2, 2, 2, 2, 2, 2
        # (the last split owns the tail in both cases).
        a = torch.randn(256, 1100, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(1100, 256, device=DEVICE, dtype=torch.bfloat16)
        self._check(_gemm, (a, b), a.float() @ b.float(), _config([128, 128, 64], 4))
        self._check(_gemm, (a, b), a.float() @ b.float(), _config([128, 128, 64], 8))

    def test_ragged_output_tiles(self) -> None:
        a = torch.randn(1000, 4096, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(4096, 1000, device=DEVICE, dtype=torch.bfloat16)
        self._check(_gemm, (a, b), a.float() @ b.float(), _config([128, 128, 64], 2))
        a = torch.randn(777, 2048, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(2048, 999, device=DEVICE, dtype=torch.bfloat16)
        self._check(_gemm, (a, b), a.float() @ b.float(), _config([128, 128, 64], 4))

    def test_rows_below_the_tile_on_64_row_tiles(self) -> None:
        # Decode GEMMs: a 64-row tile over 8 rows, 16-row TMEM datapath ops.
        a = torch.randn(8, 4096, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(4096, 2048, device=DEVICE, dtype=torch.bfloat16)
        self._check(_gemm, (a, b), a.float() @ b.float(), _config([64, 64, 128], 4))
        self._check(_gemm, (a, b), a.float() @ b.float(), _config([64, 128, 64], 8))

    def test_fused_epilogue_and_fp16(self) -> None:
        a = torch.randn(512, 8192, device=DEVICE, dtype=torch.float16)
        b = torch.randn(8192, 512, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(512, device=DEVICE, dtype=torch.float16)
        expected = torch.relu(a.float() @ b.float() + bias.float())
        self._check(_bias_relu_gemm, (a, b, bias), expected, _config([128, 128, 64], 4))

    def test_l2_grouping_decodes_the_tile_pid(self) -> None:
        a = torch.randn(1024, 16384, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(16384, 1024, device=DEVICE, dtype=torch.bfloat16)
        self._check(
            _gemm,
            (a, b),
            a.float() @ b.float(),
            _config([128, 128, 64], 4, l2_groupings=[4]),
        )

    def test_sixteen_splits(self) -> None:
        a = torch.randn(256, 131072, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(131072, 256, device=DEVICE, dtype=torch.bfloat16)
        self._check(
            _gemm, (a, b), a.float() @ b.float(), _config([128, 128, 128], 16, stages=3)
        )

    def test_cuda_graph_replay_after_an_eager_launch(self) -> None:
        # The workspace and the launch arguments are cached per (stream,
        # capture): a graph captured after eager launches must not reuse the
        # eager workspace pointer, and its replays re-run the counter reset
        # the capture recorded (the arrival tickets then restart cleanly).
        a = torch.randn(256, 16384, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(16384, 256, device=DEVICE, dtype=torch.bfloat16)
        expected = (a.float() @ b.float()).to(a.dtype)
        bound = _gemm.bind((a, b))
        bound.set_config(_config([128, 128, 64], 4))
        for _ in range(2):
            torch.testing.assert_close(bound(a, b), expected, rtol=1e-2, atol=1e-1)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            bound(a, b)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = bound(a, b)
        for _ in range(3):
            captured.zero_()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, expected, rtol=1e-2, atol=1e-1)
        torch.testing.assert_close(bound(a, b), expected, rtol=1e-2, atol=1e-1)

    def test_relaunch_on_another_stream_with_the_same_output(self) -> None:
        # The launch arguments and the workspace are cached per (stream,
        # capture): the same pointers relaunched on a second stream miss the
        # last-launch cache and take that stream's workspace.
        a = torch.randn(512, 8192, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(8192, 512, device=DEVICE, dtype=torch.bfloat16)
        out = torch.empty(512, 512, device=DEVICE, dtype=torch.bfloat16)
        expected = (a.float() @ b.float()).to(a.dtype)
        bound = _gemm_into.bind((a, b, out))
        bound.set_config(_config([128, 128, 64], 4))
        bound(a, b, out)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-1)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            out.zero_()
            bound(a, b, out)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-1)

    def test_fp8_operands(self) -> None:
        a = torch.randn(512, 16384, device=DEVICE).to(torch.float8_e4m3fn)
        # fp8 tcgen05 operands are K-major on both sides.
        b = torch.randn(512, 16384, device=DEVICE).to(torch.float8_e4m3fn).T
        expected = (a.float() @ b.float()).to(torch.bfloat16)
        bound = _gemm_fp8.bind((a, b))
        config = _config([128, 128, 128], 4)
        code = bound.to_code(config)
        self.assertIn("tcgen05_sk_slab", code)
        bound.set_config(config)
        torch.testing.assert_close(bound(a, b), expected, rtol=2e-2, atol=4.0)

    def test_unsplittable_configs_are_rejected(self) -> None:
        a = torch.randn(512, 256, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(256, 512, device=DEVICE, dtype=torch.bfloat16)
        # A tile body with a second side effect would run it once per split.
        counter = torch.zeros(1, device=DEVICE, dtype=torch.int32)
        counted = _gemm_with_counter.bind(
            (
                torch.randn(512, 65536, device=DEVICE, dtype=torch.bfloat16),
                torch.randn(65536, 512, device=DEVICE, dtype=torch.bfloat16),
                counter,
            )
        )
        with self.assertRaises(helion.exc.InvalidConfig):
            counted.to_code(_config([128, 128, 64], 4))
        bound = _gemm.bind((a, b))
        # More splits than K tiles.
        with self.assertRaises(helion.exc.InvalidConfig):
            bound.to_code(_config([128, 128, 64], 8))
        # Persistent and clustered pipelines keep whole K loops.
        with self.assertRaises(helion.exc.InvalidConfig):
            bound.to_code(
                _config([128, 128, 64], 2, pid_type="persistent_blocked", stages=2)
            )
        with self.assertRaises(
            (helion.exc.InvalidConfig, helion.exc.BackendUnsupported)
        ):
            bound.to_code(_config([128, 128, 64], 2, tcgen05_cluster_m=2, stages=2))


if __name__ == "__main__":
    unittest.main()
