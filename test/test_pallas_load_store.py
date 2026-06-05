from __future__ import annotations

import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipUnlessPallas
from helion._testing import xfailIfPallas
from helion._testing import xfailIfPallasInterpret
import helion.language as hl

# TODO(tcombes): the static matmul's emit_pipeline uses a parallel grid, which
# calls pl.program_id inside the pipeline body; JAX Pallas interpret mode can't
# trace that.  It passes on real TPU; drop the xfail when interpret supports it.
_XFAIL_INTERPRET = (
    "parallel-grid emit_pipeline calls program_id in the pipeline body, "
    "unsupported in JAX Pallas interpret mode"
)


# out[s:e] = jagged[s:e] @ dense[g] for each group g delimited by seq_offsets.
# The row tile hl.tile(s, e) has runtime bounds; unaligned group boundaries make
# adjacent groups share an output row, which the ordered carry stitches.
@helion.kernel(backend="pallas")
def jagged_dense_bmm(
    seq_offsets: torch.Tensor, jagged: torch.Tensor, dense: torch.Tensor
) -> torch.Tensor:
    L, D = jagged.shape
    B, D, K = dense.shape
    out = torch.empty((L, K), dtype=jagged.dtype, device=jagged.device)
    for g in hl.grid(B):
        s = seq_offsets[g]
        e = seq_offsets[g + 1]
        for st in hl.tile(s, e):
            for kt in hl.tile(0, K):
                acc = hl.zeros([st, kt], dtype=torch.float32)
                for dt in hl.tile(0, D):
                    acc = acc + torch.matmul(jagged[st, dt], dense[g, dt, kt])
                out[st, kt] = acc
    return out


@onlyBackends(["pallas"])
@skipUnlessPallas("JAX/Pallas TPU not available")
class TestPallasJaggedCarry(TestCase):
    # The legality gate is in place but the carry store is not, so a legal
    # map-axis store reaches the unimplemented carry and raises "carry".

    def test_bmm_store_reaches_carry(self) -> None:
        # Identity store on the jagged row is a map axis: the gate accepts it, so
        # it reaches the unimplemented carry.
        off = torch.tensor([0, 13, 25], dtype=torch.int32, device=DEVICE)
        j = torch.randn((25, 128), dtype=torch.bfloat16, device=DEVICE)
        d = torch.randn((2, 128, 128), dtype=torch.bfloat16, device=DEVICE)
        with self.assertRaisesRegex(exc.InductorLoweringError, "carry"):
            code_and_output(
                jagged_dense_bmm,
                (off, j, d),
                block_sizes=[16, 128, 128],
                pallas_loop_type="emit_pipeline",
            )

    def test_elementwise_store_reaches_carry(self) -> None:
        # A non-matmul map-axis store is accepted too, so it also reaches the
        # unimplemented carry.
        @helion.kernel(backend="pallas")
        def jagged_scale(
            seq_offsets: torch.Tensor, jagged: torch.Tensor
        ) -> torch.Tensor:
            L, D = jagged.shape
            B = seq_offsets.shape[0] - 1
            out = torch.empty((L, D), dtype=jagged.dtype, device=jagged.device)
            for g in hl.grid(B):
                s = seq_offsets[g]
                e = seq_offsets[g + 1]
                for st in hl.tile(s, e):
                    for dt in hl.tile(0, D):
                        out[st, dt] = jagged[st, dt] * 2
            return out

        off = torch.tensor([0, 13, 25], dtype=torch.int32, device=DEVICE)
        j = torch.randn((25, 128), dtype=torch.bfloat16, device=DEVICE)
        with self.assertRaisesRegex(exc.InductorLoweringError, "carry"):
            code_and_output(
                jagged_scale,
                (off, j),
                block_sizes=[16, 128],
                pallas_loop_type="emit_pipeline",
            )

    @xfailIfPallas(
        "bf16 reduction over a jagged row falls through to the f32-only existing "
        "path; the unaligned bf16 load can't compile yet"
    )
    def test_jagged_reduction_over_row_bf16(self) -> None:
        # A reduction over the jagged row (summed to a dense output) is not the
        # carry's shape, so it falls through to the existing path.  That path is
        # f32-only, so the bf16 case can't compile yet; the xfail tracks the gap.
        @helion.kernel(backend="pallas")
        def jagged_row_sum(
            seq_offsets: torch.Tensor, jagged: torch.Tensor
        ) -> torch.Tensor:
            L, D = jagged.shape
            B = seq_offsets.shape[0] - 1
            out = torch.zeros((B, D), dtype=torch.float32, device=jagged.device)
            for g in hl.grid(B):
                s = seq_offsets[g]
                e = seq_offsets[g + 1]
                for dt in hl.tile(0, D):
                    acc = hl.zeros([dt], dtype=torch.float32)
                    for st in hl.tile(s, e):
                        acc = acc + jagged[st, dt].to(torch.float32).sum(0)
                    out[g, dt] = acc
            return out

        off = torch.tensor([0, 13, 25], dtype=torch.int32, device=DEVICE)
        j = torch.randn((25, 128), dtype=torch.bfloat16, device=DEVICE)
        _code, out = code_and_output(
            jagged_row_sum,
            (off, j),
            block_sizes=[128, 16],
            pallas_loop_type="emit_pipeline",
        )
        ref = torch.zeros(2, 128, device=DEVICE)
        ref[0] = j[0:13].float().sum(0)
        ref[1] = j[13:25].float().sum(0)
        torch.testing.assert_close(out, ref)

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    def test_non_jagged_emit_pipeline_unaffected(self) -> None:
        # Static (compile-time) tile bounds never trip the gate: a plain
        # emit_pipeline matmul still lowers and runs correctly.
        @helion.kernel(backend="pallas", static_shapes=True)
        def static_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            M, K = a.shape
            K, N = b.shape
            out = torch.empty((M, N), dtype=a.dtype, device=a.device)
            for mt in hl.tile(M):
                for nt in hl.tile(N):
                    acc = hl.zeros([mt, nt], dtype=torch.float32)
                    for kt in hl.tile(K):
                        acc = acc + torch.matmul(a[mt, kt], b[kt, nt])
                    out[mt, nt] = acc.to(a.dtype)
            return out

        a = torch.randn((64, 128), dtype=torch.bfloat16, device=DEVICE)
        b = torch.randn((128, 256), dtype=torch.bfloat16, device=DEVICE)
        _code, out = code_and_output(
            static_matmul,
            (a, b),
            block_sizes=[32, 128, 128],
            pallas_loop_type="emit_pipeline",
        )
        torch.testing.assert_close(out, a @ b)
