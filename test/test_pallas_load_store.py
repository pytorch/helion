from __future__ import annotations

import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize

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

# TODO(tcombes): JAX Pallas interpret mode can't trace these emit_pipeline
# kernels.  The jagged reads use a dynamic pl.ds / BoundedSlice BlockSpec, and
# the static matmul calls pl.program_id inside the pipeline body.  They pass on
# real TPU; drop the xfail when interpret supports them.
_XFAIL_INTERPRET = (
    "emit_pipeline dynamic pl.ds / program_id BlockSpecs unsupported in JAX "
    "Pallas interpret mode"
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


def _ref(off: torch.Tensor, j: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    out = torch.empty((j.shape[0], d.shape[2]), dtype=j.dtype, device=j.device)
    for g in range(d.shape[0]):
        s, e = int(off[g]), int(off[g + 1])
        if e > s:
            out[s:e] = j[s:e] @ d[g]
    return out


def _inputs(offsets: list[int], D: int, K: int, dtype: torch.dtype):
    off = torch.tensor(offsets, dtype=torch.int32, device=DEVICE)
    L, B = offsets[-1], len(offsets) - 1
    torch.manual_seed(0)
    j = torch.randn((L, D), dtype=dtype, device=DEVICE)
    d = torch.randn((B, D, K), dtype=dtype, device=DEVICE)
    return off, j, d


def _run(off, j, d, block_sizes):
    return code_and_output(
        jagged_dense_bmm,
        (off, j, d),
        block_sizes=block_sizes,
        pallas_loop_type="emit_pipeline",
    )


# bf16 with the kernel's f32-accumulating MXU matches a single torch matmul of
# the same group bit-for-bit (single D-tile), so the default tolerances hold.
@onlyBackends(["pallas"])
@skipUnlessPallas("JAX/Pallas TPU not available")
class TestPallasJaggedCarry(TestCase):
    # Simple cases that show the logic.

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_single_group(self, dtype: torch.dtype) -> None:
        # One group: exercises the aligned-enclosing read and the tail mask,
        # with no carry between groups.
        off, j, d = _inputs([0, 20], 128, 128, dtype)
        _code, out = _run(off, j, d, [16, 128, 128])
        torch.testing.assert_close(out, _ref(off, j, d))

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_aligned_groups_no_carry(self, dtype: torch.dtype) -> None:
        # Sublane-aligned boundaries: no over-read and no shared row, so the
        # carry fold/save guards never fire.
        off, j, d = _inputs([0, 16, 32], 128, 128, dtype)
        _code, out = _run(off, j, d, [16, 128, 128])
        torch.testing.assert_close(out, _ref(off, j, d))

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    def test_carry_keeps_both_groups(self) -> None:
        # Two groups [0, 3) and [3, 16) share the [0, 16) boundary row.  With
        # identity weights out == jagged, so a clobbered carry would be obvious.
        off = torch.tensor([0, 3, 16], dtype=torch.int32, device=DEVICE)
        j = (
            torch.arange(1, 17, dtype=torch.bfloat16, device=DEVICE)[:, None]
            .expand(16, 128)
            .contiguous()
        )
        eye = torch.eye(128, dtype=torch.bfloat16, device=DEVICE)
        _code, out = _run(off, j, torch.stack([eye, eye]), [16, 128, 128])
        torch.testing.assert_close(out, j)

    # The carry across the full configuration matrix.

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize(
        "offsets",
        [
            [0, 13, 25],  # shared boundary row
            [0, 20, 50],  # multi-row groups
            [0, 3, 7, 16],  # several tiny groups in one row (cumulative carry)
        ],
    )
    def test_bmm_block_eq_sublane(self, dtype: torch.dtype, offsets: list[int]) -> None:
        off, j, d = _inputs(offsets, 128, 128, dtype)
        code, out = _run(off, j, d, [16, 128, 128])
        self.assertIn("pltpu.emit_pipeline", code)
        torch.testing.assert_close(out, _ref(off, j, d))

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_bmm_block_gt_group(self, dtype: torch.dtype) -> None:
        # Two groups (13 rows) are smaller than block_row=32, total L=200 >> block_row.
        off, j, d = _inputs([0, 13, 100, 113, 200], 128, 128, dtype)
        _code, out = _run(off, j, d, [32, 128, 128])
        torch.testing.assert_close(out, _ref(off, j, d))

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_bmm_multi_k_tile(self, dtype: torch.dtype) -> None:
        # K=256 with block_col=128 gives two output-column tiles; the carry stacks
        # the per-column-tile boundary rows along its scratch row dim.
        off, j, d = _inputs([0, 17, 40, 71], 128, 256, dtype)
        _code, out = _run(off, j, d, [16, 128, 128])
        torch.testing.assert_close(out, _ref(off, j, d))

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    def test_elementwise_map_axis(self) -> None:
        # The non-matmul map-axis store from commit 2 now runs: out[st] = 2 *
        # jagged[st], carried across the shared boundary like the matmul case.
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
        _code, out = code_and_output(
            jagged_scale,
            (off, j),
            block_sizes=[16, 128],
            pallas_loop_type="emit_pipeline",
        )
        torch.testing.assert_close(out, j * 2)

    # Rejection paths.

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

    def test_block_not_multiple_of_sublane_raises(self) -> None:
        # bf16 sublane S=16; block_row=8 is not a multiple, so the carry rejects it
        # loudly instead of clobbering the boundary with a plain store.
        off, j, d = _inputs([0, 13, 25], 128, 128, torch.bfloat16)
        with self.assertRaisesRegex(
            exc.InductorLoweringError, "block_row .* must be a multiple"
        ):
            _run(off, j, d, [8, 128, 128])

    def test_multi_grid_group_rejected(self) -> None:
        # The fold guard keys off program_id(0), so a second grid dimension is
        # refused rather than folding against the wrong program id.
        @helion.kernel(backend="pallas")
        def two_grid_bmm(
            seq_offsets: torch.Tensor, jagged: torch.Tensor, dense: torch.Tensor
        ) -> torch.Tensor:
            L, D = jagged.shape
            B, D, K = dense.shape
            out = torch.empty((L, K), dtype=jagged.dtype, device=jagged.device)
            for _b, g in hl.grid([2, B]):
                s = seq_offsets[g]
                e = seq_offsets[g + 1]
                for st in hl.tile(s, e):
                    for kt in hl.tile(0, K):
                        acc = hl.zeros([st, kt], dtype=torch.float32)
                        for dt in hl.tile(0, D):
                            acc = acc + torch.matmul(jagged[st, dt], dense[g, dt, kt])
                        out[st, kt] = acc
            return out

        off, j, d = _inputs([0, 13, 25], 128, 128, torch.bfloat16)
        with self.assertRaisesRegex(exc.InductorLoweringError, "only grid dimension"):
            code_and_output(
                two_grid_bmm,
                (off, j, d),
                block_sizes=[16, 128, 128],
                pallas_loop_type="emit_pipeline",
            )

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


instantiate_parametrized_tests(TestPallasJaggedCarry)
