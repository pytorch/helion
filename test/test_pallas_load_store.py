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


def _ref_jagged_bmm(
    seq_offsets: torch.Tensor, jagged: torch.Tensor, dense: torch.Tensor
) -> torch.Tensor:
    out = torch.empty(
        (jagged.shape[0], dense.shape[2]), dtype=jagged.dtype, device=jagged.device
    )
    for g in range(dense.shape[0]):
        s, e = int(seq_offsets[g]), int(seq_offsets[g + 1])
        if e > s:
            out[s:e] = jagged[s:e] @ dense[g]
    return out


def _inputs(offsets: list[int], D: int, K: int, dtype: torch.dtype):
    seq_offsets = torch.tensor(offsets, dtype=torch.int32, device=DEVICE)
    L, B = offsets[-1], len(offsets) - 1
    torch.manual_seed(0)
    jagged = torch.randn((L, D), dtype=dtype, device=DEVICE)
    dense = torch.randn((B, D, K), dtype=dtype, device=DEVICE)
    return seq_offsets, jagged, dense


def _run(seq_offsets, jagged, dense, block_sizes):
    return code_and_output(
        jagged_dense_bmm,
        (seq_offsets, jagged, dense),
        block_sizes=block_sizes,
        pallas_loop_type="emit_pipeline",
    )


# bf16 with the kernel's f32-accumulating MXU matches a single torch matmul of
# the same group bit-for-bit (single D-tile), so the default tolerances hold.
@onlyBackends(["pallas"])
@skipUnlessPallas("JAX/Pallas TPU not available")
class TestPallasJaggedCarrySimple(TestCase):
    """Minimal kernels that isolate one carry behaviour each."""

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_single_group(self, dtype: torch.dtype) -> None:
        # One group: exercises the aligned-enclosing read and the tail mask,
        # with no carry between groups.
        seq_offsets, jagged, dense = _inputs([0, 20], 128, 128, dtype)
        _code, out = _run(seq_offsets, jagged, dense, [16, 128, 128])
        torch.testing.assert_close(out, _ref_jagged_bmm(seq_offsets, jagged, dense))

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_aligned_groups_carry_dormant(self, dtype: torch.dtype) -> None:
        # Aligned boundaries: carry path is emitted but its runtime guard never fires.
        seq_offsets, jagged, dense = _inputs([0, 16, 32], 128, 128, dtype)
        code, out = _run(seq_offsets, jagged, dense, [16, 128, 128])
        self.assertIn("pl.program_id(0) != 0", code)
        torch.testing.assert_close(out, _ref_jagged_bmm(seq_offsets, jagged, dense))

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    def test_carry_keeps_both_groups(self) -> None:
        # Two groups [0, 3) and [3, 16) share the [0, 16) boundary.  With
        # identity weights out == jagged, so a clobbered carry would be obvious.
        seq_offsets = torch.tensor([0, 3, 16], dtype=torch.int32, device=DEVICE)
        jagged = (
            torch.arange(1, 17, dtype=torch.bfloat16, device=DEVICE)[:, None]
            .expand(16, 128)
            .contiguous()
        )
        eye = torch.eye(128, dtype=torch.bfloat16, device=DEVICE)
        _code, out = _run(seq_offsets, jagged, torch.stack([eye, eye]), [16, 128, 128])
        torch.testing.assert_close(out, jagged)

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_carry_masks_bias_add(self, dtype: torch.dtype) -> None:
        # Additive map across a shared boundary: groups [0, 3) and [3, 16) over-
        # read each other's rows (loaded as 0).  Unless the store value is masked,
        # those rows write 0 + 1.0 and the carry folds a stray 1.0 into the result.
        @helion.kernel(backend="pallas")
        def jagged_add_one(seq_offsets, jagged):
            L, D = jagged.shape
            B = seq_offsets.shape[0] - 1
            out = torch.empty((L, D), dtype=jagged.dtype, device=jagged.device)
            for g in hl.grid(B):
                s, e = seq_offsets[g], seq_offsets[g + 1]
                for st in hl.tile(s, e):
                    for dt in hl.tile(0, D):
                        out[st, dt] = jagged[st, dt] + 1.0
            return out

        seq_offsets = torch.tensor([0, 3, 16], dtype=torch.int32, device=DEVICE)
        jagged = torch.randn((16, 128), dtype=dtype, device=DEVICE)
        _code, out = code_and_output(
            jagged_add_one,
            (seq_offsets, jagged),
            block_sizes=[16, 128],
            pallas_loop_type="emit_pipeline",
        )
        torch.testing.assert_close(out, jagged + 1.0)

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_carry_separate_outputs(self, dtype: torch.dtype) -> None:
        # Two stores share one jagged tile: each must carry into its own scratch,
        # otherwise out2's boundary save clobbers out1's.
        @helion.kernel(backend="pallas")
        def jagged_add_two(seq_offsets, jagged):
            L, D = jagged.shape
            B = seq_offsets.shape[0] - 1
            out1 = torch.empty((L, D), dtype=jagged.dtype, device=jagged.device)
            out2 = torch.empty((L, D), dtype=jagged.dtype, device=jagged.device)
            for g in hl.grid(B):
                s, e = seq_offsets[g], seq_offsets[g + 1]
                for st in hl.tile(s, e):
                    for dt in hl.tile(0, D):
                        val = jagged[st, dt]
                        out1[st, dt] = val + 1.0
                        out2[st, dt] = val + 2.0
            return out1, out2

        seq_offsets = torch.tensor([0, 3, 16], dtype=torch.int32, device=DEVICE)
        jagged = torch.randn((16, 128), dtype=dtype, device=DEVICE)
        _code, (o1, o2) = code_and_output(
            jagged_add_two,
            (seq_offsets, jagged),
            block_sizes=[16, 128],
            pallas_loop_type="emit_pipeline",
        )
        torch.testing.assert_close(o1, jagged + 1.0)
        torch.testing.assert_close(o2, jagged + 2.0)

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_carry_scalar_branch_store(self, dtype: torch.dtype) -> None:
        # Two groups take different if/else branches across a shared boundary.
        @helion.kernel(backend="pallas")
        def jagged_branch(seq_offsets, jagged):
            L, D = jagged.shape
            B = seq_offsets.shape[0] - 1
            out = torch.empty((L, D), dtype=jagged.dtype, device=jagged.device)
            for g in hl.grid(B):
                s = seq_offsets[g]
                e = seq_offsets[g + 1]
                for st in hl.tile(s, e):
                    for dt in hl.tile(0, D):
                        v = jagged[st, dt]
                        if e - s > 8:
                            out[st, dt] = v + 1.0
                        else:
                            out[st, dt] = v * 2.0
            return out

        seq_offsets = torch.tensor([0, 3, 16], dtype=torch.int32, device=DEVICE)
        jagged = torch.randn((16, 128), dtype=dtype, device=DEVICE)
        code, out = code_and_output(
            jagged_branch,
            (seq_offsets, jagged),
            block_sizes=[16, 128],
            pallas_loop_type="emit_pipeline",
        )
        self.assertIn("lax.cond", code)
        expected = torch.empty_like(jagged)
        for g in range(2):
            s, e = int(seq_offsets[g]), int(seq_offsets[g + 1])
            expected[s:e] = jagged[s:e] + 1.0 if (e - s) > 8 else jagged[s:e] * 2.0
        torch.testing.assert_close(out, expected)


# bf16 with the kernel's f32-accumulating MXU matches a single torch matmul of
# the same group bit-for-bit (single D-tile), so the default tolerances hold.
@onlyBackends(["pallas"])
@skipUnlessPallas("JAX/Pallas TPU not available")
class TestPallasJaggedCarryBmm(TestCase):
    """The carry across the full configuration matrix (group counts, block vs
    group size, multiple column tiles, the non-matmul map-axis form)."""

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize(
        "offsets",
        [
            [0, 13, 25],  # shared boundary
            [0, 20, 50],  # multi-row groups
            [0, 3, 7, 16],  # several tiny groups in one boundary (cumulative carry)
        ],
    )
    def test_bmm_block_eq_sublane(self, dtype: torch.dtype, offsets: list[int]) -> None:
        seq_offsets, jagged, dense = _inputs(offsets, 128, 128, dtype)
        code, out = _run(seq_offsets, jagged, dense, [16, 128, 128])
        self.assertIn("pltpu.emit_pipeline", code)
        torch.testing.assert_close(out, _ref_jagged_bmm(seq_offsets, jagged, dense))

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_bmm_block_gt_group(self, dtype: torch.dtype) -> None:
        # Two groups (13 rows) are smaller than block_row=32, total L=200 >> block_row.
        seq_offsets, jagged, dense = _inputs([0, 13, 100, 113, 200], 128, 128, dtype)
        _code, out = _run(seq_offsets, jagged, dense, [32, 128, 128])
        torch.testing.assert_close(out, _ref_jagged_bmm(seq_offsets, jagged, dense))

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_bmm_multi_k_tile(self, dtype: torch.dtype) -> None:
        # K=256 with block_col=128 gives two output-column tiles; the carry stacks
        # the per-column-tile boundaries along its scratch row dim.
        seq_offsets, jagged, dense = _inputs([0, 17, 40, 71], 128, 256, dtype)
        _code, out = _run(seq_offsets, jagged, dense, [16, 128, 128])
        torch.testing.assert_close(out, _ref_jagged_bmm(seq_offsets, jagged, dense))

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_bmm_many_groups(self, dtype: torch.dtype) -> None:
        # 50 unaligned groups; carry scratch is per column-tile, not per group.
        offsets = list(range(0, 13 * 51, 13))  # 50 groups
        seq_offsets, jagged, dense = _inputs(offsets, 128, 128, dtype)
        code, out = _run(seq_offsets, jagged, dense, [16, 128, 128])
        self.assertIn("carry", code)
        torch.testing.assert_close(out, _ref_jagged_bmm(seq_offsets, jagged, dense))

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize(
        "offsets",
        [
            [0, 0, 16],  # leading empty
            [0, 5, 5, 16],  # interior empty
            [0, 16, 16],  # trailing empty
        ],
    )
    def test_bmm_empty_groups(self, dtype: torch.dtype, offsets: list[int]) -> None:
        # Zero-length groups (s == e) iterate no tiles; carry threads the neighbours.
        seq_offsets, jagged, dense = _inputs(offsets, 128, 128, dtype)
        _code, out = _run(seq_offsets, jagged, dense, [16, 128, 128])
        torch.testing.assert_close(out, _ref_jagged_bmm(seq_offsets, jagged, dense))

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

        seq_offsets = torch.tensor([0, 13, 25], dtype=torch.int32, device=DEVICE)
        jagged = torch.randn((25, 128), dtype=torch.bfloat16, device=DEVICE)
        _code, out = code_and_output(
            jagged_scale,
            (seq_offsets, jagged),
            block_sizes=[16, 128],
            pallas_loop_type="emit_pipeline",
        )
        torch.testing.assert_close(out, jagged * 2)

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    def test_dynamic_rows_specialized_cols(self) -> None:
        # static_shapes=False: the carry compiles once; grid is the runtime group count.
        @helion.kernel(backend="pallas", static_shapes=False)
        def jagged_scale_dyn(
            seq_offsets: torch.Tensor, jagged: torch.Tensor
        ) -> torch.Tensor:
            L, D = jagged.shape
            D = hl.specialize(D)
            B = seq_offsets.shape[0] - 1
            out = torch.empty((L, D), dtype=jagged.dtype, device=jagged.device)
            for g in hl.grid(B):
                s = seq_offsets[g]
                e = seq_offsets[g + 1]
                for st in hl.tile(s, e):
                    for dt in hl.tile(0, D):
                        out[st, dt] = jagged[st, dt] * 2.0
            return out

        seq_offsets = torch.tensor([0, 13, 25], dtype=torch.int32, device=DEVICE)
        jagged = torch.randn((25, 128), dtype=torch.bfloat16, device=DEVICE)
        code, out = code_and_output(
            jagged_scale_dyn,
            (seq_offsets, jagged),
            block_sizes=[16, 128],
            pallas_loop_type="emit_pipeline",
        )
        self.assertIn("(B,)", code)
        torch.testing.assert_close(out, jagged * 2)


@onlyBackends(["pallas"])
@skipUnlessPallas("JAX/Pallas TPU not available")
class TestPallasJaggedCarryRejects(TestCase):
    """Shapes the carry refuses (or routes elsewhere) rather than miscompiling."""

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

        seq_offsets = torch.tensor([0, 13, 25], dtype=torch.int32, device=DEVICE)
        jagged = torch.randn((25, 128), dtype=torch.bfloat16, device=DEVICE)
        _code, out = code_and_output(
            jagged_row_sum,
            (seq_offsets, jagged),
            block_sizes=[128, 16],
            pallas_loop_type="emit_pipeline",
        )
        ref = torch.zeros(2, 128, device=DEVICE)
        ref[0] = jagged[0:13].float().sum(0)
        ref[1] = jagged[13:25].float().sum(0)
        torch.testing.assert_close(out, ref)

    def test_block_not_multiple_of_sublane_raises(self) -> None:
        # bf16 sublane S=16; block_row=8 is not a multiple, so the carry rejects it
        # loudly instead of clobbering the boundary with a plain store.
        seq_offsets, jagged, dense = _inputs([0, 13, 25], 128, 128, torch.bfloat16)
        with self.assertRaisesRegex(
            exc.InductorLoweringError, "block_row .* must be a multiple"
        ):
            _run(seq_offsets, jagged, dense, [8, 128, 128])

    def test_multi_grid_group_rejected(self) -> None:
        # The fold guard keys seq_offsets program_id(0), so a second grid dimension is
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

        seq_offsets, jagged, dense = _inputs([0, 13, 25], 128, 128, torch.bfloat16)
        with self.assertRaisesRegex(exc.InductorLoweringError, "only grid dimension"):
            code_and_output(
                two_grid_bmm,
                (seq_offsets, jagged, dense),
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


@onlyBackends(["pallas"])
@skipUnlessPallas("JAX/Pallas TPU not available")
class TestPallasPartialSlice(TestCase):
    """Static bounded slices (``[:n]`` / ``[n:]``) on untiled dims."""

    def test_partial_slice_load_store(self) -> None:
        @helion.kernel(backend="pallas")
        def kernel(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(src.size(0)):
                dst[tile, :16] = src[tile, :16]
                dst[tile, 16:] = src[tile, 16:]
            return dst

        src = torch.randn((64, 32), device=DEVICE)
        dst = torch.zeros((64, 32), device=DEVICE)
        code, out = code_and_output(kernel, (src, dst), block_sizes=[16])
        self.assertIn(":16", code)
        self.assertIn("16:", code)
        torch.testing.assert_close(out, src)

    @xfailIfPallas(
        "fully-bounded slice (8:16) is lowered to an index tensor by the "
        "front-end, not a slice, so it hits the indirect scatter/gather path "
        "instead of ArbitrarySlicePattern"
    )
    def test_partial_slice_both_bounds(self) -> None:
        # Slice with both start and stop set (``8:16``): a middle window,
        # not just an open-ended prefix/suffix.
        @helion.kernel(backend="pallas")
        def kernel(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(src.size(0)):
                dst[tile, :8] = src[tile, :8]
                dst[tile, 8:16] = src[tile, 8:16] * 2.0
                dst[tile, 16:] = src[tile, 16:]
            return dst

        src = torch.randn((64, 32), device=DEVICE)
        dst = torch.zeros((64, 32), device=DEVICE)
        _code, out = code_and_output(kernel, (src, dst), block_sizes=[16])
        expected = src.clone()
        expected[:, 8:16] *= 2.0
        torch.testing.assert_close(out, expected)

    def test_partial_slice_dim0(self) -> None:
        # Bounded slices on dim 0 while tiling dim 1 (sublane-dim slicing).
        @helion.kernel(backend="pallas")
        def kernel(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(src.size(1)):
                dst[:32, tile] = src[:32, tile]
                dst[32:, tile] = src[32:, tile]
            return dst

        src = torch.randn((64, 256), device=DEVICE)
        dst = torch.zeros((64, 256), device=DEVICE)
        _code, out = code_and_output(kernel, (src, dst), block_sizes=[128])
        torch.testing.assert_close(out, src)

    def test_partial_slice_unaligned(self) -> None:
        # Lane-dim slice boundary not a multiple of 128.
        @helion.kernel(backend="pallas")
        def kernel(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(src.size(0)):
                dst[tile, :200] = src[tile, :200]
                dst[tile, 200:] = src[tile, 200:]
            return dst

        src = torch.randn((64, 512), device=DEVICE)
        dst = torch.zeros((64, 512), device=DEVICE)
        _code, out = code_and_output(kernel, (src, dst), block_sizes=[16])
        torch.testing.assert_close(out, src)

    def test_partial_slice_symbolic_bound(self) -> None:
        # Slice bound taken from a tensor size (concat pattern).
        @helion.kernel(backend="pallas")
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty(
                [x.size(0), x.size(1) + y.size(1)], dtype=x.dtype, device=x.device
            )
            n1 = x.size(1)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m, :n1] = x[tile_m, :]
                out[tile_m, n1:] = y[tile_m, :]
            return out

        x = torch.randn((64, 200), device=DEVICE)
        y = torch.randn((64, 56), device=DEVICE)
        _code, out = code_and_output(kernel, (x, y), block_sizes=[16])
        torch.testing.assert_close(out, torch.cat([x, y], dim=1))

    def test_strided_slice_rejected(self) -> None:
        @helion.kernel(backend="pallas")
        def kernel(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(src.size(0)):
                dst[tile, ::2] = src[tile, ::2]
            return dst

        src = torch.randn((64, 32), device=DEVICE)
        dst = torch.zeros((64, 32), device=DEVICE)
        with self.assertRaisesRegex(exc.BackendUnsupported, "slice expr"):
            code_and_output(kernel, (src, dst), block_sizes=[16])


instantiate_parametrized_tests(TestPallasJaggedCarrySimple)
instantiate_parametrized_tests(TestPallasJaggedCarryBmm)
instantiate_parametrized_tests(TestPallasJaggedCarryRejects)
