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
from helion._testing import skipIfPallas
from helion._testing import skipUnlessPallas
from helion._testing import xfailIfPallasInterpret
from helion._testing import xfailIfPallasTpu
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


@helion.kernel(backend="pallas")
def jagged_dense_bmm_2d_loop(
    seq_offsets: torch.Tensor, jagged: torch.Tensor, dense: torch.Tensor
) -> torch.Tensor:
    L, D = jagged.shape
    B, D, K = dense.shape
    out = torch.empty((L, K), dtype=jagged.dtype, device=jagged.device)
    for g in hl.grid(B):
        s = seq_offsets[g]
        e = seq_offsets[g + 1]
        for st, kt in hl.tile((s, 0), (e, K)):
            acc = hl.zeros([st, kt], dtype=torch.float32)
            for dt in hl.tile(0, D):
                acc = acc + torch.matmul(jagged[st, dt], dense[g, dt, kt])
            out[st, kt] = acc
    return out


_BMM_KERNELS = [
    jagged_dense_bmm,
    jagged_dense_bmm_2d_loop,
]


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


def _run(seq_offsets, jagged, dense, block_sizes, kernel=jagged_dense_bmm):
    return code_and_output(
        kernel,
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
    @parametrize("kernel", _BMM_KERNELS)
    def test_single_group(self, dtype: torch.dtype, kernel) -> None:
        # One group: exercises the aligned-enclosing read and the tail mask,
        # with no carry between groups.
        seq_offsets, jagged, dense = _inputs([0, 20], 128, 128, dtype)
        _code, out = _run(seq_offsets, jagged, dense, [16, 128, 128], kernel=kernel)
        torch.testing.assert_close(out, _ref_jagged_bmm(seq_offsets, jagged, dense))

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("kernel", _BMM_KERNELS)
    def test_aligned_groups_carry_dormant(self, dtype: torch.dtype, kernel) -> None:
        # Aligned boundaries: carry path is emitted but its runtime guard never fires.
        seq_offsets, jagged, dense = _inputs([0, 16, 32], 128, 128, dtype)
        code, out = _run(seq_offsets, jagged, dense, [16, 128, 128], kernel=kernel)
        self.assertIn("pl.program_id(0) != 0", code)
        torch.testing.assert_close(out, _ref_jagged_bmm(seq_offsets, jagged, dense))

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    @parametrize("kernel", _BMM_KERNELS)
    def test_carry_keeps_both_groups(self, kernel) -> None:
        # Two groups [0, 3) and [3, 16) share the [0, 16) boundary.  With
        # identity weights out == jagged, so a clobbered carry would be obvious.
        seq_offsets = torch.tensor([0, 3, 16], dtype=torch.int32, device=DEVICE)
        jagged = (
            torch.arange(1, 17, dtype=torch.bfloat16, device=DEVICE)[:, None]
            .expand(16, 128)
            .contiguous()
        )
        eye = torch.eye(128, dtype=torch.bfloat16, device=DEVICE)
        _code, out = _run(
            seq_offsets, jagged, torch.stack([eye, eye]), [16, 128, 128], kernel=kernel
        )
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
    @parametrize("kernel", _BMM_KERNELS)
    def test_bmm_block_eq_sublane(
        self, dtype: torch.dtype, offsets: list[int], kernel
    ) -> None:
        seq_offsets, jagged, dense = _inputs(offsets, 128, 128, dtype)
        code, out = _run(seq_offsets, jagged, dense, [16, 128, 128], kernel=kernel)
        self.assertIn("pltpu.emit_pipeline", code)
        torch.testing.assert_close(out, _ref_jagged_bmm(seq_offsets, jagged, dense))

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("kernel", _BMM_KERNELS)
    def test_bmm_block_gt_group(self, dtype: torch.dtype, kernel) -> None:
        # Two groups (13 rows) are smaller than block_row=32, total L=200 >> block_row.
        seq_offsets, jagged, dense = _inputs([0, 13, 100, 113, 200], 128, 128, dtype)
        _code, out = _run(seq_offsets, jagged, dense, [32, 128, 128], kernel=kernel)
        torch.testing.assert_close(out, _ref_jagged_bmm(seq_offsets, jagged, dense))

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("kernel", _BMM_KERNELS)
    def test_bmm_multi_k_tile(self, dtype: torch.dtype, kernel) -> None:
        # K=256 with block_col=128 gives two output-column tiles; the carry stacks
        # the per-column-tile boundaries along its scratch row dim.
        seq_offsets, jagged, dense = _inputs([0, 17, 40, 71], 128, 256, dtype)
        _code, out = _run(seq_offsets, jagged, dense, [16, 128, 128], kernel=kernel)
        torch.testing.assert_close(out, _ref_jagged_bmm(seq_offsets, jagged, dense))

    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("kernel", _BMM_KERNELS)
    def test_bmm_many_groups(self, dtype: torch.dtype, kernel) -> None:
        # 50 unaligned groups; carry scratch is per column-tile, not per group.
        offsets = list(range(0, 13 * 51, 13))  # 50 groups
        seq_offsets, jagged, dense = _inputs(offsets, 128, 128, dtype)
        code, out = _run(seq_offsets, jagged, dense, [16, 128, 128], kernel=kernel)
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
    @parametrize("kernel", _BMM_KERNELS)
    def test_bmm_empty_groups(
        self, dtype: torch.dtype, offsets: list[int], kernel
    ) -> None:
        # Zero-length groups (s == e) iterate no tiles; carry threads the neighbours.
        seq_offsets, jagged, dense = _inputs(offsets, 128, 128, dtype)
        _code, out = _run(seq_offsets, jagged, dense, [16, 128, 128], kernel=kernel)
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
    @parametrize("dynamic_cols", [True, False])
    def test_dynamic_rows(self, dynamic_cols: bool) -> None:
        @helion.kernel(backend="pallas", static_shapes=False)
        def jagged_scale_dyn(
            seq_offsets: torch.Tensor, jagged: torch.Tensor, dynamic_cols: hl.constexpr
        ) -> torch.Tensor:
            L, D = jagged.shape
            if not dynamic_cols:
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
            (seq_offsets, jagged, dynamic_cols),
            block_sizes=[16, 128],
            pallas_loop_type="emit_pipeline",
        )
        self.assertIn("(B,)", code)
        torch.testing.assert_close(out, jagged * 2)


def _ref_jagged_row_sum(seq_offsets: torch.Tensor, jagged: torch.Tensor):
    ref = torch.zeros(
        (seq_offsets.numel() - 1, jagged.shape[1]),
        dtype=torch.float32,
        device=jagged.device,
    )
    for g in range(ref.shape[0]):
        s, e = int(seq_offsets[g]), int(seq_offsets[g + 1])
        ref[g] = jagged[s:e].float().sum(0)
    return ref


@onlyBackends(["pallas"])
@skipUnlessPallas("JAX/Pallas TPU not available")
class TestPallasMultipleOfPromises(TestCase):
    """What we are willing to assert to Mosaic about a runtime row offset."""

    def test_direct_row_window_gets_no_sublane_promise(self) -> None:
        # A jagged row that is only ever read must not be promised a sublane
        # alignment, because nothing rounded its window down.
        @helion.kernel(backend="pallas")
        def nested_direct_window(
            seq_offsets: torch.Tensor,
            outer_read: torch.Tensor,
            nested_read: torch.Tensor,
            dense_out: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            _, width = nested_read.shape
            groups = seq_offsets.size(0) - 1
            scalar_out = torch.empty(
                (groups,), dtype=torch.float32, device=nested_read.device
            )
            for group in hl.grid(groups):
                begin, end = seq_offsets[group], seq_offsets[group + 1]
                for rows in hl.tile(begin, end):
                    scalar_out[group] = outer_read[rows, 0].sum()
                    for cols in hl.tile(0, width):
                        dense_out[group, cols] = dense_out[group, cols] + nested_read[
                            rows, cols
                        ].float().sum(0)
            return scalar_out, dense_out

        seq_offsets = torch.tensor([0, 13, 32], dtype=torch.int32)
        outer_read = torch.randn((32, 128), dtype=torch.float32)
        nested_read = torch.randn((32, 128), dtype=torch.float32)
        dense_out = torch.zeros((2, 128), dtype=torch.float32)
        code = nested_direct_window.bind(
            (seq_offsets, outer_read, nested_read, dense_out)
        ).to_code(
            helion.Config(block_sizes=[16, 128], pallas_loop_type="emit_pipeline")
        )
        # 8 is the f32 sublane.
        self.assertNotRegex(code, r"multiple_of\(\w+, 8\)")


@onlyBackends(["pallas"])
@skipUnlessPallas("JAX/Pallas TPU not available")
class TestPallasJaggedIndexing(TestCase):
    @xfailIfPallasInterpret(_XFAIL_INTERPRET)
    def test_rank_expanding_none_needs_no_window(self) -> None:
        # The None dimension in jagged[None, st, dt] is just for shape
        # compatibility, and should not preclude compilation.
        @helion.kernel(backend="pallas")
        def jagged_copy_expanded(
            seq_offsets: torch.Tensor, jagged: torch.Tensor
        ) -> torch.Tensor:
            L, D = jagged.shape
            B = seq_offsets.shape[0] - 1
            out = torch.zeros((B, L, D), dtype=torch.float32, device=jagged.device)
            for g in hl.grid(B):
                s = seq_offsets[g]
                e = seq_offsets[g + 1]
                for st in hl.tile(s, e):
                    for dt in hl.tile(0, D):
                        out[g, st, dt] = jagged[None, st, dt].sum(0) * 2.0
            return out

        seq_offsets = torch.tensor([0, 13, 25], dtype=torch.int32, device=DEVICE)
        jagged = torch.randn((25, 128), dtype=torch.float32, device=DEVICE)
        _code, out = code_and_output(
            jagged_copy_expanded,
            (seq_offsets, jagged),
            block_sizes=[16, 128],
            pallas_loop_type="emit_pipeline",
        )
        # Check only the rows that each group is storing to. The other rows
        # get NaNs in interpret mode, so let's skip them.
        torch.testing.assert_close(out[0, 0:13], jagged[0:13] * 2.0)
        torch.testing.assert_close(out[1, 13:25], jagged[13:25] * 2.0)


@onlyBackends(["pallas"])
@skipUnlessPallas("JAX/Pallas TPU not available")
@xfailIfPallasInterpret(_XFAIL_INTERPRET)
class TestPallasJaggedReductions(TestCase):
    """Reductions over a bf16 jagged row: aligned window, no store through the row."""

    def test_reduction_access_in_jagged_loop(self) -> None:
        # The jagged loop is innermost and slices the tensor itself.
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
        torch.testing.assert_close(
            out, _ref_jagged_row_sum(seq_offsets, jagged), rtol=1e-2, atol=1e-2
        )

    def test_reduction_access_in_nested_loop(self) -> None:
        # The jagged loop slices nothing itself; its window comes from the nested loop.
        @helion.kernel(backend="pallas")
        def jagged_col_accum(
            seq_offsets: torch.Tensor, jagged: torch.Tensor, out: torch.Tensor
        ) -> torch.Tensor:
            L, D = jagged.shape
            B = seq_offsets.shape[0] - 1
            for g in hl.grid(B):
                s = seq_offsets[g]
                e = seq_offsets[g + 1]
                for st in hl.tile(s, e):
                    for dt in hl.tile(0, D):
                        rows = jagged[st, dt].to(torch.float32)
                        out[g, dt] = out[g, dt] + rows.sum(0)
            return out

        seq_offsets = torch.tensor([0, 13, 25], dtype=torch.int32, device=DEVICE)
        jagged = torch.randn((25, 128), dtype=torch.bfloat16, device=DEVICE)
        out = torch.zeros((2, 128), dtype=torch.float32, device=DEVICE)
        _code, result = code_and_output(
            jagged_col_accum,
            (seq_offsets, jagged, out),
            block_sizes=[16, 128],
            pallas_loop_type="emit_pipeline",
        )
        torch.testing.assert_close(
            result, _ref_jagged_row_sum(seq_offsets, jagged), rtol=1e-2, atol=1e-2
        )

    def test_reduction_access_below_conditional(self) -> None:
        # The access is two graphs down: an if branch, then a nested column loop.
        @helion.kernel(backend="pallas")
        def jagged_conditional_accum(
            seq_offsets: torch.Tensor, jagged: torch.Tensor, out: torch.Tensor
        ) -> torch.Tensor:
            L, D = jagged.shape
            B = seq_offsets.shape[0] - 1
            for g in hl.grid(B):
                s = seq_offsets[g]
                e = seq_offsets[g + 1]
                for st in hl.tile(s, e):
                    if e - s > 8:
                        for dt in hl.tile(0, D):
                            rows = jagged[st, dt].to(torch.float32)
                            out[g, dt] = out[g, dt] + rows.sum(0)
            return out

        seq_offsets = torch.tensor([0, 13, 25], dtype=torch.int32, device=DEVICE)
        jagged = torch.randn((25, 128), dtype=torch.bfloat16, device=DEVICE)
        out = torch.zeros((2, 128), dtype=torch.float32, device=DEVICE)
        _code, result = code_and_output(
            jagged_conditional_accum,
            (seq_offsets, jagged, out),
            block_sizes=[16, 128],
            pallas_loop_type="emit_pipeline",
        )
        torch.testing.assert_close(
            result, _ref_jagged_row_sum(seq_offsets, jagged), rtol=1e-2, atol=1e-2
        )

    def test_reduction_remasks_after_pointwise(self) -> None:
        # Over-read rows must be re-masked after the +1.0, or they leak into the sum.
        @helion.kernel(backend="pallas")
        def jagged_plus_one_sum(
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
                        rows = jagged[st, dt].to(torch.float32) + 1.0
                        acc = acc + rows.sum(0)
                    out[g, dt] = acc
            return out

        seq_offsets = torch.tensor([0, 13, 25], dtype=torch.int32, device=DEVICE)
        jagged = torch.randn((25, 128), dtype=torch.bfloat16, device=DEVICE)
        _code, result = code_and_output(
            jagged_plus_one_sum,
            (seq_offsets, jagged),
            block_sizes=[128, 16],
            pallas_loop_type="emit_pipeline",
        )
        expected = _ref_jagged_row_sum(seq_offsets, jagged)
        lengths = (seq_offsets[1:] - seq_offsets[:-1]).to(torch.float32)[:, None]
        torch.testing.assert_close(result, expected + lengths, rtol=1e-2, atol=1e-2)


@onlyBackends(["pallas"])
@skipUnlessPallas("JAX/Pallas TPU not available")
class TestPallasJaggedCarryRejects(TestCase):
    """Shapes the carry refuses (or routes elsewhere) rather than miscompiling."""

    @parametrize("kernel", _BMM_KERNELS)
    def test_block_not_multiple_of_sublane_raises(self, kernel) -> None:
        # bf16 sublane S=16; block_row=8 is not a multiple, so the carry rejects it
        # loudly instead of clobbering the boundary with a plain store.
        seq_offsets, jagged, dense = _inputs([0, 13, 25], 128, 128, torch.bfloat16)
        with self.assertRaisesRegex(
            exc.InductorLoweringError, "block_row .* must be a multiple"
        ):
            _run(seq_offsets, jagged, dense, [8, 128, 128], kernel=kernel)

    @parametrize("out_dtype", [torch.bfloat16, torch.float32])
    def test_untiled_column_store_rejected(self, out_dtype: torch.dtype) -> None:
        # With offsets [0, 13, 32] and a 16-row bf16 block, the second group
        # logically starts at row 13 but its aligned window starts at row 0.
        # The load mask zeroes rows [0, 13), so a full store would overwrite the
        # first group with zeros. The untiled ":" column has no carry block
        # with which to save those rows; reject instead of silently corrupting.
        # An f32 output is stored DIRECT on its own, but the bf16 load rounds
        # the window, so that store writes the head rows all the same.
        @helion.kernel(backend="pallas")
        def jagged_full_row_store(
            seq_offsets: torch.Tensor, jagged: torch.Tensor, out: torch.Tensor
        ) -> torch.Tensor:
            B = seq_offsets.shape[0] - 1
            for g in hl.grid(B):
                s = seq_offsets[g]
                e = seq_offsets[g + 1]
                for st in hl.tile(s, e):
                    out[st, :] = (jagged[st, :] * 2).to(out.dtype)
            return out

        seq_offsets = torch.tensor([0, 13, 32], dtype=torch.int32)
        jagged = torch.randn((32, 128), dtype=torch.bfloat16)
        out = torch.empty((32, 128), dtype=out_dtype)
        with self.assertRaisesRegex(
            exc.InductorLoweringError,
            "written through its row dim is only supported when ordered carry",
        ):
            jagged_full_row_store.bind((seq_offsets, jagged, out)).to_code(
                helion.Config(block_sizes=[16], pallas_loop_type="emit_pipeline")
            )

    def test_nested_direct_store_under_aligned_window_rejected(self) -> None:
        # The store sits one loop down; unseen, it overwrites the preceding group.
        @helion.kernel(backend="pallas")
        def nested_mixed_dtype_row_store(
            seq_offsets: torch.Tensor, jagged: torch.Tensor
        ) -> torch.Tensor:
            L, D = jagged.shape
            B = seq_offsets.shape[0] - 1
            out = torch.empty((L, 1, D), dtype=torch.float32, device=jagged.device)
            for g in hl.grid(B):
                s = seq_offsets[g]
                e = seq_offsets[g + 1]
                for st in hl.tile(s, e):
                    for ct in hl.tile(0, D):
                        out[st, 0, ct] = jagged[st, ct].to(torch.float32)
            return out

        seq_offsets = torch.tensor([0, 13, 25], dtype=torch.int32)
        jagged = torch.randn((25, 128), dtype=torch.bfloat16)
        with self.assertRaisesRegex(
            exc.InductorLoweringError,
            "written through its row dim is only supported when ordered carry",
        ):
            nested_mixed_dtype_row_store.bind((seq_offsets, jagged)).to_code(
                helion.Config(block_sizes=[16, 128], pallas_loop_type="emit_pipeline")
            )

    def test_store_below_while_rejected(self) -> None:
        # Same store, two graphs down: inside a while body, then the column loop.
        @helion.kernel(backend="pallas")
        def while_nested_row_store(
            seq_offsets: torch.Tensor, jagged: torch.Tensor
        ) -> torch.Tensor:
            L, D = jagged.shape
            B = seq_offsets.shape[0] - 1
            out = torch.empty((L, 1, D), dtype=torch.float32, device=jagged.device)
            for g in hl.grid(B):
                s = seq_offsets[g]
                e = seq_offsets[g + 1]
                for st in hl.tile(s, e):
                    step = torch.zeros([], dtype=torch.int32, device=jagged.device)
                    while step < 1:
                        for ct in hl.tile(0, D):
                            out[st, 0, ct] = jagged[st, ct].to(torch.float32)
                        step = step + 1
            return out

        seq_offsets = torch.tensor([0, 13, 25], dtype=torch.int32)
        jagged = torch.randn((25, 128), dtype=torch.bfloat16)
        with self.assertRaisesRegex(
            exc.InductorLoweringError,
            "written through its row dim is only supported when ordered carry",
        ):
            while_nested_row_store.bind((seq_offsets, jagged)).to_code(
                helion.Config(block_sizes=[16, 128], pallas_loop_type="emit_pipeline")
            )

    def test_shifted_store_through_row_rejected(self) -> None:
        # Storing to st.index + 1 shifts every row off the row it was read
        # from, so a group's store can land on a row the previous group owns.
        # The kernel must be rejected rather than silently corrupt that group.
        @helion.kernel(backend="pallas")
        def jagged_shift_store(
            seq_offsets: torch.Tensor, jagged: torch.Tensor
        ) -> torch.Tensor:
            L, D = jagged.shape
            B = seq_offsets.shape[0] - 1
            out = torch.zeros((L, D), dtype=jagged.dtype, device=jagged.device)
            for g in hl.grid(B):
                s = seq_offsets[g]
                e = seq_offsets[g + 1]
                for st in hl.tile(s, e):
                    for dt in hl.tile(0, D):
                        out[st.index + 1, dt] = jagged[st, dt] * 2
            return out

        seq_offsets = torch.tensor([0, 13, 32], dtype=torch.int32)
        jagged = torch.randn((32, 128), dtype=torch.bfloat16)
        with self.assertRaisesRegex(
            exc.InductorLoweringError,
            "written through its row dim is only supported when ordered carry",
        ):
            jagged_shift_store.bind((seq_offsets, jagged)).to_code(
                helion.Config(block_sizes=[16, 128], pallas_loop_type="emit_pipeline")
            )

    def test_atomic_write_through_row_rejected(self) -> None:
        # An atomic through the row is a write too, so it needs the same rejection.
        @helion.kernel(backend="pallas")
        def jagged_atomic_write(
            seq_offsets: torch.Tensor, jagged: torch.Tensor, out: torch.Tensor
        ) -> torch.Tensor:
            L, D = jagged.shape
            B = seq_offsets.shape[0] - 1
            for g in hl.grid(B):
                s = seq_offsets[g]
                e = seq_offsets[g + 1]
                for st in hl.tile(s, e):
                    for dt in hl.tile(0, D):
                        hl.atomic_xchg(out, [st, dt], jagged[st, dt] + 1.0)
            return out

        seq_offsets = torch.tensor([0, 13, 32], dtype=torch.int32)
        jagged = torch.randn((32, 128), dtype=torch.bfloat16)
        out = torch.zeros_like(jagged)
        with self.assertRaisesRegex(
            exc.InductorLoweringError,
            "written through its row dim is only supported when ordered carry",
        ):
            jagged_atomic_write.bind((seq_offsets, jagged, out)).to_code(
                helion.Config(block_sizes=[16, 128], pallas_loop_type="emit_pipeline")
            )

    def test_atomic_beside_carried_store_rejected(self) -> None:
        @helion.kernel(backend="pallas")
        def carried_store_plus_atomic(
            seq_offsets: torch.Tensor, jagged: torch.Tensor, atom: torch.Tensor
        ) -> torch.Tensor:
            L, D = jagged.shape
            B = seq_offsets.shape[0] - 1
            out = torch.zeros((L, D), dtype=jagged.dtype, device=jagged.device)
            for g in hl.grid(B):
                s = seq_offsets[g]
                e = seq_offsets[g + 1]
                for st in hl.tile(s, e):
                    for dt in hl.tile(0, D):
                        out[st, dt] = jagged[st, dt] * 2
                        hl.atomic_xchg(atom, [st, dt], jagged[st, dt] + 1.0)
            return out

        seq_offsets = torch.tensor([0, 13, 32], dtype=torch.int32)
        jagged = torch.randn((32, 128), dtype=torch.bfloat16)
        atom = torch.zeros_like(jagged)
        with self.assertRaisesRegex(
            exc.InductorLoweringError,
            "an atomic write through a jagged row tile whose slices must be "
            "sublane-aligned bypasses the ordered carry",
        ):
            carried_store_plus_atomic.bind((seq_offsets, jagged, atom)).to_code(
                helion.Config(block_sizes=[16, 128], pallas_loop_type="emit_pipeline")
            )

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

    def test_partial_slice_both_bounds(self) -> None:
        @helion.kernel(backend="pallas")
        def kernel(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(src.size(0)):
                dst[tile, :8] = src[tile, :8]
                dst[tile, 8:16] = src[tile, 8:16] * 2.0
                dst[tile, 16:] = src[tile, 16:]
            return dst

        src = torch.randn((64, 32), device=DEVICE)
        dst = torch.zeros((64, 32), device=DEVICE)
        code, out = code_and_output(kernel, (src, dst), block_sizes=[16])
        self.assertIn("8:16", code)
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


@onlyBackends(["pallas"])
@skipUnlessPallas("JAX/Pallas TPU not available")
class TestPallasVmemScalarLoad(TestCase):
    """Loads with a runtime scalar index on a VMEM tensor's minor dims."""

    @parametrize("cols", [64, 256])
    @parametrize("dtype", [torch.float32, torch.int32])
    def test_runtime_row_index(self, dtype: torch.dtype, cols: int) -> None:
        @helion.kernel(backend="pallas", static_shapes=True)
        def row_sum_plus_last(x: torch.Tensor) -> torch.Tensor:
            rows, cols = x.shape
            out = torch.empty(rows, dtype=torch.float32, device=x.device)
            for row in hl.grid(rows):
                out[row] = x[row, :].to(torch.float32).sum() + x[row, cols - 1].to(
                    torch.float32
                )
            return out

        source = ((torch.arange(4 * cols).reshape(4, cols)) % 53).to(dtype)
        code, result = code_and_output(row_sum_plus_last, (source.to(DEVICE),))
        self.assertNotIn("pltpu.roll", code)
        expected = (source.float().sum(-1) + source[:, -1].float()).to(DEVICE)
        torch.testing.assert_close(result, expected)

    def test_runtime_row_index_full_row(self) -> None:
        @helion.kernel(backend="pallas", static_shapes=True)
        def row_flip_load(x: torch.Tensor) -> torch.Tensor:
            rows, cols = x.shape
            out = torch.empty_like(x)
            for row in hl.grid(rows):
                out[row, :] = x[rows - 1 - row, :]
            return out

        source = torch.arange(17 * 65, dtype=torch.float32).reshape(17, 65)
        code, result = code_and_output(row_flip_load, (source.to(DEVICE),))
        self.assertNotIn("pltpu.roll", code)
        torch.testing.assert_close(result, source.flip(0).to(DEVICE))

    @parametrize(
        "dtype",
        [torch.bfloat16, torch.int16, torch.int8, torch.uint8, torch.float8_e4m3fn],
    )
    def test_runtime_row_index_packed_dtypes(self, dtype: torch.dtype) -> None:
        @helion.kernel(backend="pallas", static_shapes=True)
        def row_sum_plus_last(x: torch.Tensor) -> torch.Tensor:
            rows, cols = x.shape
            out = torch.empty(rows, dtype=torch.float32, device=x.device)
            for row in hl.grid(rows):
                out[row] = x[row, :].to(torch.float32).sum() + x[row, cols - 1].to(
                    torch.float32
                )
            return out

        source = (torch.arange(4 * 64) % 53).reshape(4, 64).to(dtype)
        code, result = code_and_output(row_sum_plus_last, (source.to(DEVICE),))
        self.assertIn("pltpu.roll", code)
        expected = (source.float().sum(-1) + source[:, -1].float()).to(DEVICE)
        torch.testing.assert_close(result, expected)

    @xfailIfPallasTpu("pl.kernel launcher: DMAs with bool dtypes are not supported")
    def test_runtime_row_index_bool(self) -> None:
        @helion.kernel(backend="pallas", static_shapes=True)
        def row_sum_plus_last(x: torch.Tensor) -> torch.Tensor:
            rows, cols = x.shape
            out = torch.empty(rows, dtype=torch.float32, device=x.device)
            for row in hl.grid(rows):
                out[row] = x[row, :].to(torch.float32).sum() + x[row, cols - 1].to(
                    torch.float32
                )
            return out

        source = torch.arange(4 * 64).reshape(4, 64) % 3 == 0
        code, result = code_and_output(row_sum_plus_last, (source.to(DEVICE),))
        expected = (source.float().sum(-1) + source[:, -1].float()).to(DEVICE)
        torch.testing.assert_close(result, expected)

    @parametrize(
        "dtype", [torch.bfloat16, torch.float32, torch.int32, torch.float8_e4m3fn]
    )
    def test_runtime_lane_index(self, dtype: torch.dtype) -> None:
        @helion.kernel(backend="pallas", static_shapes=True)
        def column_load(x: torch.Tensor) -> torch.Tensor:
            rows, cols = x.shape
            out = torch.empty((cols, rows), dtype=torch.float32, device=x.device)
            for col in hl.grid(cols):
                out[col, :] = x[:, col].to(torch.float32)
            return out

        source = (torch.arange(17 * 65) % 53).reshape(17, 65).to(dtype)
        code, result = code_and_output(column_load, (source.to(DEVICE),))
        self.assertIn("pltpu.roll", code)
        self.assertIn("jnp.pad", code)
        torch.testing.assert_close(result, source.T.float().to(DEVICE))

    def test_runtime_lane_index_rank_one(self) -> None:
        @helion.kernel(backend="pallas", static_shapes=True)
        def rank_one_load(x: torch.Tensor) -> torch.Tensor:
            (size,) = x.shape
            out = torch.empty((size, size), dtype=torch.float32, device=x.device)
            for index in hl.grid(size):
                out[index, :] = x[index].to(torch.float32) + x[:].to(torch.float32)
            return out

        source = torch.arange(128, dtype=torch.bfloat16)
        code, result = code_and_output(rank_one_load, (source.to(DEVICE),))
        self.assertIn("pltpu.roll", code)
        self.assertIn("jnp.expand_dims", code)
        expected = source[:, None].float() + source[None, :].float()
        torch.testing.assert_close(result, expected.to(DEVICE))

    def test_runtime_row_and_lane_index(self) -> None:
        @helion.kernel(backend="pallas", static_shapes=True)
        def two_axis_load(x: torch.Tensor) -> torch.Tensor:
            rows, cols = x.shape
            out = torch.empty((rows, cols), dtype=torch.float32, device=x.device)
            for row, col in hl.grid([rows, cols]):
                out[row, col] = x[row, :].to(torch.float32).sum() + x[row, col].to(
                    torch.float32
                )
            return out

        source = (torch.arange(4 * 65).reshape(4, 65) % 53).to(torch.bfloat16)
        code, result = code_and_output(two_axis_load, (source.to(DEVICE),))
        self.assertGreaterEqual(code.count("pltpu.roll"), 2)
        expected = source.float().sum(-1, keepdim=True) + source.float()
        torch.testing.assert_close(result, expected.to(DEVICE))


@onlyBackends(["pallas"])
@skipUnlessPallas("JAX/Pallas TPU not available")
class TestPallasClampedRefs(TestCase):
    """Loads/stores through BlockSpec refs clamped to min(block, dim)."""

    def test_load_pad_store_slice_roundtrip(self) -> None:
        # dim 100 < block 128: load is zero-padded to block shape, store
        # value is sliced back to the clamped ref.
        @helion.kernel(backend="pallas")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile0, tile1 in hl.tile(x.size()):
                out[tile0, tile1] = x[tile0, tile1] * 2.0
            return out

        x = torch.randn((64, 100), device=DEVICE)
        code, out = code_and_output(kernel, (x,), block_sizes=[16, 128])
        self.assertIn("jnp.pad", code)
        self.assertIn(":100", code)
        torch.testing.assert_close(out, x * 2.0)

    def test_size1_dim_broadcasts_unpadded(self) -> None:
        # size-1 dim under a wider block must broadcast, not be zero-padded.
        @helion.kernel(backend="pallas")
        def kernel(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile0, tile1 in hl.tile(x.size()):
                out[tile0, tile1] = x[tile0, tile1] * m[tile0, :]
            return out

        x = torch.randn((64, 128), device=DEVICE)
        m = torch.randn((64, 1), device=DEVICE)
        code, out = code_and_output(kernel, (x, m), block_sizes=[16, 128])
        self.assertNotIn("jnp.pad", code)
        torch.testing.assert_close(out, x * m)

    def test_atomic_add_clamped_ref(self) -> None:
        # dim 10 < block 32: the RMW reads _prev and stores to a clamped ref,
        # so the padded value operand must be sliced back to match.
        @helion.kernel(backend="pallas")
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(x.size(0)):
                hl.atomic_add(x, [tile], y[tile])
            return x

        x = torch.zeros(10, device=DEVICE)
        y = torch.ones(10, device=DEVICE)
        _code, out = code_and_output(kernel, (x, y), block_sizes=[32])
        torch.testing.assert_close(out, torch.ones(10, device=DEVICE))

    @skipIfPallas("indirect access is unsupported by both one-hot and DMA lowering")
    def test_masked_load_spanning_tensor(self) -> None:
        # Tile spans past the tensor's extent (concat pattern): the padded
        # load composes with extra_mask / where.
        @helion.kernel(backend="pallas")
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty(
                [x.size(0), x.size(1) + y.size(1)], dtype=x.dtype, device=x.device
            )
            for tile0, tile1 in hl.tile(out.size()):
                x_part = hl.load(
                    x, [tile0, tile1], extra_mask=(tile1.index < x.size(1))[None, :]
                )
                y_part = hl.load(
                    y,
                    [tile0, tile1.index - x.size(1)],
                    extra_mask=(tile1.index >= x.size(1))[None, :],
                )
                out[tile0, tile1] = torch.where(
                    (tile1.index < x.size(1))[None, :], x_part, y_part
                )
            return out

        x = torch.randn((32, 20), device=DEVICE)
        y = torch.randn((32, 44), device=DEVICE)
        code, out = code_and_output(kernel, (x, y), block_sizes=[16, 64])
        self.assertIn("jnp.pad", code)
        torch.testing.assert_close(out, torch.cat([x, y], dim=1))


instantiate_parametrized_tests(TestPallasJaggedCarrySimple)
instantiate_parametrized_tests(TestPallasJaggedCarryBmm)
instantiate_parametrized_tests(TestPallasJaggedCarryRejects)
instantiate_parametrized_tests(TestPallasVmemScalarLoad)
