"""Row values of fragment results kept in registers (``metal_matmul=
"fragment"``, ``metal/fragment_lines.py``): row reductions of a matmul's
result folded in each lane and across a row's four lanes by
``simd_shuffle_xor``, element-wise ops of them per row slot, loop-carried
ones, element loops and updates reading them at their row, and their
stores.  Renders on CPU tensors (``helion._testing.metal.render_msl``),
with no Apple GPU; ``test_metal_fragment_mma.py`` runs the kernels."""

from __future__ import annotations

import re

import pytest
import torch

import helion
from helion import exc
from helion._testing import TestCase
from helion._testing import skipUnlessBackends
from helion._testing.metal import render_msl
from helion._testing.metal_fragment import fragment_config
from helion._testing.metal_fragment import matmul_softmax
from helion._testing.metal_fragment import online_rescale
from helion._testing.metal_fragment import row_lse
import helion.language as hl

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]

F32 = torch.float32


@helion.kernel(backend="metal", static_shapes=True)
def row_sums(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The accumulator summed over N, stored per row."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m], dtype=F32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
        out[tile_m] = acc.sum(1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def row_sums_half(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The row sums converted to half, which no line holds."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m], dtype=torch.float16, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
        out[tile_m] = acc.sum(1).to(torch.float16)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def row_sums_broadcast(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The row sums broadcast across the N tile into a full output tile."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.sum(1)[:, None]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def guarded_shift(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
    """The accumulator scaled by a float kernel argument, less its row max,
    the max replaced by 0 in rows where it is -inf (a bool row value)."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
        s = acc * scale
        row_max = s.amax(1)
        safe = torch.where(row_max == float("-inf"), 0.0, row_max)
        out[tile_m, :] = s - safe[:, None]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def cleared_bias_line(
    x: torch.Tensor, y: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """The row sums plus ``b``'s old rows, which the kernel clears before
    the K loop: a line op would re-run their load after the clear."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m], dtype=F32, device=x.device)
    for tile_m in hl.tile(m):
        old = b[tile_m]
        b[tile_m] = hl.zeros([tile_m], dtype=F32)
        acc = hl.zeros([tile_m, n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
        out[tile_m] = acc.sum(1) + old
    return out


@helion.kernel(backend="metal", static_shapes=True)
def cleared_bias_reduction(
    x: torch.Tensor, y: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """The rows of the accumulator plus ``b``'s old rows, summed: the row
    sum's element loop would re-run the cleared load."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m], dtype=F32, device=x.device)
    for tile_m in hl.tile(m):
        old = b[tile_m]
        b[tile_m] = hl.zeros([tile_m], dtype=F32)
        acc = hl.zeros([tile_m, n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
        out[tile_m] = (acc + old[:, None]).sum(1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def cleared_bias_seed(
    x: torch.Tensor, y: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """A running row max of x @ y over N tiles seeded from ``b``'s old
    rows: the seed, before the N loop, would re-run the cleared load."""
    m = x.size(0)
    hl.specialize(x.size(1))
    n = y.size(1)
    out = torch.empty([m], dtype=F32, device=x.device)
    for tile_m in hl.tile(m):
        m_i = b[tile_m]
        b[tile_m] = hl.zeros([tile_m], dtype=F32)
        q = x[tile_m, :]
        for tile_n in hl.tile(n):
            m_i = torch.maximum(m_i, torch.amax(torch.mm(q, y[:, tile_n]), 1))
        out[tile_m] = m_i
    return out


@helion.kernel(backend="metal", static_shapes=True)
def nan_or_negative_shift(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The accumulator less its row max, the max replaced by 0 in rows
    where it is NaN or negative: bool row values from a call
    (``isnan``) and from an or of it."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
        row_max = acc.amax(1)
        bad = torch.isnan(row_max) | (row_max < 0)
        out[tile_m, :] = acc - torch.where(bad, 0.0, row_max)[:, None]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def normalized_then_scaled_rhs(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """x @ y over its row sums, then, in a second tile loop, x @ (z * 2):
    the second matmul's rhs is computed, so its accumulator is transposed,
    and its epilogue reads none of the first's row values."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    out2 = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
        out[tile_m, :] = acc / acc.sum(1)[:, None]
    for tile_m2, tile_n2 in hl.tile([m, n]):
        acc2 = hl.zeros([tile_m2, tile_n2], dtype=F32)
        for tile_k2 in hl.tile(k):
            acc2 = torch.addmm(acc2, x[tile_m2, tile_k2], z[tile_k2, tile_n2] * 2.0)
        out2[tile_m2, tile_n2] = acc2 + 1.0
    return out, out2


@helion.kernel(backend="metal", static_shapes=True)
def running_max_half(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A running row max of x @ y over N tiles, converted to half after
    the N loop: no line holds half, so the carry's value after the loop is
    read in the Scalar layout."""
    m = x.size(0)
    hl.specialize(x.size(1))
    n = y.size(1)
    out = torch.empty([m], dtype=torch.float16, device=x.device)
    for tile_m in hl.tile(m):
        m_i = hl.full([tile_m], float("-inf"), dtype=F32)
        q = x[tile_m, :]
        for tile_n in hl.tile(n):
            m_i = torch.maximum(m_i, torch.amax(torch.mm(q, y[:, tile_n]), 1))
        out[tile_m] = m_i.to(torch.float16)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def running_max_read_back(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A running row max stored after the N loop, then read back from its
    output: no line store, which writes from the lane holding column 0 of
    a row, may write a tensor the kernel reads."""
    m = x.size(0)
    hl.specialize(x.size(1))
    n = y.size(1)
    out = torch.empty([m], dtype=F32, device=x.device)
    out2 = torch.empty([m], dtype=F32, device=x.device)
    for tile_m in hl.tile(m):
        m_i = hl.full([tile_m], float("-inf"), dtype=F32)
        q = x[tile_m, :]
        for tile_n in hl.tile(n):
            m_i = torch.maximum(m_i, torch.amax(torch.mm(q, y[:, tile_n]), 1))
        out[tile_m] = m_i
        out2[tile_m] = out[tile_m] * 2.0
    return out, out2


def _assert_lines(msl: str) -> None:
    """The op runs on fragments and its row values stay in registers: no
    threadgroup memory, no barrier, a fold of each row's four lanes."""
    assert "simdgroup_multiply_accumulate(" in msl
    assert "threadgroup_barrier" not in msl
    assert not re.search(r"\bthreadgroup \w+ ", msl)
    assert "simd_shuffle_xor(_lr, 1)" in msl
    assert "simd_shuffle_xor(_lr, 8)" in msl


class TestFragmentLines(TestCase):
    def test_reductions(self) -> None:
        """A row max and a row sum of the accumulator, which the store's
        element loop reads at each element's row slot, over 13 rows (a row
        tail) in 2 fragment rows; and a row sum stored by the lane holding
        column 0 of its row."""
        x = torch.randn(13, 32, dtype=torch.float16)
        y = torch.randn(32, 32, dtype=torch.float16)
        msl = render_msl(matmul_softmax, (x, y), fragment_config([16, 16]))
        _assert_lines(msl)
        self.assertIn("_lr = ::c10::metal::max(_lr, simd_shuffle_xor(_lr, 1));", msl)
        self.assertIn("_lr = (_lr + simd_shuffle_xor(_lr, 8));", msl)
        self.assertRegex(msl, r"float _line_\d+\[2\];")
        # The epilogue reads both rows' values at its element's slot.
        self.assertRegex(msl, r"\(_fv - subscript\[\(\(_fi >> 1\) / 4\)\]\)")
        msl = render_msl(row_sums, (x, y), fragment_config([16, 16]))
        _assert_lines(msl)
        self.assertIn("if (_fn == 0) {", msl)

    def test_one_row_block(self) -> None:
        """A one-row M block grows to an 8-row fragment, whose rows past
        the block hold a line reduction's identity: a row value's store
        writes only from the lane holding the block's row, as the eight
        lanes holding column 0 would all write its address."""
        x = torch.randn(1, 32)
        y = torch.randn(32, 16)
        msl = render_msl(row_sums, (x, y), fragment_config([1, 16]))
        _assert_lines(msl)
        self.assertIn("if (_fn == 0 && _e_m < 1) {", msl)
        self.assertNotIn("if (_fn == 0) {", msl)

    def test_carried(self) -> None:
        """A running max and sum that the N loop carries, seeded before it
        (-inf and 0), updated per row slot from each tile's reductions and
        stored after it.  N = 40 in tiles of 16 leaves a tail, whose
        columns the reductions skip by the N tile's mask: past N's end a
        fragment element holds no column's value."""
        x = torch.randn(13, 32)
        y = torch.randn(32, 40)
        msl = render_msl(row_lse, (x, y), fragment_config([16, 16]))
        _assert_lines(msl)
        self.assertRegex(msl, r"_line_\d+\[_ri\] = m_i;")
        self.assertRegex(msl, r"_line_\d+\[_ri\] = l_i;")
        self.assertRegex(msl, r"_lr = \(\(mask_\d+\) \? ::c10::metal::max\(_lr, ")
        self.assertRegex(msl, r"lse\[indices_\d+\] = v_\d+\[_ri\];")

    def test_carried_rejected_after_loop(self) -> None:
        """A loop-carried row value whose value after the loop has a use no
        line may have (a conversion to half; a store into a tensor the
        kernel reads back) is no line at all: its seed, its value in the
        loop and its value after the loop are one register line, so the
        whole carry falls back to the Scalar layout, which the fragment
        lowering refuses."""
        x = torch.randn(13, 32)
        y = torch.randn(32, 40)
        for kernel in (running_max_half, running_max_read_back):
            with (
                self.subTest(kernel=kernel.fn.__name__),
                self.assertRaisesRegex(
                    exc.BackendUnsupported, "read in the Scalar layout"
                ),
            ):
                render_msl(kernel, (x, y), fragment_config([16, 16]))

    def test_update(self) -> None:
        """An accumulator rescaled before each op by a row value of another
        matmul's result: the update's element loop reads it in place, so
        nothing is staged."""
        args = (
            torch.randn(16, 32),
            torch.randn(32, 32),
            torch.randn(16, 32),
            torch.randn(32, 16),
        )
        msl = render_msl(online_rescale, args, fragment_config([16, 16]))
        _assert_lines(msl)
        self.assertNotIn("_mpp_vec", msl)
        self.assertRegex(
            msl, r"auto v_\d+ = \(_fv \* subscript\[\(\(_fi >> 1\) / 2\)\]\);"
        )

    def test_scalar_use(self) -> None:
        """A row value read other than by a line op, an element loop or a
        store (a conversion to half here: a line holds float32) falls back
        to the Scalar layout, which the fragment lowering refuses."""
        x = torch.randn(16, 32, dtype=torch.float16)
        y = torch.randn(32, 16, dtype=torch.float16)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "a matmul whose result is read in the Scalar layout"
        ):
            render_msl(row_sums_half, (x, y), fragment_config([16, 16]))

    def test_broadcast_store(self) -> None:
        """A row value stored across the N tile is no line store: a line
        store writes one element per row, from the lane holding column 0,
        so the tile's other columns would stay unwritten.  The row value
        falls back to the Scalar layout, which the fragment lowering
        refuses."""
        x = torch.randn(16, 64)
        y = torch.randn(64, 16)
        config = fragment_config([16, 16, 16], num_threads=[2, 16])
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "a matmul whose result is read in the Scalar layout"
        ):
            render_msl(row_sums_broadcast, (x, y), config)

    def test_predicate_line(self) -> None:
        """A bool row value (``row_max == -inf``) is a line, its slots
        ``bool``, and the ``where`` that reads it is one; the result scaled
        by a float kernel argument, the same on every thread, stays on the
        fragments."""
        x = torch.randn(16, 32, dtype=torch.float16)
        y = torch.randn(32, 16, dtype=torch.float16)
        msl = render_msl(guarded_shift, (x, y, 0.5), fragment_config([16, 16]))
        _assert_lines(msl)
        self.assertRegex(msl, r"bool _line_\d+\[2\];|bool \w+\[2\];")

    def test_written_tensor(self) -> None:
        """A row value read from a tensor the kernel writes is no line, nor
        an input of a line reduction or a seed: a line loop, a reduction's
        element loop or a seed re-runs its load where the line is computed,
        after the write and on another thread than the write's.  The row
        values fall back to the Scalar layout, which the fragment lowering
        refuses."""
        args = (torch.randn(16, 32), torch.randn(32, 16), torch.randn(16))
        for kernel in (cleared_bias_line, cleared_bias_reduction, cleared_bias_seed):
            with (
                self.subTest(kernel=kernel.fn.__name__),
                self.assertRaisesRegex(
                    exc.BackendUnsupported, "read in the Scalar layout"
                ),
            ):
                render_msl(kernel, args, fragment_config([16, 16]))

    def test_line_dtypes(self) -> None:
        """A line's slots take its dtype, whatever computes it: ``isnan`` of
        a row max, and its or with a comparison, are ``bool`` lines; the
        ``where`` that reads them a ``float`` one."""
        x = torch.randn(16, 32, dtype=torch.float16)
        y = torch.randn(32, 16, dtype=torch.float16)
        msl = render_msl(nan_or_negative_shift, (x, y), fragment_config([16, 16]))
        _assert_lines(msl)
        kinds = {name: kind for kind, name in re.findall(r"\b(\w+) (\w+)\[2\];", msl)}

        def kind_of(pattern: str) -> str:
            match = re.search(rf"(\w+)\[_ri\] = {pattern}", msl)
            assert match is not None, pattern
            return kinds[match.group(1)]

        self.assertEqual(kind_of(r"metal::isnan\("), "bool")
        self.assertEqual(kind_of(r"\(\w+\[_ri\] \| \w+\[_ri\]\)"), "bool")
        self.assertEqual(kind_of(r"\(\w+\[_ri\] \? "), "float")

    def test_lines_scoped(self) -> None:
        """An element loop reads only the row values it uses: a transposed
        accumulator's epilogue in a later tile loop reads none of the first
        matmul's, which that matmul's epilogue reads at its row slot."""
        args = (torch.randn(16, 32), torch.randn(32, 16), torch.randn(32, 16))
        config = fragment_config([16, 16, 16, 16, 16], num_threads=[16, 2, 16, 2, 1])
        msl = render_msl(normalized_then_scaled_rhs, args, config)
        # The first run's A fragments hold M in their rows, the second's N.
        self.assertIn("int _e_m = 8 * (_fi >> 1) + _fm;", msl)
        self.assertIn("int _e_n = 8 * (_fi >> 1) + _fm;", msl)
        self.assertRegex(msl, r"\(_fv / subscript\[\(\(_fi >> 1\) / 2\)\]\)")
        self.assertRegex(msl, r"out2\[[^\]]+\] = v_\d+;")
