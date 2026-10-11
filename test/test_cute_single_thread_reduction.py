"""Regression coverage for CuTe reductions shrunk to a single live thread.

``CuteBackend.adjust_reduction_thread_count`` collapses a reduction's thread
count to 1 once the competing tile/reduction axes exhaust the CTA thread
budget. ``TileStrategy._compute_thread_axis_offset`` then lets a tile strategy
share that reduction's thread axis, so the reduction must index with a constant
0 rather than ``thread_idx()`` (which would alias the tile's thread id and
scatter the slice store across the wrong columns / out of bounds).  Both the
persistent (synthetic lane) and the rolled (``reduction_loops``) index forms
are covered.

A single-thread reduction also claims no thread axis of its own
(``ReductionStrategy._claims_thread_axis``): an attention kernel whose value
head dim differs from the q/k head dim holds four reduction dims (the two head
dims plus the two tile-sized dims its causal mask's ``tile.index[...]`` slices
allocate), and handing each a private axis ran past the three hardware axes.

The launch layout (``TileStrategyDispatch.thread_axis_for_strategy``, which
sizes the launch block) must not count an axis for it either:
``x[tile, :] @ w[:, :]`` with a 32-thread K reduction beside a one-thread N
lane put the row tile on axis 1 in the body and on axis 2 in the launch, which
ran ``block=(32, 1, 32)`` and stored one row per tile.  The launch also
reserves the kernel-wide reduction axes in a root loop without reductions, as
the body does.
"""

from __future__ import annotations

import math
import re
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion import exc
from helion._compiler.cute.backend import launch_drops_root_threads
from helion._compiler.reduction_strategy import ReductionStrategy
from helion._compiler.tile_dispatch import TileStrategyDispatch
from helion._compiler.tile_strategy import BlockSizeTileStrategy
from helion._compiler.tile_strategy import TileStrategy
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True)
def _concat2d_dim1_slices(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.size(0) == y.size(0)
    out = torch.empty(
        [x.size(0), x.size(1) + y.size(1)], dtype=x.dtype, device=x.device
    )
    n1 = x.size(1)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m, :n1] = x[tile_m, :]
        out[tile_m, n1:] = y[tile_m, :]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _row_sum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m] = x[tile_m, :].sum(-1)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _full_slice_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), w.size(1)], device=x.device)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m, :] = x[tile_m, :] @ w[:, :]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _softmax_full_slice_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), w.size(1)], device=x.device)
    for tile_m in hl.tile(x.size(0)):
        p = torch.softmax(x[tile_m, :], dim=-1)
        out[tile_m, :] = p @ w[:, :] + x[tile_m, :].sum(-1, keepdim=True)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _scaled_rows_and_inner_tiles(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(y)
    shifted = torch.empty_like(z)
    for tile_m in hl.tile(x.size(0)):
        rows = x[tile_m, :]
        out[tile_m, :] = y[tile_m, :] * rows.sum(-1, keepdim=True)
        for tile_d in hl.tile(z.size(1)):
            shifted[tile_m, tile_d] = z[tile_m, tile_d] + 1
    return out, shifted


@helion.kernel(backend="cute", static_shapes=True)
def _doubled_then_incremented(x: torch.Tensor) -> torch.Tensor:
    doubled = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        doubled[tile] = x[tile] * 2
    hl.barrier()
    for tile in hl.tile(x.size(0)):
        out[tile] = doubled[tile] + 1
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _row_sums_then_doubled(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    sums = torch.empty([x.size(0)], device=x.device)
    doubled = torch.empty_like(y)
    for tile_m in hl.tile(x.size(0)):
        sums[tile_m] = x[tile_m, :].sum(-1)
    for tile_n in hl.tile(y.size(0)):
        doubled[tile_n] = y[tile_n] * 2
    return sums, doubled


@helion.kernel(backend="cute", static_shapes=True)
def _full_slice_matmul_grads(
    grad_out: torch.Tensor, x: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, _k = x.size()
    grad_x = torch.empty_like(x)
    block_m = hl.register_block_size(m)
    grad_w_parts = torch.zeros(
        [(m + block_m - 1) // block_m, *w.shape], dtype=torch.float32, device=w.device
    )
    for tile_m in hl.tile(m, block_size=block_m):
        g = grad_out[tile_m, :]
        grad_x[tile_m, :] = g @ w[:, :].T
        grad_w_parts[tile_m.id, :, :] = x[tile_m, :].T @ g
    return grad_x, grad_w_parts


def _root_launch_threads(code: str) -> tuple[int, int]:
    """The launch's thread count on the root tile's axis and the root block size."""
    axis = re.search(
        r"_BLOCK_SIZE_0 \+ cutlass\.Int32\(cute\.arch\.thread_idx\(\)\[(\d)\]\)",
        code,
    )
    block_size = re.search(r"^_BLOCK_SIZE_0 = (\d+)$", code, re.MULTILINE)
    launch = re.search(r"block=\((\d+), (\d+), (\d+)\)", code)
    assert axis is not None and block_size is not None and launch is not None, code
    return int(launch.group(int(axis.group(1)) + 1)), int(block_size.group(1))


_plan_thread_axis = TileStrategyDispatch.thread_axis_for_strategy


def _shift_tile_axes(self: TileStrategyDispatch, target: TileStrategy) -> int | None:
    """A launch layout one axis too high for every tile."""
    axis = _plan_thread_axis(self, target)
    if axis is not None and not isinstance(target, ReductionStrategy):
        return axis + 1
    return axis


def _synthetic_lane_index_lines(code: str) -> list[str]:
    """Return the index assignments driven by a synthetic reduction lane loop."""
    return [
        line
        for line in code.splitlines()
        if "indices_" in line
        and "synthetic_lane_" in line
        and "=" in line
        and not line.lstrip().startswith("for ")
    ]


def _rolled_lane_index_lines(code: str) -> list[str]:
    """Return the index assignments driven by a rolled reduction's lane loop."""
    return [
        line
        for line in code.splitlines()
        if "reduction_lane_" in line
        and "=" in line
        and not line.lstrip().startswith("for ")
    ]


@helion.kernel(backend="cute", static_shapes=True)
def _attention_mla(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    """Causal attention whose value head dim differs from the q/k head dim."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    v_dim = hl.specialize(v_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, v_dim])
    out = torch.empty(
        [q_view.size(0), m_dim, v_dim], dtype=q_in.dtype, device=q_in.device
    )
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], -1e30, dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, v_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(n_dim):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            # The ``tile.index`` slices allocate reduction dims sized by the
            # tile symbols, each left with a single live thread.
            qk = torch.where(
                tile_m.index[None, :, None] >= tile_n.index[None, None, :],
                qk,
                float("-inf"),
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            p = torch.exp2(qk - m_ij_keepdim)
            m_ij = m_ij_keepdim.squeeze(-1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + torch.sum(p, -1)
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc * alpha[:, :, None], p.to(vt.dtype), vt)
            m_i = m_ij
        out[tile_b, tile_m, :] = (acc / l_i[:, :, None]).to(out.dtype)
    return out.view([q_in.size(0), q_in.size(1), m_dim, v_dim])


@helion.kernel(backend="cute", static_shapes=True)
def _attention_rope(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    qr_in: torch.Tensor,
    kr_in: torch.Tensor,
) -> torch.Tensor:
    """Causal attention whose scores add a second matmul over a shorter
    (RoPE) head dim."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    rope_dim = hl.specialize(qr_in.size(-1))
    v_dim = hl.specialize(v_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    qr_view = qr_in.reshape([-1, m_dim, rope_dim])
    kr_view = kr_in.reshape([-1, n_dim, rope_dim])
    v_view = v_in.reshape([-1, n_dim, v_dim])
    out = torch.empty(
        [q_view.size(0), m_dim, v_dim], dtype=q_in.dtype, device=q_in.device
    )
    qk_scale = (1.0 / math.sqrt(head_dim + rope_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], -1e30, dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, v_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        qr = qr_view[tile_b, tile_m, :]
        for tile_n in hl.tile(n_dim):
            kt = k_view[tile_b, tile_n, :]
            kr = kr_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = qk + torch.bmm(qr * qk_scale, kr.transpose(1, 2), torch.float32)
            # The ``tile.index`` slices allocate reduction dims sized by the
            # tile symbols, each left with a single live thread.
            qk = torch.where(
                tile_m.index[None, :, None] >= tile_n.index[None, None, :],
                qk,
                float("-inf"),
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            p = torch.exp2(qk - m_ij_keepdim)
            m_ij = m_ij_keepdim.squeeze(-1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + torch.sum(p, -1)
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc * alpha[:, :, None], p.to(vt.dtype), vt)
            m_i = m_ij
        out[tile_b, tile_m, :] = (acc / l_i[:, :, None]).to(out.dtype)
    return out.view([q_in.size(0), q_in.size(1), m_dim, v_dim])


def _attention_inputs(
    device: torch.device | str, *dims: int
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(0)
    return tuple(
        torch.randn(1, 2, 256, dim, generator=generator, dtype=torch.bfloat16).to(
            device
        )
        for dim in dims
    )


def _cpu_render(
    kernel: helion.Kernel, args: tuple[object, ...], **overrides: object
) -> str:
    """Render a kernel without a GPU, from the default config plus ``overrides``."""
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU test")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = _cpu_bind(kernel, args)
        settings = dict(bound.config_spec.default_config().config)
        settings.update(overrides)
        return bound.to_code(helion.Config.from_dict(settings))


@skipUnlessBackends(["cute"])
def test_single_thread_reductions_claim_no_thread_axis() -> None:
    """The q/k head dim (192, 256 threads) and the value head dim (128, 4
    threads once the first reduction took the budget) own axes 0 and 1; the
    two tile-sized dims of the causal mask are left with one live thread
    each and claim no axis, so the launch is ``(256, 4, 1)`` and no
    coordinate reads a fourth axis.  ``torch.bmm(..., torch.float32)`` folds
    its transposed operand like ``bmm.default`` (no shared-memory shuffle)."""
    q, k, v = _attention_inputs("cpu", 192, 192, 128)
    code = _cpu_render(_attention_mla, (q, k, v), block_sizes=[1, 128, 128])
    assert "block=(256, 4, 1)" in code
    assert "thread_idx()[3]" not in code
    assert "permute_smem" not in code


@skipUnlessBackends(["cute"])
def test_second_matmul_contraction_reaches_the_lane_scheduler() -> None:
    """A RoPE side matmul adds a fifth reduction dim (64); with the single
    thread dims out of the way the four axes are not exhausted any more and
    codegen reaches the lane-split scheduler, which declines the side
    matmul's serial K fold (a loop, not a straight-line assignment) inside the
    key lane loop.  Pins where the limitation now lies."""
    q, k, v, qr, kr = _attention_inputs("cpu", 128, 128, 128, 64, 64)
    with pytest.raises(helion.exc.BackendUnsupported, match="staged lane schedule"):
        _cpu_render(_attention_rope, (q, k, v, qr, kr), block_sizes=[1, 128, 128])


@onlyBackends(["cute"])
class TestCuteSingleThreadReduction(TestCase):
    def _check(self, rows: int, x_cols: int, y_cols: int) -> None:
        torch.manual_seed(0)
        x = torch.randn(rows, x_cols, device=DEVICE)
        y = torch.randn(rows, y_cols, device=DEVICE)
        # 32 tile rows x 32 rolled-reduction threads fill the 1024-thread CTA
        # budget, so the full-slice ``y`` dim is shrunk to one live thread that
        # shares the tile's thread axis.
        code, out = code_and_output(
            _concat2d_dim1_slices,
            (x, y),
            block_sizes=[32],
            reduction_loops=[32],
        )
        torch.testing.assert_close(out, torch.cat((x, y), dim=1))
        lane_lines = _synthetic_lane_index_lines(code)
        self.assertTrue(lane_lines, code)
        for line in lane_lines:
            self.assertNotIn("thread_idx", line, code)

    def test_slice_store_single_thread_reduction_matches_cat(self) -> None:
        self._check(256, 128, 256)

    def test_slice_store_single_thread_reduction_ragged(self) -> None:
        self._check(64, 33, 257)

    def test_rolled_single_thread_reduction_indexes_with_constant(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(2048, 192, device=DEVICE)
        # A 1024-thread tile exhausts the CTA budget, so the rolled column
        # reduction is shrunk to one live thread: its per-iteration index is
        # ``roffset + 0 + reduction_lane * 1``, never ``thread_idx()``.
        code, out = code_and_output(
            _row_sum, (x,), block_sizes=[1024], reduction_loops=[64]
        )
        torch.testing.assert_close(out, x.sum(-1), rtol=1e-4, atol=1e-4)
        lane_lines = _rolled_lane_index_lines(code)
        self.assertTrue(lane_lines, code)
        for line in lane_lines:
            self.assertNotIn("thread_idx", line, code)

    @skipIfNotCUDA()
    def test_attention_with_narrower_value_head_dim_matches_sdpa(self) -> None:
        """The generic SIMT path (the flash plan declines a value head dim
        that differs from the q/k head dim) with the two single-thread mask
        dims sharing no launch axis."""
        q, k, v = _attention_inputs(DEVICE, 192, 192, 128)
        code, out = code_and_output(
            _attention_mla, (q, k, v), block_sizes=[1, 128, 128]
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), is_causal=True
        )
        torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
        self.assertIn("block=(256, 4, 1)", code)
        self.assertNotIn("thread_idx()[3]", code)

    def test_full_slice_matmul_launches_every_row(self) -> None:
        # The 32-thread K reduction claims axis 0 and the N lane (one thread
        # once K x rows fill the budget) claims none, so the rows are on
        # axis 1 for the body and the launch alike.
        for m, k, n in ((64, 32, 16), (32, 32, 8), (37, 20, 10), (128, 32, 3)):
            torch.manual_seed(0)
            x = torch.randn(m, k, device=DEVICE)
            w = torch.randn(k, n, device=DEVICE)
            with self.subTest(m=m, k=k, n=n):
                code, out = code_and_output(_full_slice_matmul, (x, w))
                torch.testing.assert_close(out, x @ w, rtol=1e-4, atol=1e-4)
                launched, rows = _root_launch_threads(code)
                self.assertEqual(launched, rows, code)

    def test_full_slice_matmul_after_reductions_launches_every_row(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(64, 32, device=DEVICE)
        w = torch.randn(32, 16, device=DEVICE)
        code, out = code_and_output(_softmax_full_slice_matmul, (x, w))
        expected = torch.softmax(x, dim=-1) @ w + x.sum(-1, keepdim=True)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
        launched, rows = _root_launch_threads(code)
        self.assertEqual(launched, rows, code)

    def test_full_slice_matmul_grads_launch_every_row(self) -> None:
        # The shape of helion.experimental.backward's kernel for
        # ``out[tile_m, :] = x[tile_m, :] @ w[:, :]``: two contractions over
        # transposed full slices, one of them shrunk to a single thread.
        torch.manual_seed(0)
        x = torch.randn(64, 48, device=DEVICE)
        w = torch.randn(48, 32, device=DEVICE)
        grad_out = torch.randn(64, 32, device=DEVICE)
        code, (grad_x, grad_w_parts) = code_and_output(
            _full_slice_matmul_grads, (grad_out, x, w), block_sizes=[32]
        )
        torch.testing.assert_close(grad_x, grad_out @ w.T, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            grad_w_parts.sum(0), x.T @ grad_out, rtol=1e-4, atol=1e-4
        )
        launched, rows = _root_launch_threads(code)
        self.assertEqual(launched, rows, code)

    def test_inner_tile_loop_beside_single_thread_reduction(self) -> None:
        # The 32-thread x reduction and the one-thread y slice are both live
        # when the inner tile loop is placed: it goes right above the rows, as
        # in the launch layout, not one axis higher per live reduction.
        torch.manual_seed(0)
        x = torch.randn(64, 32, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        z = torch.randn(64, 16, device=DEVICE)
        for block_sizes in ([32, 4], [64, 2], [8, 4]):
            with self.subTest(block_sizes=block_sizes):
                code, (out, shifted) = code_and_output(
                    _scaled_rows_and_inner_tiles, (x, y, z), block_sizes=block_sizes
                )
                torch.testing.assert_close(
                    out, y * x.sum(-1, keepdim=True), rtol=1e-4, atol=1e-4
                )
                torch.testing.assert_close(shifted, z + 1)

    def test_root_without_reduction_beside_a_reducing_root(self) -> None:
        # The body places the second root's 256 threads above the first
        # root's reduction axis; the launch used to put them on axis 0.
        torch.manual_seed(0)
        x = torch.randn(64, 32, device=DEVICE)
        y = torch.randn(100, device=DEVICE)
        code, (sums, doubled) = code_and_output(
            _row_sums_then_doubled, (x, y), block_sizes=[1, 256]
        )
        torch.testing.assert_close(sums, x.sum(-1), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(doubled, y * 2)
        self.assertIn("block=(4, 256, 1)", code)

    def test_launch_layout_mismatch_is_rejected(self) -> None:
        # A launch layout that disagrees with the body is refused where the
        # body places the tile, and, past that check, by the launch guard
        # rather than launching too few rows.
        x = torch.randn(64, 32, device=DEVICE)
        w = torch.randn(32, 16, device=DEVICE)
        with patch.object(
            TileStrategyDispatch, "thread_axis_for_strategy", _shift_tile_axes
        ):
            with self.assertRaisesRegex(
                exc.BackendUnsupported, "thread-axis layout mismatch"
            ):
                code_and_output(_full_slice_matmul, (x, w), block_sizes=[32])
            with (
                patch.object(
                    BlockSizeTileStrategy,
                    "_check_launch_thread_axis",
                    lambda self, offset: offset,
                ),
                self.assertRaisesRegex(
                    exc.BackendUnsupported, "drops threads of a root tile"
                ),
            ):
                code_and_output(_full_slice_matmul, (x, w), block_sizes=[32])

    def test_barrier_launch_dropping_root_threads_is_rejected(self) -> None:
        # hl.barrier() kernels size their launch separately
        # (``_multi_phase_block_dims``); the root guard checks that shape too.
        x = torch.arange(64, device=DEVICE, dtype=torch.float32)
        _code, out = code_and_output(
            _doubled_then_incremented, (x,), block_sizes=[32, 32]
        )
        torch.testing.assert_close(out, x * 2 + 1)
        with (
            patch(
                "helion._compiler.cute.backend._multi_phase_block_dims",
                return_value=(1, 1, 1),
            ),
            self.assertRaisesRegex(
                exc.BackendUnsupported, "drops threads of a root tile"
            ),
        ):
            code_and_output(_doubled_then_incremented, (x,), block_sizes=[32, 32])

    def test_launch_drops_root_threads(self) -> None:
        self.assertEqual(launch_drops_root_threads((32, 1, 32), [{1: 32}]), (1, 32))
        self.assertIsNone(launch_drops_root_threads((32, 32, 1), [{1: 32}]))
        # A wider sibling root that never reads axis 1 records no extent for
        # it; the narrower reader's 8 threads are launched.
        self.assertIsNone(launch_drops_root_threads((8, 8, 1), [{1: 8}, {0: 8, 1: 4}]))
        self.assertEqual(
            launch_drops_root_threads((8, 2, 1), [{1: 8}, {0: 8, 1: 4}]), (1, 8)
        )
