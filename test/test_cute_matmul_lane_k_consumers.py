"""Scalar CuTe matmul fallback with a K axis split across a serial lane loop.

The fallback sums the per-K-lane products in a per-thread ``dot_acc`` running
sum that is zeroed outside the K loop.  That is only the contraction when the
K lane loop is the innermost open scope and the consumer sees the final value
(a loop-carried accumulator or a last-write-wins store).  An ``hl.atomic_add``
consumer would add every prefix of the sum, and a free-axis loop nested inside
the K lane loop (attention backward's ``dq`` with the key tile as the grid
block) would be folded into the sum as well.  A K axis that is fully mapped to
threads has no running sum, so its consumers are unrestricted.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True)
def _outer_lane_k_atomic(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # out[m, d] = sum_n s[n, m] * k[n, d] with s = k @ q.T, contracted over the
    # key tile that owns the grid (its lane loop wraps the query tile loop).
    m_dim, head_dim = q.shape
    n_dim = k.size(0)
    head_dim = hl.specialize(k.size(1))
    out = torch.zeros((m_dim, head_dim), device=q.device, dtype=torch.float32)
    block_m = hl.register_block_size(m_dim)
    block_n = hl.register_block_size(n_dim)
    for tile_n in hl.tile(n_dim, block_size=block_n):
        k_j = k[tile_n, :]
        for tile_m in hl.tile(m_dim, block_size=block_m):
            q_i = q[tile_m, :]
            s_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
            r = hl.dot(s_t.to(q.dtype).T, k_j, out_dtype=torch.float32)
            hl.atomic_add(out, [tile_m, slice(None)], r * 0.5)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _outer_lane_k_store(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # Same contraction, consumed by a plain store (one key tile covers n_dim).
    m_dim, head_dim = q.shape
    n_dim = k.size(0)
    head_dim = hl.specialize(k.size(1))
    out = torch.zeros((m_dim, head_dim), device=q.device, dtype=torch.float32)
    block_m = hl.register_block_size(m_dim)
    block_n = hl.register_block_size(n_dim)
    for tile_n in hl.tile(n_dim, block_size=block_n):
        k_j = k[tile_n, :]
        for tile_m in hl.tile(m_dim, block_size=block_m):
            q_i = q[tile_m, :]
            s_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
            out[tile_m, :] = hl.dot(s_t.to(q.dtype).T, k_j, out_dtype=torch.float32)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _inner_lane_k_atomic(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # The key tile is the innermost loop, so its lane loop wraps the atomic.
    m_dim, head_dim = q.shape
    n_dim = k.size(0)
    head_dim = hl.specialize(k.size(1))
    out = torch.zeros((m_dim, head_dim), device=q.device, dtype=torch.float32)
    block_m = hl.register_block_size(m_dim)
    block_n = hl.register_block_size(n_dim)
    for tile_m in hl.tile(m_dim, block_size=block_m):
        q_i = q[tile_m, :]
        for tile_n in hl.tile(n_dim, block_size=block_n):
            k_j = k[tile_n, :]
            s_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
            r = hl.dot(s_t.to(q.dtype).T, k_j, out_dtype=torch.float32)
            hl.atomic_add(out, [tile_m, slice(None)], r)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _inner_lane_k_store(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    m_dim, head_dim = q.shape
    n_dim = k.size(0)
    head_dim = hl.specialize(k.size(1))
    out = torch.zeros((m_dim, head_dim), device=q.device, dtype=torch.float32)
    block_m = hl.register_block_size(m_dim)
    block_n = hl.register_block_size(n_dim)
    for tile_m in hl.tile(m_dim, block_size=block_m):
        q_i = q[tile_m, :]
        for tile_n in hl.tile(n_dim, block_size=block_n):
            k_j = k[tile_n, :]
            s_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
            out[tile_m, :] = hl.dot(s_t.to(q.dtype).T, k_j, out_dtype=torch.float32)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _inner_lane_k_atomic_relu(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # A non-linear op between the matmul and the atomic: per-lane partials
    # cannot be split, and a running sum would add every prefix.
    m_dim, head_dim = q.shape
    n_dim = k.size(0)
    head_dim = hl.specialize(k.size(1))
    out = torch.zeros((m_dim, head_dim), device=q.device, dtype=torch.float32)
    block_m = hl.register_block_size(m_dim)
    block_n = hl.register_block_size(n_dim)
    for tile_m in hl.tile(m_dim, block_size=block_m):
        q_i = q[tile_m, :]
        for tile_n in hl.tile(n_dim, block_size=block_n):
            k_j = k[tile_n, :]
            s_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
            r = hl.dot(s_t.to(q.dtype).T, k_j, out_dtype=torch.float32)
            hl.atomic_add(out, [tile_m, slice(None)], torch.relu(r))
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _inner_lane_k_atomic_max(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # ``max`` does not distribute over the K lanes' partial sums.
    m_dim, head_dim = q.shape
    n_dim = k.size(0)
    head_dim = hl.specialize(k.size(1))
    out = torch.zeros((m_dim, head_dim), device=q.device, dtype=torch.float32)
    block_m = hl.register_block_size(m_dim)
    block_n = hl.register_block_size(n_dim)
    for tile_m in hl.tile(m_dim, block_size=block_m):
        q_i = q[tile_m, :]
        for tile_n in hl.tile(n_dim, block_size=block_n):
            k_j = k[tile_n, :]
            s_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
            r = hl.dot(s_t.to(q.dtype).T, k_j, out_dtype=torch.float32)
            hl.atomic_max(out, [tile_m, slice(None)], r)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _inner_lane_k_atomic_value_kwarg(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # The atomic value passed by keyword is the same linear chain.
    m_dim, head_dim = q.shape
    n_dim = k.size(0)
    head_dim = hl.specialize(k.size(1))
    out = torch.zeros((m_dim, head_dim), device=q.device, dtype=torch.float32)
    block_m = hl.register_block_size(m_dim)
    block_n = hl.register_block_size(n_dim)
    for tile_m in hl.tile(m_dim, block_size=block_m):
        q_i = q[tile_m, :]
        for tile_n in hl.tile(n_dim, block_size=block_n):
            k_j = k[tile_n, :]
            s_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
            r = hl.dot(s_t.to(q.dtype).T, k_j, out_dtype=torch.float32)
            hl.atomic_add(out, [tile_m, slice(None)], value=r * 2.0)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _inner_lane_k_atomic_half(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # A half-precision matmul result: splitting the atomic per lane would
    # round every lane's partial before adding it.
    m_dim, head_dim = q.shape
    n_dim = k.size(0)
    head_dim = hl.specialize(k.size(1))
    out = torch.zeros((m_dim, head_dim), device=q.device, dtype=torch.float32)
    block_m = hl.register_block_size(m_dim)
    block_n = hl.register_block_size(n_dim)
    for tile_m in hl.tile(m_dim, block_size=block_m):
        q_i = q[tile_m, :]
        for tile_n in hl.tile(n_dim, block_size=block_n):
            k_j = k[tile_n, :]
            s_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
            r = hl.dot(s_t.to(q.dtype).T, k_j, out_dtype=torch.float16)
            hl.atomic_add(out, [tile_m, slice(None)], r)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _inner_lane_k_dot_feeds_dot_atomic(
    q: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    # The lane-split ``r`` feeds a second dot (over head_dim) inside the same
    # lane loop, so that dot would consume every prefix of ``r``.
    m_dim = q.size(0)
    n_dim = k.size(0)
    out = torch.zeros((m_dim, n_dim), device=q.device, dtype=torch.float32)
    block_m = hl.register_block_size(m_dim)
    block_n = hl.register_block_size(n_dim)
    for tile_m in hl.tile(m_dim, block_size=block_m):
        q_i = q[tile_m, :]
        for tile_n in hl.tile(n_dim, block_size=block_n):
            k_j = k[tile_n, :]
            s_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
            r = hl.dot(s_t.to(q.dtype).T, k_j, out_dtype=torch.float32)
            r2 = hl.dot(r.to(q.dtype), k_j.T, out_dtype=torch.float32)
            hl.atomic_add(out, [tile_m, tile_n], r2)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _inner_lane_k_atomic_in_nested_loop(
    q: torch.Tensor, k: torch.Tensor, reps: torch.Tensor
) -> torch.Tensor:
    # The atomic sits in a loop nested inside the K lane loop: its subgraph
    # takes the lane-split ``r`` as an argument.
    m_dim, head_dim = q.shape
    n_dim = k.size(0)
    head_dim = hl.specialize(k.size(1))
    j_dim = reps.size(0)
    out = torch.zeros((m_dim, head_dim), device=q.device, dtype=torch.float32)
    block_m = hl.register_block_size(m_dim)
    block_n = hl.register_block_size(n_dim)
    for tile_m in hl.tile(m_dim, block_size=block_m):
        q_i = q[tile_m, :]
        for tile_n in hl.tile(n_dim, block_size=block_n):
            k_j = k[tile_n, :]
            s_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
            r = hl.dot(s_t.to(q.dtype).T, k_j, out_dtype=torch.float32)
            for _tile_j in hl.tile(j_dim, block_size=1):
                hl.atomic_add(out, [tile_m, slice(None)], r)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _inner_lane_k_addmm_atomic(
    q: torch.Tensor, k: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    # ``addmm`` carries an accumulator: the lane loop would add every prefix
    # of ``bias + s @ k`` instead of the single sum.
    m_dim, head_dim = q.shape
    n_dim = k.size(0)
    head_dim = hl.specialize(k.size(1))
    out = torch.zeros((m_dim, head_dim), device=q.device, dtype=torch.float32)
    block_m = hl.register_block_size(m_dim)
    block_n = hl.register_block_size(n_dim)
    for tile_m in hl.tile(m_dim, block_size=block_m):
        q_i = q[tile_m, :]
        bias_i = bias[tile_m, :]
        for tile_n in hl.tile(n_dim, block_size=block_n):
            k_j = k[tile_n, :]
            s_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
            r = torch.addmm(bias_i, s_t.to(q.dtype).T, k_j)
            hl.atomic_add(out, [tile_m, slice(None)], r)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _lane_k_online_softmax_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    # examples/attention.py without the batch dim.  head_dim takes the first
    # thread axis, so each score is a warp reduction; the key tile is threads
    # x serial lanes, and the running max, the running sum and the ``p @ v``
    # contribution each reduce across its lane loop.
    m_dim = q.size(0)
    n_dim = k.size(0)
    head_dim = hl.specialize(q.size(1))
    out = torch.empty_like(q)
    for tile_m in hl.tile(m_dim):
        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        l_i = hl.zeros([tile_m], dtype=torch.float32)
        acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
        q_i = q[tile_m, :]
        for tile_n in hl.tile(n_dim):
            qk = torch.mm(q_i, k[tile_n, :].T)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            p = torch.exp(qk - m_ij[:, None])
            alpha = torch.exp(m_i - m_ij)
            l_i = l_i * alpha + torch.sum(p, -1)
            acc = torch.addmm(acc * alpha[:, None], p, v[tile_n, :])
            m_i = m_ij
        out[tile_m, :] = acc / l_i[:, None]
    return out


def _lane_loop_count(code: str) -> int:
    return sum(
        1
        for line in code.splitlines()
        if line.lstrip().startswith("for lane_") and " in range(" in line
    )


@onlyBackends(["cute"])
class TestCuteMatmulLaneKConsumers(TestCase):
    def _inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        q = torch.randn(128, 64, device=DEVICE, dtype=torch.float16) * 0.1
        k = torch.randn(32, 64, device=DEVICE, dtype=torch.float16) * 0.1
        s = (k.float() @ q.float().T).to(torch.float16).float()
        return q, k, s.T @ k.float()

    def test_outer_lane_k_atomic_add_contributes_per_lane_partials(self) -> None:
        q, k, expected = self._inputs()
        # head_dim=64 takes the first thread axis, so the 32-wide key and query
        # tiles each become 4 threads x 8 serial lanes.  The key lane loop wraps
        # the query tile loop, which carries its own lane loop.
        code, out = code_and_output(_outer_lane_k_atomic, (q, k), block_sizes=[32, 32])
        self.assertEqual(_lane_loop_count(code), 2)
        self.assertNotIn("dot_acc", code)
        self.assertIn("atomic_add(", code)
        torch.testing.assert_close(out, expected * 0.5, rtol=1e-2, atol=1e-2)

    def test_outer_lane_k_store_consumer_is_rejected(self) -> None:
        q, k, _ = self._inputs()
        with pytest.raises(exc.BackendUnsupported, match="lane owner"):
            code_and_output(_outer_lane_k_store, (q, k), block_sizes=[32, 32])

    def test_inner_lane_k_atomic_add_contributes_per_lane_partials(self) -> None:
        q, k, expected = self._inputs()
        code, out = code_and_output(_inner_lane_k_atomic, (q, k), block_sizes=[32, 32])
        self.assertEqual(_lane_loop_count(code), 2)
        self.assertNotIn("dot_acc", code)
        # The atomic varies along the key lane loop (each lane adds its own
        # partial), so the lane placement must not pin it to the first lane
        # as it would an atomic that is uniform along an uncovered axis.
        self.assertNotIn("lane_1 == 0", code)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    def test_inner_lane_k_store_keeps_running_sum(self) -> None:
        q, k, expected = self._inputs()
        code, out = code_and_output(_inner_lane_k_store, (q, k), block_sizes=[32, 32])
        self.assertEqual(_lane_loop_count(code), 2)
        self.assertIn("dot_acc", code)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    def test_non_linear_atomic_consumer_is_rejected(self) -> None:
        q, k, _ = self._inputs()
        with pytest.raises(exc.BackendUnsupported, match="cannot split per lane"):
            code_and_output(_inner_lane_k_atomic_relu, (q, k), block_sizes=[32, 32])

    def test_atomic_max_consumer_is_rejected(self) -> None:
        q, k, _ = self._inputs()
        with pytest.raises(exc.BackendUnsupported, match="cannot split per lane"):
            code_and_output(_inner_lane_k_atomic_max, (q, k), block_sizes=[32, 32])

    def test_atomic_value_keyword_contributes_per_lane_partials(self) -> None:
        q, k, expected = self._inputs()
        code, out = code_and_output(
            _inner_lane_k_atomic_value_kwarg, (q, k), block_sizes=[32, 32]
        )
        self.assertNotIn("dot_acc", code)
        self.assertNotIn("lane_1 == 0", code)
        torch.testing.assert_close(out, expected * 2.0, rtol=1e-2, atol=1e-2)

    def test_thread_mapped_k_non_linear_atomic_consumer_compiles(self) -> None:
        # head_dim=16 leaves the 32-wide key tile to 32 threads, so K is not
        # split across lanes: the grouped reduction is complete at every
        # thread and ``relu`` before the atomic is fine.
        torch.manual_seed(0)
        q = torch.randn(128, 16, device=DEVICE, dtype=torch.float16) * 0.1
        k = torch.randn(32, 16, device=DEVICE, dtype=torch.float16) * 0.1
        s = (k.float() @ q.float().T).to(torch.float16).float()
        expected = torch.relu(s.T @ k.float())
        code, out = code_and_output(
            _inner_lane_k_atomic_relu,
            (q, k),
            block_sizes=[32, 32],
            num_threads=[1, 32],
        )
        self.assertEqual(_lane_loop_count(code), 1)
        self.assertNotIn("dot_acc", code)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)

    def test_half_precision_output_atomic_add_takes_owned_route(self) -> None:
        # A half-precision result is not split per lane.  The owned
        # product-sum marker completes the K sum first or its lane scheduler
        # declines; either way no per-lane rounding reaches the atomic.
        q, k, expected = self._inputs()
        try:
            code, out = code_and_output(
                _inner_lane_k_atomic_half, (q, k), block_sizes=[32, 32]
            )
        except exc.BackendUnsupported as e:
            self.assertIn("staged lane schedule", str(e))
            return
        self.assertNotIn("dot_acc", code)
        self.assertIn("dot_sum", code)
        torch.testing.assert_close(
            out, expected.to(torch.float16).float(), rtol=1e-2, atol=1e-2
        )

    def test_lane_k_dot_feeding_second_dot_atomic_is_rejected(self) -> None:
        q, k, _ = self._inputs()
        with pytest.raises(exc.BackendUnsupported, match="cannot split per lane"):
            code_and_output(
                _inner_lane_k_dot_feeds_dot_atomic, (q, k), block_sizes=[32, 32]
            )

    def test_lane_k_atomic_in_nested_loop_is_rejected(self) -> None:
        q, k, _ = self._inputs()
        reps = torch.zeros(2, device=DEVICE)
        with pytest.raises(exc.BackendUnsupported, match="cannot split per lane"):
            code_and_output(
                _inner_lane_k_atomic_in_nested_loop, (q, k, reps), block_sizes=[32, 32]
            )

    def test_lane_k_addmm_atomic_consumer_is_rejected(self) -> None:
        q, k, _ = self._inputs()
        bias = torch.randn(128, 64, device=DEVICE, dtype=torch.float16) * 0.1
        with pytest.raises(exc.BackendUnsupported, match="cannot split per lane"):
            code_and_output(
                _inner_lane_k_addmm_atomic, (q, k, bias), block_sizes=[32, 32]
            )

    def test_warp_reduced_scores_feed_lane_k_softmax_and_product(self) -> None:
        # The score's warp reduction runs once per key lane into a register
        # stash; the max, sum and ``p @ v`` passes read the stash.
        torch.manual_seed(0)
        q, k, v = (
            torch.randn(64, 32, device=DEVICE, dtype=torch.float32) for _ in range(3)
        )
        code, out = code_and_output(
            _lane_k_online_softmax_attention, (q, k, v), block_sizes=[16, 16]
        )
        self.assertIn("warp_reduction_sum", code)
        self.assertIn("_lane_stash", code)
        expected = torch.softmax(q @ k.T, -1) @ v
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
