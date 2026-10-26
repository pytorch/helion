"""Loop carries beside the lane loops CuTe builds around a device loop's body.

A device loop whose tile is wider than its threads walks the rest of the tile
in a per-thread lane loop, built around the loop body before the body exists.
A carry the body updates without reading the lane (a counter or a scalar sum
next to the scalar matmul fallback's per-K-lane products, or next to a
per-lane copy) must be updated once per tile, not once per lane:
``cnt = cnt + 1.0`` beside ``torch.addmm`` counts ``tiles_k``.  The body
updates the carry through a copy that the phi after the loop merges with it,
so the nest check reads the carries through both names.  An inexact nest is
placed like a grid body (each statement inside only the lane loops it depends
on) or rejected when a statement cannot leave a loop that repeats it.
"""

from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True)
def _matmul_with_counter(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty((a.size(0), b.size(1)), dtype=torch.float32, device=a.device)
    count = torch.empty_like(out)
    for tile_m, tile_n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        cnt = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
            cnt = cnt + 1.0
        out[tile_m, tile_n] = acc
        count[tile_m, tile_n] = cnt
    return out, count


@helion.kernel(backend="cute", static_shapes=True)
def _matmul_with_scalar_sum(
    a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, side: torch.Tensor
) -> torch.Tensor:
    out = torch.empty((a.size(0), b.size(1)), dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        ks = hl.zeros([], dtype=torch.float32)
        for tile_k in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
            ks = ks + w[tile_k.id]
        out[tile_m, tile_n] = acc
        side[tile_m.id, tile_n.id] = ks
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _matmul_with_conditional_count(
    a: torch.Tensor, b: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty((a.size(0), b.size(1)), dtype=torch.float32, device=a.device)
    count = torch.empty_like(out)
    for tile_m, tile_n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        cnt = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        ks = hl.zeros([], dtype=torch.float32)
        for tile_k in hl.tile(a.size(1)):
            ks = ks + w[tile_k.id]
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
            cnt = torch.where(w[tile_k.id] > 0, cnt * 0.5 + ks, cnt)
        out[tile_m, tile_n] = acc
        count[tile_m, tile_n] = cnt
    return out, count


@helion.kernel(backend="cute", static_shapes=True)
def _matmul_with_k_loop_store_and_atomic(
    a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, side: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty((a.size(0), b.size(1)), dtype=torch.float32, device=a.device)
    doubled = torch.zeros(a.shape, dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
            doubled[tile_m, tile_k] = a[tile_m, tile_k].to(torch.float32) * 2.0
            hl.atomic_add(side, [tile_m.id, tile_n.id], w[tile_k.id])
        out[tile_m, tile_n] = acc
    return out, doubled


@helion.kernel(backend="cute", static_shapes=True)
def _copy_with_counter(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.size()
    y = torch.empty_like(x)
    count = torch.empty(m, device=x.device)
    for tile_m in hl.tile(m):
        cnt = hl.zeros([tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            y[tile_m, tile_k] = x[tile_m, tile_k] * 2.0
            cnt = cnt + 1.0
        count[tile_m] = cnt
    return y, count


@helion.kernel(backend="cute", static_shapes=True)
def _matmul_with_inner_loop_counter(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty((a.size(0), b.size(1)), dtype=torch.float32, device=a.device)
    count = torch.empty_like(out)
    for tile_m, tile_n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        cnt = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
            for _tile_j in hl.tile(64, block_size=16):
                cnt = cnt + 1.0
        out[tile_m, tile_n] = acc
        count[tile_m, tile_n] = cnt
    return out, count


@helion.kernel(backend="cute", static_shapes=True)
def _per_lane_inner_loop_with_counter(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.size()
    y = torch.empty_like(x)
    count = torch.empty(m, device=x.device)
    for tile_m in hl.tile(m):
        cnt = hl.zeros([tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            for _tile_j in hl.tile(64, block_size=16):
                y[tile_m, tile_k] = x[tile_m, tile_k] * 2.0
                cnt = cnt + 1.0
        count[tile_m] = cnt
    return y, count


@helion.kernel(backend="cute", static_shapes=True)
def _row_sum_with_runtime_branch(x: torch.Tensor, beta: float) -> torch.Tensor:
    rows, cols = x.shape
    block_cols = hl.register_block_size(cols)
    block_rows = hl.register_block_size(rows)
    out = torch.zeros([rows], dtype=torch.float32, device=x.device)
    for tile_rows in hl.tile(rows, block_size=block_rows):
        acc = hl.zeros([tile_rows, block_cols], dtype=torch.float32)
        for tile_cols in hl.tile(cols, block_size=block_cols):
            v = x[tile_rows, tile_cols]
            if beta == 0.0:
                acc += v
            else:
                acc += v * 2.0
        out[tile_rows] = torch.sum(acc, dim=1)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _counter_in_runtime_branch(
    x: torch.Tensor, beta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.size()
    y = torch.empty_like(x)
    count = torch.empty(m, device=x.device)
    for tile_m in hl.tile(m):
        cnt = hl.zeros([tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            y[tile_m, tile_k] = x[tile_m, tile_k] * 2.0
            if beta == 0.0:
                cnt = cnt + 1.0
            else:
                cnt = cnt + 2.0
        count[tile_m] = cnt
    return y, count


def _kernel_function(code: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_"):
            return node
    raise AssertionError(code)


def _loop(function: ast.FunctionDef, prefix: str) -> ast.For:
    (loop,) = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith(prefix)
    ]
    return loop


def _matmul_inputs(
    m: int, n: int, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    a = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
    b = torch.randn(k, n, device=DEVICE, dtype=torch.float16)
    w = torch.randn(64, device=DEVICE)
    return a, b, w


_SHAPES = [(256, 256, 512), (200, 136, 96)]
# The default config (one K thread, a 16-lane serial K loop), a wider K tile,
# a row tile split across lanes too, and a collective config whose epilogue
# falls back to the same scalar path.
_MATMUL_CONFIGS = [
    {"block_sizes": [16, 16, 16]},
    {"block_sizes": [16, 16, 32]},
    {"block_sizes": [32, 16, 16], "num_threads": [8, 16, 0]},
    {
        "block_sizes": [64, 64, 32],
        "cute_collective_mma": True,
        "cute_collective_compute": "warp",
        "cute_collective_copy": "async_cached",
        "cute_collective_stages": 2,
    },
]


@onlyBackends(["cute"])
@skipIfRefEager("counts the configured tiles; ref mode runs one tile per loop")
class TestCuteDeviceLoopLaneCarries(TestCase):
    def test_counter_beside_scalar_matmul_counts_k_tiles(self) -> None:
        for m, n, k in _SHAPES:
            a, b, _ = _matmul_inputs(m, n, k)
            for config in _MATMUL_CONFIGS:
                with self.subTest(shape=(m, n, k), config=config):
                    code, (out, count) = code_and_output(
                        _matmul_with_counter, (a, b), **config
                    )
                    block_k = config["block_sizes"][2]
                    torch.testing.assert_close(
                        count, torch.full_like(count, -(-k // block_k))
                    )
                    torch.testing.assert_close(
                        out, a.float() @ b.float(), rtol=1e-2, atol=5e-2
                    )

    def test_counter_update_leaves_the_k_lane_loop(self) -> None:
        a, b, _ = _matmul_inputs(256, 256, 512)
        code, _ = code_and_output(
            _matmul_with_counter, (a, b), block_sizes=[16, 16, 16]
        )
        function = _kernel_function(code)
        k_loop = _loop(function, "tile_offset_2")
        lane_loop = _loop(k_loop, "lane_2")
        # The products stay per lane; the counter runs once per K tile.
        self.assertIn("dot_acc", ast.unparse(lane_loop))
        self.assertNotIn("cnt", ast.unparse(lane_loop))
        self.assertIn("cnt", ast.unparse(k_loop))

    def test_scalar_sum_beside_scalar_matmul_adds_each_tile_once(self) -> None:
        for m, n, k in _SHAPES:
            a, b, w = _matmul_inputs(m, n, k)
            for config in _MATMUL_CONFIGS:
                with self.subTest(shape=(m, n, k), config=config):
                    side = torch.zeros(64, 64, device=DEVICE)
                    _, out = code_and_output(
                        _matmul_with_scalar_sum, (a, b, w, side), **config
                    )
                    block_m, block_n, block_k = config["block_sizes"]
                    tiles_m, tiles_n = -(-m // block_m), -(-n // block_n)
                    expected = torch.zeros_like(side)
                    expected[:tiles_m, :tiles_n] = w[: -(-k // block_k)].sum()
                    torch.testing.assert_close(side, expected)
                    torch.testing.assert_close(
                        out, a.float() @ b.float(), rtol=1e-2, atol=5e-2
                    )

    def test_conditional_count_beside_scalar_matmul(self) -> None:
        m, n, k = 200, 136, 96
        a, b, w = _matmul_inputs(m, n, k)
        for config in _MATMUL_CONFIGS[:3]:
            with self.subTest(config=config):
                _, (out, count) = code_and_output(
                    _matmul_with_conditional_count, (a, b, w), **config
                )
                cnt = ks = 0.0
                for weight in w[: -(-k // config["block_sizes"][2])].tolist():
                    ks += weight
                    if weight > 0:
                        cnt = cnt * 0.5 + ks
                torch.testing.assert_close(count, torch.full_like(count, cnt))
                torch.testing.assert_close(
                    out, a.float() @ b.float(), rtol=1e-2, atol=5e-2
                )

    def test_k_loop_store_and_uniform_atomic_beside_scalar_matmul(self) -> None:
        m, n, k = 200, 136, 96
        a, b, w = _matmul_inputs(m, n, k)
        for config in _MATMUL_CONFIGS[:2]:
            with self.subTest(config=config):
                side = torch.zeros(64, 64, device=DEVICE)
                _, (out, doubled) = code_and_output(
                    _matmul_with_k_loop_store_and_atomic, (a, b, w, side), **config
                )
                block_m, block_n, block_k = config["block_sizes"]
                expected = torch.zeros_like(side)
                expected[: -(-m // block_m), : -(-n // block_n)] = w[
                    : -(-k // block_k)
                ].sum()
                torch.testing.assert_close(side, expected)
                torch.testing.assert_close(doubled, a.float() * 2.0)
                torch.testing.assert_close(
                    out, a.float() @ b.float(), rtol=1e-2, atol=5e-2
                )

    def test_counter_beside_per_lane_copy_counts_k_tiles(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(64, 256, device=DEVICE)
        for num_threads in ([16, 8], [16, 4]):
            with self.subTest(num_threads=num_threads):
                _, (y, count) = code_and_output(
                    _copy_with_counter,
                    (x,),
                    block_sizes=[16, 32],
                    num_threads=num_threads,
                )
                torch.testing.assert_close(y, x * 2.0)
                torch.testing.assert_close(count, torch.full_like(count, 256 // 32))

    def test_lane_invariant_inner_loop_counter_leaves_the_k_lane_loop(self) -> None:
        a, b, _ = _matmul_inputs(64, 64, 128)
        _, (out, count) = code_and_output(
            _matmul_with_inner_loop_counter, (a, b), block_sizes=[16, 16, 16]
        )
        torch.testing.assert_close(count, torch.full_like(count, (128 // 16) * 4))
        torch.testing.assert_close(out, a.float() @ b.float(), rtol=1e-2, atol=5e-2)

    def test_counter_in_a_per_lane_inner_loop_is_rejected(self) -> None:
        # The inner loop stores per lane, so it runs whole once per lane and
        # its counter update cannot leave the lane loop.
        x = torch.randn(64, 256, device=DEVICE)
        with pytest.raises(exc.BackendUnsupported, match="loop-carried"):
            code_and_output(
                _per_lane_inner_loop_with_counter,
                (x,),
                block_sizes=[16, 32],
                num_threads=[16, 8],
            )

    def test_carry_assigned_in_both_branches_of_a_runtime_if(self) -> None:
        # The if-join assigns the carry a default before the ``if``; both
        # branches reassign it before anything reads it, so the default is
        # no per-lane update of the carry (jsd_forward's beta branches).
        x = torch.randint(-4, 5, (64, 512), device=DEVICE).float()
        for block_sizes, num_threads in (
            ([16, 1], [2, 1]),
            ([32, 2], [4, 2]),
            ([64, 1], [32, 1]),
        ):
            with self.subTest(block_sizes=block_sizes, num_threads=num_threads):
                _, out = code_and_output(
                    _row_sum_with_runtime_branch,
                    (x, 0.5),
                    block_sizes=block_sizes,
                    num_threads=num_threads,
                )
                torch.testing.assert_close(out, (x * 2.0).sum(1), rtol=0, atol=0)

    def test_counter_updated_in_both_branches_of_a_runtime_if(self) -> None:
        # Each branch updates the counter from its own value: that is a
        # lane-invariant update, placed once per K tile.
        torch.manual_seed(0)
        x = torch.randn(64, 256, device=DEVICE)
        for num_threads in ([16, 8], [16, 4]):
            with self.subTest(num_threads=num_threads):
                _, (y, count) = code_and_output(
                    _counter_in_runtime_branch,
                    (x, 0.5),
                    block_sizes=[16, 32],
                    num_threads=num_threads,
                )
                torch.testing.assert_close(y, x * 2.0)
                torch.testing.assert_close(
                    count, torch.full_like(count, 2.0 * (256 // 32))
                )
