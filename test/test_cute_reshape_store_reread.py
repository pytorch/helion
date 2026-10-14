"""Stores of reshapes whose elements sit on other CuTe threads.

A merging or splitting reshape that no proof keeps in place hands each thread
the element at its *source* position (``cute_reshape``,
``is_cute_thread_moving_reshape_value``).  The generic store re-reads the
element its slot names from the loads below the reshape
(``memory_ops._cute_reread_moved_reshape``); a store it cannot re-read, or one
whose slot binds a block twice, whose index math would read a runtime or
padded size, or whose loaded tensor another argument may alias is refused,
and so are the affine store fast paths that would write each thread's own
element.
"""

from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


@helion.kernel(autotune_effort="none", static_shapes=False)
def merge_store_dyn(x: torch.Tensor) -> torch.Tensor:
    m, a, b = x.shape
    out = torch.empty([m, a * b], device=x.device, dtype=x.dtype)
    for t in hl.tile(m):
        out[t, :] = x[t, :, :].reshape(t, a * b)
    return out


@helion.kernel(autotune_effort="none", static_shapes=False)
def merge_store_specialized_dyn(x: torch.Tensor) -> torch.Tensor:
    m, a, b = x.shape
    a = hl.specialize(a)
    b = hl.specialize(b)
    out = torch.empty([m, a * b], device=x.device, dtype=x.dtype)
    for t in hl.tile(m):
        out[t, :] = x[t, :, :].reshape(t, a * b)
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def merge_store_out(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, a, b = x.shape
    for t in hl.tile(m):
        out[t, :] = x[t, :, :].reshape(t, a * b)
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def merge_store(x: torch.Tensor) -> torch.Tensor:
    m, a, b = x.shape
    out = torch.empty([m, a * b], device=x.device, dtype=x.dtype)
    for t in hl.tile(m):
        out[t, :] = x[t, :, :].reshape(t, a * b)
    return out


@helion.kernel(autotune_effort="none", static_shapes=False)
def split4_store_dyn(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m, 4, n // 4], device=x.device, dtype=x.dtype)
    for t in hl.tile(m):
        out[t, :, :] = x[t, :].reshape(t, 4, n // 4)
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def affine_rows_from_load(x3: torch.Tensor) -> torch.Tensor:
    m, _, n = x3.shape
    out = torch.zeros(m * 2, n, dtype=x3.dtype, device=x3.device)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            v = x3[tile_m, :, tile_n].reshape(tile_m.block_size * 2, tile_n.block_size)
            out[tile_m.begin * 2 : tile_m.begin * 2 + tile_m.block_size * 2, tile_n] = v
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def affine_range_from_row(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.zeros(m * n, dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        v = x[tm, :].reshape(tm.block_size * n)
        out[tm.begin * n : tm.begin * n + tm.block_size * n] = v
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def affine_range_from_load(x2: torch.Tensor) -> torch.Tensor:
    m, _ = x2.shape
    out = torch.zeros(m * 2, dtype=x2.dtype, device=x2.device)
    for tile_m in hl.tile(m):
        v = x2[tile_m, :].reshape(tile_m.block_size * 2)
        out[tile_m.begin * 2 : tile_m.begin * 2 + tile_m.block_size * 2] = v
    return out


@onlyBackends(["cute"])
class TestCuteReshapeStoreReread(TestCase):
    def test_dynamic_merge_store_of_runtime_slices_is_refused(self) -> None:
        # The re-read lays the flat position out at compile-time sizes; a
        # kernel reused at another runtime slice size (12 in a bucket compiled
        # at 16) would split it up wrongly.
        config = helion.Config(
            block_sizes=[1],
            cute_lane_layouts=["blocked"] * 4,
            cute_vector_widths=[1, 8, 4, 1],
            num_threads=[1, 1, 2, 4],
            reduction_loops=[None],
        )
        x = torch.randn(48, 4, 16, device=DEVICE)
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported, "moves elements between threads"
        ):
            merge_store_dyn.bind((x,)).to_triton_code(config)

    def test_specialized_dynamic_merge_store_is_reused_exactly(self) -> None:
        config = helion.Config(
            block_sizes=[32],
            cute_lane_layouts=["blocked"] * 4,
            cute_vector_widths=[1, 8, 4, 1],
            num_threads=[32, 1, 2, 4],
            reduction_loops=[None],
        )
        x = torch.randn(48, 4, 16, device=DEVICE)
        bound = merge_store_specialized_dyn.bind((x,))
        code = bound.to_triton_code(config)
        compiled = bound.compile_config(config)
        self.assertEqual(bound.to_triton_code(config), code)
        torch.testing.assert_close(compiled(x), x.reshape(48, -1), rtol=0, atol=0)
        # Another (non power of two, partial last tile) row count reuses the
        # kernel and stays exact.
        y = torch.randn(80, 4, 16, device=DEVICE)
        self.assertIs(merge_store_specialized_dyn.bind((y,)), bound)
        torch.testing.assert_close(compiled(y), y.reshape(80, -1), rtol=0, atol=0)
        # Another slice size is a new specialization, never a silent reuse.
        z = torch.randn(48, 4, 8, device=DEVICE)
        self.assertIsNot(merge_store_specialized_dyn.bind((z,)), bound)
        torch.testing.assert_close(
            merge_store_specialized_dyn(z), z.reshape(48, -1), rtol=0, atol=0
        )

    def test_preallocated_out_is_reread_unless_it_aliases(self) -> None:
        # The bound kernel keys the storage overlap of its arguments, so a
        # disjoint ``out`` lets the store re-read ``x``; an ``out`` sharing
        # its storage would see other threads' writes and is refused.
        x = torch.randn(64, 4, 32, device=DEVICE)
        out = torch.empty(64, 128, device=DEVICE)
        code_and_output(merge_store_out, (x, out), block_sizes=[16])
        torch.testing.assert_close(out, x.reshape(64, -1), rtol=0, atol=0)
        storage = torch.randn(64 * 128, device=DEVICE)
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported, "moves elements between threads"
        ):
            code_and_output(
                merge_store_out,
                (storage.view(64, 4, 32), storage.view(64, 128)),
                block_sizes=[16],
            )

    def test_merge_store_under_constexpr_lane_loops(self) -> None:
        # The store runs in unrolled constexpr lane loops; each iteration
        # re-reads its element from the load, with no shared-memory exchange.
        x = torch.randn(64, 8, 32, device=DEVICE)
        for extra in ({"cute_vector_widths": [1, 4, 2, 2]}, {}):
            with self.subTest(**extra):
                _, result = code_and_output(
                    merge_store,
                    (x,),
                    block_sizes=[16],
                    num_threads=[16, 8, 0, 64],
                    **extra,
                )
                torch.testing.assert_close(result, x.reshape(64, -1), rtol=0, atol=0)

    def test_split_store_binding_one_block_twice_is_refused(self) -> None:
        # Both ``:`` dims of the slot resolve to one reduction block at n=16.
        x = torch.randn(64, 16, device=DEVICE)
        for block_size in (32, 64):
            with (
                self.subTest(block_size=block_size),
                self.assertRaisesRegex(
                    helion.exc.BackendUnsupported, "moves elements between threads"
                ),
            ):
                code_and_output(split4_store_dyn, (x,), block_sizes=[block_size])

    def test_affine_store_of_a_loaded_reshape_is_refused(self) -> None:
        # The affine fast paths would write each thread's own element to every
        # row lane; only a stack may select by the lane.
        cases = [
            (affine_rows_from_load, (torch.randn(16, 2, 32, device=DEVICE),), [4, 16]),
            (affine_range_from_load, (torch.randn(64, 2, device=DEVICE),), [16]),
        ]
        for fn, args, block_sizes in cases:
            with (
                self.subTest(fn=fn.fn.__name__),
                self.assertRaisesRegex(
                    helion.exc.BackendUnsupported, "moves elements between threads"
                ),
            ):
                code_and_output(fn, args, block_sizes=block_sizes)

    def test_affine_range_store_of_a_one_row_slice_is_refused(self) -> None:
        # At block size 1 the reshape only drops a unit dim, but the row's 16
        # elements still sit on 16 threads, not on the affine store's lanes.
        x = torch.randn(64, 16, device=DEVICE)
        with self.assertRaises(helion.exc.BackendUnsupported):
            code_and_output(affine_range_from_row, (x,), block_sizes=[1])
