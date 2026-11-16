"""Metal kernels with several roots, without a GPU: the store guards that keep
the surplus threads of a root laid out in fewer threads than the launch from
writing (``metal/multi_root.py``).  Binds CPU tensors only;
``test_metal.py`` runs the same kernels (``TestMetalMultiRoot``)."""

from __future__ import annotations

import ast
import textwrap
import unittest

import pytest
import torch

import helion
from helion import exc
from helion._compiler.metal.lane_fold import is_store
from helion._compiler.metal.lane_fold import pointer_base
from helion._compiler.metal.multi_root import guard_surplus_stores
from helion._testing import skipUnlessBackends
from helion._testing.metal import render_msl
from helion._testing.metal_multi_root import grid_beside_row
from helion._testing.metal_multi_root import matmul_then_scale
from helion._testing.metal_multi_root import scale_then_accumulate_then_tile
from helion._testing.metal_multi_root import scale_then_add_masked_rowsum
from helion._testing.metal_multi_root import scale_then_add_rowsum
from helion._testing.metal_multi_root import scale_then_copy_then_tile
from helion._testing.metal_multi_root import scale_then_round_trip_then_tile
from helion._testing.metal_multi_root import scale_then_toggle_then_tile
from helion._testing.metal_multi_root import scale_then_update_then_tile
from helion._testing.metal_multi_root import scatter_beside_scale
from helion._testing.metal_multi_root import update_beside_rowsum
from helion._testing.metal_multi_root import update_then_scale
from helion._testing.metal_multi_root import write_then_read_back

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]


class TestMultiRootCodegen(unittest.TestCase):
    def test_surplus_threads_dont_write(self) -> None:
        """The first root, laid out in one thread, runs on all 64 the second
        needs: only its own thread stores, or the in-place update of ``x``
        could land once per SIMD group.  The second root takes every
        thread, so its store needs no guard.  A persistent launch, whose
        loop holds the root dispatch, guards the same store."""
        x, y = torch.randn(1 << 12), torch.randn(1 << 12)
        for pid_type in ("flat", "persistent_interleaved"):
            with self.subTest(pid_type=pid_type):
                msl = render_msl(
                    update_then_scale,
                    (x, y),
                    helion.Config(block_sizes=[1, 64], pid_type=pid_type),
                )
                self.assertRegex(msl, r"if \(\(tid\[0\] < 1\)\) \{\s*x\[\w+\] = ")
                self.assertRegex(msl, r"\n *out\[\w+\] = ")
                self.assertNotIn("tid[0] < 64", msl)

    def test_reduction_beside(self) -> None:
        """A row sum over 1024 threads beside an update one thread walks in
        lanes.  Every root runs the sum's setup, which records the update as
        1024 threads wide and leaves a read of ``tid[0]`` no store depends
        on in its body.  The update's stores depend on no thread, so they
        take ``tid[0] < 1``."""
        y, x = torch.randn(4096), torch.randn(64, 1024)
        config = helion.Config(block_sizes=[16, 1], num_warps=32)
        msl = render_msl(update_beside_rowsum, (y, x), config)
        self.assertIn("required_threads_per_threadgroup(1024, 1, 1)", msl)
        self.assertRegex(msl, r"if \(\(tid\[0\] < 1\)\) \{\s*y\[\w+\] = ")
        self.assertRegex(msl, r"\n *s\[\w+\] = ")

    def test_equal_layouts_unguarded(self) -> None:
        """Roots laid out in the same threads leave none surplus."""
        x, y = torch.randn(1 << 12), torch.randn(1 << 12)
        msl = render_msl(update_then_scale, (x, y), helion.Config(block_sizes=[64, 64]))
        self.assertNotRegex(msl, r"tid\[\d\] < \d")

    def test_narrow_matmul_root_kept(self) -> None:
        """An MPP matmul spreads its result over the launch's SIMD groups and
        writes it through its own markers, which no store guard reaches: its
        root, laid out in 8 x 16 threads of a 64 x 16 launch, keeps its
        stores, as before.  The elementwise root beside it lays out no
        threads along y, so its store takes ``tid[1] < 1``."""
        x, y = torch.randn(64, 64, dtype=torch.float16), torch.randn(1 << 12)
        config = helion.Config(block_sizes=[32, 32, 32, 64])
        code = matmul_then_scale.bind((x, x.clone(), y)).to_code(config)
        self.assertIn("_block_dims=(64, 16, 1)", code)
        self.assertIn("_helion_mpp_run(", code)
        self.assertRegex(code, r"tl\.store\(out \+ [^\n]*, acc, mask_\d+ & mask_\d+\)")
        self.assertRegex(code, r"tl\.store\(ys \+ [^\n]*, tid\[1\] < 1 and mask_\d+\)")

    def test_phased_repeated_update_refused(self) -> None:
        """A barrier kernel's second phase updates an element of ``c`` in
        place, then lays ``y``'s tiles out on threads: 64 of them, or 2 that
        walk 32 lanes each, would repeat the update, so codegen refuses it.
        With the tiles walked in lanes by one thread, both stores take
        ``tid[0] < 1``."""
        x, c, y = torch.randn(1 << 12), torch.randn(64), torch.randn(1 << 12)
        args = (x, c, y)
        for num_threads in ([0, 0, 64], [0, 0, 0]):
            with self.subTest(num_threads=num_threads):
                threads = helion.Config(
                    block_sizes=[64, 1, 64], num_threads=num_threads, pid_type="flat"
                )
                with self.assertRaisesRegex(exc.BackendUnsupported, "in-place update"):
                    scale_then_update_then_tile.bind(args).to_code(threads)
        lanes = helion.Config(
            block_sizes=[64, 1, 64], num_threads=[0, 0, 1], pid_type="flat"
        )
        msl = render_msl(scale_then_update_then_tile, args, lanes)
        self.assertRegex(msl, r"if \(\(tid\[0\] < 1\)\) \{\s*c\[\w+\] = ")
        self.assertRegex(msl, r"if \(\(tid\[0\] < 1\)\) \{\s*out\[\w+\] = ")

    def test_phased_update_of_reduction_refused(self) -> None:
        """A barrier kernel's second phase adds a row sum into an element of
        ``s``, unmasked or masked by the sum.  The store's pointer depends on
        no thread, though its value and mask read the sum, which all 1024
        threads of the reduction get: each would apply the update, so
        codegen refuses it."""
        y, x, s = torch.randn(1 << 12), torch.randn(64, 1024), torch.randn(64)
        config = helion.Config(block_sizes=[64, 1], num_warps=32, pid_type="flat")
        for kernel in (scale_then_add_rowsum, scale_then_add_masked_rowsum):
            with (
                self.subTest(kernel.fn.__name__),
                self.assertRaisesRegex(exc.BackendUnsupported, "in-place update"),
            ):
                kernel.bind((y, x, s)).to_code(config)

    def test_phased_update_by_test_scratch_or_loop_refused(self) -> None:
        """Barrier kernels whose second phase rewrites an element of ``c`` on
        all 64 threads of ``y``'s tiles: a sign flip by a test of the element
        (its stores write constants), an update through a scratch tensor
        written and read back on the way, and one carried through a loop.
        Stores the threads repeat load what such stores write, so codegen
        refuses all three."""
        x, y = torch.randn(1 << 12), torch.randn(1 << 12)
        c, tmp = torch.randn(64), torch.randn(64)
        cases = (
            (
                scale_then_toggle_then_tile,
                (x, c, y),
                helion.Config(
                    block_sizes=[64, 64], num_threads=[0, 64], pid_type="flat"
                ),
            ),
            (
                scale_then_round_trip_then_tile,
                (x, c, tmp, y),
                helion.Config(
                    block_sizes=[64, 1, 64], num_threads=[0, 0, 64], pid_type="flat"
                ),
            ),
            (
                scale_then_accumulate_then_tile,
                (x, c, y),
                helion.Config(
                    block_sizes=[64, 1, 4, 64],
                    num_threads=[0, 0, 1, 64],
                    pid_type="flat",
                ),
            ),
        )
        for kernel, args, config in cases:
            with (
                self.subTest(kernel.fn.__name__),
                self.assertRaisesRegex(exc.BackendUnsupported, "in-place update"),
            ):
                kernel.bind(args).to_code(config)

    def test_aliased_update_refused_at_call(self) -> None:
        """A barrier kernel's second phase copies an element of ``d`` into
        ``c`` on the 64 threads of ``y``'s tiles, unguarded: separate tensors
        make every thread write the same value.  Called with ``c`` and ``d``
        one tensor, the copy is an in-place update each of those threads
        would apply, which shows only at the call: metal_jit refuses it."""
        x, y = torch.randn(1 << 12), torch.randn(1 << 12)
        c, d = torch.randn(64), torch.randn(64)
        config = helion.Config(
            block_sizes=[64, 1, 64], num_threads=[0, 0, 64], pid_type="flat"
        )
        msl = render_msl(scale_then_copy_then_tile, (x, c, d, y), config)
        self.assertRegex(msl, r"\n *c\[\w+\] = ")
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "c shares its storage with d"
        ):
            render_msl(
                scale_then_copy_then_tile, (x, c, d, y), config, run_args=(x, c, c, y)
            )

    def test_repeated_updates(self) -> None:
        """An in-place update whose pointer depends on no thread is guarded
        when no store of its root reads a thread, and reported when the
        threads of a tiled store or of a matmul would repeat it, or those of
        a reduction whose result its value, mask, test or pointer reads; so
        are a flip by a test of the element and an update through a scratch
        tensor.  Repeated stores that load other tensors are recorded with
        them, for the check at the call."""
        update = """
            load = tl.load(c + 0, None, other=0)
            v = load * 2.0
            tl.store(c + 0, v, None)
        """
        tiled = """
            load_y = tl.load(y + tid[0], None, other=0)
            tl.store(out + tid[0], load_y, None)
        """
        matmul = """
            acc = _helion_mpp_acc(x)
        """
        reduced = """
            load_x = tl.load(x + tid[0], None, other=0)
            r = helion_red.group_sum(site, partials, load_x, tid, 0, 64, 64, 1, 1)
        """
        adds_result = """
            load = tl.load(c + 0, None, other=0)
            tl.store(c + 0, r + load, None)
        """
        masked_by_result = """
            load = tl.load(c + 0, None, other=0)
            tl.store(c + 0, load * 2.0, r > 0)
        """
        tests_result = """
            if r > 0:
                load = tl.load(c + 0, None, other=0)
                tl.store(c + 0, load * 2.0, None)
        """
        at_result = """
            i = helion_red.group_argmax(s, p, q, load_x, tid[0], tid, 0, 64, 64, 1, 1)
            load = tl.load(c + i, None, other=0)
            tl.store(c + i, load + 1.0, None)
        """
        copy = """
            load_d = tl.load(d + 0, None, other=0)
            tl.store(c + 0, load_d * 2.0, None)
        """
        flipped = """
            load = tl.load(c + 0, None, other=0)
            if load > 0:
                tl.store(c + 0, -1.0, None)
        """
        round_trip = """
            load_c = tl.load(c + 0, None, other=0)
            tl.store(tmp + 0, load_c + 1.0, None)
            load_tmp = tl.load(tmp + 0, None, other=0)
            tl.store(c + 0, load_tmp * 2.0, None)
        """
        tested_copy = """
            load_d = tl.load(d + 0, None, other=0)
            if load_d > 0:
                tl.store(c + 0, 1.0, None)
        """
        frame = (64, 1, 1)
        for name, parts, repeated, reads, guard in (
            ("alone", (update,), [], {}, "tid[0] < 1"),
            ("before a tiled store", (update, tiled), [0], {}, "None"),
            ("beside a matmul", (update, matmul), [0], {}, "None"),
            ("adding a reduction's result", (reduced, adds_result), [0], {}, "None"),
            ("masked by it", (reduced, masked_by_result), [0], {}, "r > 0"),
            ("under a test of it", (reduced, tests_result), [0], {}, "None"),
            ("at an index it gives", (reduced, at_result), [0], {}, "None"),
            ("copy before a tiled store", (copy, tiled), [], {"c": {"d"}}, "None"),
            ("flipped by a test of it", (flipped, tiled), [0], {}, "None"),
            ("through a scratch tensor", (round_trip, tiled), [0], {}, "None"),
            ("copied by a test", (tested_copy, tiled), [], {"c": {"d"}}, "None"),
        ):
            with self.subTest(name):
                body = ast.parse("".join(textwrap.dedent(p) for p in parts)).body
                result = guard_surplus_stores([body], [frame], [frame], body, {})
                self.assertEqual(result.repeated_updates, repeated)
                self.assertEqual(result.repeated_reads, reads)
                (store,) = [
                    node
                    for node in ast.walk(ast.Module(body=body, type_ignores=[]))
                    if is_store(node) and pointer_base(node.args[0]) == "c"
                ]
                self.assertEqual(ast.unparse(store.args[2]), guard)

    def test_loop_carried_address_unguarded(self) -> None:
        """A root as wide as the launch whose scatter reaches its row's
        thread only through a sum the loop carries: the store depends on
        that thread, so it takes no guard and every row scatters.  Inside the
        loop the sum is a temporary that takes the name ``acc`` only when
        the kernel is emitted."""
        x = torch.zeros(256, 32, dtype=torch.int64)
        out, y = torch.zeros(256), torch.randn(1 << 12)
        config = helion.Config(block_sizes=[64, 8, 64], num_threads=[64, 1, 64])
        code = scatter_beside_scale.bind((x, out, y)).to_code(config)
        (scatter,) = [line for line in code.splitlines() if "tl.store(out" in line]
        self.assertNotIn("tid[", scatter)

    def test_loop_carried_update(self) -> None:
        """An update that loads its element only inside a loop, into a value
        the loop carries, is seen under the names the kernel is emitted with
        (``DeviceFunction.merge_variable_names``), and missed without them."""
        source = textwrap.dedent("""
            acc = 0.0
            for k in range(4):
                load = tl.load(c + 0, None, other=0)
                v_1 = acc + load
            tl.store(c + 0, acc, None)
            load_y = tl.load(y + tid[0], None, other=0)
            tl.store(out + tid[0], load_y, None)
        """)
        frame = (64, 1, 1)
        for renames, repeated in (({}, []), ({"v_1": "acc"}, [0])):
            with self.subTest(renames=renames):
                body = ast.parse(source).body
                result = guard_surplus_stores([body], [frame], [frame], body, renames)
                self.assertEqual(result.repeated_updates, repeated)

    def test_written_row_read_back(self) -> None:
        """A row of ``s`` written, then read back by every thread of a tile
        laid out in 8 x 16 threads of a 64 x 16 launch: the row's store
        depends on no column thread, but the root's other store reads every
        thread, so both take the guard that keeps the root's 8 x 16 threads,
        and each thread reads back its own write.  The other root, laid out
        in 64 threads, takes ``tid[1] < 1``."""
        x, s_src, s = torch.randn(64, 64), torch.randn(64), torch.randn(64)
        y = torch.randn(1 << 12)
        config = helion.Config(block_sizes=[16, 16, 64])
        code = write_then_read_back.bind((x, s_src, s, y)).to_code(config)
        self.assertIn("_block_dims=(64, 16, 1)", code)
        self.assertRegex(code, r"tl\.store\(s \+ [^\n]*, tid\[0\] < 8 and mask_\d+\)")
        self.assertRegex(
            code, r"tl\.store\(out \+ [^\n]*, tid\[0\] < 8 and mask_\d+ & mask_\d+\)"
        )
        self.assertRegex(
            code, r"tl\.store\(out_y \+ [^\n]*, tid\[1\] < 1 and mask_\d+\)"
        )

    def test_launch_past_threadgroup_refused(self) -> None:
        """Roots laid out in 8 x 32 and 512 threads would share a 512 x 32
        launch, more than a threadgroup's 1024 threads: refused at codegen."""
        x, y = torch.randn(64, 64), torch.randn(1 << 12)
        config = helion.Config(block_sizes=[8, 32, 512], num_warps=32)
        with self.assertRaisesRegex(exc.BackendUnsupported, "1024 threads"):
            grid_beside_row.bind((x, y)).to_code(config)


if __name__ == "__main__":
    unittest.main()
