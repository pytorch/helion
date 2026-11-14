"""The gluon backend's analyses as plain functions: the ring's register budgets,
waits and paced fills, the pipeliner's load arguments, the native kernels'
compile options and the names its ``tl`` resolves. They run on CPU."""

from __future__ import annotations

import ast
from pathlib import Path
import re
from types import SimpleNamespace
from typing import cast
import unittest

import pytest
import torch

import helion
from helion._compat import gluon_api_available
from helion._testing import TestCase
from helion.runtime.settings import _get_backend

if _get_backend() != "triton" or not gluon_api_available():
    raise unittest.SkipTest(
        "the gluon backend's units run in the triton lane on Triton >= 3.8"
    )

from helion._compiler.cross_loop_codegen import TaggedPeerFallback
from helion._compiler.device_function import SyncRole
from helion._compiler.triton import gluon_ops
from helion._compiler.triton import gluon_pipeline
from helion._compiler.triton import gluon_ring
from helion._compiler.triton.gluon_ops import tl_namespace

pytestmark = pytest.mark.cpu_only


class TestWorkerRegisters(TestCase):
    def test_workers_shrink_only_for_a_body_below_the_cap(self) -> None:

        ring = [(2, False), (2, False), (4, True)]
        # A 4-warp body reaches the 256 cap with 88-register workers: keep them.
        self.assertEqual(gluon_ring._worker_registers(4, ring), [88, 88, 88])
        self.assertEqual(gluon_ring._body_registers(4, ring, [88, 88, 88]), 256)
        # A one-warp claimer pads the workers' warpgroups: still at the cap.
        self.assertEqual(
            gluon_ring._worker_registers(4, [*ring, (1, False)]), [88, 88, 88, 24]
        )
        # An 8-warp body sits at 168: the floors give it 208.
        self.assertEqual(gluon_ring._worker_registers(8, ring), [40, 40, 48])
        self.assertEqual(gluon_ring._body_registers(8, ring, [40, 40, 48]), 208)
        # Two CTAs a SM halve each one's register file: the floors, 152 left.
        self.assertEqual(gluon_ring._worker_registers(4, ring, 2), [40, 40, 48])
        self.assertEqual(gluon_ring._body_registers(4, ring, [40, 40, 48], 2), 152)
        # Scalar (one-warp) partitions take 24 whatever the budget.
        scalar = [(1, False), (1, False), (1, True)]
        self.assertEqual(gluon_ring._worker_registers(8, scalar), [24, 24, 24])

    def test_unswizzled_slot_keeps_whole_rows(self) -> None:
        """A tcgen05.cp scale slot keeps its contiguous rows as the TMA box
        (16-byte rows stream slower) in an explicit swizzle-free layout."""

        scales = torch.zeros([4, 2, 64, 256], dtype=torch.uint8)
        desc = gluon_ring.ring_descriptor(
            scales, [4, 2, 64, 256], [32768, 16384, 256, 1], [1, 2, 8, 256], False
        )
        self.assertEqual(list(desc.block_shape), [1, 2, 8, 256])
        self.assertEqual(desc.layout.swizzle_byte_width, 0)

    def test_tagged_pollers_take_only_what_the_body_leaves(self) -> None:

        # Two pollers (the gated producer and one added) beside the weights
        # producer and the MMA side; a 4-warp body stays at the cap.
        ring = [(2, False), (2, False), (2, False), (4, True)]
        extra = [False, False, True, False]
        floors = [40, 40, 40, 48]
        self.assertEqual(gluon_ring._worker_registers(4, ring), floors)
        self.assertEqual(
            gluon_ring._poller_registers(4, ring, floors, [1, 2], extra),
            [40, 104, 104, 48],
        )
        self.assertEqual(gluon_ring._body_registers(4, ring, [40, 104, 104, 48]), 256)
        # An 8-warp body would lose registers to the added warpgroup: no tagging.
        with self.assertRaises(TaggedPeerFallback):
            gluon_ring._poller_registers(8, ring, floors, [1, 2], extra)


class TestFirstWaits(TestCase):
    def test_a_step_waits_once_per_block(self) -> None:
        """Every operand of a step shares the slot's full barrier: only the
        block's first acquire of the step waits; one after the step counter
        moves, in a nested block or after it waits again."""

        def acquire(slot: str) -> str:
            return f"{slot} = _helion_ring.ring_acquire(r, full, e, step, S, d, (), True, B, X)"

        body = ast.parse(
            "\n".join(
                [
                    acquire("a"),
                    acquire("b"),
                    "step = step + 1",
                    acquire("c"),
                    "for k in range(n):",
                    f"    {acquire('d')}",
                    f"    {acquire('f')}",
                    acquire("g"),
                ]
            )
        ).body
        gluon_ring._first_waits(body)
        skipped = {
            ast.unparse(statement.targets[0])
            for statement in ast.walk(ast.Module(body=body, type_ignores=[]))
            if isinstance(statement, ast.Assign)
            and "WAIT=False" in ast.unparse(statement)
        }
        self.assertEqual(skipped, {"b", "f"})


class TestPartitionWaits(TestCase):
    def test_partitions_keep_only_waits_ordering_their_reads(self) -> None:
        """A partition keeps a wait only if it orders a read the partition makes;
        a wait also ordering its root's barrier publication stays in the body,
        which makes every publication; an unannotated wait orders everything."""

        class Function:
            def __init__(self) -> None:
                self.sync_roles = {
                    "reads": SyncRole("wait", covers=frozenset([1])),
                    "publishes": SyncRole(
                        "wait", covers=frozenset([1]), publication=True
                    ),
                    "unknown": SyncRole("wait"),
                }

            def sync_names(self, *kinds: str) -> set[str]:
                return {n for n, r in self.sync_roles.items() if r.kind in kinds}

        def orders(name: str, reads: frozenset[int], publishes: bool) -> bool:
            wait = ast.parse(f"{name} = 0").body[0]
            return gluon_ring._orders(Function(), wait, reads, publishes)  # pyrefly: ignore

        # The weight producer (reads 2) drops both covered waits ...
        self.assertFalse(orders("reads", frozenset([2]), False))
        self.assertFalse(orders("publishes", frozenset([2]), False))
        # ... the gated producer (reads 1) keeps them ...
        self.assertTrue(orders("reads", frozenset([1, 2]), False))
        self.assertTrue(orders("publishes", frozenset([1]), False))
        # ... and the body keeps the publication's even with no read after it.
        self.assertFalse(orders("reads", frozenset(), True))
        self.assertTrue(orders("publishes", frozenset(), True))
        self.assertTrue(orders("unknown", frozenset([2]), False))


class TestPacedFill(TestCase):
    def test_paced_fill_stops_at_a_kept_wait(self) -> None:
        """A weight producer's first fill is a paced run up to the first
        statement keeping a dependency wait after (or among) its paced puts:
        the owed arrivals flush ahead of it, and later steps are plain."""

        put = "_helion_ring.ring_put(r, _helion_ring_full0, e, s, 2, d, c, True, 8, b)"
        skip = "_helion_ring.ring_skip(_helion_ring_full0, e, s, 2, 8)"

        def run(source: str) -> list[str]:
            body = ast.parse(source).body
            out = gluon_ring._paced_fill(body, [0], 1, {"w"}, ast.Constant(True))
            return [ast.unparse(st) for st in out]

        # A wait before any put leaves the fill paced (the counters start at
        # zero); a put is flushed after it, a skip flushes before it.
        out = run(f"w = 0\n{put}\n{skip}")
        self.assertEqual(
            out[:2],
            [f"_helion_ring_{n}0 = tl.full([], 0, tl.int32)" for n in ("paced", "run")],
        )
        self.assertEqual(out[2], "w = 0")
        self.assertIn(
            "_helion_ring_paced0, _helion_ring_run0 = _helion_ring.ring_put_paced(r, "
            "_helion_ring_full0, _helion_ring_pace0, e, s, 2, d, c, True, 8, b, "
            "_helion_ring_paced0, _helion_ring_run0, True, 1)",
            out[3],
        )
        # The flush after the put leaves the one before the skip nothing to do.
        self.assertIn("ring_pace_flush", out[4])
        self.assertIn("ring_skip", out[5])
        self.assertEqual(len(out), 6)
        # A wait past a put: flush ahead of it, then plain puts and skips.
        out = run(f"{put}\nw = 0\n{put}\nfor j in y:\n    {skip}")
        self.assertIn("ring_put_paced", out[2])
        self.assertIn("ring_pace_flush", out[3])
        self.assertEqual(out[4], "w = 0")
        self.assertEqual(out[5], put)
        self.assertEqual(out[6], f"for j in y:\n    {skip}")
        self.assertEqual(len(out), 7)
        # A wait among the first puts (one loop): nothing paced at all.
        body = f"for i in x:\n    w = 0\n    {put}"
        self.assertEqual(run(body), [body])
        # No put (a claimer): unchanged.
        self.assertEqual(run("w = 0"), ["w = 0"])


class TestTlNamespace(TestCase):
    def test_every_emitted_name_resolves(self) -> None:
        """Every ``tl.X`` the Triton lowering emits resolves in a gluon kernel's
        ``tl``, but the block pointers and device-side descriptors gluon never
        emits (its descriptors are host ones) and a name only Triton < 3.4 has."""

        root = Path(helion.__file__).parent
        skip = {"cute", "pallas", "metal", "flydsl"}
        files = [
            p
            for p in (root / "_compiler").rglob("*.py")
            if not skip & set(p.parts) and not p.name.startswith("gluon_")
        ]
        files += [*(root / "language").glob("*.py"), *(root / "runtime").glob("*.py")]
        names = {
            name
            for p in files
            for name in re.findall(
                r"(?<![\w.])tl\.([A-Za-z_]\w*)(?![\w{])", p.read_text()
            )
        }
        unresolved = {n for n in names if not hasattr(tl_namespace, n)}
        self.assertEqual(
            unresolved,
            {
                "make_block_ptr",
                "make_tensor_descriptor",
                "_experimental_make_tensor_descriptor",
            },
        )


class TestPipelinedLoads(TestCase):
    def test_load_arguments_by_keyword(self) -> None:
        """A pipelined ``tl.load`` keeps its mask and other given by keyword."""
        for source in ("tl.load(p, mask=m, other=0.0)", "tl.load(p, m, 0.0)"):
            load = cast("ast.Call", cast("ast.Expr", ast.parse(source).body[0]).value)
            self.assertEqual(
                [ast.unparse(a) for a in gluon_pipeline._load_args(load)],
                ["p", "m", "0.0"],
            )
        load = cast("ast.Call", cast("ast.Expr", ast.parse("tl.load(p)").body[0]).value)
        self.assertEqual(
            [ast.unparse(a) for a in gluon_pipeline._load_args(load)],
            ["p", "None", "None"],
        )


class TestNativeKernel(TestCase):
    def test_compile_options_carry_maxnreg(self) -> None:
        """Launches and the autotuner's precompiler pack the same options: the
        kernel's ``maxnreg`` among them."""

        @gluon_ops.native_kernel(maxnreg=128)
        def kernel(x):
            pass

        options: dict[str, object] = {}

        def parse_options(kwargs: dict[str, object]) -> SimpleNamespace:
            options.update(kwargs)
            return SimpleNamespace(**kwargs)

        backend = SimpleNamespace(parse_options=parse_options, parse_attr=str)
        kernel._pack_args(backend, {"num_warps": 4}, {"x": 1}, [("i32", "")], None)
        self.assertEqual(options, {"maxnreg": 128, "num_warps": 4})


if __name__ == "__main__":
    unittest.main()
