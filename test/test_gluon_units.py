"""The gluon backend on CPU: its analyses as plain functions (the ring's
register budgets, waits and paced fills, the pipeliner's load arguments, the
native kernels' compile options and the names its ``tl`` resolves), and whole
kernels' generated Gluon source (``gluon_cpu_codegen``)."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
import unittest

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore

import helion
from helion._compat import gluon_api_available
from helion._testing import TestCase
from helion._testing.distributed_kernels import TAGGED_RING
from helion._testing.distributed_kernels import tagged_ring_exchange_kernel
from helion._testing.gluon_kernels import dot_mm
from helion._testing.gluon_kernels import mm_then_mm
from helion._testing.gluon_kernels import nvfp4_mm
from helion._testing.gluon_kernels import scale_then_nested_wx
from helion._testing.gluon_kernels import scale_then_two_wx
from helion._testing.gluon_kernels import scale_then_wx
from helion._testing.gluon_kernels import scale_then_wx_and_f16_vx
from helion._testing.gluon_kernels import scale_then_wx_and_nvfp4
from helion._testing.triton_codegen import GLUON_RING_CONFIG as _RING
from helion._testing.triton_codegen import gluon_cpu_codegen
from helion._testing.triton_codegen import gluon_tail_ring_config
import helion.language as hl
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
from helion._compiler.triton.backend import GluonBackend
from helion._compiler.triton.gluon_ops import tl_namespace
from helion.autotuner.config_spec import ConfigSpec

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only


@helion.kernel(static_shapes=True, backend="gluon")
def _scale_then_wtx(x: torch.Tensor, wt: torch.Tensor) -> torch.Tensor:
    """``scale_then_wx`` with the weight stored transposed ([k, n])."""
    m, k = x.size()
    n = wt.size(1)
    y = torch.empty_like(x)
    out = torch.empty([n, m], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([m, k]):
        y[tile_r, tile_c] = x[tile_r, tile_c] * 2
    for tile_n, tile_m in hl.tile([n, m]):
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(wt[tile_k, tile_n].T, y[tile_m, tile_k].T, acc=acc)
        out[tile_n, tile_m] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def _scale_then_selected_wx(
    x: torch.Tensor, w: torch.Tensor, sel: torch.Tensor
) -> torch.Tensor:
    """``scale_then_wx`` with the weight one of a stack, picked by an index the
    first root writes (as a MoE's expert id)."""
    m, k = x.size()
    n = w.size(1)
    y = torch.empty_like(x)
    out = torch.empty([n, m], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([m, k]):
        y[tile_r, tile_c] = x[tile_r, tile_c] * 2
        sel[0] = 1
    for tile_n, tile_m in hl.tile([n, m]):
        e = sel[0]
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(w[e, tile_n, tile_k], y[tile_m, tile_k].T, acc=acc)
        out[tile_n, tile_m] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def _scale_then_guarded_wx(
    x: torch.Tensor, w: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    """``scale_then_wx`` with each task's product behind a loaded bound (the
    root's tasks wait for the first root as a whole, ahead of their loop)."""
    m, k = x.size()
    n = w.size(0)
    y = torch.empty_like(x)
    out = torch.empty([n, m], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([m, k]):
        y[tile_r, tile_c] = x[tile_r, tile_c] * 2
    for tile_n, tile_m in hl.tile([n, m]):
        if tile_n.begin < rows[0]:
            acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = hl.dot(w[tile_n, tile_k], y[tile_m, tile_k].T, acc=acc)
            out[tile_n, tile_m] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def _w1x_then_w3x(
    x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``scale_then_wx`` with a second weight's K loop after the first's, in
    the same task (tiles of one shape)."""
    m, k = x.size()
    n = w1.size(0)
    y = torch.empty_like(x)
    out1 = torch.empty([n, m], dtype=torch.float32, device=x.device)
    out3 = torch.empty([n, m], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([m, k]):
        y[tile_r, tile_c] = x[tile_r, tile_c] * 2
    for tile_n, tile_m in hl.tile([n, m]):
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(w1[tile_n, tile_k], y[tile_m, tile_k].T, acc=acc)
        out1[tile_n, tile_m] = acc
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(w3[tile_n, tile_k], y[tile_m, tile_k].T, acc=acc)
        out3[tile_n, tile_m] = acc
    return out1, out3


@helion.kernel(static_shapes=True, backend="gluon")
def _w13x(
    x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two weights of one shape read in one K loop (a gate/up pair)."""
    m, k = x.size()
    n = w1.size(0)
    out1 = torch.empty([n, m], dtype=torch.float32, device=x.device)
    out3 = torch.empty([n, m], dtype=torch.float32, device=x.device)
    for tile_n, tile_m in hl.tile([n, m]):
        acc1 = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        acc3 = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc1 = hl.dot(w1[tile_n, tile_k], x[tile_m, tile_k].T, acc=acc1)
            acc3 = hl.dot(w3[tile_n, tile_k], x[tile_m, tile_k].T, acc=acc3)
        out1[tile_n, tile_m] = acc1
        out3[tile_n, tile_m] = acc3
    return out1, out3


@helion.kernel(static_shapes=True, backend="gluon")
def _two_pairs(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    """A task with two K loops, one after the other, each reading two weights
    of one shape (two same-shape dots each)."""
    m, k = x.size()
    n = w1.size(0)
    out = torch.empty([n, m], dtype=torch.float32, device=x.device)
    for tile_n, tile_m in hl.tile([n, m]):
        a1 = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        a3 = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            a1 = hl.dot(w1[tile_n, tile_k], x[tile_m, tile_k].T, acc=a1)
            a3 = hl.dot(w3[tile_n, tile_k], x[tile_m, tile_k].T, acc=a3)
        b1 = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        b3 = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            b1 = hl.dot(w3[tile_n, tile_k], x[tile_m, tile_k].T, acc=b1)
            b3 = hl.dot(w1[tile_n, tile_k], x[tile_m, tile_k].T, acc=b3)
        out[tile_n, tile_m] = a1 + a3 - b1 * b3
    return out


class TestAccumulatorRings(TestCase):
    def test_a_dot_takes_a_ring_without_a_live_accumulator(self) -> None:
        """Two K loops of a task, one after the other, with two same-shape dots
        each, when tensor memory holds only two rings: the first loop's dots
        take one each; in the second, the first dot shares the first ring and
        the second, live with it, takes the other (not the first again)."""
        x = torch.empty([256, 1024], dtype=torch.bfloat16)
        w1 = torch.empty([256, 1024], dtype=torch.bfloat16)
        with gluon_cpu_codegen():
            code = _two_pairs.bind((x, w1, torch.empty_like(w1))).to_triton_code(
                helion.Config(
                    **_RING,
                    # Two 128x128 accumulator rings take all 512 columns.
                    block_sizes=[128, 128, 64, 64],
                    range_warp_specializes=[True, None, None],
                )
            )
        self.assertIn("warp_specialize", code)
        self.assertEqual(code.count("= _helion_gluon.acc_buffers("), 2)
        claims = re.findall(r"acc_claim\((_helion_acc\d)", code)
        self.assertEqual(
            claims, ["_helion_acc0", "_helion_acc1", "_helion_acc0", "_helion_acc1"]
        )

    def test_dots_of_one_loop_share_no_accumulator_ring(self) -> None:
        """Two dots of one K loop whose accumulator rings tensor memory cannot
        both hold do not share one (both accumulators are live through the
        loop): the ring falls back."""
        x = torch.empty([256, 1024], dtype=torch.bfloat16)
        w1 = torch.empty([256, 1024], dtype=torch.bfloat16)
        config = helion.Config(
            **_RING,
            # A 128x256 accumulator ring takes all 512 columns.
            block_sizes=[128, 256, 64],
            range_warp_specializes=[True, None],
        )
        with gluon_cpu_codegen():
            bound = _w13x.bind((x, w1, torch.empty_like(w1)))
            with self.assertLogs("helion._compiler.triton.gluon_ring", "INFO") as logs:
                code = bound.to_triton_code(config)
        self.assertIn("holds an accumulator live with this one", "\n".join(logs.output))
        self.assertNotIn("warp_specialize", code)


def _worker_budgets(code: str) -> dict[str, tuple[int, int]]:
    """Each ring worker's (warps, registers) in ``gl.warp_specialize``, by
    function name, in the order the kernel passes them."""
    call = next(
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "gl.warp_specialize"
    )
    warps, regs = (ast.literal_eval(arg) for arg in call.args[1:])
    tasks = cast("ast.List", call.args[0]).elts[1:]  # the body's first
    names = [ast.unparse(cast("ast.Tuple", task).elts[0]) for task in tasks]
    return dict(zip(names, zip(warps, regs, strict=True), strict=True))


class TestTmemPrefetch(TestCase):
    """``cross_loop_tmem_prefetch``'s candidates, search and generated code (its
    outputs: test_gluon_ring's test_cross_loop_tmem_prefetch)."""

    _LOG = "helion._compiler.triton.gluon_ring"

    @staticmethod
    def _ring(bound: BoundKernel, **extra: object) -> helion.Config:
        return helion.Config(**gluon_tail_ring_config(bound, **extra))

    def test_a_tail_streams_a_k_loops_last_steps(self) -> None:
        """The K loop behind its task's gate, its weight (operand A) streamed
        ungated, is a candidate; the key is a hand-set override (no search
        dimension, no default entry, an all-off entry dropped). Enabled, its
        last 14 of 16 steps (448 columns beside the accumulator ring's 64; 2
        with three CTAs an SM, 128 columns each) read A from tensor memory: the
        MMA partition copies them in a fill loop, the gated producer skips the
        fill steps ahead of its dependency wait, and only the producer's put of
        a step past the fill takes ``HEAD`` (paced or not)."""
        x = torch.empty([32, 1024], dtype=torch.bfloat16)
        w = torch.empty([256, 1024], dtype=torch.bfloat16)
        with gluon_cpu_codegen():
            bound = scale_then_wx.bind((x, w))
            spec = bound.config_spec
            self.assertEqual(len(spec.cross_loop_tmem_prefetch), 1)
            self.assertNotIn(
                "cross_loop_tmem_prefetch",
                {d.name for d in spec.iter_search_dimensions()},
            )
            self.assertNotIn("cross_loop_tmem_prefetch", spec.default_config().config)
            for entry, kept in (([True], True), ([False], False)):
                config = self._ring(bound, cross_loop_tmem_prefetch=entry).config
                spec.normalize(config)
                self.assertEqual("cross_loop_tmem_prefetch" in config, kept)
            plain = bound.to_triton_code(self._ring(bound))
            code = bound.to_triton_code(
                self._ring(bound, cross_loop_tmem_prefetch=[True])
            )
            shared = bound.to_triton_code(
                self._ring(bound, num_sm_multiplier=3, cross_loop_tmem_prefetch=[True])
            )
            # The first root loading x by pointer: the weight producer paces its
            # first fill (``ring_put_paced``).
            paced = bound.to_triton_code(
                self._ring(
                    bound,
                    indexing=["pointer", *["tensor_descriptor"] * 4],
                    cross_loop_tmem_prefetch=[True],
                )
            )
        self.assertNotIn("tail_fill(", plain)
        functions = {
            chunk.split("(", 1)[0]: chunk for chunk in code.split("\ndef ")[1:]
        }
        mma = functions["_helion_scale_then_wx_mma"]
        self.assertEqual(mma.count("tail_fill("), 1)
        self.assertIn("TAIL=_helion_ring_tail0", mma)
        self.assertNotIn("HEAD=", mma)
        self.assertIn("HEAD=", functions["_helion_scale_then_wx_producer0"])
        self.assertIn("tail_buffers(w_ring_desc, 14, 128, 64)", code)
        self.assertIn("tail_buffers(w_ring_desc, 2, 128, 64)", shared)
        gated = functions["_helion_scale_then_wx_producer1"]
        self.assertLess(gated.index("ring_skip("), gated.index("ld.acquire"))
        # Paced, the K loop's put still passes over the tail steps.
        weights = paced.split("def _helion_scale_then_wx_producer0")[1]
        puts = [line for line in weights.splitlines() if "ring_put_paced(" in line]
        self.assertTrue(puts)
        self.assertTrue(any("HEAD=" in line for line in puts))

    def test_loops_that_take_no_tail(self) -> None:
        """A 64-row weight tile and one of 256-byte rows (two swizzle units a
        row) take no tail, nor a loop under cross_loop_pipeline="barrier" (its
        phase loops share the fill's statements), a dot reading no ring slot
        (both operands loaded by pointer), a K loop inside another loop of its
        task, the second of two loops of a task sharing a buffer, a loop whose
        task loop follows a root barrier's wait (each one's fill would go
        behind the task's waits), or one whose A coordinates read what another
        root writes (an expert id: the fill would load ahead of the wait for
        it), and each is logged; a view of a load (the
        weight stored transposed) is no candidate; a config that lowers no ring
        (one stage) rejects an enabled tail. An enabled loop turned away logs a
        warning; the others keep their tails."""
        x = torch.empty([32, 1024], dtype=torch.bfloat16)
        w = torch.empty([256, 1024], dtype=torch.bfloat16)
        with gluon_cpu_codegen():
            bound = scale_then_wx.bind((x, w))
            rows = "tensor memory takes 128 rows of 128 bytes"
            pointers = ["tensor_descriptor"] * 2 + ["pointer"] * 2
            for extra, reason in (
                ({"block_sizes": [32, 64, 64, 32, 64]}, rows),
                ({"block_sizes": [32, 64, 128, 32, 128]}, rows),
                (
                    {"indexing": [*pointers, "tensor_descriptor"]},
                    "loop 4: its dot reads no ring slot",
                ),
                (
                    {"cross_loop_pipeline": "barrier"},
                    "loop 4: cross_loop_pipeline is 'barrier'",
                ),
            ):
                config = self._ring(bound, **extra, cross_loop_tmem_prefetch=[True])
                with (
                    self.subTest(**extra),
                    self.assertLogs(self._LOG, "WARNING") as logs,
                ):
                    code = bound.to_triton_code(config)
                    self.assertNotIn("tail_fill(", code)
                    self.assertIn(reason, "\n".join(logs.output))
            pair = _w1x_then_w3x.bind((x, w, torch.empty_like(w)))
            config = self._ring(pair, task_loop=1, cross_loop_tmem_prefetch=[True] * 2)
            with self.assertLogs(self._LOG, "WARNING") as logs:
                code = pair.to_triton_code(config)
            self.assertIn(
                "loop 5: an earlier loop reading its tensor-memory buffer follows a "
                "dependency wait",
                "\n".join(logs.output),
            )
            self.assertEqual(code.count("tail_fill("), 1)
            guarded = _scale_then_guarded_wx.bind(
                (x, w, torch.empty([1], dtype=torch.int32))
            )
            # The bound loaded by pointer: the ring streams the rest.
            config = self._ring(
                guarded,
                indexing=[*pointers[:2], "pointer", *["tensor_descriptor"] * 3],
                cross_loop_tmem_prefetch=[True],
            )
            with self.assertLogs(self._LOG, "WARNING") as logs:
                code = guarded.to_triton_code(config)
            self.assertIn(
                "loop 4: a dependency wait precedes its task loop",
                "\n".join(logs.output),
            )
            self.assertNotIn("tail_fill(", code)
            selected = _scale_then_selected_wx.bind(
                (x, w.view(2, 128, 1024), torch.empty([1], dtype=torch.int32))
            )
            with self.assertLogs(self._LOG, "WARNING") as logs:
                code = selected.to_triton_code(
                    self._ring(selected, cross_loop_tmem_prefetch=[True])
                )
            self.assertIn(
                "loop 4: A's coordinates follow a dependency wait",
                "\n".join(logs.output),
            )
            self.assertNotIn("tail_fill(", code)
            nested = scale_then_nested_wx.bind((x, w, torch.empty_like(w)))
            # One entry per loop in block-id order (the outer loop first, as
            # in the range_* keys); both enabled, the outer one takes a tail,
            # the inner alone none.
            self.assertEqual(
                [s.block_ids for s in nested.config_spec.cross_loop_tmem_prefetch],
                [[4], [5]],
            )
            for entry, fills in (([True, True], 1), ([False, True], 0)):
                config = self._ring(nested, task_loop=1, cross_loop_tmem_prefetch=entry)
                with (
                    self.subTest(entry=entry),
                    self.assertLogs(self._LOG, "WARNING") as logs,
                ):
                    code = nested.to_triton_code(config)
                    self.assertIn(
                        "loop 5: inside another loop of its task",
                        "\n".join(logs.output),
                    )
                    self.assertEqual(code.count("tail_fill("), fills)
            with self.assertLogs(self._LOG, "INFO") as logs:
                transposed = _scale_then_wtx.bind((x, w.T.contiguous()))
            self.assertEqual(len(transposed.config_spec.cross_loop_tmem_prefetch), 0)
            self.assertIn("operand A is no load", "\n".join(logs.output))
            with self.assertRaisesRegex(
                helion.exc.InvalidConfig,
                "cross_loop_tmem_prefetch sets a Gluon ring's K loops' tails, but "
                "this kernel lowers no ring for this config",
            ):
                bound.to_triton_code(
                    self._ring(bound, num_stages=1, cross_loop_tmem_prefetch=[True])
                )

    def test_block_scales_take_from_the_tails_budget(self) -> None:
        """An nvfp4 dot in the K loop's task keeps its block scales in tensor
        memory beside the two accumulator rings (2 x 64 columns): the scales'
        48 columns (a_scale's two [128, 16] buffers of 16, b_scale's two
        [32, 16] ones of 8: Triton spreads each over all 128 lanes) leave 336,
        10 tiles of 32 (12 without them)."""
        x = torch.empty([32, 1024], dtype=torch.bfloat16)
        w = torch.empty([256, 1024], dtype=torch.bfloat16)
        a = torch.empty([256, 256], dtype=torch.uint8)
        b = torch.empty([256, 32], dtype=torch.uint8)
        a_scale, b_scale = (
            torch.empty([rows, 32], dtype=torch.float8_e4m3fn) for rows in (256, 32)
        )
        with gluon_cpu_codegen():
            bound = scale_then_wx_and_nvfp4.bind((x, w, a, a_scale, b, b_scale))
            code = bound.to_triton_code(
                self._ring(bound, task_loop=1, cross_loop_tmem_prefetch=[True])
            )
        self.assertIn("_helion_gluon.tmem_dot_scaled(", code)
        self.assertEqual(code.count("tail_fill("), 1)
        self.assertIn("tail_buffers(w_ring_desc, 10, 128, 64)", code)

    def test_tails_share_a_buffer_per_tile_shape(self) -> None:
        """Two tasks' K loops of one tile shape fill one buffer (14 of 16 steps
        each, beside the accumulator ring they share), with more tasks than
        workers (every worker runs both roots' tasks: the first root's waits,
        ahead of the second's task loop, are not the second's); an fp16 loop
        after a bf16 one in a task takes a buffer of its own, ahead of the
        task's waits too, from the columns the first leaves (4 of 4 steps, then
        8 of 16)."""
        x = torch.empty([32, 1024], dtype=torch.bfloat16)
        w = torch.empty([256, 1024], dtype=torch.bfloat16)
        many = torch.empty([152 * 128, 1024], dtype=torch.bfloat16)
        with gluon_cpu_codegen():
            tasks = scale_then_two_wx.bind((x, many, torch.empty_like(many)))
            shared = tasks.to_triton_code(
                self._ring(
                    tasks,
                    block_sizes=[32, 64, *[128, 32, 64] * 2],
                    range_warp_specializes=[None, True, None, True, None],
                    cross_loop_tmem_prefetch=[True, True],
                )
            )
            mixed = scale_then_wx_and_f16_vx.bind(
                (
                    x[:, :256],
                    w[:, :256],
                    torch.empty([32, 1024], dtype=torch.float16),
                    torch.empty([256, 1024], dtype=torch.float16),
                )
            )
            split = mixed.to_triton_code(
                self._ring(
                    mixed,
                    task_loop=2,
                    block_sizes=[32, 64, 32, 64, 128, 32, 64, 64],
                    cross_loop_tmem_prefetch=[True, True],
                )
            )
        fills = re.findall(r"tail_fill\(.*", shared)
        self.assertEqual(len(fills), 2)
        self.assertTrue(all("_helion_ring_tail0" in fill for fill in fills))
        self.assertIn("tail_buffers(w1_ring_desc, 14, 128, 64)", shared)
        self.assertNotIn("_helion_ring_tail1", shared)
        self.assertEqual(split.count("tail_fill("), 2)
        self.assertIn("tail_buffers(w_ring_desc, 4, 128, 64)", split)
        self.assertIn("tail_buffers(v_ring_desc, 8, 128, 64)", split)
        self.assertIn("TAIL=_helion_ring_tail1", split)
        # Both fills' steps: the gated producer skips them all before its wait.
        gated = split.split("def _helion_scale_then_wx_and_f16_vx_producer1")[1]
        gated = gated.split("\ndef ")[0]
        self.assertLess(gated.rindex("ring_skip("), gated.index("ld.acquire"))


class TestRingTimeline(TestCase):
    def test_entry_clock_joins_the_launch_state(self) -> None:
        """A timeline keeps a ring megakernel with peer launch state: only its
        entry clock joins the launch state's function, ahead of every epoch
        read (the peer launch state's too); the entry stamps stay in the body,
        and the in-band poll's spin ends a wait (its ready stamp follows)."""
        if dist.is_initialized():
            self.skipTest("needs exclusive control of the process group")
        dist.init_process_group(backend="fake", store=FakeStore(), rank=0, world_size=4)
        self.addCleanup(dist.destroy_process_group)
        world, m, k, n = 4, 128, 64, 128
        bf16 = torch.bfloat16
        args = (
            torch.zeros(m, world * k, dtype=bf16),
            torch.zeros(256),
            torch.ones(m, k, dtype=bf16),
            torch.ones(world * k, n, dtype=bf16),
            torch.zeros(256),
            dist.group.WORLD.group_name,
            0,
        )
        kernel = helion.kernel(
            tagged_ring_exchange_kernel.fn,
            autotune_effort="none",
            static_shapes=True,
            backend="gluon",
            megakernel_timeline=os.path.join(os.getcwd(), "unused"),
        )
        with (
            gluon_cpu_codegen(),
            self.assertLogs("helion.megakernel", "INFO") as logs,
        ):
            bound = kernel.bind(args)
            config = bound.config_spec.default_config().config
            config.pop("range_warp_specializes")
            code = bound.to_triton_code(helion.Config(**{**config, **TAGGED_RING}))
        log = "\n".join(logs.output)
        self.assertIn("]: 2 tasks\n", log)
        # 148 workers' 3 words and 12 tasks' 4, 3936 bytes: rounded up.
        self.assertIn("148 workers, 12 tasks, 4 KiB a launch", log)
        self.assertEqual(code.count("_helion_ring.tagged_load("), 2)
        name = "_helion_tagged_ring_exchange_kernel"
        launch = code.split(f"def {name}_launch(")[1].split("\n@")[0]
        clock = launch.index("%globaltimer")
        self.assertLess(clock, launch.index("tile_dependency_epoch ="))
        self.assertLess(clock, launch.index("tile_dependency_peer_epoch ="))
        self.assertNotIn("atomic_max", launch)
        lines = [line for line in code.splitlines() if "# src[" not in line]
        (ready,) = [i for i, line in enumerate(lines) if "* 4 + 2)" in line]
        self.assertRegex(lines[ready - 1], r"inband_word\w*, .* = tl\.inline_asm")


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

    def test_worker_registers_by_hand(self) -> None:

        parts = [ast.parse(f"def p{i}(): pass").body[0] for i in range(4)]
        self.assertEqual(
            gluon_ring._manual_registers([72, 104, 104, 80], parts), [72, 104, 104, 80]
        )
        with self.assertRaisesRegex(helion.exc.InvalidConfig, "3 values for 4"):
            gluon_ring._manual_registers([40, 96, 64], parts)
        # normalize: an empty list is the rule's; budgets outside setmaxnreg's
        # range (multiples of 8 in [24, 256]) are rejected.
        spec = ConfigSpec(backend=GluonBackend())
        config = {"worker_registers": []}
        spec.normalize(config)
        self.assertNotIn("worker_registers", config)
        # A hand override only: the search never proposes it.
        self.assertNotIn(
            "worker_registers", {d.name for d in spec.iter_search_dimensions()}
        )
        self.assertNotIn("worker_registers", spec.default_config().config)
        for wrong in ([20, 72], [60, 72], 64):
            with (
                self.subTest(worker_registers=wrong),
                self.assertRaises(helion.exc.InvalidConfig),
            ):
                spec.normalize({"worker_registers": wrong})

    def test_worker_registers_reach_their_partitions(self) -> None:
        """``worker_registers`` sets the workers' budgets, one per worker in the
        order the ring makes them (producers, then the MMA side), packed into
        warpgroups; an empty list is the rule's, another count or budgets that
        leave the body too few are rejected. A multi-warp worker (a partition
        that computes tensors) grows into what the capped body cannot take."""
        x = torch.empty([256, 512], dtype=torch.bfloat16)
        y = torch.empty([512, 128], dtype=torch.bfloat16)
        config = {
            **_RING,
            "block_sizes": [128, 64, 64],
            "range_warp_specializes": [None, True],  # a producer and the MMA side
        }
        with gluon_cpu_codegen():
            bound = dot_mm.bind((x, y))
            default = bound.to_triton_code(helion.Config(**config))
            producer, mma = sorted(_worker_budgets(default), key=lambda n: "mma" in n)
            self.assertTrue(producer.endswith("producer0") and mma.endswith("mma"))
            code = bound.to_triton_code(
                helion.Config(**config, worker_registers=[56, 72])
            )
            budgets = _worker_budgets(code)
            self.assertEqual((budgets[producer][1], budgets[mma][1]), (56, 72))
            # Packed: the higher budget goes first.
            self.assertEqual(list(budgets), [mma, producer])
            self.assertEqual(
                bound.to_triton_code(helion.Config(**config, worker_registers=[])),
                default,
            )
            with self.assertRaisesRegex(helion.exc.InvalidConfig, "3 values for 2"):
                bound.to_triton_code(
                    helion.Config(**config, worker_registers=[56, 72, 32])
                )
            # Two CTAs a SM: budgets the body cannot keep setmaxnreg's minimum beside.
            with self.assertRaisesRegex(
                helion.exc.InvalidConfig, "leave the body 24 registers a thread short"
            ):
                bound.to_triton_code(
                    helion.Config(
                        **{**config, "num_sm_multiplier": 2},
                        worker_registers=[256, 256],
                    )
                )
            # One stage lowers no ring: the budgets would name no workers.
            with self.assertRaisesRegex(helion.exc.InvalidConfig, "lowers no ring"):
                bound.to_triton_code(
                    helion.Config(
                        **{**config, "num_stages": 1}, worker_registers=[56, 72]
                    )
                )
            self.assertNotIn(
                "warp_specialize",
                bound.to_triton_code(
                    helion.Config(**{**config, "num_stages": 1}, worker_registers=[])
                ),
            )
            # A block-scaled dot whose scales reach tensor memory through
            # registers gives the MMA side 4 warps: it grows past the rule's 88.
            x = torch.empty([256, 256], dtype=torch.uint8)
            y = torch.empty([256, 128], dtype=torch.uint8)
            x_scale, y_scale = (
                torch.empty([rows, 32], dtype=torch.float8_e4m3fn)
                for rows in (256, 128)
            )
            nvfp4 = nvfp4_mm.bind((x, x_scale, y, y_scale))
            nvfp4_config = {**config, "block_sizes": [128, 64], "num_stages": 3}
            code = nvfp4.to_triton_code(helion.Config(**nvfp4_config))
            # Hand-set budgets are reordered, never grown.
            manual = nvfp4.to_triton_code(
                helion.Config(**nvfp4_config, worker_registers=[24, 88])
            )
            # A ring that falls back names its reason.
            with self.assertRaisesRegex(
                helion.exc.InvalidConfig,
                "lowers no ring for this config \\(its ring fell back: no TMA load",
            ):
                bound.to_triton_code(
                    helion.Config(
                        **{
                            **config,
                            "indexing": ["pointer", "pointer", "tensor_descriptor"],
                        },
                        worker_registers=[56, 72],
                    )
                )
        self.assertEqual(list(_worker_budgets(code).values()), [(4, 224), (1, 24)])
        self.assertEqual(list(_worker_budgets(manual).values()), [(4, 88), (1, 24)])

    def test_equal_budgets_share_warpgroups_and_the_rest_goes_to_workers(
        self,
    ) -> None:

        # The pollers' warpgroup also holds the 40-register producer (and the
        # padding the other poller): two warpgroups at 104.
        ring = [(2, False), (2, False), (2, False), (4, True)]
        regs = [40, 104, 104, 48]
        order = gluon_ring._warpgroup_order(4, ring, regs)
        self.assertEqual(order, [3, 1, 2, 0])
        ordered = [ring[i] for i in order]
        # The pollers together, the producer with the padding: the body would
        # take 320, past the cap; the lowest-budget workers take what it cannot.
        self.assertEqual(
            gluon_ring._surplus_registers(4, ordered, [regs[i] for i in order]),
            [80, 104, 104, 72],
        )
        self.assertEqual(
            gluon_ring._body_registers(4, ordered, [80, 104, 104, 72]), 256
        )
        # Hand-set budgets pack the same way: in the given order these leave the
        # body 16 registers (below setmaxnreg's minimum), packed 24.
        manual = [24, 32, 208, 256]
        order = gluon_ring._warpgroup_order(4, ring, manual)
        self.assertEqual(gluon_ring._body_registers(4, ring, manual), 16)
        self.assertEqual(
            gluon_ring._body_registers(
                4, [ring[i] for i in order], [manual[i] for i in order]
            ),
            24,
        )
        # The order is the best within each warp count, never worse than the
        # given one: the high-budget one-warp claimer (240) shares the padding's
        # warpgroup with the highest two-warp budget (48) so the two 24s share a
        # cheap one (highest budget first would pair a 24 with the claimer and
        # leave the body 8).
        claimed = [*ring, (1, False)]
        manual = [24, 24, 48, 216, 240]
        order = gluon_ring._warpgroup_order(4, claimed, manual)
        self.assertEqual(gluon_ring._body_registers(4, claimed, manual), 32)
        self.assertEqual(
            gluon_ring._body_registers(
                4, [claimed[i] for i in order], [manual[i] for i in order]
            ),
            32,
        )
        # A body below the cap keeps every register: nothing to hand out.
        self.assertEqual(
            gluon_ring._surplus_registers(8, ring, [40, 40, 40, 48]), [40, 40, 40, 48]
        )


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


class TestBarrierMode(TestCase):
    def test_other_roots_writes_stay_plain_loads(self) -> None:
        """Barrier mode orders the roots by a grid barrier only the body runs,
        with no dependency waits a producer could keep: the first root's
        output, which the second dot reads, stays a plain load in the body
        (no gated producer, so no skips), where the static pipeline streams it
        from the gated producer, which skips the first root's steps."""
        x = torch.empty([128, 8192], dtype=torch.bfloat16)
        w = torch.empty([8192, 128], dtype=torch.bfloat16)
        v = torch.empty([128, 128], dtype=torch.bfloat16)
        codes = {}
        with gluon_cpu_codegen():
            bound = mm_then_mm.bind((x, w, v))
            slots = len(bound.config_spec.range_warp_specialize)
            for pipeline in ("barrier", "static"):
                codes[pipeline] = bound.to_triton_code(
                    helion.Config(
                        **_RING,
                        block_sizes=[128, 128, 64, 128, 64, 64],
                        num_warps=4,
                        range_warp_specializes=[True] * slots,
                        cross_loop_pipeline=pipeline,
                    )
                )
        gated = "def _helion_mm_then_mm_producer1("
        barrier, static = codes["barrier"], codes["static"]
        self.assertIn("warp_specialize", barrier)
        self.assertIn("x_grid_barrier(", barrier)
        self.assertNotIn(gated, barrier)
        self.assertNotIn("ring_skip(", barrier)
        self.assertIn(gated, static)
        self.assertEqual(static.count("ring_skip("), 1)


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
