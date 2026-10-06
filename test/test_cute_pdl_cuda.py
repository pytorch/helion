"""GPU semantics of ``cute_pdl`` (programmatic dependent launch).

A programmatic dependent launch starts as soon as the kernel ahead of it in
the stream releases its dependents (``griddepcontrol.launch_dependents``), so
a producer that releases them at entry and writes late exposes every read a
dependent issues ahead of its ``griddepcontrol.wait``.  Such a producer (a
Triton kernel) runs ahead of the knob-on kernels here, which must match the
knob-off kernels bit for bit, in the stream and inside a CUDA graph, where
the launch attribute becomes a programmatic edge.  A control that launches
the knob-on module without its wait has to read stale data: that proves the
producer really overlaps the launch and the wait is load-bearing.  The
pure-launch chain at the end shows why a kernel that releases its dependents
at entry has to stay an ordinary launch (the materialized-fission producer,
which ``_stage_config`` keeps away from the knob).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from torch._inductor.codecache import PyCodeCache

pytest.importorskip("triton")

import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents
from triton.language.extra.cuda import gdc_wait

from test._cute_vloop_sink_kernels import col_reduce_max_dynamic
from test._cute_vloop_sink_kernels import col_reduce_sum_static

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion.autotuner.benchmarking import _make_cudagraph_replay

if TYPE_CHECKING:
    from collections.abc import Callable

cutlass = pytest.importorskip("cutlass")

_COLUMN = {
    "block_sizes": [1024, 16],
    "num_threads": [128, 4],
    "cute_vector_widths": [1, 4],
    "cute_lane_layouts": ["strided", "blocked"],
    "cute_vloop_sink": True,
    "cute_lane_unroll": 8,
}
_WAIT = "cute.arch.griddepcontrol_wait()"
_ATTRIBUTE = "._helion_cute_use_pdl = True"
_PRODUCER_GRID = 128
_BLOCK = 1024
# The positive control's producer spins this many rounds ahead of every store
# (about a millisecond per run) so a dependent launched without the wait is
# certain to read ``x`` before the producer rewrites it; at the default 64
# rounds the producer can finish inside the consumer's host launch latency
# and the control is a coin flip.
_CONTROL_DELAY = 4096


@triton.jit
def _late_writer(
    src_ptr,
    dst_ptr,
    n,
    per_program,
    delay,
    RELEASE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copies ``src`` to ``dst`` late: releases its dependents at entry (when
    ``RELEASE``) and spins ``delay`` rounds ahead of every store."""
    if RELEASE:
        gdc_launch_dependents()
    base = tl.program_id(0) * per_program
    for chunk in range(0, per_program, BLOCK):
        offsets = base + chunk + tl.arange(0, BLOCK)
        mask = offsets < n
        values = tl.load(src_ptr + offsets, mask=mask)
        spin = tl.zeros([BLOCK], dtype=tl.float32)
        for _ in range(delay):
            spin = tl.sin(spin + 0.5)
        values = values + (spin * 0.0).to(values.dtype)
        tl.store(dst_ptr + offsets, values, mask=mask)


@triton.jit
def _middle(flag_ptr, RELEASE_FIRST: tl.constexpr):
    """Waits on its predecessor and releases its dependents, in either order."""
    if RELEASE_FIRST:
        gdc_launch_dependents()
    gdc_wait()
    tl.atomic_add(flag_ptr, 1)
    if not RELEASE_FIRST:
        gdc_launch_dependents()


@triton.jit
def _copy_without_wait(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out_ptr + offsets, tl.load(x_ptr + offsets, mask=mask), mask=mask)


def _produce(
    src: torch.Tensor, dst: torch.Tensor, *, delay: int = 64, release: bool = True
) -> None:
    n = dst.numel()
    programs = (n + _PRODUCER_GRID - 1) // _PRODUCER_GRID
    per_program = (programs + _BLOCK - 1) // _BLOCK * _BLOCK
    _late_writer[(_PRODUCER_GRID,)](
        src.view(-1),
        dst.view(-1),
        n,
        per_program,
        delay,
        RELEASE=release,
        BLOCK=_BLOCK,
    )


def _build(
    kernel: helion.Kernel, x: torch.Tensor, **config: object
) -> tuple[helion.runtime.kernel.BoundKernel, Callable[..., torch.Tensor], str]:
    bound = kernel.bind((x,))
    normalized = bound.config_spec.normalized_config(helion.Config(**config))
    return bound, bound.compile_config(normalized), bound.to_code(normalized)


def _attribute_only_control(
    bound: helion.runtime.kernel.BoundKernel, on: str
) -> Callable[..., torch.Tensor]:
    """The knob-on module minus its wait: a programmatic dependent launch
    that reads its input right away."""
    lines = on.splitlines()
    kept = [line for line in lines if _WAIT not in line]
    assert len(kept) == len(lines) - 1
    source = "\n".join(kept) + "\n"
    module = PyCodeCache.load(source)
    bound.env.backend.annotate_compiled_module(module, source, bound.kernel.name)
    return getattr(module, bound.kernel.name)


def _stale_runs(
    fn: Callable[..., torch.Tensor],
    x: torch.Tensor,
    sources: list[torch.Tensor],
    references: list[torch.Tensor],
    iterations: int,
    *,
    delay: int = 64,
) -> int:
    """Runs producer -> ``fn`` back to back and counts runs whose output
    differs (bitwise) from the reference of the source just produced.

    ``delay`` is the producer's spin count ahead of every store: the
    zero-stale assertions hold at any delay, while the positive control (a
    programmatic launch without the wait) needs a producer that outlives the
    consumer's host launch latency, or the race it demonstrates may never
    be lost."""
    stale = torch.zeros((), dtype=torch.int64, device=DEVICE)
    for i in range(iterations):
        j = i & 1
        _produce(sources[j], x, delay=delay)
        stale += (fn(x) != references[j]).any()
    torch.cuda.synchronize()
    return int(stale)


@onlyBackends(["cute"])
class TestCutePdl(TestCase):
    def test_knob_on_matches_knob_off_behind_a_producer_that_releases_at_entry(
        self,
    ) -> None:
        iterations = 100
        # The max kernel writes into ``torch.empty``: nothing sits between
        # the producer and the reduction, so a missing wait reads stale
        # data (the control proves it).  The sum zero-fills its output first;
        # that fill node is what the knob's launch overlaps.
        for kernel, shape, control in (
            (col_reduce_max_dynamic, (2000, 1008), True),
            (col_reduce_sum_static, (1024, 1024), False),
        ):
            x = torch.empty(*shape, dtype=torch.bfloat16, device=DEVICE)
            sources = [
                torch.randn(*shape, dtype=torch.bfloat16, device=DEVICE),
                torch.randn(*shape, dtype=torch.bfloat16, device=DEVICE) * 7,
            ]
            bound, off, off_code = _build(kernel, x, **_COLUMN)
            _bound, on, on_code = _build(kernel, x, **_COLUMN, cute_pdl=True)
            self.assertNotIn(_WAIT, off_code)
            self.assertNotIn(_ATTRIBUTE, off_code)
            self.assertEqual(on_code.count(_WAIT), 1)
            self.assertEqual(on_code.count(_ATTRIBUTE), 1)
            references = []
            for source in sources:
                _produce(source, x)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(x, source))
                references.append(off(x).clone())
            self.assertEqual(_stale_runs(off, x, sources, references, iterations), 0)
            self.assertEqual(_stale_runs(on, x, sources, references, iterations), 0)
            # One graph holding producer -> kernel -> producer -> kernel: the
            # launch attribute becomes a programmatic edge inside the graph.
            side = torch.cuda.Stream()
            with torch.cuda.stream(side):
                _produce(sources[0], x)
                on(x)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                _produce(sources[0], x)
                out0 = on(x)
                _produce(sources[1], x)
                out1 = on(x)
            for _ in range(iterations // 4):
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(out0, references[0]))
                self.assertTrue(torch.equal(out1, references[1]))
            if control:
                attribute_only = _attribute_only_control(bound, on_code)
                self.assertGreater(
                    _stale_runs(
                        attribute_only,
                        x,
                        sources,
                        references,
                        iterations,
                        delay=_CONTROL_DELAY,
                    ),
                    0,
                )

    def test_graph_replay_sees_the_fill_before_the_kernel(self) -> None:
        x = torch.randn(1024, 1024, dtype=torch.bfloat16, device=DEVICE)
        bound = col_reduce_sum_static.bind((x,))
        config = bound.config_spec.normalized_config(
            helion.Config(**_COLUMN, cute_pdl=True)
        )
        fn = bound.compile_config(config)
        # ``col_reduce_sum`` zero-fills its output in a kernel that precedes
        # the reduction inside the captured graph; the reduction's launch
        # overlaps that fill and its wait orders the two.
        replay = _make_cudagraph_replay(lambda: fn(x))
        expected = x.float().sum(0).to(torch.bfloat16)
        for _ in range(3):
            torch.testing.assert_close(replay(), expected, atol=2e-2, rtol=2e-2)

    def test_a_kernel_that_releases_its_dependents_at_entry_must_be_an_ordinary_launch(
        self,
    ) -> None:
        # Producer (releases at entry, writes x late) -> middle kernel ->
        # copy of x launched as a dependent with no wait of its own, the
        # shape of the tcgen05 consumer's ordinary-operand loads ahead of
        # its wait.
        n = 512 * 1024
        x = torch.empty(n, dtype=torch.bfloat16, device=DEVICE)
        out = torch.empty_like(x)
        flag = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        sources = [
            torch.randn(n, dtype=torch.bfloat16, device=DEVICE),
            torch.randn(n, dtype=torch.bfloat16, device=DEVICE),
        ]
        iterations = 50

        def stale_runs(*, middle_is_dependent: bool, release_first: bool) -> int:
            stale = torch.zeros((), dtype=torch.int64, device=DEVICE)
            for i in range(iterations):
                j = i & 1
                _produce(sources[j], x, delay=256)
                _middle[(_PRODUCER_GRID,)](
                    flag, RELEASE_FIRST=release_first, launch_pdl=middle_is_dependent
                )
                _copy_without_wait[(n // _BLOCK,)](
                    x, out, n, BLOCK=_BLOCK, launch_pdl=True
                )
                stale += (out != sources[j]).any()
            torch.cuda.synchronize()
            return int(stale)

        # The fission producer's shape: an ordinary launch that releases at
        # entry.  It starts after the producer completes, so its dependents
        # cannot start earlier either.
        self.assertEqual(stale_runs(middle_is_dependent=False, release_first=True), 0)
        # The shape ``cute_pdl`` gave it before ``_stage_config`` kept the
        # knob away: launched as a dependent it starts while the producer
        # still runs, releases at once, and the copy reads x unwritten.
        self.assertGreater(stale_runs(middle_is_dependent=True, release_first=True), 0)
        # A dependent launch that waits before it releases is safe.
        self.assertEqual(stale_runs(middle_is_dependent=True, release_first=False), 0)
