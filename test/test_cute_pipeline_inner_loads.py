"""Tests for the CuTe ``pipeline_inner_loads`` AST pass.

Software-pipelines the per-iteration vec load by one stage:

- Pre-issues the iter 0 vec load above the inner loop (prologue).
- Inside the body issues the NEXT iter's load BEFORE the current
  iter's compute body runs.
- Drains the last prefetched load at end-of-loop.

The SASS scheduler then has multiple ``ld.global`` instructions in
flight per warp, hiding HBM round-trip latency on kernels where the
inner-loop body has a sequential dependency chain that otherwise
prevents the compiler from issuing the next load early.

Measured ~18-37% drop in
``smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio``
on the (4096, 12672) softmax shape (NCU) and +20% wall-clock gain on
wide-N shapes like (4096, 10240) and (4096, 16384).

P19 adds an opt-in loop-carried-scalar-write gate
(``HELION_LOAD_PIPELINE_CARRIED_ONLY=1``) that restricts pipelining to
outer loops with a loop-carried scalar write (reduction accumulators
like ``mi``/``di``). Skips pure stream-store loops where the inter-iter
slack isn't useful. Default remains pipeline-everything.

Lives in ``helion/_compiler/cute/pipeline_inner_loads.py``.
"""

from __future__ import annotations

import os
import re

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


@pytest.fixture(autouse=True)
def _disable_online_to_3pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests in this file pin codegen details of the ORIGINAL online
    two-pass form.  The ``online_to_3pass`` rewrite would change them,
    so disable it here; the rewrite itself is covered in
    ``test_cute_online_to_3pass.py``.
    """
    monkeypatch.setenv("HELION_DISABLE_ONLINE_TO_3PASS", "1")


@helion.kernel(backend="cute")
def _reduction_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
    return out


def _shadow_var_count(code: str, kind: str) -> int:
    """Count ``_pipe_<kind>_N = `` occurrences (prologue + per-iter
    prefetch assignments, so 2 per pipelined load site)."""
    return len(re.findall(rf"_pipe_{kind}_\d+ = ", code))


@onlyBackends(["cute"])
class TestCutePipelineInnerLoads(TestCase):
    """Default behavior: pipeline every qualifying inner load."""

    def test_pipeline_fires_on_two_pass_pattern(self) -> None:
        """Both the reduce and consume sweeps of the two-pass kernel
        match the (single inner-lane loop, single load) shape so the
        pass rewrites both sweeps.  Each rewrite emits a prologue
        snapshot and a per-iter prefetch — 2 ``_pipe_lane_base_*`` assigns
        and 2 ``_pipe_load_*`` assigns per pipelined site, for 4 total
        of each across the two sweeps.
        """
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # 2 sweeps * 2 assigns each = 4 _pipe_lane_base_* writes and
        # 4 _pipe_load_* writes.
        self.assertEqual(_shadow_var_count(code, "lane_base"), 4)
        self.assertEqual(_shadow_var_count(code, "load"), 4)
        # The snapshot ``lane_base_<N> = _pipe_lane_base_<M>`` form must
        # appear inside the loop body — that's the substitution that
        # gives the SASS scheduler one full iter of slack to issue the
        # next load.
        self.assertRegex(code, r"lane_base_\d+ = _pipe_lane_base_\d+")
        self.assertRegex(code, r"_tile_unroll_vec_\d+_\d+ = _pipe_load_\d+")

    def test_pipeline_skips_when_lane_reps_greater_than_one(self) -> None:
        """V=4 with block_size=256 forces the inner lane loop to
        ``for lane in range(2):`` (block=256, V=4 → 2 lanes per
        thread).  The pipeline transform must SKIP this case: the
        per-iter prefetch advances ``_pipe_lane_base`` to the next
        outer iter but uses the CURRENT ``lane`` index, which would
        corrupt the snapshot read by the NEXT lane index in the same
        outer iter.
        """
        x = torch.randn(4096, 8192, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 256],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # lane_reps == 2, so the pass bails -- no _pipe_* vars.
        self.assertNotIn("_pipe_lane_base_", code)
        self.assertNotIn("_pipe_load_", code)

    def test_pipeline_disable_env(self) -> None:
        """``HELION_DISABLE_LOAD_PIPELINE=1`` disables the pass for
        experimentation.  The kernel must still compile and produce
        correct output without the pipeline rewrite.
        """
        old = os.environ.get("HELION_DISABLE_LOAD_PIPELINE")
        os.environ["HELION_DISABLE_LOAD_PIPELINE"] = "1"
        try:
            x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
            code, out = code_and_output(
                _reduction_kernel,
                (x,),
                block_sizes=[1, 128],
                num_threads=[0, 32],
                cute_vector_widths=[1, 4],
            )
            ref = torch.nn.functional.softmax(x, dim=1)
            torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
            # No pipeline vars when the pass is disabled.
            self.assertNotIn("_pipe_lane_base_", code)
            self.assertNotIn("_pipe_load_", code)
        finally:
            if old is None:
                os.environ.pop("HELION_DISABLE_LOAD_PIPELINE", None)
            else:
                os.environ["HELION_DISABLE_LOAD_PIPELINE"] = old


@onlyBackends(["cute"])
class TestCutePipelineCarriedOnlyGate(TestCase):
    """Opt-in carried-only gate (``HELION_LOAD_PIPELINE_CARRIED_ONLY=1``).

    The default pipelines BOTH sweeps of a two-pass softmax: the reduce
    sweep (which writes the ``mi``/``di`` accumulator) and the consume
    sweep (a pure stream-store). With the gate enabled, only the reduce
    sweep is pipelined; the consume sweep falls back to inline loads.
    """

    def setUp(self) -> None:
        super().setUp()
        self._env_to_restore: dict[str, str | None] = {}

    def tearDown(self) -> None:
        for name, prev in self._env_to_restore.items():
            if prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prev
        super().tearDown()

    def _set_env(self, name: str, value: str | None) -> None:
        """Restore-aware env var setter."""
        if name not in self._env_to_restore:
            self._env_to_restore[name] = os.environ.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def test_carried_only_pipelines_reduce_sweep(self) -> None:
        """With the gate enabled, the reduce sweep (which writes ``mi``
        and ``di`` — both defined in the outer function-body scope
        before the loop) MUST still be pipelined.
        """
        self._set_env("HELION_LOAD_PIPELINE_CARRIED_ONLY", "1")
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # The reduce sweep MUST still be pipelined — there's a
        # loop-carried scalar write to ``mi`` (pre-rename: ``v_1_0
        # = v_1``) so the gate fires positive.
        self.assertIn("_pipe_lane_base_", code)
        self.assertIn("_pipe_load_", code)
        # And the per-iter snapshot ``lane_base_<N> = _pipe_lane_base_<M>``
        # form appears inside the reduce loop body.
        self.assertRegex(code, r"lane_base_\d+ = _pipe_lane_base_\d+")

    def test_carried_only_skips_consume_sweep(self) -> None:
        """With the gate enabled, the consume sweep (which only stores
        into ``out[k]`` and never writes a scalar accumulator in the
        outer scope) MUST NOT be pipelined.

        The reduce-sweep pipeline writes ``_pipe_lane_base_0`` and
        ``_pipe_load_0``; the consume sweep would have written
        ``_pipe_lane_base_1`` and ``_pipe_load_1`` under the default
        behavior.  Under the gate, only the ``_0`` indices appear.
        """
        self._set_env("HELION_LOAD_PIPELINE_CARRIED_ONLY", "1")
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, _ = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        # ONLY the reduce-sweep pipe (index 0) — no index 1.
        self.assertIn("_pipe_load_0", code)
        self.assertNotIn("_pipe_load_1", code)
        self.assertNotIn("_pipe_lane_base_1", code)
        # And the consume sweep retains its original inline
        # ``cute.arch.load`` (no snapshot/prefetch rewrite).
        self.assertIn("_tile_unroll_vec_1_1 = cute.arch.load(", code)

    def test_pipeline_all_overrides_carried_only(self) -> None:
        """``HELION_LOAD_PIPELINE_ALL=1`` must explicitly override the
        gate even when ``HELION_LOAD_PIPELINE_CARRIED_ONLY`` is also
        set, restoring the pre-P19 "pipeline every qualifying loop"
        behavior.
        """
        self._set_env("HELION_LOAD_PIPELINE_CARRIED_ONLY", "1")
        self._set_env("HELION_LOAD_PIPELINE_ALL", "1")
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, _ = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        # Both sweeps pipelined → both _pipe_load_0 and _pipe_load_1
        # appear.
        self.assertIn("_pipe_load_0", code)
        self.assertIn("_pipe_load_1", code)
