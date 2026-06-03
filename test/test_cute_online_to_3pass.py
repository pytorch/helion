"""Tests for the CuTe ``online_to_3pass`` AST rewrite.

The CuTe backend pre-processes the user's AST before tracing.  When
the body matches the canonical two-loop online softmax pattern
(a running ``mi``/``di`` update sweep followed by a normalize sweep)
AND the reduction-axis extent is at or above the cutoff (default 2048),
the pass rewrites the body into THREE inner ``for tile_n`` loops:

  1. max-only sweep
  2. sum-only sweep
  3. consume sweep (unchanged)

The 3-pass form's two reductions are independent (no rescale data
dependency between them) and compile to materially faster code on
CuTe for large-N inputs — the autotuner can now pick layouts that
benefit from the parallelism the online form blocked.

Lives in ``helion/_compiler/cute/online_to_3pass.py``.
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


@helion.kernel(backend="cute")
def _online_two_pass_kernel(x: torch.Tensor) -> torch.Tensor:
    """The exact canonical online two-pass softmax pattern the rewrite
    detects: outer tile loop holding ``mi`` and ``di``, an online update
    sweep, then a normalize sweep.
    """
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


@onlyBackends(["cute"])
class TestCuteOnlineTo3PassRewrite(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._env_to_restore: dict[str, str | None] = {}
        # The Kernel decorator caches bound kernels by input signature
        # (shape + dtype + stride).  Each test in this class flips env
        # vars that change the compiler pipeline behavior, so we must
        # invalidate the cache before every test or stale compilations
        # leak across cases.
        _online_two_pass_kernel.reset()

    def tearDown(self) -> None:
        for name, prev in self._env_to_restore.items():
            if prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prev
        _online_two_pass_kernel.reset()
        super().tearDown()

    def _set_env(self, name: str, value: str | None) -> None:
        if name not in self._env_to_restore:
            self._env_to_restore[name] = os.environ.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def _generated_loop_count(self, code: str) -> int:
        """Count the number of inner ``for tile_offset_N in range(`` sites in
        the generated CuTe source.  Each inner sweep yields one of these
        loops, so this is the number of inner sweeps the rewrite emitted.
        """
        return len(re.findall(r"for tile_offset_\d+ in range\(", code))

    def test_rewrite_fires_on_canonical_pattern(self) -> None:
        """For (4096, 12672) the rewrite MUST fire and the generated
        kernel MUST have THREE inner ``for tile_offset`` loops (max,
        sum, consume) instead of TWO.
        """
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _online_two_pass_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # 3 inner sweeps now (was 2 in the original online form).
        self.assertEqual(self._generated_loop_count(code), 3)

    def test_shape_gate_skips_small_n(self) -> None:
        """For (4096, 256) the rewrite MUST be skipped by the
        reduction-axis-extent gate (256 < default cutoff 2048), so the
        generated kernel still has TWO inner sweeps (the original
        online form).
        """
        x = torch.randn(4096, 256, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _online_two_pass_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        self.assertEqual(self._generated_loop_count(code), 2)

    def test_disable_env_skips_rewrite(self) -> None:
        """``HELION_DISABLE_ONLINE_TO_3PASS=1`` MUST skip the pass even
        when the shape and pattern would otherwise match — the kernel
        falls back to the original 2-loop online form.
        """
        self._set_env("HELION_DISABLE_ONLINE_TO_3PASS", "1")
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _online_two_pass_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        self.assertEqual(self._generated_loop_count(code), 2)

    def test_min_n_env_override(self) -> None:
        """``HELION_ONLINE_TO_3PASS_MIN_N=0`` MUST always fire the rewrite
        regardless of reduction-axis extent (useful for A/B sweeps).
        Verify by running a small-N shape which the default gate would
        skip and observe the 3-loop form.
        """
        self._set_env("HELION_ONLINE_TO_3PASS_MIN_N", "0")
        x = torch.randn(4096, 256, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _online_two_pass_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        self.assertEqual(self._generated_loop_count(code), 3)

    def test_correctness_after_rewrite(self) -> None:
        """Tight-tolerance correctness check on multiple large-N
        shapes.  Guards against any silent numeric drift introduced by
        the rewrite (the 3-pass and online forms have different
        rounding paths in fp16 — the diff must stay inside the existing
        tolerance).
        """
        for n_val in (4096, 6400, 12672, 16384):
            x = torch.randn(4096, n_val, device=DEVICE, dtype=HALF_DTYPE)
            _, out = code_and_output(
                _online_two_pass_kernel,
                (x,),
                block_sizes=[1, 128],
                num_threads=[0, 32],
                cute_vector_widths=[1, 4],
            )
            ref = torch.nn.functional.softmax(x, dim=1)
            torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
