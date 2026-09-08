"""Tests for the CuTe ``"streaming"`` load eviction policy.

``"streaming"`` lowers to the ``ld.global.cs`` cache operator (``cop='cs'``
on ``cute.arch.load``): evict-first at both L1 and L2, so single-use
streaming reads stop displacing useful L2 lines.  Scalar sites route
through ``cute.arch.load`` (``(ptr).load()`` has no hint kwargs), and the
cross-sweep load fuser must keep matching the hinted scalar form against
its unhinted twin in the consume sweep.

Lives in ``helion/_compiler/cute/memory_ops.py`` (emission),
``helion/language/memory_ops.py`` (scalar form), and
``helion/_compiler/cute/fuse_two_pass_loads.py`` (matching).
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


@helion.kernel(
    backend="cute",
    config={
        "block_sizes": [1],
        "reduction_loops": [1024],
        "load_eviction_policies": ["streaming", "streaming", "streaming"],
    },
)
def _logsumexp_scalar_kernel(x: torch.Tensor) -> torch.Tensor:
    m, _n = x.shape
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        rows = x[tile_m, :].to(torch.float32)
        max_x = torch.amax(rows, dim=-1)
        sum_exp = torch.sum(torch.exp(rows - max_x[:, None]), dim=-1)
        out[tile_m] = max_x + torch.log(sum_exp)
    return out


@helion.kernel(
    backend="cute",
    config={
        "block_sizes": [1],
        "reduction_loops": [512],
        "num_threads": [0, 64],
        "cute_vector_widths": [8, 1],
        "load_eviction_policies": ["streaming", "streaming", "streaming"],
    },
)
def _logsumexp_vec_kernel(x: torch.Tensor) -> torch.Tensor:
    m, _n = x.shape
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        rows = x[tile_m, :].to(torch.float32)
        max_x = torch.amax(rows, dim=-1)
        sum_exp = torch.sum(torch.exp(rows - max_x[:, None]), dim=-1)
        out[tile_m] = max_x + torch.log(sum_exp)
    return out


@onlyBackends(["cute"])
class TestCuteStreamingLoads(TestCase):
    def test_streaming_in_choices(self) -> None:
        from helion.autotuner.config_spec import get_valid_eviction_policies

        self.assertIn("streaming", get_valid_eviction_policies("cute"))
        self.assertNotIn("streaming", get_valid_eviction_policies("triton"))

    def test_scalar_streaming_load_keeps_fusion(self) -> None:
        x = torch.randn(64, 2048, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(_logsumexp_scalar_kernel, (x,))
        # Scalar sites route through cute.arch.load to carry the hint...
        self.assertIn("cop='cs'", code)
        # ...and the cross-sweep register cache must still fire (the hinted
        # reduce-sweep load matches the unhinted consume-sweep load).
        self.assertIn("_fuse_cache_0", code)
        torch.testing.assert_close(
            out, torch.logsumexp(x, dim=-1), rtol=1e-3, atol=1e-3
        )

    def test_vec_streaming_load(self) -> None:
        x = torch.randn(64, 4096, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(_logsumexp_vec_kernel, (x,))
        self.assertIn("cop='cs'", code)
        torch.testing.assert_close(
            out, torch.logsumexp(x.float(), dim=-1), rtol=1e-3, atol=1e-3
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
