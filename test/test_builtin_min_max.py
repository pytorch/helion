from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl

if TYPE_CHECKING:
    from helion import Kernel


class TestBuiltinMinMax(RefEagerTestDisabled, TestCase):
    def _type_propagation_report(self, fn: Kernel, *args: object) -> str:
        return fn.bind(args)._debug_str()

    def test_gdn_kernel_matches_reference(self) -> None:
        @helion.kernel(autotune_effort="none")
        def helion_gdn_fwd_h_kernel(g_c):
            nchunks, chunk_size = g_c.shape
            chunk_size = hl.specialize(chunk_size)
            seqlen = chunk_size * nchunks
            out = torch.zeros(nchunks, dtype=g_c.dtype, device=g_c.device)
            for chunk in hl.grid(nchunks):
                last_idx = min((chunk + 1) * chunk_size, seqlen) - 1
                out[chunk] = g_c[last_idx // chunk_size, last_idx % chunk_size]
            return out

        def ref_gdn_fwd_h(g):
            nchunks, chunk_size = g.shape
            chunk_size = int(chunk_size)
            seqlen = chunk_size * nchunks
            out = torch.zeros(nchunks, dtype=g.dtype, device=g.device)
            for chunk in range(nchunks):
                last_idx = min((chunk + 1) * chunk_size, seqlen) - 1
                out[chunk] = g[last_idx // chunk_size, last_idx % chunk_size]
            return out

        device = DEVICE
        dtype = torch.float32
        nchunks, chunk_size = 3, 2
        g = torch.arange(nchunks * chunk_size, dtype=dtype, device=device).reshape(
            nchunks, chunk_size
        )

        code, helion_out = code_and_output(helion_gdn_fwd_h_kernel, (g,))
        ref_out = ref_gdn_fwd_h(g)

        self.assertTrue(
            torch.allclose(helion_out, ref_out, rtol=1e-3, atol=1e-3),
            msg=f"max diff {torch.max(torch.abs(helion_out - ref_out)).item()}",
        )
        self.assertExpectedJournal(code)

    def test_max_kernel_matches_reference(self) -> None:
        @helion.kernel(autotune_effort="none")
        def helion_max_kernel(x_c):
            nchunks, chunk_size = x_c.shape
            chunk_size = hl.specialize(chunk_size)
            seqlen = chunk_size * nchunks
            out = torch.zeros(nchunks, dtype=x_c.dtype, device=x_c.device)
            for chunk in hl.grid(nchunks):
                first_idx = chunk * chunk_size
                last_idx = max(first_idx, seqlen - 1)
                out[chunk] = x_c[last_idx // chunk_size, last_idx % chunk_size]
            return out

        def ref_max(x):
            nchunks, chunk_size = x.shape
            chunk_size = int(chunk_size)
            seqlen = chunk_size * nchunks
            out = torch.zeros(nchunks, dtype=x.dtype, device=x.device)
            for chunk in range(nchunks):
                first_idx = chunk * chunk_size
                last_idx = max(first_idx, seqlen - 1)
                out[chunk] = x[last_idx // chunk_size, last_idx % chunk_size]
            return out

        device = DEVICE
        dtype = torch.float32
        nchunks, chunk_size = 3, 2
        x = torch.arange(nchunks * chunk_size, dtype=dtype, device=device).reshape(
            nchunks, chunk_size
        )

        code, helion_out = code_and_output(helion_max_kernel, (x,))
        ref_out = ref_max(x)

        self.assertTrue(
            torch.allclose(helion_out, ref_out, rtol=1e-3, atol=1e-3),
            msg=f"max diff {torch.max(torch.abs(helion_out - ref_out)).item()}",
        )
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    import unittest

    unittest.main()
