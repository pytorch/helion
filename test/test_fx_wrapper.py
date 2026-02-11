from __future__ import annotations

import unittest

import torch
from torch.fx import symbolic_trace

import helion
from helion._compat import requires_torch_version
from helion._compiler._fx.wrapper import _is_fx_tracing
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
import helion.language as hl


@helion.kernel(autotune_effort="none")
def k_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


class TestCreateFxProxyForKernel(RefEagerTestDisabled, TestCase):
    """Tests for create_fx_proxy_for_kernel."""

    def setUp(self) -> None:
        super().setUp()
        if not requires_torch_version("2.11"):
            self.skipTest("HOP support requires PyTorch >= 2.11")

        from helion._compiler._dynamo.higher_order_ops import (
            helion_kernel_side_table,
        )

        self.side_table = helion_kernel_side_table
        self.side_table.reset_table()

    def _make_wrapper(self, kernel: helion.Kernel) -> object:
        """Build a callable that dispatches to create_fx_proxy_for_kernel
        when it receives Proxy args, otherwise calls the kernel normally."""
        from helion._compiler._fx.wrapper import create_fx_proxy_for_kernel

        class _Wrapper:
            def __init__(self, k: helion.Kernel) -> None:
                self._kernel = k

            def __call__(self, *args: object) -> object:
                if _is_fx_tracing(args):
                    # TODO(gmagogsfm): Support multiple outputs.
                    return create_fx_proxy_for_kernel(
                        self._kernel, None, args, {}
                    )[0]
                return self._kernel(*args)

        return _Wrapper(kernel)

    def test_trace_and_execute(self) -> None:
        """Traced graph produces correct results when executed."""
        from helion._compiler._dynamo.higher_order_ops import (
            helion_kernel_wrapper_mutation,
        )

        wrapper = self._make_wrapper(k_add)

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return wrapper(x, y)

        gm = symbolic_trace(f)

        # Verify HOP node exists
        hop_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is helion_kernel_wrapper_mutation
        ]
        self.assertEqual(len(hop_nodes), 1)

        # Verify execution correctness
        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        torch.testing.assert_close(gm(x, y), x + y)


if __name__ == "__main__":
    unittest.main()
