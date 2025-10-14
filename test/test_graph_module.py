from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfRefEager
import helion.language as hl


@helion.kernel(autotune_effort="none")
def apply_graph_module(func_m, x):
    """Kernel that applies a GraphModule function to tensor elements."""
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = func_m(x[tile])
    return out


class TestGraphModule(RefEagerTestBase, TestCase):
    def test_graph_module_arg(self):
        """Test that GraphModule arguments work in kernels."""
        x = torch.randn(1000, device=DEVICE)

        # Create a GraphModule with a simple computation
        gm = torch.fx.symbolic_trace(lambda x: torch.sin(x + 1))

        # This should work - GraphModule is treated like a function call
        code, result = code_and_output(apply_graph_module, (gm, x))
        expected = torch.sin(x + 1)

        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(code)

    def test_graph_module_with_multiple_ops(self):
        """Test GraphModule with multiple operations."""
        x = torch.randn(512, device=DEVICE)

        # Create a more complex GraphModule
        def complex_func(x):
            return torch.cos(torch.relu(x * 2) + 1)

        gm = torch.fx.symbolic_trace(complex_func)

        code, result = code_and_output(apply_graph_module, (gm, x))
        expected = complex_func(x)

        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(code)

    def test_graph_module_specialization(self):
        """Test that different GraphModules get specialized separately."""
        x = torch.randn(256, device=DEVICE)

        # Create two different GraphModules
        gm1 = torch.fx.symbolic_trace(lambda x: torch.sin(x))
        gm2 = torch.fx.symbolic_trace(lambda x: torch.cos(x))

        # Each should get its own specialization
        code1, result1 = code_and_output(apply_graph_module, (gm1, x))
        code2, result2 = code_and_output(apply_graph_module, (gm2, x))

        torch.testing.assert_close(result1, torch.sin(x))
        torch.testing.assert_close(result2, torch.cos(x))

    @skipIfRefEager("doesn't make required call")
    def test_graph_module_with_unsupported_ops(self):
        """Test that GraphModules with unsupported ops raise an error."""
        x = torch.randn(128, device=DEVICE)

        # Create a module with call_module (unsupported)
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(128, 128)

            def forward(self, x):
                return self.linear(x)

        module = MyModule()
        gm = torch.fx.symbolic_trace(module)

        # This should raise an error due to call_module op
        with self.assertRaises(helion.exc.GraphModuleUnsupportedOps) as cm:
            apply_graph_module(gm, x)

        self.assertIn("call_module", str(cm.exception))

    def test_graph_module_caching(self):
        """Test that GraphModule hash caching works correctly."""
        x = torch.randn(256, device=DEVICE)

        # Create a GraphModule
        gm = torch.fx.symbolic_trace(lambda x: torch.sin(x))

        # Call the kernel twice with the same GraphModule
        # Should use cached hash the second time
        code1, result1 = code_and_output(apply_graph_module, (gm, x))
        code2, result2 = code_and_output(apply_graph_module, (gm, x))

        torch.testing.assert_close(result1, torch.sin(x))
        torch.testing.assert_close(result2, torch.sin(x))

        # Same GraphModule should produce same code
        self.assertEqual(code1, code2)


if __name__ == "__main__":
    unittest.main()
