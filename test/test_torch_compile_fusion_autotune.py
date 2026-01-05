"""Tests for autotuning with prologue/epilogue fusion in torch.compile.

These tests verify that:
1. DURING autotuning, the generated Triton code already contains the fused operations
2. Multiple configs are searched during autotuning, all with fusion applied
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
import helion.language as hl


# Kernel with quick autotune effort - should autotune but quickly
@helion.kernel(
    static_shapes=True, allow_torch_compile_fusion=True, autotune_effort="quick"
)
def elementwise_add_autotune(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Simple elementwise add kernel that should be autotuned."""
    m, n = x.size()
    out = torch.empty_like(x)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :]
        y_tile = y[tile_m, :]
        out[tile_m, :] = x_tile + y_tile

    return out


class TestFusionDuringAutotuning(RefEagerTestDisabled, TestCase):
    """Test that DURING autotuning, generated code already has fusion ops."""

    def setUp(self):
        super().setUp()
        # Reset dynamo and clear kernel caches between tests
        torch._dynamo.reset()
        # Clear the bound kernel cache
        elementwise_add_autotune._bound_kernels.clear()

    def test_epilogue_fusion_in_code_during_autotuning(self):
        """Test that code generated DURING autotuning contains epilogue fusion."""
        m, n = 64, 64
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        y = torch.randn(m, n, device=DEVICE, dtype=torch.float32)

        # Track all Triton code generated during autotuning
        generated_codes_during_autotuning: list[str] = []

        from helion.runtime.kernel import BoundKernel

        original_compile_config = BoundKernel.compile_config

        def tracking_compile_config(
            self, config=None, *, allow_print=True, template_buffer=None
        ):
            # Call original to get the compiled function
            result = original_compile_config(
                self,
                config,
                allow_print=allow_print,
                template_buffer=template_buffer,
            )

            # If template_buffer is provided, this is fusion code generation during autotuning
            if template_buffer is not None:
                # Get the generated Triton code
                triton_code = self.to_triton_code(
                    config if config is not None else self._require_implicit_config(),
                    emit_repro_caller=False,
                    template_buffer=template_buffer,
                )
                generated_codes_during_autotuning.append(triton_code)

            return result

        def f(x, y):
            out = elementwise_add_autotune(x, y)
            # Epilogue: relu (should be fused)
            return torch.relu(out)

        with patch.object(BoundKernel, "compile_config", tracking_compile_config):
            compiled_f = torch.compile(f)
            result = compiled_f(x, y)

        # Verify correctness
        expected = torch.relu(x + y)
        torch.testing.assert_close(result, expected)

        # Verify that we captured code during autotuning
        self.assertGreater(
            len(generated_codes_during_autotuning),
            0,
            "Should have captured Triton code generated during autotuning",
        )

        # Verify EACH generated code during autotuning contains the epilogue
        for i, code in enumerate(generated_codes_during_autotuning):
            # relu is typically implemented as tl.maximum(x, 0)
            has_relu = "maximum" in code or "relu" in code.lower()
            self.assertTrue(
                has_relu,
                f"Code #{i} generated during autotuning should contain relu epilogue, "
                f"but got:\n{code[:1500]}...",
            )

    def test_prologue_fusion_in_code_during_autotuning(self):
        """Test that code generated DURING autotuning contains prologue fusion."""
        m, n = 64, 64
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        y = torch.randn(m, n, device=DEVICE, dtype=torch.float32)

        # Track all Triton code generated during autotuning
        generated_codes_during_autotuning: list[str] = []

        from helion.runtime.kernel import BoundKernel

        original_compile_config = BoundKernel.compile_config

        def tracking_compile_config(
            self, config=None, *, allow_print=True, template_buffer=None
        ):
            result = original_compile_config(
                self,
                config,
                allow_print=allow_print,
                template_buffer=template_buffer,
            )

            if template_buffer is not None:
                triton_code = self.to_triton_code(
                    config if config is not None else self._require_implicit_config(),
                    emit_repro_caller=False,
                    template_buffer=template_buffer,
                )
                generated_codes_during_autotuning.append(triton_code)

            return result

        def f(x, y):
            # Prologue: multiply by 2.0 (should be fused)
            x_scaled = x * 2.0
            out = elementwise_add_autotune(x_scaled, y)
            return out

        with patch.object(BoundKernel, "compile_config", tracking_compile_config):
            compiled_f = torch.compile(f)
            result = compiled_f(x, y)

        # Verify correctness
        expected = (x * 2.0) + y
        torch.testing.assert_close(result, expected)

        # Verify that we captured code during autotuning
        self.assertGreater(
            len(generated_codes_during_autotuning),
            0,
            "Should have captured Triton code generated during autotuning",
        )

        # Verify EACH generated code during autotuning contains the prologue
        for i, code in enumerate(generated_codes_during_autotuning):
            # The prologue multiplies by 2.0
            has_prologue = "2.0" in code or "* 2" in code
            self.assertTrue(
                has_prologue,
                f"Code #{i} generated during autotuning should contain prologue (* 2.0), "
                f"but got:\n{code[:1500]}...",
            )

    def test_captured_buffer_epilogue_fusion_in_code_during_autotuning(self):
        """Test that code during autotuning contains epilogue with captured buffers."""
        m, n = 64, 64
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        y = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

        # Track all Triton code generated during autotuning
        generated_codes_during_autotuning: list[str] = []

        from helion.runtime.kernel import BoundKernel

        original_compile_config = BoundKernel.compile_config

        def tracking_compile_config(
            self, config=None, *, allow_print=True, template_buffer=None
        ):
            result = original_compile_config(
                self,
                config,
                allow_print=allow_print,
                template_buffer=template_buffer,
            )

            if template_buffer is not None:
                triton_code = self.to_triton_code(
                    config if config is not None else self._require_implicit_config(),
                    emit_repro_caller=False,
                    template_buffer=template_buffer,
                )
                generated_codes_during_autotuning.append(triton_code)

            return result

        def f(x, y, bias):
            out = elementwise_add_autotune(x, y)
            # Epilogue with captured buffer (bias)
            return torch.relu(out) + bias

        with patch.object(BoundKernel, "compile_config", tracking_compile_config):
            compiled_f = torch.compile(f)
            result = compiled_f(x, y, bias)

        # Verify correctness
        expected = torch.relu(x + y) + bias
        torch.testing.assert_close(result, expected)

        # Verify that we captured code during autotuning
        self.assertGreater(
            len(generated_codes_during_autotuning),
            0,
            "Should have captured Triton code generated during autotuning",
        )

        # Verify EACH generated code during autotuning contains the epilogue
        for i, code in enumerate(generated_codes_during_autotuning):
            # Should have relu AND the addition with bias
            has_relu = "maximum" in code or "relu" in code.lower()
            # The bias is added as part of the epilogue - look for the captured buffer load
            # The captured buffer should appear as an additional argument
            self.assertTrue(
                has_relu,
                f"Code #{i} generated during autotuning should contain relu epilogue, "
                f"but got:\n{code[:1500]}...",
            )

    def test_multiple_configs_all_have_fusion(self):
        """Test that ALL configs searched during autotuning have fusion applied."""
        m, n = 64, 64
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        y = torch.randn(m, n, device=DEVICE, dtype=torch.float32)

        # Track configs and whether each has fusion
        configs_with_fusion: list[tuple[object, bool]] = []

        from helion.runtime.kernel import BoundKernel

        original_compile_config = BoundKernel.compile_config

        def tracking_compile_config(
            self, config=None, *, allow_print=True, template_buffer=None
        ):
            result = original_compile_config(
                self,
                config,
                allow_print=allow_print,
                template_buffer=template_buffer,
            )

            if template_buffer is not None:
                triton_code = self.to_triton_code(
                    config if config is not None else self._require_implicit_config(),
                    emit_repro_caller=False,
                    template_buffer=template_buffer,
                )
                has_relu = "maximum" in triton_code or "relu" in triton_code.lower()
                configs_with_fusion.append((config, has_relu))

            return result

        def f(x, y):
            out = elementwise_add_autotune(x, y)
            return torch.relu(out)

        with patch.object(BoundKernel, "compile_config", tracking_compile_config):
            compiled_f = torch.compile(f)
            result = compiled_f(x, y)

        # Verify correctness
        expected = torch.relu(x + y)
        torch.testing.assert_close(result, expected)

        # Verify multiple configs were searched
        self.assertGreater(
            len(configs_with_fusion),
            1,
            f"Expected multiple configs to be tried, but only got {len(configs_with_fusion)}",
        )

        # Verify ALL configs had fusion applied
        for config, has_fusion in configs_with_fusion:
            self.assertTrue(
                has_fusion,
                f"Config {config} was benchmarked during autotuning but did NOT have "
                "fusion applied - the code did not contain relu epilogue!",
            )

    def test_multiple_captured_buffers_with_differing_shapes(self):
        """Test fusion during autotuning with multiple captured buffers of different shapes."""
        m, n = 64, 64
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        y = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        # Captured buffers with different shapes
        bias_row = torch.randn(n, device=DEVICE, dtype=torch.float32)  # Shape: [64]
        scale_col = torch.randn(m, 1, device=DEVICE, dtype=torch.float32)  # Shape: [64, 1]

        # Track all Triton code generated during autotuning
        generated_codes_during_autotuning: list[str] = []

        from helion.runtime.kernel import BoundKernel

        original_compile_config = BoundKernel.compile_config

        def tracking_compile_config(
            self, config=None, *, allow_print=True, template_buffer=None
        ):
            result = original_compile_config(
                self,
                config,
                allow_print=allow_print,
                template_buffer=template_buffer,
            )

            if template_buffer is not None:
                triton_code = self.to_triton_code(
                    config if config is not None else self._require_implicit_config(),
                    emit_repro_caller=False,
                    template_buffer=template_buffer,
                )
                generated_codes_during_autotuning.append(triton_code)

            return result

        def f(x, y, bias_row, scale_col):
            out = elementwise_add_autotune(x, y)
            # Epilogue with multiple captured buffers of different shapes
            # bias_row broadcasts along rows, scale_col broadcasts along columns
            return (torch.relu(out) + bias_row) * scale_col

        with patch.object(BoundKernel, "compile_config", tracking_compile_config):
            compiled_f = torch.compile(f)
            result = compiled_f(x, y, bias_row, scale_col)

        # Verify correctness
        expected = (torch.relu(x + y) + bias_row) * scale_col
        torch.testing.assert_close(result, expected)

        # Verify that we captured code during autotuning
        self.assertGreater(
            len(generated_codes_during_autotuning),
            0,
            "Should have captured Triton code generated during autotuning",
        )

        # Verify EACH generated code during autotuning contains the epilogue ops
        for i, code in enumerate(generated_codes_during_autotuning):
            has_relu = "maximum" in code or "relu" in code.lower()
            self.assertTrue(
                has_relu,
                f"Code #{i} generated during autotuning should contain relu epilogue, "
                f"but got:\n{code[:1500]}...",
            )


# Kernel that mutates an input tensor (in-place addition)
@helion.kernel(
    static_shapes=True, allow_torch_compile_fusion=True, autotune_effort="quick"
)
def inplace_add_autotune(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Kernel that adds y to x in-place and returns x."""
    m, n = x.size()

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :]
        y_tile = y[tile_m, :]
        x[tile_m, :] = x_tile + y_tile

    return x


class TestFusionWithMutation(RefEagerTestDisabled, TestCase):
    """Test autotuning with kernels that mutate inputs.

    Note: Mutation kernels with epilogue fusion during autotuning has limitations.
    These tests verify correctness rather than checking for specific fusion patterns.
    """

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        inplace_add_autotune._bound_kernels.clear()

    def test_mutation_with_epilogue_produces_correct_result(self):
        """Test that mutation kernel with epilogue produces correct results during autotuning."""
        m, n = 64, 64
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        y = torch.randn(m, n, device=DEVICE, dtype=torch.float32)

        def f(x, y):
            # Kernel mutates x in-place
            out = inplace_add_autotune(x, y)
            # Epilogue: relu
            return torch.relu(out)

        # Clone x since it will be mutated
        x_clone = x.clone()

        compiled_f = torch.compile(f)
        result = compiled_f(x_clone, y)

        # Verify correctness - x_clone should have been mutated
        expected = torch.relu(x + y)
        torch.testing.assert_close(result, expected)
        # Also verify x_clone was mutated
        torch.testing.assert_close(x_clone, x + y)

    def test_mutation_with_captured_buffer_produces_correct_result(self):
        """Test that mutation kernel with captured buffer epilogue produces correct results."""
        m, n = 64, 64
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        y = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, y, bias):
            # Kernel mutates x in-place
            out = inplace_add_autotune(x, y)
            # Epilogue with captured buffer
            return torch.relu(out) + bias

        # Clone x since it will be mutated
        x_clone = x.clone()

        compiled_f = torch.compile(f)
        result = compiled_f(x_clone, y, bias)

        # Verify correctness
        expected = torch.relu(x + y) + bias
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
