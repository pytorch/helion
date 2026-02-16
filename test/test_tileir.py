from __future__ import annotations

from unittest.mock import patch

import torch

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipUnlessTileIR
import helion.language as hl


@onlyBackends(["triton"])
class TestTileIR(TestCase):
    @skipUnlessTileIR("Test requires tileir")
    def test_tileir_tunables_in_kernel(self) -> None:
        """Test that tileir tunables are supported."""

        @helion.kernel(
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[128, 128],
                num_ctas=2,
                occupancy=2,
            ),
        )
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                result[tile] = x[tile] + y[tile]
            return result

        x = torch.randn(128, 128, device=DEVICE, dtype=torch.float32)
        y = torch.randn(128, 128, device=DEVICE, dtype=torch.float32)

        code, result = code_and_output(add_kernel, (x, y))
        expected = x + y

        torch.testing.assert_close(result, expected)

        # Verify that the tunables are passed to Triton
        self.assertIn("num_ctas=2", code)
        self.assertIn("occupancy=2", code)

    def test_tileir_tunables_error_when_not_supported(self) -> None:
        """Test that specifying tileir tunables on non-tileir backend raises an error."""
        device = torch.device("cuda")
        settings = helion.Settings()

        with patch(
            "helion.autotuner.config_spec.use_tileir_tunables",
            return_value=False,
        ):
            env = CompileEnvironment(device, settings)

            config = helion.Config(num_ctas=2)
            with self.assertRaisesRegex(
                helion.exc.InvalidConfig,
                "num_ctas is not supported on this target hardware",
            ):
                env.config_spec.normalize(config)

            config = helion.Config(occupancy=16)
            with self.assertRaisesRegex(
                helion.exc.InvalidConfig,
                "occupancy is not supported on this target hardware",
            ):
                env.config_spec.normalize(config)
