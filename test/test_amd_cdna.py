from __future__ import annotations

from unittest.mock import patch

import torch

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipUnlessAMDCDNA
import helion.language as hl


@onlyBackends(["triton"])
class TestAMDCDNA(TestCase):
    @skipUnlessAMDCDNA("Test requires AMD CDNA GPU (MI200/MI300 series)")
    def test_amd_cdna_tunables_in_kernel(self) -> None:
        """Test that AMD CDNA tunables are supported."""

        @helion.kernel(
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[32, 32],
                waves_per_eu=2,
                matrix_instr_nonkdim=16,
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
        self.assertIn("waves_per_eu=2", code)
        self.assertIn("matrix_instr_nonkdim=16", code)

    def test_amd_tunables_error_when_not_supported(self) -> None:
        """Test that specifying AMD tunables on non-AMD hardware raises an error."""
        device = torch.device("cuda")
        settings = helion.Settings()

        with patch(
            "helion.autotuner.config_spec.supports_amd_cdna_tunables",
            return_value=False,
        ):
            env = CompileEnvironment(device, settings)

            config = helion.Config(waves_per_eu=2)
            with self.assertRaisesRegex(
                helion.exc.InvalidConfig,
                "waves_per_eu is not supported on this target hardware",
            ):
                env.config_spec.normalize(config)

            config = helion.Config(matrix_instr_nonkdim=16)
            with self.assertRaisesRegex(
                helion.exc.InvalidConfig,
                "matrix_instr_nonkdim is not supported on this target hardware",
            ):
                env.config_spec.normalize(config)
