from __future__ import annotations

import unittest

import pytest
import torch

import helion
from helion._compat import supports_ptxas
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion.exc import InvalidConfig
import helion.language as hl
from helion.runtime.ptxas_configs import _advanced_compiler_configuration_path


@helion.kernel()
def _copy_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    x_flat = x.view(-1)
    out_flat = out.view(-1)
    for tile in hl.tile(x_flat.numel()):
        out_flat[tile] = x_flat[tile]
    return out


class TestPtxasConfig(TestCase):
    @unittest.skipUnless(
        supports_ptxas(DEVICE), "PTXAS controls are only available on NVIDIA GPUs"
    )
    def test_ptxas_config_apply_controls_flag(self) -> None:
        x = torch.randn(128, device=DEVICE)
        code, result = code_and_output(
            _copy_kernel, (x,), ptxas_config=1, block_size=32
        )
        torch.testing.assert_close(result, x)

        option = f"--apply-controls {_advanced_compiler_configuration_path(1)}"
        self.assertIn(option, code)

        self.assertExpectedJournal(
            code.replace(_advanced_compiler_configuration_path(1), "<path>")
        )

    def test_ptxas_config_invalid_value(self) -> None:
        x = torch.randn(2, device=DEVICE)
        bound = _copy_kernel.bind((x,))
        base = bound.config_spec.default_config()

        options = base.config.copy()
        options["ptxas_config"] = "a"
        flagged = helion.Config(**options)

        with pytest.raises(InvalidConfig):
            bound.config_spec.normalize(flagged)
