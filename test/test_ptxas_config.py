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
from helion.runtime.ptxas_configs import search_ptxas_configs


@helion.kernel()
def _copy_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    x_flat = x.view(-1)
    out_flat = out.view(-1)
    for tile in hl.tile(x_flat.numel()):
        out_flat[tile] = x_flat[tile]
    return out


class TestAdvancedCompilerConfiguration(TestCase):
    @unittest.skipUnless(
        supports_ptxas(DEVICE), "PTXAS controls are only available on NVIDIA GPUs"
    )
    def test_configuration_apply_controls_flag(self) -> None:
        available_configs = search_ptxas_configs()
        self.assertTrue(available_configs)
        config_id = available_configs[0]
        x = torch.randn(128, device=DEVICE)
        code, result = code_and_output(
            _copy_kernel,
            (x,),
            advanced_compiler_configuration=config_id,
            block_size=32,
        )
        torch.testing.assert_close(result, x)

        option = f"--apply-controls {_advanced_compiler_configuration_path(config_id)}"
        self.assertIn(option, code)

        self.assertExpectedJournal(
            code.replace(_advanced_compiler_configuration_path(config_id), "<path>")
        )

    def test_configuration_invalid_value(self) -> None:
        x = torch.randn(2, device=DEVICE)
        bound = _copy_kernel.bind((x,))
        base = bound.config_spec.default_config()

        options = base.config.copy()
        options["advanced_compiler_configuration"] = "a"
        flagged = helion.Config(**options)

        with pytest.raises(InvalidConfig):
            bound.config_spec.normalize(flagged)

    def test_autotune_search_acc_flag_disables_generation(self) -> None:
        x = torch.randn(4, device=DEVICE)

        kernel_with_acc = helion.kernel()(_copy_kernel.fn)
        bound_with_acc = kernel_with_acc.bind((x,))
        bound_with_acc.config_spec.ptxas_supported = True
        config_with_acc = bound_with_acc.config_spec.flat_config(
            lambda fragment: fragment.default(),
            include_advanced_compiler_configuration=bound_with_acc.kernel.settings.autotune_search_acc,
        )

        kernel_without_acc = helion.kernel(autotune_search_acc=False)(_copy_kernel.fn)
        bound_without_acc = kernel_without_acc.bind((x,))
        bound_without_acc.config_spec.ptxas_supported = True
        config_without_acc = bound_without_acc.config_spec.flat_config(
            lambda fragment: fragment.default(),
            include_advanced_compiler_configuration=bound_without_acc.kernel.settings.autotune_search_acc,
        )

        self.assertIn(
            "advanced_compiler_configuration",
            config_with_acc.config,
        )
        self.assertNotIn(
            "advanced_compiler_configuration",
            config_without_acc.config,
        )

        kernel_with_acc.reset()
        kernel_without_acc.reset()
