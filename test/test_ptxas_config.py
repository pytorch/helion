from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion.exc import InvalidConfig
import helion.language as hl


@helion.kernel()
def _copy_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    x_flat = x.view(-1)
    out_flat = out.view(-1)
    for tile in hl.tile(x_flat.numel()):
        out_flat[tile] = x_flat[tile]
    return out


class TestAdvancedCompilerConfiguration(TestCase):
    def test_configuration_apply_controls_flag(self) -> None:
        config_path = "/some/test/path.bin"
        x = torch.randn(128, device=DEVICE)
        bound = _copy_kernel.bind((x,))
        code = bound.to_triton_code(
            {
                "advanced_compiler_configuration": config_path,
                "block_size": 32,
            }
        )

        option = f"--apply-controls {config_path}"
        self.assertIn(option, code)
        self.assertExpectedJournal(code)

    def test_configuration_invalid_value(self) -> None:
        x = torch.randn(2, device=DEVICE)
        bound = _copy_kernel.bind((x,))
        base = bound.config_spec.default_config()

        options = base.config.copy()
        options["advanced_compiler_configuration"] = 123
        flagged = helion.Config(**options)

        with self.assertRaises(InvalidConfig):
            bound.config_spec.normalize(flagged)

    def test_autotune_search_acc_enables_generation(self) -> None:
        x = torch.randn(4, device=DEVICE)

        kernel_with_acc = helion.kernel(
            autotune_search_acc=["", "/some/path.bin"],
        )(_copy_kernel.fn)
        bound_with_acc = kernel_with_acc.bind((x,))
        config_with_acc = bound_with_acc.config_spec.flat_config(
            lambda fragment: fragment.default(),
            advanced_compiler_configurations=bound_with_acc.kernel.settings.autotune_search_acc,
        )

        kernel_without_acc = helion.kernel(autotune_search_acc=[])(_copy_kernel.fn)
        bound_without_acc = kernel_without_acc.bind((x,))
        config_without_acc = bound_without_acc.config_spec.flat_config(
            lambda fragment: fragment.default(),
            advanced_compiler_configurations=bound_without_acc.kernel.settings.autotune_search_acc,
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

    def test_empty_string_means_no_config(self) -> None:
        x = torch.randn(128, device=DEVICE)
        code, result = code_and_output(
            _copy_kernel,
            (x,),
            advanced_compiler_configuration="",
            block_size=32,
        )
        torch.testing.assert_close(result, x)
        self.assertNotIn("ptx_options", code)


if __name__ == "__main__":
    unittest.main()
