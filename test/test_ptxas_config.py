from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion.autotuner.config_spec import EnumFragment
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
                "advanced_controls_file": config_path,
                "block_size": 32,
            }
        )

        option = f"--apply-controls {config_path}"
        self.assertIn(option, code)
        self.assertIn(config_path, code)

    def test_configuration_invalid_value(self) -> None:
        x = torch.randn(2, device=DEVICE)
        bound = _copy_kernel.bind((x,))
        base = bound.config_spec.default_config()

        options = base.config.copy()
        options["advanced_controls_file"] = 123
        flagged = helion.Config(**options)

        with self.assertRaises(InvalidConfig):
            bound.config_spec.normalize(flagged)

    def test_autotune_search_acf_empty_list_means_no_acf(self) -> None:
        x = torch.randn(2, device=DEVICE)
        kernel = helion.kernel(autotune_search_acf=[])(_copy_kernel.fn)
        bound = kernel.bind((x,))
        config = bound.config_spec.flat_config(
            lambda fragment: fragment.default(),
            advanced_controls_files=bound.kernel.settings.autotune_search_acf,
        )
        self.assertNotIn("advanced_controls_file", config.config)

    def test_autotune_search_acf_enables_generation(self) -> None:
        x = torch.randn(4, device=DEVICE)

        kernel_with_acf = helion.kernel(
            autotune_search_acf=["", "/some/path.bin"],
        )(_copy_kernel.fn)
        bound_with_acf = kernel_with_acf.bind((x,))
        config_with_acf = bound_with_acf.config_spec.flat_config(
            lambda fragment: fragment.default(),
            advanced_controls_files=bound_with_acf.kernel.settings.autotune_search_acf,
        )

        kernel_without_acf = helion.kernel(autotune_search_acf=[])(_copy_kernel.fn)
        bound_without_acf = kernel_without_acf.bind((x,))
        config_without_acf = bound_without_acf.config_spec.flat_config(
            lambda fragment: fragment.default(),
            advanced_controls_files=bound_without_acf.kernel.settings.autotune_search_acf,
        )

        self.assertIn(
            "advanced_controls_file",
            config_with_acf.config,
        )
        self.assertNotIn(
            "advanced_controls_file",
            config_without_acf.config,
        )

        kernel_with_acf.reset()
        kernel_without_acf.reset()

    def test_empty_string_means_no_config(self) -> None:
        x = torch.randn(128, device=DEVICE)
        code, result = code_and_output(
            _copy_kernel,
            (x,),
            advanced_controls_file="",
            block_size=32,
        )
        torch.testing.assert_close(result, x)
        self.assertNotIn("ptx_options", code)

    def test_empty_string_appended_when_missing(self) -> None:
        """When autotune_search_acf omits "", it is appended automatically so
        the no-ACF baseline is always part of the search space."""
        x = torch.randn(4, device=DEVICE)
        kernel = helion.kernel(
            autotune_search_acf=["/some/path.bin"],
        )(_copy_kernel.fn)
        bound = kernel.bind((x,))

        seen_choices: list[tuple[object, ...]] = []

        def collect_choices(fragment: object) -> object:
            if isinstance(fragment, EnumFragment):
                seen_choices.append(fragment.choices)
            return fragment.default()  # type: ignore[union-attr]

        bound.config_spec.flat_config(
            collect_choices,
            advanced_controls_files=bound.kernel.settings.autotune_search_acf,
        )

        acf_choices = next(
            (c for c in seen_choices if "/some/path.bin" in c),
            None,
        )
        self.assertIsNotNone(
            acf_choices, "advanced_controls_file not found in config space"
        )
        self.assertIn("", acf_choices)

        kernel.reset()

    def test_empty_string_not_duplicated(self) -> None:
        """When autotune_search_acf already contains "", it must not be duplicated."""
        x = torch.randn(4, device=DEVICE)
        kernel = helion.kernel(
            autotune_search_acf=["", "/some/path.bin"],
        )(_copy_kernel.fn)
        bound = kernel.bind((x,))

        seen_choices: list[tuple[object, ...]] = []

        def collect_choices(fragment: object) -> object:
            if isinstance(fragment, EnumFragment):
                seen_choices.append(fragment.choices)
            return fragment.default()  # type: ignore[union-attr]

        bound.config_spec.flat_config(
            collect_choices,
            advanced_controls_files=bound.kernel.settings.autotune_search_acf,
        )

        acf_choices = next(
            (c for c in seen_choices if "/some/path.bin" in c),
            None,
        )
        self.assertIsNotNone(
            acf_choices, "advanced_controls_file not found in config space"
        )
        self.assertEqual(acf_choices.count(""), 1)

        kernel.reset()


if __name__ == "__main__":
    unittest.main()
