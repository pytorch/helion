from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import helion
from helion import _compat
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion.autotuner.config_fragment import EnumFragment
import helion.language as hl


@helion.kernel(static_shapes=True)
def scaled_copy(x: torch.Tensor) -> torch.Tensor:
    """Registers a launch option of a Triton build with the AutoWS patches."""
    hl.register_tunable("_triton_config_maxRegAutoWS", EnumFragment(choices=(152,)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * 2.0
    return out


@onlyBackends(["triton"])
class TestTritonLaunchOptions(TestCase):
    @skipIfRefEager("codegen test")
    def test_unsupported_launch_option_is_dropped(self) -> None:
        # A ``_triton_config_<option>`` key the installed Triton rejects is left
        # out of the launch (with a warning) instead of failing the kernel.
        x = torch.randn([256], device=DEVICE)
        with patch.object(
            _compat, "_supported_launch_options", return_value=frozenset({"num_warps"})
        ):
            code, result = code_and_output(
                scaled_copy, (x,), block_sizes=[64], _triton_config_maxRegAutoWS=152
            )
        self.assertNotIn("maxRegAutoWS", code)
        torch.testing.assert_close(result, x * 2.0)

    @skipIfRefEager("codegen test")
    def test_supported_launch_option_is_emitted(self) -> None:
        x = torch.randn([256], device=DEVICE)
        with patch.object(
            _compat,
            "_supported_launch_options",
            return_value=frozenset({"num_warps", "maxRegAutoWS"}),
        ):
            code = scaled_copy.bind((x,)).to_triton_code(
                helion.Config(block_sizes=[64], _triton_config_maxRegAutoWS=152)
            )
        self.assertIn("maxRegAutoWS=152", code)

    @skipIfRefEager("codegen test")
    def test_unknown_backend_keeps_the_option(self) -> None:
        # Without a probe result the key is passed through as before.
        x = torch.randn([256], device=DEVICE)
        with patch.object(_compat, "_supported_launch_options", return_value=None):
            code = scaled_copy.bind((x,)).to_triton_code(
                helion.Config(block_sizes=[64], _triton_config_maxRegAutoWS=152)
            )
        self.assertIn("maxRegAutoWS=152", code)


if __name__ == "__main__":
    unittest.main()
