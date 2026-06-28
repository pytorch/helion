from __future__ import annotations

from types import SimpleNamespace
from typing import cast
import unittest
from unittest.mock import patch

import torch

import helion.runtime


def _tpu_device() -> torch.device:
    try:
        return torch.device("tpu")
    except RuntimeError:
        return cast("torch.device", SimpleNamespace(type="tpu", index=None))


class TestRuntimeGetNumSm(unittest.TestCase):
    def test_pallas_interpret_cpu_returns_one(self) -> None:
        with patch("helion.runtime._module_is_pallas_interpret", return_value=True):
            self.assertEqual(helion.runtime.get_num_sm(torch.device("cpu")), 1)
            self.assertEqual(
                helion.runtime.get_num_sm(torch.device("cpu"), reserved_sms=8),
                1,
            )

    def test_normal_cpu_still_unsupported(self) -> None:
        with (
            patch("helion.runtime._module_is_pallas_interpret", return_value=False),
            self.assertRaisesRegex(
                AssertionError,
                "TODO: implement for other devices",
            ),
        ):
            helion.runtime.get_num_sm(torch.device("cpu"))

    def test_tpu_returns_one(self) -> None:
        device = _tpu_device()

        self.assertEqual(helion.runtime.get_num_sm(device), 1)
        self.assertEqual(helion.runtime.get_num_sm(device, reserved_sms=8), 1)
