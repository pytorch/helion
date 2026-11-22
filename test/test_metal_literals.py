"""Python-number literals in 16-bit Metal kernels, which lowering rounds to
the tensors' dtype on the host (``inductor_lowering._HostScalarTensors``).
Renders on meta tensors: like MPS on a machine without CUDA, a device fake
mode doesn't copy constants to, so a rounding on the kernel's device would
read back an unbacked SymFloat."""

from __future__ import annotations

import re

import pytest
import torch

from helion._testing import TestCase
from helion._testing import skipUnlessBackends
from helion._testing.metal import literal_math

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]


class TestLiterals(TestCase):
    def test_rounded_on_the_host(self) -> None:
        """A comparison's literal and remainder's divisor, which 16-bit math
        rounds on every device, are numbers in the generated code (0.3 and
        0.7 in bf16), not unbacked symbols (``zuf0``)."""
        x = torch.empty(1024, dtype=torch.bfloat16, device=torch.device("meta"))
        code = literal_math.bind((x, x)).to_code()
        self.assertIsNone(re.search(r"\bzuf\d", code))
        self.assertIn("0.30078125", code)
        self.assertIn("0.69921875", code)
