from __future__ import annotations

import os
import unittest

import pytest
import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl


class TestLoops(RefEagerTestBase, TestCase):
    def test_loop_dependency_error1(self):
        @helion.kernel
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)

            for tile in hl.tile(x.size()):
                out[tile] += x[tile]

            for tile in hl.tile(y.size()):
                out[tile] += y[tile]

            return out

        x = torch.randn(4, device=DEVICE)
        y = torch.randn(4, device=DEVICE)

        with pytest.raises(
            expected_exception=exc.LoopDependencyError,
            match="Loop dependency detected: 'out' was written in a previous loop.",
        ):
            code_and_output(kernel, (x, y))

    def test_loop_dependency_error2(self):
        @helion.kernel
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)

            for tile in hl.tile(x.size()):
                y[tile] += x[tile]

            for tile in hl.tile(y.size()):
                out[tile] += y[tile]

            return out

        x = torch.randn(4, device=DEVICE)
        y = torch.randn(4, device=DEVICE)

        with pytest.raises(
            expected_exception=exc.LoopDependencyError,
            match="Loop dependency detected: 'y' was written in a previous loop.",
        ):
            code_and_output(kernel, (x, y))

    def test_loop_dependency_error3(self):
        @helion.kernel
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(x.size()):
                x[tile] += x[tile]

            x.sum()

            for tile in hl.tile(y.size()):
                y[tile] += y[tile]

            return x + y

        x = torch.randn(4, device=DEVICE)
        y = torch.randn(4, device=DEVICE)

        with pytest.raises(
            expected_exception=exc.TopLevelStatementBetweenLoops,
            match="Statements cannot appear between top level loops.",
        ):
            code_and_output(kernel, (x, y))


if __name__ == "__main__":
    unittest.main()
