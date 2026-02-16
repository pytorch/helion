from __future__ import annotations

import builtins
from contextlib import contextmanager
import os
import sys
from typing import TYPE_CHECKING
import unittest
from unittest import mock

import torch
import triton.runtime.interpreter as triton_interpreter

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import onlyBackends
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel


@onlyBackends(["triton"])
class TestBreakpoint(RefEagerTestDisabled, TestCase):
    @staticmethod
    @contextmanager
    def _auto_resume_breakpoint() -> None:
        """Temporarily suppress interactive debuggers so tests can run unattended."""
        original_builtin = builtins.breakpoint
        original_hook = sys.breakpointhook

        def _noop_breakpoint(*_args: object, **_kwargs: object) -> None:
            return None

        def _noop_hook(*_args: object, **_kwargs: object) -> None:
            return None

        builtins.breakpoint = _noop_breakpoint
        sys.breakpointhook = _noop_hook
        try:
            yield
        finally:
            builtins.breakpoint = original_builtin
            sys.breakpointhook = original_hook

    def _make_device_breakpoint_kernel(self) -> Kernel:
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                breakpoint()
                out[tile] = x[tile]
            return out

        return kernel

    def _make_host_breakpoint_kernel(self) -> Kernel:
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            breakpoint()
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile]
            return out

        return kernel

    def _run_device_breakpoint_test(
        self, triton_interpret: int, helion_interpret: int
    ) -> None:
        """Common test logic for device breakpoint tests with different interpret modes."""
        x = torch.randn(8, device=DEVICE, dtype=torch.float32)
        env = {
            "TRITON_INTERPRET": str(triton_interpret),
            "HELION_INTERPRET": str(helion_interpret),
        }
        with mock.patch.dict(os.environ, env, clear=False):
            kernel = self._make_device_breakpoint_kernel()
            kernel.reset()
            if triton_interpret == helion_interpret == 0:
                with self.assertRaises(exc.BreakpointInDeviceLoopRequiresInterpret):
                    kernel.bind((x,))
            else:
                bound = kernel.bind((x,))
                if triton_interpret == 1:
                    self.assertIn(
                        "breakpoint()",
                        bound.to_triton_code(bound.config_spec.default_config()),
                    )
                with self._auto_resume_breakpoint():
                    out = bound(x)
                torch.testing.assert_close(out, x)

    def test_device_breakpoint_no_interpret(self) -> None:
        self._run_device_breakpoint_test(triton_interpret=0, helion_interpret=0)

    @unittest.skipUnless(
        hasattr(triton_interpreter, "_MISSING"),
        "https://github.com/triton-lang/triton/pull/8735",
    )
    def test_device_breakpoint_triton_interpret(self) -> None:
        self._run_device_breakpoint_test(triton_interpret=1, helion_interpret=0)

    def test_device_breakpoint_helion_interpret(self) -> None:
        self._run_device_breakpoint_test(triton_interpret=0, helion_interpret=1)

    def _run_host_breakpoint_test(
        self, triton_interpret: int, helion_interpret: int
    ) -> None:
        """Common test logic for host breakpoint tests with different interpret modes."""
        x = torch.randn(8, device=DEVICE, dtype=torch.float32)
        env = {
            "TRITON_INTERPRET": str(triton_interpret),
            "HELION_INTERPRET": str(helion_interpret),
        }
        with mock.patch.dict(os.environ, env, clear=False):
            kernel = self._make_host_breakpoint_kernel()
            kernel.reset()
            bound = kernel.bind((x,))
            with self._auto_resume_breakpoint():
                out = bound(x)
            torch.testing.assert_close(out, x)

    def test_host_breakpoint_no_interpret(self) -> None:
        self._run_host_breakpoint_test(triton_interpret=0, helion_interpret=0)

    @unittest.skipUnless(
        hasattr(triton_interpreter, "_MISSING"),
        "https://github.com/triton-lang/triton/pull/8735",
    )
    def test_host_breakpoint_triton_interpret(self) -> None:
        self._run_host_breakpoint_test(triton_interpret=1, helion_interpret=0)

    def test_host_breakpoint_helion_interpret(self) -> None:
        self._run_host_breakpoint_test(triton_interpret=0, helion_interpret=1)

    def test_incompatible_interpret_modes(self) -> None:
        env = {
            "TRITON_INTERPRET": "1",
            "HELION_INTERPRET": "1",
        }
        with (
            mock.patch.dict(os.environ, env, clear=False),
            self.assertRaises(exc.IncompatibleInterpretModes),
        ):

            @helion.kernel(autotune_effort="none")
            def kernel(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                for tile in hl.tile(x.size()):
                    out[tile] = x[tile]
                return out

            x = torch.randn(8, device=DEVICE, dtype=torch.float32)
            kernel(x)


if __name__ == "__main__":
    unittest.main()
