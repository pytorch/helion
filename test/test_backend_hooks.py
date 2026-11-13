"""The finalize_device_body contract an out-of-tree backend relies on."""

from __future__ import annotations

import ast
from typing import ClassVar

import pytest
import torch

import helion
from helion import backend_api
from helion._compiler import backend_registry
from helion._compiler.backend_registry import unregister_compiler_backend
from helion._testing import TestCase
from helion._testing import onlyBackends
import helion.language as hl

pytestmark = pytest.mark.cpu_only

_SCALAR_MARKER = "_scalar_marker"
_BODY_MARKER = "_body_marker"


class _FinalizeBackend(backend_api.TritonBackend):
    # Reuses Triton codegen under a new name, like TileIRBackend.
    calls: ClassVar[list[tuple[int, int, int]]] = []

    @property
    def name(self) -> str:
        return "_test_finalize"

    @property
    def codegen_name(self) -> str:
        return "triton"

    def scalar_arg_preamble(self, arg: backend_api.Argument) -> list[ast.AST]:
        if arg.name == "alpha":
            return [backend_api.statement_from_string(f"{_SCALAR_MARKER} = 0")]
        return []

    def finalize_device_body(
        self, body: list[ast.stmt], ctx: backend_api.DeviceBodyContext
    ) -> tuple[list[ast.stmt], list[ast.arg]]:
        preamble = len(ctx.device_function.preamble)
        type(self).calls.append((ctx.body_start, len(ctx.args), preamble))
        assert ctx.device_function.name.startswith("_helion_")
        marker = backend_api.statement_from_string(f"{_BODY_MARKER} = 0")
        at = ctx.body_start
        # pyrefly: ignore[bad-argument-type]
        return [*body[:at], marker, *body[at:]], [ast.arg("_extra_param")]


def _axpy(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + alpha * y[tile]
    return out


def _kernel_def(code: str) -> ast.FunctionDef:
    (kernel,) = [
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    ]
    return kernel


@onlyBackends(["triton"])  # backend-independent; run in one lane
class TestFinalizeDeviceBody(TestCase):
    def setUp(self) -> None:
        super().setUp()
        _FinalizeBackend.calls = []
        # Registering a "triton"-codegen backend clears triton's repaired mark.
        self.addCleanup(
            setattr,
            backend_registry,
            "_REPAIRED_CODEGEN_NAMES",
            backend_registry._REPAIRED_CODEGEN_NAMES,
        )
        backend_api.register_compiler_backend(_FinalizeBackend)
        self.addCleanup(unregister_compiler_backend, "_test_finalize")

    def _code(self, pid_type: str) -> str:
        x = torch.randn(64, 64)
        bound = helion.kernel(backend="_test_finalize")(_axpy).bind((x, x, 2.0))
        config = bound.config_spec.default_config()
        config.config["pid_type"] = pid_type
        return bound.to_triton_code(config)

    def test_body_start_marks_the_traced_body(self) -> None:
        # persistent_blocked puts the pid setup into fn.preamble.
        code = self._code("persistent_blocked")
        stmts = [ast.unparse(s) for s in _kernel_def(code).body]
        ((body_start, _, preamble),) = _FinalizeBackend.calls
        # Exactly the scalar_arg_preamble statement, then fn.preamble (pid
        # setup), come before body_start; the traced body follows the marker.
        self.assertGreater(preamble, 0)
        self.assertEqual(body_start, 1 + preamble)
        self.assertEqual(stmts[0], f"{_SCALAR_MARKER} = 0", code)
        self.assertTrue(stmts[1].startswith("total_pids"), code)
        self.assertEqual(stmts[body_start], f"{_BODY_MARKER} = 0", code)
        self.assertGreater(len(stmts), body_start + 1, code)
        self.assertEqual(code.count(_BODY_MARKER), 1, code)

    def test_returned_args_extend_the_signature(self) -> None:
        code = self._code("flat")
        params = [a.arg for a in _kernel_def(code).args.args]
        ((_, n_args, _),) = _FinalizeBackend.calls
        # ctx.args is the signature so far; the returned params follow it.
        self.assertEqual(len(params), n_args + 1, params)
        self.assertEqual(params[-1], "_extra_param")

    def test_called_once_per_compile(self) -> None:
        self._code("flat")
        self._code("persistent_blocked")
        self.assertEqual(len(_FinalizeBackend.calls), 2)


if __name__ == "__main__":
    import unittest

    unittest.main()
