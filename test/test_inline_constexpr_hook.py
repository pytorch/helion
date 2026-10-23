from __future__ import annotations

import ast
import unittest
from unittest.mock import patch

import torch

import helion
from helion._testing import onlyBackends
import helion.language as hl
from helion.runtime.config import Config


@helion.kernel(config=Config(block_sizes=[16]))
def _add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        result[tile] = x[tile] + y[tile]
    return result


def _kernel_def(code: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion")
    )


def _launcher_call(code: str) -> ast.Call:
    return next(
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_launcher"
    )


def _passed_names(call: ast.Call) -> list[str]:
    return [arg.id for arg in call.args if isinstance(arg, ast.Name)]


@onlyBackends(["triton"])
class TestInlineConstexprHookGating(unittest.TestCase):
    """inline_constexpr_at_module_level() must gate def and call sides together.

    A backend that opts out keeps literal constexpr args as kernel parameters,
    so the launcher call must pass them positionally — otherwise the kernel
    fails at launch with a missing-argument error (the parameter has no
    default). Pure codegen: bind on CPU tensors and inspect to_code() output.
    """

    def test_hook_off_keeps_constexpr_params_and_passes_them(self) -> None:
        x = torch.randn(32)
        y = torch.randn(32)
        bound = _add_kernel.bind((x, y))
        backend_cls = type(bound.env.backend)
        with patch.object(
            backend_cls, "inline_constexpr_at_module_level", return_value=False
        ):
            code = bound.to_code(Config(block_sizes=[16]))

        # Def side: _BLOCK_SIZE_0 stays a kernel parameter instead of a
        # module-level tl.constexpr assignment.
        self.assertNotIn("_BLOCK_SIZE_0 = tl.constexpr(", code)
        self.assertIn("_BLOCK_SIZE_0", [arg.arg for arg in _kernel_def(code).args.args])
        # Call side: the launcher call passes it positionally.
        self.assertIn("_BLOCK_SIZE_0", _passed_names(_launcher_call(code)))

    def test_hook_on_inlines_constexpr_at_module_level(self) -> None:
        x = torch.randn(32)
        y = torch.randn(32)
        bound = _add_kernel.bind((x, y))
        code = bound.to_code(Config(block_sizes=[16]))

        # Def side inlines; call side does not pass it.
        self.assertIn("_BLOCK_SIZE_0 = tl.constexpr(", code)
        self.assertNotIn(
            "_BLOCK_SIZE_0", [arg.arg for arg in _kernel_def(code).args.args]
        )
        self.assertNotIn("_BLOCK_SIZE_0", _passed_names(_launcher_call(code)))


if __name__ == "__main__":
    unittest.main()
