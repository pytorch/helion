"""Out-of-tree thread-level backends (like helion-flydsl) override hooks no
in-tree backend overrides; a stand-in drives them through real codegen."""

from __future__ import annotations

import ast
from typing import ClassVar

import pytest
import torch

import helion
from helion import backend_api
from helion._compiler.backend_registry import unregister_compiler_backend
from helion._testing import TestCase
from helion._testing import onlyBackends
import helion.language as hl

pytestmark = pytest.mark.cpu_only


class _ThreadLevelHooksBackend(backend_api.TritonBackend):
    # Overrides the hooks no in-tree backend overrides (out-of-tree
    # thread-level backends such as helion-flydsl do), so Helion's own tests
    # exercise their non-default paths.
    calls: ClassVar[dict[str, list[object]]] = {}

    @property
    def name(self) -> str:
        return "_test_thread_level_hooks"

    @property
    def codegen_name(self) -> str:
        return "triton"

    def max_reduction_threads(self) -> int | None:
        return 64

    def reduction_block_size_is_inlined_constexpr(self) -> bool:
        return True

    def looped_reduction_thread_count(self, **kwargs: object) -> int | None:
        type(self).calls.setdefault("looped", []).append(kwargs["requested"])
        return 32

    def wrap_reduction_accumulator(self, acc_full: str, **kwargs: object) -> str:
        type(self).calls.setdefault("wrap", []).append(kwargs["thread_count"])
        return f"_wrapped_acc({acc_full})"

    def register_reduction_loop_config_slots(
        self, env: backend_api.CompileEnvironment, block_id: int, size_hint: int
    ) -> None:
        type(self).calls.setdefault("slots", []).append(size_hint)

    def expands_broadcast_dims(self) -> bool:
        return False

    def cast_scalar_ast(self, x: ast.AST, target_dtype: torch.dtype) -> ast.AST:
        type(self).calls.setdefault("cast_scalar", []).append(target_dtype)
        return backend_api.expr_from_string("_scalar_cast({x})", x=x)


def _rowsum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], device=x.device, dtype=x.dtype)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m] = x[tile_m, :].sum(-1)
    return out


def _add_row(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        out[tile_m, tile_n] = x[tile_m, tile_n] + b[tile_n]
    return out


def _sub_mean(x: torch.Tensor, begin: int, end: int) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(begin, end):
        v = x[tile]
        # The mean divides by the tile's extent, a scalar from an index expr.
        out[tile] = v - v.mean()
    return out


def _kernel_params(code: str) -> list[str]:
    (kernel,) = [
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    ]
    return [arg.arg for arg in kernel.args.args]


@onlyBackends(["triton"])  # backend-independent; run in one lane
class TestThreadLevelBackendHooks(TestCase):
    def setUp(self) -> None:
        super().setUp()
        _ThreadLevelHooksBackend.calls = {}
        backend_api.register_compiler_backend(_ThreadLevelHooksBackend)
        self.addCleanup(unregister_compiler_backend, "_test_thread_level_hooks")

    def _code(self, fn: object, args: tuple[torch.Tensor, ...], backend: str) -> str:
        bound = helion.kernel(backend=backend)(fn).bind(args)
        config = bound.config_spec.default_config()
        if "reduction_loops" in config.config:
            config.config["reduction_loops"] = [64]
        return bound.to_triton_code(config)

    def test_looped_reduction_hooks(self) -> None:
        x = torch.randn(32, 512)
        code = self._code(_rowsum, (x,), "_test_thread_level_hooks")
        calls = _ThreadLevelHooksBackend.calls
        self.assertEqual(calls["slots"], [512])
        self.assertEqual(calls["looped"], [64])
        # The thread count the backend returned is the one codegen used.
        self.assertEqual(calls["wrap"], [32])
        self.assertIn("_wrapped_acc(", code)
        # Inlined: a module-level constant, not a kernel parameter.
        self.assertIn("_REDUCTION_BLOCK_1 = tl.constexpr(64)", code)
        self.assertNotIn("_REDUCTION_BLOCK_1", _kernel_params(code))
        self.assertIn(
            "_REDUCTION_BLOCK_1", _kernel_params(self._code(_rowsum, (x,), "triton"))
        )

    def test_expands_broadcast_dims(self) -> None:
        x = torch.randn(32, 64)
        args = (x, x[0].contiguous())
        self.assertIn("load_1[None, :]", self._code(_add_row, args, "triton"))
        plugin = self._code(_add_row, args, "_test_thread_level_hooks")
        self.assertNotIn("load_1[None, :]", plugin)

    def test_cast_scalar_ast(self) -> None:
        args = (torch.randn(40), 3, 14)
        self.assertNotIn("_scalar_cast(", self._code(_sub_mean, args, "triton"))
        plugin = self._code(_sub_mean, args, "_test_thread_level_hooks")
        self.assertEqual(_ThreadLevelHooksBackend.calls["cast_scalar"], [torch.float32])
        self.assertIn("_scalar_cast(", plugin)


if __name__ == "__main__":
    import unittest

    unittest.main()
