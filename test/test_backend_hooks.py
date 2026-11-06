from __future__ import annotations

import ast
from typing import ClassVar

import torch

import helion
from helion import backend_api
from helion._compiler import backend_registry
from helion._compiler.backend_registry import _REGISTRY
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
import helion.language as hl

_SCALAR_MARKER = "_scalar_marker"
_PROLOGUE_MARKER = "_prologue_marker"


class _PrologueBackend(backend_api.TritonBackend):
    # Reuses Triton codegen under a new name, like TileIRBackend.
    seen: ClassVar[list[tuple[int, int]]] = []
    body_starts: ClassVar[list[int]] = []

    @property
    def name(self) -> str:
        return "_test_prologue"

    @property
    def codegen_name(self) -> str:
        return "triton"

    def scalar_arg_preamble(self, arg: backend_api.Argument) -> list[ast.AST]:
        if arg.name == "alpha":
            return [backend_api.statement_from_string(f"{_SCALAR_MARKER} = 0")]
        return []

    def kernel_prologue(self, fn: backend_api.DeviceFunction) -> list[ast.stmt]:
        type(self).seen.append((len(fn.preamble), len(fn.body)))
        return [backend_api.statement_from_string(f"{_PROLOGUE_MARKER} = 0")]

    def finalize_device_body(
        self, body: list[ast.stmt], ctx: object
    ) -> tuple[list[ast.stmt], list[ast.arg]]:
        type(self).body_starts.append(ctx.body_start)  # pyrefly: ignore[missing-attribute]
        return body, []


class _FlagBackend(backend_api.TritonBackend):
    # Mirrors FlyDSL: body codegen sets a flag that kernel_prologue reads.
    calls: ClassVar[list[tuple[str, int]]] = []

    def __init__(self) -> None:
        super().__init__()
        self._needs_marker = False
        self._marker_emitted = False

    @property
    def name(self) -> str:
        return "_test_prologue_flag"

    @property
    def codegen_name(self) -> str:
        return "triton"

    def pre_codegen(
        self,
        graphs: list[backend_api.GraphInfo],
        config: backend_api.Config,
        tile_strategy: backend_api.TileStrategyDispatch,
    ) -> None:
        type(self).calls.append(("pre_codegen", id(self)))
        self._needs_marker = False
        self._marker_emitted = False
        super().pre_codegen(graphs, config, tile_strategy)

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        type(self).calls.append(("program_id_expr", id(self)))
        self._needs_marker = True
        return super().program_id_expr(dim, index_dtype=index_dtype)

    def kernel_prologue(self, fn: backend_api.DeviceFunction) -> list[ast.stmt]:
        type(self).calls.append(("kernel_prologue", id(self)))
        # Emit-once guard: only correct because pre_codegen resets it.
        if not self._needs_marker or self._marker_emitted:
            return []
        self._marker_emitted = True
        return [backend_api.statement_from_string(f"{_PROLOGUE_MARKER} = 0")]


class TestKernelPrologueDefault(TestCase):
    def test_default_is_empty(self) -> None:
        self.assertEqual(backend_api.TritonBackend().kernel_prologue(None), [])  # pyrefly: ignore[bad-argument-type]


@onlyBackends(["triton"])
class TestKernelPrologue(TestCase):
    def test_placement_and_call_order(self) -> None:
        _PrologueBackend.seen = []
        _PrologueBackend.body_starts = []
        backend_api.register_compiler_backend(_PrologueBackend)
        try:

            @helion.kernel(backend="_test_prologue")
            def axpy(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
                out = torch.empty_like(x)
                for tile in hl.tile(out.size()):
                    out[tile] = x[tile] + alpha * y[tile]
                return out

            x = torch.randn(64, 64, device=DEVICE)
            bound = axpy.bind((x, x, 2.0))
            config = bound.config_spec.default_config()
            # persistent_blocked puts the pid setup into fn.preamble.
            config.config["pid_type"] = "persistent_blocked"
            code = bound.to_triton_code(config)
        finally:
            _REGISTRY.pop("_test_prologue", None)

        # Called once, after both the preamble and the body were generated.
        self.assertEqual(len(_PrologueBackend.seen), 1)
        preamble_len, body_len = _PrologueBackend.seen[0]
        self.assertGreater(preamble_len, 0)
        self.assertGreater(body_len, 0)

        self.assertEqual(code.count(_PROLOGUE_MARKER), 1, code)
        (kernel,) = [
            node
            for node in ast.parse(code).body
            if isinstance(node, ast.FunctionDef)
            and any(_PROLOGUE_MARKER in ast.unparse(s) for s in node.body)
        ]
        stmts = [ast.unparse(s) for s in kernel.body]
        # scalar preamble -> kernel_prologue -> fn.preamble (pid setup) -> body
        self.assertEqual(stmts[0], f"{_SCALAR_MARKER} = 0", code)
        self.assertEqual(stmts[1], f"{_PROLOGUE_MARKER} = 0", code)
        self.assertTrue(stmts[2].startswith("total_pids"), code)
        # body_start counts the prologue too: scalar preamble + prologue + fn.preamble.
        self.assertEqual(_PrologueBackend.body_starts, [2 + preamble_len])

    def test_recompile_reemits_prologue(self) -> None:
        # One backend instance serves every pass of a bound kernel, so a flag
        # kept on the instance must be reset in pre_codegen (FlyDSL pattern).
        _FlagBackend.calls = []
        repaired_before = backend_registry._REPAIRED_CODEGEN_NAMES
        backend_api.register_compiler_backend(_FlagBackend)
        try:

            @helion.kernel(backend="_test_prologue_flag")
            def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                for tile in hl.tile(out.size()):
                    out[tile] = x[tile] + y[tile]
                return out

            x = torch.randn(64, 64, device=DEVICE)
            bound = add.bind((x, x))
            config = bound.config_spec.default_config()
            code1 = bound.to_triton_code(config)
            code2 = bound.to_triton_code(config)
        finally:
            _REGISTRY.pop("_test_prologue_flag", None)
            backend_registry._REPAIRED_CODEGEN_NAMES = repaired_before

        self.assertEqual(code1.count(_PROLOGUE_MARKER), 1, code1)
        self.assertEqual(code2.count(_PROLOGUE_MARKER), 1, code2)
        self.assertEqual(len({i for _, i in _FlagBackend.calls}), 1)
        hooks = [h for h, _ in _FlagBackend.calls]
        self.assertEqual(hooks.count("pre_codegen"), 2)
        self.assertEqual(hooks.count("kernel_prologue"), 2)


if __name__ == "__main__":
    import unittest

    unittest.main()
