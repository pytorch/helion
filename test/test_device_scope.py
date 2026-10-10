from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfNotTriton
from helion._testing import skipIfRefEager
from helion._testing import xfailIfMetal
import helion.exc as exc
import helion.language as hl
from helion.language import device_scope as direct_device_scope


@helion.kernel()
def routing_between_loops(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)
    for t in hl.tile(x.size(0)):
        tmp[t] = x[t] * 2
    with hl.device_scope():
        scores = torch.sigmoid(tmp[:, :].to(torch.float32))
        tmp[:, :] = (scores * scale[:, :]).to(tmp.dtype)
    for t in hl.tile(x.size(0)):
        out[t] = tmp[t] + 1
    return out


@helion.kernel()
def routing_between_loops_grid(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)
    for t in hl.tile(x.size(0)):
        tmp[t] = x[t] * 2
    for _unused_program in hl.grid(1):
        scores = torch.sigmoid(tmp[:, :].to(torch.float32))
        tmp[:, :] = (scores * scale[:, :]).to(tmp.dtype)
    for t in hl.tile(x.size(0)):
        out[t] = tmp[t] + 1
    return out


@helion.kernel()
def wrapped_gather_scatter(
    x: torch.Tensor, state: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    with hl.device_scope():
        for tile_tokens in hl.tile(x.size(0)):
            idx = hl.load(indices, [tile_tokens])
            selected = hl.load(state, [idx, slice(None)])
            out[tile_tokens, :] = selected + x[tile_tokens, :]
            state[idx, :] = selected * 2
    return out


@helion.kernel()
def scope_only(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    with hl.device_scope():
        out[:, :] = x[:, :] + 1
    return out


def _error_case_direct_import() -> None:
    @helion.kernel()
    def kernel(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        with direct_device_scope():
            out[:, :] = x[:, :] + 1
        return out

    kernel(torch.empty(8, 8, device=DEVICE, dtype=torch.float32))


def _error_case_as_binding() -> None:
    @helion.kernel()
    def kernel(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        with hl.device_scope() as scope:  # noqa: F841
            out[:, :] = x[:, :] + 1
        return out

    kernel(torch.empty(8, 8, device=DEVICE, dtype=torch.float32))


def _error_case_args() -> None:
    @helion.kernel()
    def kernel(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        with hl.device_scope(1):
            out[:, :] = x[:, :] + 1
        return out

    kernel(torch.empty(8, 8, device=DEVICE, dtype=torch.float32))


def _error_case_multiple_items() -> None:
    @helion.kernel()
    def kernel(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        with hl.device_scope(), torch.no_grad():
            out[:, :] = x[:, :] + 1
        return out

    kernel(torch.empty(8, 8, device=DEVICE, dtype=torch.float32))


def _error_case_nested_in_loop() -> None:
    @helion.kernel()
    def kernel(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for t in hl.tile(x.size(0)):
            with hl.device_scope():
                out[t] = x[t] + 1
        return out

    kernel(torch.empty(8, 8, device=DEVICE, dtype=torch.float32))


def _error_case_nested_in_scope() -> None:
    @helion.kernel()
    def kernel(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        with hl.device_scope():
            out[:, :] = x[:, :] + 1
            with hl.device_scope():
                out[:, :] = out[:, :] * 2
        return out

    kernel(torch.empty(8, 8, device=DEVICE, dtype=torch.float32))


def _strip_source_comments(code: str) -> list[str]:
    return [line for line in code.splitlines() if "# src[" not in line]


class TestDeviceScope(RefEagerTestBase, TestCase):
    @skipIfNotTriton("cross-loop dependency scheduling requires NVIDIA Triton")
    @skipIfNotCUDA()
    def test_between_loops_dependency(self) -> None:
        x = torch.randn(32, 16, device=DEVICE, dtype=torch.float32)
        scale = torch.rand(32, 16, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(routing_between_loops, (x, scale))
        expected = torch.sigmoid(x * 2) * scale + 1
        torch.testing.assert_close(out, expected)
        self.assertIn("device_scope", code)

    @skipIfNotTriton("cross-loop dependency scheduling requires NVIDIA Triton")
    @skipIfNotCUDA()
    def test_equivalent_to_grid_scaffolding(self) -> None:
        x = torch.randn(32, 16, device=DEVICE, dtype=torch.float32)
        scale = torch.rand(32, 16, device=DEVICE, dtype=torch.float32)
        _, out = code_and_output(routing_between_loops, (x, scale))
        _, out_grid = code_and_output(routing_between_loops_grid, (x, scale))
        torch.testing.assert_close(out, out_grid)
        expected = torch.sigmoid(x * 2) * scale + 1
        torch.testing.assert_close(out, expected)

    @skipIfRefEager("generated code is only available in compiled mode")
    @skipIfNotTriton("cross-loop dependency scheduling requires NVIDIA Triton")
    @skipIfNotCUDA()
    def test_generates_identical_code_as_grid(self) -> None:
        x = torch.randn(32, 16, device=DEVICE, dtype=torch.float32)
        scale = torch.rand(32, 16, device=DEVICE, dtype=torch.float32)
        code, _ = code_and_output(routing_between_loops, (x, scale))
        code_grid, _ = code_and_output(routing_between_loops_grid, (x, scale))
        normalized = code.replace("routing_between_loops", "kernel")
        normalized_grid = code_grid.replace("routing_between_loops_grid", "kernel")
        self.assertEqual(
            _strip_source_comments(normalized), _strip_source_comments(normalized_grid)
        )

    def test_wrapped_single_program_gather_scatter(self) -> None:
        n, d = 16, 8
        x = torch.randn(n, d, device=DEVICE, dtype=torch.float32)
        indices = torch.randperm(n, device=DEVICE, dtype=torch.int32)
        state0 = torch.randn(n, d, device=DEVICE, dtype=torch.float32)

        expected_out = state0[indices] + x
        expected_state = state0.clone()
        expected_state[indices] = state0[indices] * 2

        state = state0.clone()
        code, out = code_and_output(
            wrapped_gather_scatter, (x.clone(), state, indices.clone())
        )
        torch.testing.assert_close(out, expected_out)
        torch.testing.assert_close(state, expected_state)

    @xfailIfMetal("Metal allows at most one reduction dimension per kernel")
    def test_scope_only_kernel(self) -> None:
        x = torch.randn(8, 16, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(scope_only, (x,))
        torch.testing.assert_close(out, x + 1)

    @skipIfRefEager("device_scope misuse errors are raised at compile time")
    def test_error_direct_import(self) -> None:
        with self.assertRaisesRegex(
            exc.DeviceScopeInvalidUsage, "use hl.device_scope()"
        ):
            _error_case_direct_import()

    @skipIfRefEager("device_scope misuse errors are raised at compile time")
    def test_error_as_binding(self) -> None:
        with self.assertRaisesRegex(
            exc.DeviceScopeInvalidUsage, "'as' bindings are not supported"
        ):
            _error_case_as_binding()

    @skipIfRefEager("device_scope misuse errors are raised at compile time")
    def test_error_args(self) -> None:
        with self.assertRaisesRegex(
            exc.DeviceScopeInvalidUsage, "does not accept arguments"
        ):
            _error_case_args()

    @skipIfRefEager("device_scope misuse errors are raised at compile time")
    def test_error_multiple_items(self) -> None:
        with self.assertRaisesRegex(
            exc.DeviceScopeInvalidUsage, "only context manager"
        ):
            _error_case_multiple_items()

    @skipIfRefEager("nested device_scope is rejected during type propagation")
    def test_error_nested_in_loop(self) -> None:
        with self.assertRaisesRegex(exc.NotAllowedOnDevice, "statement With"):
            _error_case_nested_in_loop()

    @skipIfRefEager("nested device_scope is rejected during type propagation")
    def test_error_nested_in_scope(self) -> None:
        with self.assertRaisesRegex(exc.NotAllowedOnDevice, "statement With"):
            _error_case_nested_in_scope()
