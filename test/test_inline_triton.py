from __future__ import annotations

import torch
import triton

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


@onlyBackends(["triton"])
class TestInlineTriton(RefEagerTestDisabled, TestCase):
    def test_inline_triton_simple(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                y_val = y[tile]
                result = hl.inline_triton(
                    """
                    tmp = {lhs} + {rhs}
                    tmp
                    """,
                    args={"lhs": x_val, "rhs": y_val},
                    output_like=x_val,
                )
                out[tile] = result
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        y = torch.randn_like(x)
        code, result = code_and_output(kernel, (x, y))
        self.assertExpectedJournal(code)
        torch.testing.assert_close(result, x + y)

    def test_inline_triton_multi_output(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(
            a: torch.Tensor, b: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            sum_out = torch.empty_like(a)
            diff_out = torch.empty_like(a)
            for tile in hl.tile(a.shape):
                a_val = a[tile]
                b_val = b[tile]
                sum_val, diff_val = hl.inline_triton(
                    """
                    sum_val = {0} + {1}
                    diff_val = {0} - {1}
                    sum_val, diff_val
                    """,
                    args=(a_val, b_val),
                    output_like=(a_val, a_val),
                )
                sum_out[tile] = sum_val
                diff_out[tile] = diff_val
            return sum_out, diff_out

        a = torch.randn(64, device=DEVICE, dtype=torch.float32)
        b = torch.randn_like(a)
        code, (sum_result, diff_result) = code_and_output(kernel, (a, b))
        self.assertExpectedJournal(code)
        torch.testing.assert_close(sum_result, a + b)
        torch.testing.assert_close(diff_result, a - b)

    def test_inline_triton_list_args_reuse(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)

            for tile in hl.tile(x.shape):
                x_val = x[tile]
                y_val = y[tile]
                out[tile] = hl.inline_triton(
                    """
                    triple = {0} + {0} + {0}
                    triple + {1}
                    """,
                    args=[x_val, y_val],
                    output_like=x_val,
                )

            return out

        x = torch.randn(16, device=DEVICE, dtype=torch.float32)
        y = torch.randn_like(x)
        code, out = code_and_output(kernel, (x, y))
        self.assertExpectedJournal(code)
        torch.testing.assert_close(out, 3 * x + y)

    def test_inline_triton_invalid_output_like(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                out[tile] = hl.inline_triton(
                    "{0}\n",
                    args=(x_val,),
                    output_like="not a tensor",
                )
            return out

        x = torch.randn(8, device=DEVICE, dtype=torch.float32)
        with self.assertRaises(helion.exc.InvalidAPIUsage):
            code_and_output(kernel, (x,))

    def test_inline_triton_invalid_mapping_key(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                out[tile] = hl.inline_triton(
                    "{bad}\n",
                    args={0: x_val},
                    output_like=x_val,
                )
            return out

        x = torch.randn(8, device=DEVICE, dtype=torch.float32)
        with self.assertRaises(helion.exc.InvalidAPIUsage):
            code_and_output(kernel, (x,))

    def test_inline_triton_static_assert_mismatch(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                out_like = x_val
                out[tile] = hl.inline_triton(
                    """
                    reshaped = tl.reshape({0}, (1, {0}.shape[0]))
                    reshaped
                    """,
                    args=(x_val,),
                    output_like=out_like,
                )
            return out

        x = torch.randn(8, device=DEVICE, dtype=torch.float32)
        with self.assertRaises(
            (
                triton.compiler.errors.CompilationError,
                RuntimeError,
                helion.exc.InternalError,
            )
        ):
            code_and_output(kernel, (x,))

    def test_inline_triton_side_effect_only(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            flag = torch.zeros(1, device=x.device, dtype=x.dtype)
            for tile in hl.tile(x.shape):
                val = x[tile]
                _ = hl.inline_triton(
                    "tl.store({0}, {1}[0])",
                    args=(flag, val),
                    output_like=None,
                )
            return flag

        x = torch.randn(1, device=DEVICE, dtype=torch.float32)
        bound = kernel.bind((x,))
        code = bound.to_triton_code(bound.config_spec.default_config())
        self.assertIn("tl.store(", code)

    def test_inline_triton_none_output_allows_terminal_statement(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(grad_x_lock: torch.Tensor) -> torch.Tensor:
            for _ in hl.tile(grad_x_lock.shape):
                hl.inline_triton(
                    """
                    while tl.atomic_cas({0} + {1}, 0, 1) == 1:
                        pass
                    """,
                    args=(grad_x_lock, 0),
                    output_like=None,
                )
            return grad_x_lock

        grad_x_lock = torch.ones(4, device=DEVICE, dtype=torch.int32)
        bound = kernel.bind((grad_x_lock,))
        code = bound.to_triton_code(bound.config_spec.default_config())
        self.assertIn("while tl.atomic_cas", code)
        self.assertNotIn("_host_tensor", code)

    def test_inline_ctx(self) -> None:
        """Aspirational test modeled after _attn_fwd_ws_persistent in tritonbench.

        Uses nested hl.tile: outer GRID loop (one CTA per row-group) with
        sibling DEVICE loops inside each async_task -- all in the same kernel.
        Tasks communicate via barrier arrive/wait through hl.inline_triton.

        Each group performs a distinct global-memory transformation so the
        generated code has visible tl.load / tl.store with real operations:

        Pipeline: out = (x + y) * 2 + 1
          load (out=x) --(data_full)--> compute (out+=y)
          --(acc_full)--> correction (out*=2) --(o_full)--> epilogue (out+=1)
        """

        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            M, N = x.shape
            out = torch.empty_like(x)
            for tile_m in hl.tile(M):
                with hl.inline_ctx("tlx.async_tasks()"):
                    # Allocate barriers for inter-task communication
                    hl.inline_triton(
                        "{barrier} = tlx.alloc_barrier(1)\n",
                        args={"barrier": "data_full"},
                        output_like=None,
                    )
                    hl.inline_triton(
                        "{barrier} = tlx.alloc_barrier(1)\n",
                        args={"barrier": "data_empty"},
                        output_like=None,
                    )
                    hl.inline_triton(
                        "{barrier} = tlx.alloc_barrier(1)\n",
                        args={"barrier": "acc_full"},
                        output_like=None,
                    )
                    hl.inline_triton(
                        "{barrier} = tlx.alloc_barrier(1)\n",
                        args={"barrier": "o_full"},
                        output_like=None,
                    )
                    hl.inline_triton(
                        "{barrier} = tlx.alloc_barrier(1)\n",
                        args={"barrier": "o_empty"},
                        output_like=None,
                    )
                    # Load task: stage x into output buffer
                    with hl.inline_ctx(
                        "tlx.async_task(num_warps={num_warps}, registers={registers})",
                        args={"num_warps": 1, "registers": 24},
                    ):
                        for tile_n in hl.tile(N):
                            hl.inline_triton(
                                "tlx.barrier_wait({barrier}[0], 0)\n",
                                args={"barrier": "data_empty"},
                                output_like=None,
                            )
                            out[tile_m, tile_n] = x[tile_m, tile_n]
                            hl.inline_triton(
                                "tlx.barrier_arrive({barrier}[0])\n",
                                args={"barrier": "data_full"},
                                output_like=None,
                            )
                    # Compute task: add y to staged value
                    with hl.inline_ctx(
                        "tlx.async_task(num_warps={num_warps}, registers={registers}, replicate={replicate})",
                        args={
                            "num_warps": 4,
                            "registers": 168,
                            "replicate": 2,
                        },
                    ):
                        for tile_n in hl.tile(N):
                            hl.inline_triton(
                                "tlx.barrier_wait({barrier}[0], 0)\n",
                                args={"barrier": "data_full"},
                                output_like=None,
                            )
                            staged = out[tile_m, tile_n]
                            y_val = y[tile_m, tile_n]
                            out[tile_m, tile_n] = staged + y_val
                            hl.inline_triton(
                                "tlx.barrier_arrive({barrier1}[0])\ntlx.barrier_arrive({barrier2}[0])\n",
                                args={"barrier1": "acc_full", "barrier2": "data_empty"},
                                output_like=None,
                            )
                    # Correction task: scale the accumulated result
                    with hl.inline_ctx("tlx.async_task('default')"):
                        for tile_n in hl.tile(N):
                            hl.inline_triton(
                                "tlx.barrier_wait({barrier}[0], 0)\n",
                                args={"barrier": "acc_full"},
                                output_like=None,
                            )
                            acc = out[tile_m, tile_n]
                            out[tile_m, tile_n] = acc * 2.0
                            hl.inline_triton(
                                "tlx.barrier_arrive({barrier}[0])\n",
                                args={"barrier": "o_full"},
                                output_like=None,
                            )
                    # Epilogue task: apply final bias and store
                    with hl.inline_ctx(
                        "tlx.async_task(num_warps={num_warps}, registers={registers})",
                        args={"num_warps": 1, "registers": 24},
                    ):
                        for tile_n in hl.tile(N):
                            hl.inline_triton(
                                "tlx.barrier_wait({barrier}[0], 0)\n",
                                args={"barrier": "o_full"},
                                output_like=None,
                            )
                            result = out[tile_m, tile_n]
                            out[tile_m, tile_n] = result + 1.0
                            hl.inline_triton(
                                "tlx.barrier_arrive({barrier}[0])\n",
                                args={"barrier": "o_empty"},
                                output_like=None,
                            )
            return out

        x = torch.randn(4, 32, device=DEVICE, dtype=torch.float32)
        y = torch.randn_like(x)
        bound = kernel.bind((x, y))
        code = bound.to_triton_code(bound.config_spec.default_config())
        self.assertIn("with tlx.async_tasks():", code)
        self.assertIn("with tlx.async_task('default'):", code)
        self.assertIn(
            "with tlx.async_task(num_warps=4, registers=168, replicate=2):", code
        )
        self.assertIn("with tlx.async_task(num_warps=1, registers=24):", code)
        self.assertIn("tlx.barrier_wait(", code)
        self.assertIn("tlx.barrier_arrive(", code)
        self.assertExpectedJournal(code)
