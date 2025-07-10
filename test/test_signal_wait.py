from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl


class TestWait(TestCase):
    def test_wait_basic(self):
        @helion.kernel
        def gmem_wait_kernel(signal_pad: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(signal_pad)
            (n,) = signal_pad.shape
            for i in hl.grid(n):
                hl.wait(signal_pad, [i], signal=1)
                out[i] = i

            return out

        signal_pad = torch.ones(4, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(gmem_wait_kernel, (signal_pad,))
        torch.testing.assert_close(
            result, torch.arange(4, device=DEVICE, dtype=torch.int32)
        )
        self.maxDiff = None
        self.assertExpectedJournal(code)

    def test_wait_2d_tile(self):
        @helion.kernel
        def wait_for_2d_tile_kernel(
            signal_pad: torch.Tensor, x: torch.Tensor
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            (n, m) = x.shape
            for tile_n, tile_m in hl.tile([n, m]):
                hl.wait(signal_pad, [tile_n.id, tile_m.id], signal=1)
                out[tile_n, tile_m] = x[tile_n, tile_m]
            return out

        signal_pad = torch.ones([4, 4], device=DEVICE, dtype=torch.int32)
        x = torch.randn([64, 64], device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            wait_for_2d_tile_kernel,
            (signal_pad, x),
            block_size=[16, 16],
        )

        torch.testing.assert_close(result, x)
        self.assertExpectedJournal(code)

    def test_signal_basic(self):
        @helion.kernel
        def gmem_signal_scalar_bar_kernel(signal_pad: torch.Tensor) -> torch.Tensor:
            (n,) = signal_pad.shape
            for i in hl.grid(n):
                hl.signal(signal_pad, [i], signal=1)
            return signal_pad

        signal_pad = torch.zeros(4, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(gmem_signal_scalar_bar_kernel, (signal_pad,))
        torch.testing.assert_close(
            result, torch.ones(4, device=DEVICE, dtype=torch.int32)
        )
        self.assertExpectedJournal(code)

    def test_signal_cas(self):
        @helion.kernel
        def gmem_signal_cas_kernel(signal_pad: torch.Tensor) -> torch.Tensor:
            (n,) = signal_pad.shape
            for i in hl.grid(n):
                hl.signal(signal_pad, [i], signal=1, wait_for=0, op="atomic_cas")
            return signal_pad

        signal_pad = torch.zeros(4, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(gmem_signal_cas_kernel, (signal_pad,))
        torch.testing.assert_close(
            result, torch.ones(4, device=DEVICE, dtype=torch.int32)
        )
        self.assertExpectedJournal(code)

    def test_signal_multiple(self):
        @helion.kernel
        def gmem_signal_tensor_bar_kernel(signal_pad: torch.Tensor) -> torch.Tensor:
            (n,) = signal_pad.shape
            for tile in hl.tile(n):
                hl.signal(signal_pad, [tile], signal=1)
            return signal_pad

        signal_pad = torch.zeros(16, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(
            gmem_signal_tensor_bar_kernel,
            (signal_pad,),
            block_size=[4],
        )
        torch.testing.assert_close(
            result, torch.ones(16, device=DEVICE, dtype=torch.int32)
        )
        self.assertExpectedJournal(code)

    def test_sent_recieve_cta(self):
        @helion.kernel
        def gmem_signal_n_wait_kernel(signal_pad: torch.Tensor) -> torch.Tensor:
            (n,) = signal_pad.shape
            for i in hl.grid(n):  # first N ctas sends signal
                hl.signal(signal_pad, [i], signal=1)
            for i in hl.grid(n):  # last N ctas waits for signal
                hl.wait(signal_pad, [i], signal=1)
            return signal_pad

        signal_pad = torch.zeros(4, device=DEVICE, dtype=torch.int32)

        code, result = code_and_output(gmem_signal_n_wait_kernel, (signal_pad,))
        torch.testing.assert_close(
            result, torch.ones(4, device=DEVICE, dtype=torch.int32)
        )
        self.assertIn("helion.runtime.triton_send_signal", code)
        self.assertIn("helion.runtime.triton_wait_signal", code)


if __name__ == "__main__":
    unittest.main()
