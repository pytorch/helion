"""A cross-loop kernel whose launcher state is first requested inside a CUDA graph capture.

The capture stream has no state yet (the kernel ran eagerly on another stream), so the launcher allocates it during the
capture. The state must be zeroed outside the graph: replays continue the epochs instead of re-zeroing them.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
import helion.language as hl
import helion.runtime.triton.launcher as launcher


def _grouped_affine_chain(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    m, hidden = x.size()
    _, twice_intermediate = w13.size()
    intermediate = twice_intermediate // 2
    _, out_features = w2.size()
    hl.specialize(group_size)
    groups = intermediate // group_size
    gate_up = torch.empty((m, twice_intermediate), dtype=x.dtype, device=x.device)
    activation = torch.empty((m, intermediate), dtype=x.dtype, device=x.device)
    activation_scale = torch.empty((m, groups), dtype=torch.float32, device=x.device)
    out = torch.empty((m, out_features), dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, twice_intermediate]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(hidden, block_size=32):
            acc = torch.addmm(acc, x[tile_m, tile_k], w13[tile_k, tile_n])
        gate_up[tile_m, tile_n] = acc.to(x.dtype)
    for tile_m, tile_i in hl.tile([m, intermediate], block_size=[1, group_size]):
        gate = gate_up[tile_m, tile_i].to(torch.float32)
        up = gate_up[tile_m, tile_i + intermediate].to(torch.float32)
        activated = gate * up
        activation[tile_m, tile_i] = activated.to(x.dtype)
        activation_scale[tile_m, tile_i.id] = (
            torch.amax(torch.abs(activated), dim=-1) + 1
        )
    for tile_m, tile_n in hl.tile([m, out_features]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(intermediate, block_size=group_size):
            values = activation[tile_m, tile_k].to(torch.float32)
            scale = activation_scale[tile_m, tile_k.id].to(torch.float32)
            acc = torch.addmm(
                acc, values * scale[:, None], w2[tile_k, tile_n].to(torch.float32)
            )
        out[tile_m, tile_n] = acc
    return out


@skipIfNotCUDA()
@skipIfRefEager("persistent tile-dependency codegen is unavailable")
class TestPersistentStateCapture(unittest.TestCase):
    def test_capture_allocated_state_is_not_reset_by_replay(self) -> None:
        torch.manual_seed(0)
        x = torch.rand((2, 64), device=DEVICE, dtype=torch.float16)
        w13 = torch.rand((64, 256), device=DEVICE, dtype=torch.float16)
        w2 = torch.rand((128, 64), device=DEVICE, dtype=torch.float16)
        kernel = helion.kernel(
            _grouped_affine_chain,
            config=helion.Config(
                block_sizes=[1, 16, 1, 16],
                pid_type="persistent_blocked",
                cross_loop_pipeline="static",
                num_sm_multiplier=1,
                num_warps=4,
                num_stages=2,
            ),
            static_shapes=True,
        )
        args = (x, w13, w2, 32)
        self.assertIn(
            "_persistent_state_specs",
            kernel.bind(args).to_triton_code(kernel.configs[0]),
        )
        gate, up = (x.float() @ w13.float()).half().chunk(2, dim=-1)
        activated = gate.float() * up.float()
        scale = activated.abs().reshape(2, 4, 32).amax(dim=-1) + 1
        expected = (
            activated.half().float().reshape(2, 4, 32) * scale[:, :, None]
        ).reshape(2, 128) @ w2.float()
        torch.testing.assert_close(kernel(*args), expected, rtol=3e-2, atol=3e-2)
        torch.cuda.synchronize()

        captured = []
        get_state = launcher._get_persistent_state

        def record(*a: object, **kw: object) -> torch.Tensor:
            state = get_state(*a, **kw)  # type: ignore[arg-type]
            if torch.cuda.is_current_stream_capturing():
                captured.append(state)
            return state

        graph = torch.cuda.CUDAGraph()
        with (
            patch.object(launcher, "_get_persistent_state", record),
            torch.cuda.graph(graph),
        ):
            out = kernel(*args)
        self.assertEqual(len(captured), 1)
        state = captured[0].view(torch.int32)
        self.assertEqual(torch.count_nonzero(state).item(), 0)
        after = []
        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(out, expected, rtol=3e-2, atol=3e-2)
            after.append(state.clone())
        # Epochs and counters grow by one launch per replay; a recorded fill would reset them every replay.
        self.assertGreater(torch.count_nonzero(after[0]).item(), 0)
        self.assertTrue(torch.equal(after[1], 2 * after[0]))
        self.assertTrue(torch.equal(after[2], 3 * after[0]))


if __name__ == "__main__":
    unittest.main()
