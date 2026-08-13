"""Real-TPU JAX export coverage for resident Pallas Ref views."""

from __future__ import annotations

import unittest

import torch

import helion
import helion.language as hl

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    HAS_TPU = any(device.platform == "tpu" for device in jax.devices())
except Exception:  # pragma: no cover - JAX is optional or TPU is busy
    HAS_TPU = False


@unittest.skipUnless(HAS_TPU, "requires a real JAX TPU device")
class TestResidentRefJaxExport(unittest.TestCase):
    def test_flatten_worklist_grouping_two_variants(self) -> None:
        """Both physical resident-Ref shapes execute through JAX export."""

        def grouped_worklist(q: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(q)
            for sequence in hl.grid(offsets.size(0) - 1):
                begin = offsets[sequence]
                end = offsets[sequence + 1]
                for tile_q in hl.tile(begin, end):
                    q_block = q[tile_q, :, :]
                    live = tile_q.end - tile_q.begin
                    seed = hl.zeros([1, q.size(1), q.size(2)], dtype=torch.float32)
                    if live >= 1:
                        seed = q_block[live - 1 : live, :, :].float()
                    state = hl.zeros(
                        [tile_q, q.size(1), q.size(2)], dtype=torch.float32
                    )
                    state += seed
                    out[tile_q, :, :] = state.to(out.dtype)
            return out

        kernel = helion.kernel(
            grouped_worklist,
            config=helion.Config(
                block_sizes=[4],
                pallas_loop_type="fori_loop",
                pallas_worklist_grouping=2,
            ),
            static_shapes=True,
            backend="pallas",
        )
        q = jnp.arange(24 * 2 * 128, dtype=jnp.float32).reshape(24, 2, 128)
        offsets = jnp.asarray([0, 4, 12, 24], dtype=jnp.int32)
        out = jax.block_until_ready(jax.jit(kernel.jax_fn)(q, offsets))

        expected = np.empty((24, 2, 128), dtype=np.float32)
        q_host = np.asarray(q)
        for begin, end in ((0, 4), (4, 12), (12, 24)):
            for tile_begin in range(begin, end, 8):
                tile_end = min(tile_begin + 8, end)
                expected[tile_begin:tile_end] = q_host[tile_end - 1 : tile_end]
        np.testing.assert_array_equal(np.asarray(out), expected)
