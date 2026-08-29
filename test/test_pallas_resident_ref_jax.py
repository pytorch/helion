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

    def test_indirect_dma_scratch_composes_resident_subviews(self) -> None:
        """Grouped state DMA remains addressable through resident Ref views."""

        def state_roundtrip(
            indices: torch.Tensor, table: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            out = torch.empty(
                [indices.size(0), *table.shape[2:]],
                dtype=table.dtype,
                device=table.device,
            )
            for _ in hl.grid(1):
                for tile_b in hl.tile(indices.size(0)):
                    selected_indices = hl.load(indices, [tile_b])
                    selected = hl.load(
                        table,
                        [selected_indices, slice(None), slice(None), slice(None)],
                    )
                    state_0 = selected[:, 0, :, :]
                    state_1 = selected[:, 1, :, :]
                    state_2 = selected[:, 2, :, :]
                    out[tile_b, :, :] = state_0 + state_1 + state_2
                    history = hl.arange(3)[None, :, None, None]
                    updated = torch.where(
                        history == 0,
                        state_1[:, None, :, :],
                        torch.where(
                            history == 1,
                            state_2[:, None, :, :],
                            state_0[:, None, :, :],
                        ),
                    )
                    hl.store(
                        table,
                        [selected_indices, slice(None), slice(None), slice(None)],
                        updated,
                    )
            return out, table

        kernel = helion.kernel(
            state_roundtrip,
            config=helion.Config(
                block_sizes=[128],
                pallas_loop_type="fori_loop",
                pallas_load_buffer_count=[1, 2],
                pallas_indirect_access_mode="dma",
            ),
            static_shapes=True,
            backend="pallas",
        )
        indices = jnp.tile(jnp.arange(128, dtype=jnp.int32), 2)
        table = jax.random.normal(
            jax.random.key(0),
            (512, 3, 16, 128),
            dtype=jnp.bfloat16,
        )
        first = table[indices[:128]]
        after_first = table.at[indices[:128]].set(
            jnp.stack([first[:, 1], first[:, 2], first[:, 0]], axis=1)
        )
        second = after_first[indices[128:]]
        expected_out = jnp.concatenate(
            (
                first[:, 0] + first[:, 1] + first[:, 2],
                second[:, 0] + second[:, 1] + second[:, 2],
            )
        )
        expected_table = after_first.at[indices[128:]].set(
            jnp.stack([second[:, 1], second[:, 2], second[:, 0]], axis=1)
        )

        out, updated_table = jax.block_until_ready(
            jax.jit(kernel.jax_fn, donate_argnums=(1,))(indices, table)
        )
        np.testing.assert_array_equal(np.asarray(out), np.asarray(expected_out))
        np.testing.assert_array_equal(
            np.asarray(updated_table), np.asarray(expected_table)
        )
