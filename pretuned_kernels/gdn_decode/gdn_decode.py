"""Decode-only Gated Delta Network with an indirectly indexed state cache."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import helion
import helion.language as hl

GDN_DECODE_SHAPES = ((512, 2, 128, 128),)

GDN_DECODE_CONFIG = helion.Config(
    block_sizes=[32],
    pallas_loop_type="fori_loop",
    pallas_load_buffer_count=[2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1],
    pallas_indirect_access_mode="dma",
)


@helion.kernel(
    backend="pallas",
    static_shapes=True,
    dot_precision="default",
    config=GDN_DECODE_CONFIG,
)
def gdn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
    distribution: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    has_initial_state: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one recurrent GDN update per active decode token.

    Shapes follow tpu-inference's public decode kernel, with expanded
    ``a_log`` and ``dt_bias`` tensors of shape ``[heads, key_dim]``. The
    returned recurrent state aliases ``recurrent_state``.
    """
    tokens, heads, key_dim = q.size()
    value_tokens, value_heads, value_dim = v.size()
    state_count, state_heads, state_key_dim, state_value_dim = recurrent_state.size()
    assert k.size() == q.size() and g.size() == q.size()
    assert beta.size() == v.size()
    assert value_tokens == tokens and value_heads == heads
    assert state_heads == heads
    assert state_key_dim == key_dim and state_value_dim == value_dim
    assert state_indices.size(0) == tokens
    assert has_initial_state.size(0) == tokens and state_count >= tokens
    assert a_log.size() == (heads, key_dim)
    assert dt_bias.size() == (heads, key_dim)
    out = torch.empty_like(v)
    for _ in hl.grid(1):
        for tile_tokens in hl.tile(tokens):
            indices = hl.load(state_indices, [tile_tokens])
            gathered_state = hl.load(
                recurrent_state,
                [indices, slice(None), slice(None), slice(None)],
            )
            active = tile_tokens.index < distribution[0]
            has_state = has_initial_state[tile_tokens] != 0
            valid_scale = (active & has_state).to(torch.float32)[:, None, None, None]
            state_tile = gathered_state * valid_scale
            q_tile = q[tile_tokens, :, :].to(torch.float32)
            k_tile = k[tile_tokens, :, :].to(torch.float32)
            v_tile = v[tile_tokens, :, :].to(torch.float32)
            gate_input = g[tile_tokens, :, :].to(torch.float32)

            q_tile = q_tile * torch.rsqrt(
                torch.sum(q_tile * q_tile, dim=-1, keepdim=True) + 1e-6
            )
            k_tile = k_tile * torch.rsqrt(
                torch.sum(k_tile * k_tile, dim=-1, keepdim=True) + 1e-6
            )
            q_tile = q_tile * scale
            gate_input = gate_input + dt_bias[:, :].to(torch.float32)
            gate = -torch.exp(a_log[:, :].to(torch.float32)) * F.softplus(gate_input)

            decayed_state = state_tile * torch.exp(gate.unsqueeze(-1))
            batch_heads = tile_tokens.block_size * heads
            projected = hl.dot(
                k_tile.reshape(batch_heads, 1, key_dim),
                decayed_state.reshape(batch_heads, key_dim, value_dim),
                out_dtype=torch.float32,
            ).reshape(tile_tokens.block_size, heads, value_dim)
            residual = torch.sigmoid(beta[tile_tokens, :, :].to(torch.float32)) * (
                v_tile - projected
            )
            new_state = decayed_state + k_tile.unsqueeze(-1) * residual.unsqueeze(-2)
            output = hl.dot(
                q_tile.reshape(batch_heads, 1, key_dim),
                new_state.reshape(batch_heads, key_dim, value_dim),
                out_dtype=torch.float32,
            ).reshape(tile_tokens.block_size, heads, value_dim)

            active_scale = active.to(out.dtype)[:, None, None]
            out[tile_tokens, :, :] = output.to(out.dtype) * active_scale
            hl.store(
                recurrent_state,
                [indices, slice(None), slice(None), slice(None)],
                new_state * active_scale[:, None, :, :]
                + gathered_state * (1 - active_scale[:, None, :, :]),
            )
    return out, recurrent_state


def gdn_decode_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
    distribution: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    has_initial_state: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for correctness checks."""
    indices = state_indices.to(torch.int64)
    gathered = recurrent_state[indices]
    active = torch.arange(q.size(0), device=q.device) < distribution[0]
    valid = active & has_initial_state.to(torch.bool)
    state = torch.where(valid[:, None, None, None], gathered, 0.0)

    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)
    v_f32 = v.to(torch.float32)
    q_f32 = q_f32 * torch.rsqrt(torch.sum(q_f32 * q_f32, dim=-1, keepdim=True) + 1e-6)
    k_f32 = k_f32 * torch.rsqrt(torch.sum(k_f32 * k_f32, dim=-1, keepdim=True) + 1e-6)
    q_f32 = q_f32 * scale
    gate = -torch.exp(a_log.to(torch.float32)) * F.softplus(
        g.to(torch.float32) + dt_bias.to(torch.float32)
    )
    decayed = state * torch.exp(gate.unsqueeze(-1))
    projected = torch.matmul(k_f32.unsqueeze(-2), decayed).squeeze(-2)
    residual = torch.sigmoid(beta.to(torch.float32)) * (v_f32 - projected)
    updated = decayed + k_f32.unsqueeze(-1) * residual.unsqueeze(-2)
    out = torch.matmul(q_f32.unsqueeze(-2), updated).squeeze(-2).to(v.dtype)
    out = torch.where(active[:, None, None], out, torch.zeros_like(out))

    selected_out = torch.where(active[:, None, None, None], updated, gathered)
    state_out = recurrent_state.clone()
    state_out[indices] = selected_out
    return out, state_out


def main() -> None:
    """Run the representative TPU correctness case through JAX export."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    tokens, heads, key_dim, value_dim = GDN_DECODE_SHAPES[0]
    assert key_dim == value_dim
    dim = key_dim
    keys = jax.random.split(jax.random.key(1), 8)
    q, k, v, g = (
        jax.random.normal(key, (tokens, heads, dim), dtype=jnp.bfloat16) * 0.1
        for key in keys[:4]
    )
    beta = jax.random.normal(keys[4], (tokens, heads, dim), dtype=jnp.bfloat16)
    a_log = jnp.broadcast_to(
        jax.random.normal(keys[5], (heads,), dtype=jnp.float32)[:, None],
        (heads, dim),
    )
    dt_bias = jnp.broadcast_to(
        jax.random.normal(keys[6], (heads,), dtype=jnp.float32)[:, None],
        (heads, dim),
    )
    state = (
        jax.random.normal(keys[7], (tokens, heads, dim, dim), dtype=jnp.float32) * 0.01
    )
    indices = jnp.arange(tokens - 1, -1, -1, dtype=jnp.int32)
    distribution = jnp.asarray([tokens, tokens, tokens], dtype=jnp.int32)
    has_state = jnp.ones((tokens,), dtype=jnp.int32)
    scale = dim**-0.5

    def run_kernel(
        q_arg: jax.Array,
        k_arg: jax.Array,
        v_arg: jax.Array,
        g_arg: jax.Array,
        state_arg: jax.Array,
        indices_arg: jax.Array,
        distribution_arg: jax.Array,
        beta_arg: jax.Array,
        a_log_arg: jax.Array,
        dt_bias_arg: jax.Array,
        has_state_arg: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return gdn_decode.jax_fn(
            q_arg,
            k_arg,
            v_arg,
            g_arg,
            state_arg,
            indices_arg,
            distribution_arg,
            beta_arg,
            a_log_arg,
            dt_bias_arg,
            has_state_arg,
            hl.constexpr(scale),
        )

    run = jax.jit(run_kernel, donate_argnums=(4,))
    out, state_out = jax.block_until_ready(
        run(
            q,
            k,
            v,
            g,
            jnp.copy(state),
            indices,
            distribution,
            beta,
            a_log,
            dt_bias,
            has_state,
        )
    )

    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    q_f32 *= jax.lax.rsqrt(jnp.sum(q_f32 * q_f32, axis=-1, keepdims=True) + 1e-6)
    k_f32 *= jax.lax.rsqrt(jnp.sum(k_f32 * k_f32, axis=-1, keepdims=True) + 1e-6)
    q_f32 *= scale
    gate = -jnp.exp(a_log) * jax.nn.softplus(g.astype(jnp.float32) + dt_bias)
    selected = state[indices]
    decayed = selected * jnp.exp(gate[..., None])
    projected = jnp.einsum("thk,thkv->thv", k_f32, decayed)
    residual = jax.nn.sigmoid(beta.astype(jnp.float32)) * (
        v.astype(jnp.float32) - projected
    )
    expected_selected = decayed + k_f32[..., None] * residual[..., None, :]
    expected_out = jnp.einsum("thk,thkv->thv", q_f32, expected_selected).astype(v.dtype)
    expected_state = state.at[indices].set(expected_selected)
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(expected_out), rtol=2e-2, atol=2e-2
    )
    np.testing.assert_allclose(
        np.asarray(state_out), np.asarray(expected_state), rtol=2e-2, atol=2e-2
    )


if __name__ == "__main__":
    main()
