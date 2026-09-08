"""Decode-only causal Conv1D with an indirectly indexed state cache."""

from __future__ import annotations

import torch

import helion
import helion.language as hl

CAUSAL_CONV1D_SHAPES = ((512, 4, 128, 4),)

CAUSAL_CONV1D_CONFIG = helion.Config(
    block_sizes=[128],
    pallas_loop_type="fori_loop",
    pallas_load_buffer_count=[2, 1, 1, 1, 1, 1, 1],
    pallas_indirect_access_mode="dma",
)


@helion.kernel(
    backend="pallas",
    static_shapes=True,
    config=CAUSAL_CONV1D_CONFIG,
)
def _causal_conv1d_decode_kernel(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    conv_bias: torch.Tensor,
    state_indices: torch.Tensor,
    distribution: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, heads, head_dim = x.size()
    kernel_size, weight_heads, weight_dim = conv_weight.size()
    state_count, history, state_heads, state_dim = conv_state.size()
    assert weight_heads == heads and weight_dim == head_dim
    assert state_heads == heads and state_dim == head_dim
    assert conv_bias.size() == (heads, head_dim)
    assert history == kernel_size - 1
    assert state_indices.size(0) == tokens
    assert has_initial_state.size(0) == tokens and state_count >= tokens
    kernel_size = hl.specialize(kernel_size)
    assert kernel_size == 4

    out = torch.empty_like(x)
    for _ in hl.grid(1):
        for tile_tokens in hl.tile(tokens):
            indices = hl.load(state_indices, [tile_tokens])
            selected = hl.load(
                conv_state,
                [indices, slice(None), slice(None), slice(None)],
            )
            active = tile_tokens.index < distribution[0]
            has_state = has_initial_state[tile_tokens] != 0
            valid_scale = (active & has_state).to(selected.dtype)[:, None, None]

            original_0 = selected[:, 0, :, :]
            original_1 = selected[:, 1, :, :]
            original_2 = selected[:, 2, :, :]
            state_0 = original_0 * valid_scale
            state_1 = original_1 * valid_scale
            state_2 = original_2 * valid_scale
            x_tile = x[tile_tokens, :, :]

            # Loading the compact weight block once avoids treating four static
            # slices as interchangeable streamed operands in the fori planner.
            weight_block = conv_weight[:, :, :]
            weight_index = hl.arange(4)[:, None, None]
            weight_0 = torch.sum(weight_block * (weight_index == 0), dim=0).to(
                torch.float32
            )
            weight_1 = torch.sum(weight_block * (weight_index == 1), dim=0).to(
                torch.float32
            )
            weight_2 = torch.sum(weight_block * (weight_index == 2), dim=0).to(
                torch.float32
            )
            weight_3 = torch.sum(weight_block * (weight_index == 3), dim=0).to(
                torch.float32
            )

            acc = x_tile.to(torch.float32) * weight_3
            acc = acc + state_0.to(torch.float32) * weight_0
            acc = acc + state_1.to(torch.float32) * weight_1
            acc = acc + state_2.to(torch.float32) * weight_2
            acc = acc + conv_bias[:, :].to(torch.float32)
            active_scale = active.to(x.dtype)[:, None, None]
            out[tile_tokens, :, :] = acc.to(x.dtype) * active_scale

            history_index = hl.arange(3)[None, :, None, None]
            updated = torch.where(
                history_index == 0,
                state_1[:, None, :, :],
                torch.where(
                    history_index == 1,
                    state_2[:, None, :, :],
                    x_tile[:, None, :, :],
                ),
            )
            original = torch.where(
                history_index == 0,
                original_0[:, None, :, :],
                torch.where(
                    history_index == 1,
                    original_1[:, None, :, :],
                    original_2[:, None, :, :],
                ),
            )
            hl.store(
                conv_state,
                [indices, slice(None), slice(None), slice(None)],
                updated * active_scale[:, None, :, :]
                + original * (1 - active_scale[:, None, :, :]),
            )
    return out, conv_state


def causal_conv1d_decode(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    state_indices: torch.Tensor,
    distribution: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run width-four depthwise Conv1D for one decode token per request.

    The public tensors follow tpu-inference's flat ``ragged_causal_conv1d``
    layout. The wrapper exposes a TPU-native ``[heads, 128]`` view to Helion
    without changing storage or the returned input/output signature.
    """
    tokens, dim = x.size()
    state_count, history, state_dim = conv_state.size()
    weight_dim, weight_channels, kernel_size = conv_weight.size()
    assert state_dim == dim and weight_dim == dim
    assert weight_channels == 1 and history == kernel_size - 1
    assert dim % 128 == 0
    heads = dim // 128
    x_view = x.reshape(tokens, heads, 128)
    state_view = conv_state.reshape(state_count, history, heads, 128)
    weight_view = conv_weight[:, 0, :].transpose(0, 1).reshape(kernel_size, heads, 128)
    bias_view = conv_bias.reshape(heads, 128)
    out, state_out = _causal_conv1d_decode_kernel(
        x_view,
        weight_view,
        state_view,
        bias_view,
        state_indices,
        distribution,
        has_initial_state,
    )
    return out.reshape_as(x), state_out.reshape_as(conv_state)


def causal_conv1d_decode_reference(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    state_indices: torch.Tensor,
    distribution: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for correctness checks."""
    indices = state_indices.to(torch.int64)
    selected = conv_state[indices]
    active = torch.arange(x.size(0), device=x.device) < distribution[0]
    valid = active & has_initial_state.to(torch.bool)
    selected_for_compute = torch.where(
        valid[:, None, None], selected, torch.zeros_like(selected)
    )
    lhs = torch.cat((selected_for_compute, x[:, None, :]), dim=1)
    weights = conv_weight[:, 0, :].transpose(0, 1)
    out = torch.sum(
        lhs.to(torch.float32) * weights[None, :, :].to(torch.float32), dim=1
    )
    out = (out + conv_bias.to(torch.float32)).to(x.dtype)
    out = torch.where(active[:, None], out, torch.zeros_like(out))

    updated = torch.cat((selected_for_compute[:, 1:, :], x[:, None, :]), dim=1)
    selected_out = torch.where(active[:, None, None], updated, selected)
    state_out = conv_state.clone()
    state_out[indices] = selected_out
    return out, state_out


def main() -> None:
    """Run the representative TPU correctness case through JAX export."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    tokens, heads, head_dim, kernel_size = CAUSAL_CONV1D_SHAPES[0]
    dim = heads * head_dim
    keys = jax.random.split(jax.random.key(0), 4)
    x = jax.random.normal(keys[0], (tokens, dim), dtype=jnp.bfloat16) * 0.1
    state = (
        jax.random.normal(keys[1], (tokens, kernel_size - 1, dim), dtype=jnp.bfloat16)
        * 0.1
    )
    weight = jax.random.normal(keys[2], (dim, 1, kernel_size), dtype=jnp.bfloat16) * 0.1
    bias = jax.random.normal(keys[3], (dim,), dtype=jnp.bfloat16) * 0.1
    indices = jnp.arange(tokens - 1, -1, -1, dtype=jnp.int32)
    distribution = jnp.asarray([tokens, tokens, tokens], dtype=jnp.int32)
    has_state = jnp.ones((tokens,), dtype=jnp.int32)

    x_view = x.reshape(tokens, heads, head_dim)
    state_view = state.reshape(tokens, kernel_size - 1, heads, head_dim)
    weight_view = weight[:, 0, :].T.reshape(kernel_size, heads, head_dim)
    bias_view = bias.reshape(heads, head_dim)
    run = jax.jit(_causal_conv1d_decode_kernel.jax_fn, donate_argnums=(2,))
    out, state_out = jax.block_until_ready(
        run(
            x_view,
            weight_view,
            jnp.copy(state_view),
            bias_view,
            indices,
            distribution,
            has_state,
        )
    )
    selected = state[indices]
    lhs = jnp.concatenate((selected, x[:, None, :]), axis=1)
    expected_out = jnp.sum(
        lhs.astype(jnp.float32) * weight[:, 0, :].T[None, :, :].astype(jnp.float32),
        axis=1,
    ).astype(x.dtype)
    expected_out = (expected_out + bias).astype(x.dtype)
    expected_selected = jnp.concatenate((selected[:, 1:], x[:, None, :]), axis=1)
    expected_state = state.at[indices].set(expected_selected)
    np.testing.assert_allclose(
        np.asarray(out.reshape(tokens, dim)),
        np.asarray(expected_out),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_array_equal(
        np.asarray(state_out.reshape(state.shape)), np.asarray(expected_state)
    )


if __name__ == "__main__":
    main()
