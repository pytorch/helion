"""Tokamax SplashAttention invocation for the comparison benchmark."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import Any

from .configs import CAUSAL_TOKAMAX_CONFIGS
from .configs import DENSE_TOKAMAX_CONFIGS

if TYPE_CHECKING:
    from collections.abc import Callable


def make_tokamax_attention(
    mode: str,
    head_dim: int,
    sequence_length: int,
    *,
    batching: str = "vmap_batch",
) -> tuple[Callable[[Any, Any, Any], Any], dict[str, Any]]:
    """Create one JIT-compiled SplashAttention call with a fixed configuration."""
    import jax
    from tokamax._src.ops.experimental.tpu.splash_attention import (
        splash_attention_kernel,
    )
    from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask

    configs = DENSE_TOKAMAX_CONFIGS if mode == "dense" else CAUSAL_TOKAMAX_CONFIGS
    config_values = configs[head_dim, sequence_length]
    config_kwargs = dict(config_values)
    for name in ("q_layout", "k_layout", "v_layout"):
        config_kwargs[name] = splash_attention_kernel.QKVLayout[config_kwargs[name]]

    mask_shape = (sequence_length, sequence_length)
    if mode == "dense":
        mask = splash_attention_mask.FullMask(mask_shape)
    else:
        mask = splash_attention_mask.CausalMask(mask_shape)
    kernel = splash_attention_kernel.make_splash_mha_single_device(
        mask,
        config=splash_attention_kernel.SplashConfig(**config_kwargs),
    )

    @jax.jit
    def attention(query: jax.Array, key: jax.Array, value: jax.Array) -> jax.Array:
        # Keep the recorded dense path unchanged; causal inputs are unscaled Q.
        if mode == "causal":
            query = query * (1.0 / math.sqrt(head_dim))
        if batching == "vmap_batch":
            return jax.vmap(kernel, in_axes=(0, 0, 0))(query, key, value)
        shape = query.shape
        flat = (-1, sequence_length, head_dim)
        return kernel(
            query.reshape(flat), key.reshape(flat), value.reshape(flat)
        ).reshape(shape)

    return attention, config_values
