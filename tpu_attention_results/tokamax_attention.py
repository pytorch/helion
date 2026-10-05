"""Tokamax SplashAttention invocation for the comparison benchmark."""

from __future__ import annotations

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

    if mode == "dense":

        @jax.jit
        def attention(query: object, key: object, value: object) -> object:
            return jax.vmap(kernel, in_axes=(0, 0, 0))(query, key, value)

    else:

        @jax.jit
        def attention(query: object, key: object, value: object) -> object:
            return kernel(query, key, value)

    return attention, config_values
