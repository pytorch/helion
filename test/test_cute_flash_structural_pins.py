"""A decorator config is normalized onto a backend's structural block-size pin.

``examples/attention.py`` pins Triton-shaped configs
(``block_sizes=[64, 64]``) on its kernels. The CuTe fused flash-attention
surfaces admit exactly one tile shape (128x128) and used to pin only the
autotuner's range, so a provided config kept its 64x64 tiles and the kernel
landed in the scalar SIMT lowering (117 ms vs 0.03 ms for the backward).
``BlockSizeSpec.pin_structurally`` now records the pin: ``normalize`` maps
any other provided value onto it with a logged warning naming the key and
the surface. These tests render on the CPU (no CUDA work).
"""

from __future__ import annotations

import logging

from examples.attention import attention
from examples.attention import attention_backward
import pytest
import torch

from test._cute_binding import _cpu_bind

import helion
from helion._testing import skipUnlessBackends

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

TRITON_SHAPED_BACKWARD = helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=2)


def _backward_args(shape: tuple[int, int, int, int]) -> tuple[object, ...]:
    q, k, v, o, do = (torch.empty(shape, dtype=torch.bfloat16) for _ in range(5))
    lse, delta = (torch.empty(shape[:-1], dtype=torch.float32) for _ in range(2))
    return (q, k, v, o, lse, do, delta, 0.125)


def _forward_args(shape: tuple[int, int, int, int]) -> tuple[object, ...]:
    return tuple(torch.empty(shape, dtype=torch.bfloat16) for _ in range(3))


def _bind(fn, config: helion.Config, args: tuple[object, ...], *, backend: str):
    kernel = helion.kernel(
        fn,
        config=config,
        backend=backend,
        static_shapes=True,
        autotune_accuracy_check=False,
        force_autotune=False,
    )
    return _cpu_bind(kernel, args)


@skipUnlessBackends(["cute"])
def test_backward_decorator_config_normalizes_onto_the_pin(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert attention_backward.configs[0].block_sizes == [64, 64]
    bound = _bind(
        attention_backward.fn,
        TRITON_SHAPED_BACKWARD,
        _backward_args((1, 2, 256, 64)),
        backend="cute",
    )
    spec = bound.config_spec
    assert spec.cute_flash_bwd_search_enabled
    assert [s.structural_pin for s in spec.block_sizes] == [128, 128]
    config = bound._implicit_config()
    assert config is not None and config.block_sizes == [64, 64]
    with caplog.at_level(logging.WARNING, logger="helion.autotuner.config_spec"):
        normalized = bound._normalized_config_copy(config)
        bound._normalized_config_copy(config)
    assert normalized.block_sizes == [128, 128]
    warnings = [r.getMessage() for r in caplog.records if "normalized onto" in r.msg]
    # One warning per block size, not one per normalization.
    assert len(warnings) == 2, warnings
    assert "config[block_sizes][0]=64" in warnings[0]
    assert "backward (backend 'cute')" in warnings[0]
    assert "'kind': 'helion_flash_bwd'" in bound.to_code(config)
    # A pinned block size cannot back off: no spurious shrink retries.
    assert spec.shrink_block_sizes_once(normalized) is None


@skipUnlessBackends(["cute"])
def test_forward_decorator_config_normalizes_onto_the_pin() -> None:
    bound = _bind(
        attention.fn,
        helion.Config(block_sizes=[1, 64, 64], num_warps=4, num_stages=2),
        _forward_args((1, 2, 256, 64)),
        backend="cute",
    )
    spec = bound.config_spec
    assert spec.cute_flash_search_enabled
    assert [s.structural_pin for s in spec.block_sizes] == [1, 128, 128]
    config = bound._implicit_config()
    assert config is not None
    assert bound._normalized_config_copy(config).block_sizes == [1, 128, 128]
    # Too-large tiles normalize down onto the pin as well.
    big = bound._normalized_config_copy(helion.Config(block_sizes=[2, 256, 256]))
    assert big.block_sizes == [1, 128, 128]
    assert "'kind': 'helion_flash'" in bound.to_code(config)


@skipUnlessBackends(["triton"])
def test_triton_keeps_the_decorator_config() -> None:
    bound = _bind(
        attention_backward.fn,
        TRITON_SHAPED_BACKWARD,
        _backward_args((1, 2, 256, 64)),
        backend="triton",
    )
    spec = bound.config_spec
    assert not spec.cute_flash_bwd_search_enabled
    assert [s.structural_pin for s in spec.block_sizes] == [None, None]
    config = bound._implicit_config()
    assert config is not None
    assert bound._normalized_config_copy(config).block_sizes == [64, 64]
