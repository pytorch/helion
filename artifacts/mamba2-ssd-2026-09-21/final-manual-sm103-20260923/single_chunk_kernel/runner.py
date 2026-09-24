"""Stateless explicit-config runner; each invocation returns a fresh output."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import Any

import torch

from . import kernels
import helion

if TYPE_CHECKING:
    from collections.abc import Callable

Inputs = tuple[torch.Tensor, ...]
Configs = dict[str, dict[str, Any]]
HELION_KERNELS = (kernels.mamba2_ssd_cute_scan_fused,)
HELION_REFERENCES = (kernels.tuning_reference,)


def _meta(inputs: Inputs) -> tuple[tuple[Any, ...], ...]:
    return tuple((v.shape, v.stride(), v.dtype, v.device) for v in inputs)


def _validate_inputs(inputs: Inputs) -> None:
    if len(inputs) != 6:
        raise ValueError("Expected exactly X, dt, A, B, C and D")
    x, dt, a, b, c, d = inputs
    batch, length, heads, _dim = x.shape
    groups, state = b.shape[2:]
    if length != 128 or state != 128 or groups <= 0 or heads % groups:
        raise ValueError("SingleChunkNative requires one chunk128 and grouped state128")
    if (
        dt.shape != (batch, length, heads)
        or a.shape != (heads,)
        or b.shape != (batch, length, groups, state)
        or c.shape != b.shape
        or d.shape != (heads,)
    ):
        raise ValueError("Inconsistent native input shapes")
    if any(not value.is_contiguous() for value in inputs):
        raise ValueError("SingleChunkNative requires contiguous native inputs")
    if any(value.device != x.device for value in inputs):
        raise ValueError("All inputs must be on the same device")
    if x.dtype != torch.bfloat16 or b.dtype != x.dtype or c.dtype != x.dtype:
        raise ValueError("This private precision experiment requires BF16 X/B/C")


class SingleChunkNative:
    """One stage, no retained tensors, persistent scratch, or shared settings."""

    def __init__(
        self,
        inputs: Inputs,
        *,
        configs: Configs | None = None,
        autotune: bool = False,
        settings: helion.Settings | None = None,
        stage_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        _validate_inputs(inputs)
        self.meta = _meta(inputs)
        kernel = HELION_KERNELS[0]
        name = kernel.fn.__name__
        if not autotune and (configs is None or set(configs) != {name}):
            raise KeyError(f"Expected exactly the explicit config for {name}")
        if autotune and configs is not None:
            raise ValueError("Cold full autotuning does not accept injected configs")
        base_settings = settings or kernel.settings
        options: dict[str, Any] = base_settings.to_dict()
        stage_seed = base_settings.autotune_random_seed
        options.update(
            backend="cute",
            static_shapes=True,
            autotune_baseline_fn=kernels.tuning_reference,
        )
        if autotune:
            options.update(autotune_effort="full", autotune_random_seed=stage_seed)
        private = helion.aot_kernel(kernel.fn, **options)
        bound = private.bind(inputs)
        self.autotune_stage_seconds: dict[str, float] = {}
        self.autotune_stage_seeds: dict[str, int] = {}
        if autotune:
            started = time.perf_counter()
            config = bound.autotune(inputs, force=True)
            self.autotune_stage_seconds[name] = time.perf_counter() - started
            self.autotune_stage_seeds[name] = stage_seed
        else:
            assert configs is not None
            config = helion.Config(**configs[name])
        self.fn = bound.compile_config(config)
        self.fn(*inputs)
        self.configs = {name: dict(config)}
        self.generated_code = {name: bound.to_code(config)}
        if stage_callback is not None:
            stage_callback(
                {
                    "stage": 0,
                    "kernel": name,
                    "config": dict(config),
                    "autotune_seconds": self.autotune_stage_seconds.get(name),
                    "autotune_seed": self.autotune_stage_seeds.get(name),
                    "autotune_force": autotune,
                    "code": self.generated_code[name],
                }
            )

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        if _meta(inputs) != self.meta:
            raise ValueError("Input shape, strides, dtype or device changed")
        return self.fn(*inputs)
