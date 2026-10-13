"""Fixed RetNet T128 fixtures and the four original prepared FLA boundaries."""

from __future__ import annotations

import copy
import hashlib
import importlib
import os
from pathlib import Path
from typing import Any
from typing import Literal
from typing import cast

from examples.linear.retention_prefill import retention_prefill_fixed_decay
import torch

import helion

STAGES = ("retention_fused_native",)
BASELINES = {
    "fla-prepared": ("chunk", "chunk_simple_gla", False),
    "fla-recurrent-prepared": ("fused_recurrent", "fused_recurrent_simple_gla", False),
    "fla-chunk-prepared-gamma": ("chunk", "chunk_simple_gla", True),
    "fla-recurrent-prepared-gamma": (
        "fused_recurrent",
        "fused_recurrent_simple_gla",
        True,
    ),
}


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases() -> list[dict[str, Any]]:
    return [
        {
            "name": "retention-prefill-b1-t128-h8-d128-fp32",
            "family": "retention-prefill",
            "batch": 1,
            "tokens": 128,
            "heads": 8,
            "value_heads": 8,
            "key_dim": 128,
            "value_dim": 128,
            "chunk_size": 64,
            "implementations": ["helion", *BASELINES],
            "source": "RetNet/FLA fixed per-head decay; H8/D128, T128",
        }
    ]


def _retention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP64 recurrence with FLA's FP32 fixed-head decay coefficients."""
    batch, tokens, heads, _ = q.shape
    decay = 1.0 - 2.0 ** (
        -5.0 - torch.arange(heads, device=q.device, dtype=torch.float32)
    )
    state = initial_state.double().clone()
    output = torch.empty(
        (batch, tokens, heads, v.size(3)), device=v.device, dtype=v.dtype
    )
    for token in range(tokens):
        state = state * decay.double()[None, :, None, None]
        state = state + torch.einsum(
            "bhk,bhv->bhkv", k[:, token].double(), v[:, token].double()
        )
        output[:, token] = (
            torch.einsum("bhk,bhkv->bhv", q[:, token].double(), state) * scale
        ).to(v.dtype)
    return output, state.to(initial_state.dtype)


def prepare_fixed_prefix(
    heads: int, tokens: int, chunk_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Only immutable H/T/C/device-dependent FP32 preparation; no raw inputs."""
    two = torch.tensor(2.0, dtype=torch.float32, device=device)
    head_index = torch.arange(heads, dtype=torch.float32, device=device)
    log_gamma = (1 - two.pow(-5.0 - head_index)).log()
    chunks = (tokens + chunk_size - 1) // chunk_size
    token = torch.arange(chunks * chunk_size, device=device).view(chunks, chunk_size)
    increments = torch.where(token[None, :, :] < tokens, log_gamma[:, None, None], 0.0)
    prefix = increments.cumsum(-1, dtype=torch.float32)
    return log_gamma, prefix


def arguments(
    raw: dict[str, Any], case: dict[str, Any], scale: float
) -> tuple[tuple[Any, ...], ...]:
    return (
        (
            raw["q"],
            raw["k"],
            raw["v"],
            raw["retention_prepared_prefix"],
            raw["state"],
            raw["final_state"],
            scale,
            case["chunk_size"],
        ),
    )


def make_kernel(
    *,
    effort: Literal["none", "quick", "full"] = "none",
    source_workers: int = 0,
    native_workers: int = 0,
    compile_timeout: int = 120,
    log: str | None = None,
) -> helion.Kernel:
    """Clone the public settings; use ordinary defaults and the complete seed set."""
    settings = copy.deepcopy(retention_prefill_fixed_decay.settings)
    settings.autotune_effort = effort
    settings.autotune_benchmark_subprocess = False
    settings.autotune_baseline_fn = None
    settings.autotune_log = log
    settings.autotune_cute_source_workers = source_workers
    settings.autotune_cute_native_workers = native_workers
    settings.autotune_compile_timeout = compile_timeout
    kernel = helion.kernel(retention_prefill_fixed_decay.fn, settings=settings)
    assert (
        settings.static_shapes and settings.backend == "cute" and not settings.fast_math
    )
    assert not kernel.configs and not settings.autotune_seed_configs
    assert not settings.autotune_config_overrides
    assert settings.autotune_budget_seconds is None
    assert (
        settings.autotune_max_generations is None and settings.autotune_best_of_k == 1
    )
    return kernel


def build(case: dict[str, Any], implementation: str = "helion") -> dict[str, Any]:
    if implementation not in ("helion", *BASELINES):
        raise NotImplementedError(implementation)
    assert case["family"] == "retention-prefill"
    assert case["tokens"] == case["key_dim"] == 128 and case["chunk_size"] == 64
    assert case["heads"] == case["value_heads"]
    device = torch.device(case.get("device", "cuda"))
    generator = torch.Generator(device=device).manual_seed(case.get("seed", 20261002))
    batch, tokens = case["batch"], case["tokens"]
    heads, value_heads = case["heads"], case["value_heads"]
    key_dim, value_dim = case["key_dim"], case["value_dim"]
    scale = key_dim**-0.5

    def normal(
        shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16, std: float = 0.25
    ) -> torch.Tensor:
        return torch.randn(shape, dtype=dtype, device=device, generator=generator).mul_(
            std
        )

    q = normal((batch, tokens, heads, key_dim))
    k = normal((batch, tokens, heads, key_dim))
    v = normal((batch, tokens, value_heads, value_dim))
    state = normal((batch, heads, key_dim, value_dim), torch.float32, 0.02)
    original_state = state.clone()
    output = torch.zeros(
        (batch, tokens, value_heads, value_dim), dtype=v.dtype, device=device
    )
    final_state = torch.zeros_like(state)
    inputs = {
        "q": q,
        "k": k,
        "v": v,
        "state": state,
        "output": output,
        "final_state": final_state,
    }
    kernels = {}
    metadata = {
        "family": "retention-prefill",
        "implementation": implementation,
        "seed": case.get("seed", 20261002),
        "scale": scale,
        "io_dtype": "bfloat16",
        "state_dtype": "float32",
        "input_layout": "[batch,token,head,channel], contiguous",
        "state_layout": "[batch,head,key,value] (K-major)",
        "input_layout_copies_outside_timing": False,
        "initial_state_readonly": True,
        "return_final_state": True,
        "qk_l2norm_in_kernel": False,
        "chunk_size": case["chunk_size"],
        "reference": "independent FP64 token-by-token recurrence, final cast only",
        "fixed_decay": "gamma[h] = 1 - 2**(-5-h), constructed in FP32",
        "input_generation": "unnormalized BF16 q/k/v; no input gating or normalization",
    }

    def reference() -> tuple[torch.Tensor, torch.Tensor]:
        return _retention_reference(q, k, v, original_state, scale)

    def reset() -> None:
        output.zero_()
        final_state.zero_()

    if implementation == "helion":
        log_gamma, prefix = prepare_fixed_prefix(
            heads, tokens, case["chunk_size"], device
        )
        inputs.update(
            retention_prepared_log_gamma=log_gamma,
            retention_prepared_prefix=prefix,
            retention_chunk_states=torch.empty(
                (batch * heads, prefix.size(1), key_dim, value_dim),
                dtype=q.dtype,
                device=device,
            ),
        )
        kernel = make_kernel()
        kernels[STAGES[0]] = kernel

        def run() -> tuple[torch.Tensor, torch.Tensor]:
            return kernel(*arguments(inputs, case, scale)[0])

        metadata.update(
            route="public retention_prefill_fixed_decay; one complete raw-input call",
            timed_boundary="BF16 q/k/v, readonly FP32 prefix/initial state -> fresh BF16 output and supplied FP32 final state",
            outside_timing="Exact fixed FP32 log-gamma and chunk-prefix recipe; invariant H/T/C/device only",
            prefix_reciprocal_ln2_prescale=False,
            fixed_prefix_only=True,
            arithmetic_note="Fixed RetNet distance formula with FP32 distance multiplication; not arbitrary-prefix semantics",
        )
    else:
        # The original prepared routes disable optional alternative FLA kernels.
        os.environ.update(FLA_FLASH_KDA="0", FLA_TILELANG="0", FLA_USE_FAST_OPS="0")
        suffix, callee, prepared_gamma = BASELINES[implementation]
        wrapper = importlib.import_module(f"fla.ops.retention.{suffix}")
        fn = getattr(wrapper, callee)
        numerical = importlib.import_module(fn.__module__)
        two = q.new_tensor(2.0, dtype=torch.float)
        head_index = q.new_tensor(range(q.shape[2]), dtype=torch.float)
        inputs.update(retention_constant_two=two, retention_head_index=head_index)
        gamma = (1 - two.pow(-5.0 - head_index)).log() if prepared_gamma else None
        if prepared_gamma:
            inputs["retention_prepared_gamma"] = cast("torch.Tensor", gamma)

        def run() -> tuple[torch.Tensor, torch.Tensor]:
            g_gamma = (
                gamma if prepared_gamma else (1 - two.pow(-5.0 - head_index)).log()
            )
            if suffix == "chunk":
                return fn(
                    q=q,
                    k=k,
                    v=v,
                    scale=scale,
                    g_gamma=g_gamma,
                    initial_state=state,
                    output_final_state=True,
                    cu_seqlens=None,
                    chunk_size=None,
                )
            return fn(
                q=q,
                k=k,
                v=v,
                g_gamma=g_gamma,
                scale=scale,
                initial_state=state,
                output_final_state=True,
                reverse=False,
                cu_seqlens=None,
                state_v_first=False,
            )

        metadata.update(
            route=f"installed FLA {callee}",
            gamma_prepared=prepared_gamma,
            prepared_gamma_expression="(1 - two.pow(-5.0 - head_index)).log()",
            timed_boundary="Whole FLA simple-GLA call with K-major state; fixed-gamma arithmetic also timed for literal-only prepared routes",
            outside_timing="Two FP32 device literals; fixed gamma only for -gamma routes",
            wrapper_sources=[
                {"path": module.__file__, "sha256": sha(cast("str", module.__file__))}
                for module in (wrapper, numerical)
            ],
            historical_qualification=False,
        )
    return {
        "run": run,
        "reset": reset,
        "reference": reference,
        "kernels": kernels,
        "metadata": metadata,
        "inputs": inputs,
        "atol": 0.02,
        "rtol": 0.01,
    }
