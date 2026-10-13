"""SSD fixtures and compatible installed baselines for the output-only contract."""

from __future__ import annotations

import copy
import hashlib
import importlib
from pathlib import Path
from typing import Any
from typing import cast

from benchmarks.cute import mamba2_ssd_reference as reference
from examples.linear import mamba2_ssd as program
import torch

import helion

RAW_NAMES = ("x", "dt", "a", "b", "c", "d")
SHORT = ("mamba2_ssd_cute_scan_fused",)
LONG = (
    "mamba_native_state",
    "mamba_native_pass",
    "mamba_native_cb",
    "mamba_native_output",
)
BASELINES = ("flashinfer-cute", "mamba", "vllm", "flashinfer-cake", "sglang")


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stages(case: dict[str, Any]) -> tuple[str, ...]:
    return SHORT if case["program"] == "single" else LONG


def cases() -> list[dict[str, Any]]:
    return [
        {
            "name": f"mamba2-ssd-b1-t{tokens}-h{heads}-p64-n128-g{groups}",
            "family": "mamba2-ssd",
            "batch": 1,
            "tokens": tokens,
            "heads": heads,
            "value_dim": 64,
            "state_dim": 128,
            "groups": groups,
            "chunk_size": 128,
            "program": variant,
            "implementations": ["helion", *BASELINES],
        }
        for tokens, heads, groups, variant in (
            (128, 128, 8, "single"),
            (2048, 64, 1, "staged"),
        )
    ]


def configure(kernel: helion.Kernel) -> None:
    kernel.settings.backend = "cute"
    kernel.settings.fast_math = False
    kernel.settings.autotune_effort = "full"
    kernel.settings.autotune_benchmark_subprocess = False
    assert kernel.settings.autotune_budget_seconds is None
    assert not kernel.settings.autotune_config_overrides
    assert not kernel.settings.autotune_seed_configs
    assert not kernel.configs


def build(case: dict[str, Any], implementation: str = "helion") -> dict[str, Any]:
    assert implementation in ("helion", *BASELINES)
    assert case["program"] in ("single", "staged")
    device = case.get("device", "cuda")
    generator = torch.Generator(device=device).manual_seed(20261002)

    def rand(
        shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16, scale: float = 1.0
    ) -> torch.Tensor:
        return (torch.randn(shape, generator=generator, device=device) * scale).to(
            dtype
        )

    B, T, H = case["batch"], case["tokens"], case["heads"]
    G, P, N, C = (case[k] for k in ("groups", "value_dim", "state_dim", "chunk_size"))
    assert T > 0 and T % C == 0 and H % G == 0
    assert case["program"] != "single" or T == C
    x = rand((B, T, H, P), scale=0.5)
    dt = (torch.rand((B, T, H), generator=generator, device=device) * 0.08 + 0.01).to(
        torch.float32
    )
    a = -torch.rand((H,), generator=generator, device=device) - 0.1
    b, c = (rand((B, T, G, N), scale=0.1) for _ in range(2))
    d = rand((H,), scale=0.5)
    inputs = dict(zip(RAW_NAMES, (x, dt, a, b, c, d), strict=True))
    kernels = {}
    metadata = {
        "family": "mamba2-ssd",
        "implementation": implementation,
        "boundary": "Native token-major x/dt/A/B/C/D -> output only; dt is already processed, zero initial state, z off",
        "limitation": "Final state, raw-dt softplus, packed sequences and optional z are not assessed by these SSD rows",
        "state_dtype": "BF16 inter-chunk state; FP32 accumulator",
    }

    def reset() -> None:
        pass

    def oracle() -> tuple[torch.Tensor, ...]:
        return (
            cast(
                "torch.Tensor",
                reference._mamba2_ssd_recurrent_fused_reference(x, dt, a, b, c, d, C),
            ),
        )

    refs = (
        (reference.tuning_reference,)
        if stages(case) == SHORT
        else (
            reference.state_reference,
            reference.pass_reference,
            reference.cb_reference,
            reference.output_reference,
        )
    )
    if implementation == "helion":
        for name, ref in zip(stages(case), refs, strict=True):
            original = getattr(program, name)
            settings = copy.deepcopy(original.settings)
            settings.autotune_baseline_fn = ref
            settings.autotune_baseline_atol = 0.002
            settings.autotune_baseline_rtol = 0.01
            kernels[name] = helion.kernel(original.fn, settings=settings)

        def run() -> tuple[torch.Tensor, ...]:
            from benchmarks.cute.mamba2_ssd_objective import stage_arguments

            values = []
            for index, kernel in enumerate(kernels.values()):
                values.append(kernel(*stage_arguments(inputs, case, index, values)))
            return (values[-1],)

    elif implementation.startswith("flashinfer-"):
        from flashinfer.mamba.ssd_combined import (
            SSDCombined,  # pyrefly: ignore[missing-import]
        )

        backend = "cake" if implementation == "flashinfer-cake" else "cute"
        operation = SSDCombined(
            C,
            H,
            P,
            N,
            G,
            io_dtype=x.dtype,
            state_dtype=torch.bfloat16,
            has_d=True,
            backend=backend,
        )

        def run() -> tuple[torch.Tensor, ...]:
            result = operation.run(x, dt, a, b, c, D=d, return_final_states=False)
            return (result[0],)

        metadata["backend"] = backend
    elif implementation == "mamba":
        mamba_chunk_scan_combined = importlib.import_module(
            "mamba_ssm.ops.triton.ssd_combined"
        ).mamba_chunk_scan_combined

        def run() -> tuple[torch.Tensor, ...]:
            return (
                mamba_chunk_scan_combined(
                    x, dt, a, b, c, C, D=d, state_dtype=torch.bfloat16
                ),
            )

    else:
        output = torch.zeros_like(x)
        if implementation == "vllm":
            from vllm.model_executor.layers.mamba.ops.ssd_combined import (  # pyrefly: ignore[missing-import]
                mamba_chunk_scan_combined_varlen,
            )

            assert B == 1, "This compatible varlen adapter represents one sequence"
            boundaries = [*range(0, T, C), T]
            cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)
            cu_chunk_seqlens = torch.tensor(
                boundaries, dtype=torch.int32, device=device
            )
            seq_idx = torch.zeros(len(boundaries) - 1, dtype=torch.int32, device=device)
            last_chunk_indices = torch.tensor(
                [len(boundaries) - 2], dtype=torch.int32, device=device
            )

            def run() -> tuple[torch.Tensor, ...]:
                mamba_chunk_scan_combined_varlen(
                    x=x.view(T, H, P),
                    dt=dt.view(T, H),
                    A=a,
                    B=b.view(T, G, N),
                    C=c.view(T, G, N),
                    chunk_size=C,
                    cu_seqlens=cu_seqlens,
                    cu_chunk_seqlens=cu_chunk_seqlens,
                    last_chunk_indices=last_chunk_indices,
                    seq_idx=seq_idx,
                    out=output.view(T, H, P),
                    D=d,
                    z=None,
                    dt_bias=None,
                    initial_states=None,
                    dt_softplus=False,
                    state_dtype=torch.bfloat16,
                )
                return (output,)

            inputs.update(
                work_cu_seqlens=cu_seqlens,
                work_cu_chunk_seqlens=cu_chunk_seqlens,
                work_seq_idx=seq_idx,
                work_last_chunk_indices=last_chunk_indices,
            )
            metadata["timed_layout_adaptation"] = (
                "Views only; final-state gather remains inside the call and is discarded"
            )
        else:
            from sglang.kernels.ops.mamba.triton_ops.ssd_combined import (  # pyrefly: ignore[missing-import]
                mamba_chunk_scan_combined,
            )

            def run() -> tuple[torch.Tensor, ...]:
                mamba_chunk_scan_combined(
                    x,
                    dt,
                    a,
                    b,
                    c,
                    C,
                    D=d,
                    z=None,
                    dt_bias=None,
                    initial_states=None,
                    seq_idx=None,
                    cu_seqlens=None,
                    dt_softplus=False,
                    out=output,
                    return_final_states=False,
                    state_dtype=torch.bfloat16,
                )
                return (output,)

        inputs["work_output"] = output

        def reset() -> None:
            output.zero_()

    copied = {}
    for name, original in kernels.items():
        settings = copy.deepcopy(original.settings)
        settings.autotune_baseline_fn = None
        copied[name] = helion.kernel(original.fn, settings=settings)
    return {
        "run": run,
        "reset": reset,
        "reference": oracle,
        "kernels": kernels,
        "metadata": metadata,
        "inputs": inputs,
        "atol": 0.002,
        "rtol": 0.01,
        "objective_binding": {"kernels": copied, "references": refs},
    }
