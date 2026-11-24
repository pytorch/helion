"""Run one causal forward or backward case with the bundled kernels."""

from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
from typing import TYPE_CHECKING

import kernels
import torch
from torch.nn.attention.flex_attention import create_block_mask
from triton.testing import do_bench

import helion

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def reference_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, length, dimension = q.shape
    out = torch.empty_like(q, dtype=torch.float32)
    lse = torch.empty((batch, heads, length), device=q.device, dtype=torch.float32)
    kv_idx = torch.arange(length, device=q.device)[None, :]
    scale = 1.0 / math.sqrt(dimension)
    for b in range(batch):
        for h in range(heads):
            key = k[b, h].float()
            value = v[b, h].float()
            for begin in range(0, length, 1024):
                end = min(begin + 1024, length)
                scores = (q[b, h, begin:end].float() @ key.T) * scale
                q_idx = torch.arange(begin, end, device=q.device)[:, None]
                scores = torch.where(q_idx >= kv_idx, scores, float("-inf"))
                probabilities = torch.softmax(scores, dim=-1)
                out[b, h, begin:end] = probabilities @ value
                lse[b, h, begin:end] = torch.logsumexp(scores, dim=-1)
    return out.to(q.dtype), lse


def reference_backward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, do: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, length, dimension = q.shape
    gradients = tuple(
        torch.zeros_like(tensor, dtype=torch.float32) for tensor in (q, k, v)
    )
    q_idx = torch.arange(length, device=q.device)[:, None]
    kv_idx = torch.arange(length, device=q.device)[None, :]
    for b in range(batch):
        for h in range(heads):
            query = q[b, h].detach().float().requires_grad_(True)
            key = k[b, h].detach().float().requires_grad_(True)
            value = v[b, h].detach().float().requires_grad_(True)
            scores = (query @ key.T) * (1.0 / math.sqrt(dimension))
            scores = torch.where(q_idx >= kv_idx, scores, float("-inf"))
            out = torch.softmax(scores, dim=-1) @ value
            result = torch.autograd.grad(out, (query, key, value), do[b, h].float())
            for destination, gradient in zip(gradients, result, strict=True):
                destination[b, h] = gradient
    return gradients


def forward_call(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, config: dict
) -> tuple[Callable[[], torch.Tensor], dict[str, str]]:
    arguments = (q, k, v, kernels.causal_stage)
    bound = kernels.attn_stage.bind(arguments)
    selected = helion.Config.from_json(json.dumps(config))
    bound.set_config(selected)
    code = bound.to_code(selected)
    assert any(
        marker in code
        for marker in ("_flash_num_kv_tiles", "helion_flash", "row_mma", "ws_overlap")
    ), "CuTe fused forward path did not fire"
    return lambda: kernels.attn_stage(*arguments), {"forward": code}


def backward_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    config: dict,
) -> tuple[
    Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]], dict[str, str]
]:
    length = q.shape[2]
    mask = torch.compile(create_block_mask)(
        kernels.causal_mask,
        None,
        None,
        length,
        length,
        device=str(q.device),
        BLOCK_SIZE=(128, 128),
    )
    tables = tuple(
        table.expand(q.shape[0], q.shape[1], *table.shape[2:]).contiguous()
        for table in (
            mask.q_num_blocks,
            mask.q_indices,
            mask.full_q_num_blocks,
            mask.full_q_indices,
        )
    )
    delta = (out.float() * do.float()).sum(-1)
    lse2 = lse * 1.44269504
    tail = (
        kernels.score_mod,
        kernels.score_mod_bwd,
        *tables,
        kernels.causal_mask,
        128,
        128,
        1.0 / math.sqrt(q.shape[-1]),
        False,
    )
    arguments = (q, k, v, do, lse2, delta, *tail)
    codes = {}
    for name, kernel, inputs in (
        ("preprocess", kernels._flex_attention_bwd_preprocess, (out, do, lse)),
        ("main", kernels.helion_flex_attention_backward_kernel, arguments),
    ):
        bound = kernel.bind(inputs)
        selected = helion.Config.from_json(json.dumps(config[name]))
        bound.set_config(selected)
        codes[name] = bound.to_code(selected)
    assert any(marker in codes["main"] for marker in ("fbwd_", "helion_flash_bwd")), (
        "CuTe fused backward path did not fire"
    )

    def run() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta, lse2 = kernels._flex_attention_bwd_preprocess(out, do, lse)
        dq, dk, dv = kernels.helion_flex_attention_backward_kernel(
            q, k, v, do, lse2, delta, *tail
        )
        return dq.to(q.dtype), dk, dv

    return run, codes


def gpu_warmup() -> None:
    matrix = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    start = time.time()
    while (time.time() - start) * 1000 < 5000:
        for _ in range(20):
            matrix = matrix @ matrix
        torch.cuda.synchronize()


def benchmark(fn: Callable[[], object]) -> dict:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    gpu_warmup()
    times = [
        float(do_bench(fn, warmup=500, rep=500, return_mode="median")) for _ in range(5)
    ]
    return {
        "median_ms": statistics.median(times),
        "best_ms": min(times),
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times),
        "runs_ms": times,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("forward", "backward"), required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    pins = json.loads((ROOT / "pins.json").read_text())
    expected = pins["software_versions"][args.mode]
    assert platform.python_version() == pins["python"]
    assert platform.machine() == pins["architecture"]
    assert (
        Path(helion.__file__).resolve().parent
        == ROOT / ".engines" / args.mode / "helion"
    )
    assert os.environ["HELION_BACKEND"] == "cute"
    actual = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": version("triton"),
        "cutlass": version("nvidia-cutlass-dsl"),
        "cudnn": version("nvidia-cudnn-cu13"),
        "cuda": torch.version.cuda,
        **{
            name: version(name)
            for name in (
                "flash-attn-4",
                "quack-kernels",
                "cuda-python",
                "cuda-bindings",
                "cuda-core",
            )
        },
    }
    assert all(actual[name] == expected[name] for name in actual), actual
    assert all(
        version(name) == expected["cutlass"]
        for name in (
            "nvidia-cutlass-dsl-libs-base",
            "nvidia-cutlass-dsl-libs-core",
            "nvidia-cutlass-dsl-libs-cu13",
        )
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    shape = (4, 16, args.sequence, 64)
    generator = torch.Generator(device=device).manual_seed(0)
    q, k, v = tuple(
        torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
        for _ in range(3)
    )
    out, lse = reference_forward(q, k, v)
    config_path = (
        ROOT
        / "configs"
        / args.mode
        / f"causal.b4h16s{args.sequence}d64.helion-cute.json"
    )
    assert (
        hashlib.sha256(config_path.read_bytes()).hexdigest()
        == pins["configs"][str(config_path.relative_to(ROOT))]
    )
    config = json.loads(config_path.read_text())
    if args.mode == "forward":
        fn, codes = forward_call(q, k, v, config)
        error = (fn().float() - out.float()).abs().max().item()
        assert math.isfinite(error) and error < 0.03, error
    else:
        grad_generator = torch.Generator(device=device).manual_seed(1)
        do = torch.randn(
            shape, device=device, dtype=torch.bfloat16, generator=grad_generator
        )
        fn, codes = backward_call(q, k, v, do, out, lse, config)
        reference = reference_backward(q, k, v, do)
        errors = [
            (got.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
            for got, ref in zip(fn(), reference, strict=True)
        ]
        assert all(math.isfinite(error) and error < 0.03 for error in errors), errors
        error = max(errors)
    for name, code in codes.items():
        (args.result.parent / f"s{args.sequence}.{name}.py").write_text(code)
    stats = benchmark(fn)
    factor = 256 if args.mode == "forward" else 640
    flops = 4 * 16 * (args.sequence * (args.sequence + 1) // 2) * factor
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "variant": "causal",
        "mode": args.mode,
        "backend": "helion-cute",
        "shape": list(shape),
        "dtype": "bfloat16",
        "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_name": torch.cuda.get_device_name(0),
        "helion_commit": pins["sources"][args.mode]["commit"],
        "software_versions": actual,
        "accuracy": "PASS",
        "error": error,
        "config": config,
        "helion_autotuning": "skipped",
        "flops": flops,
        "median_tflops": flops / stats["median_ms"] / 1e9,
        **stats,
    }
    args.result.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
