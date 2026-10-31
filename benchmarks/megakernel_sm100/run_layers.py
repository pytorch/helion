"""Cold-L2 single-call timing of the one-GPU workloads, for Helion or vLLM.

    python run_layers.py --system vllm   --workload qwen3_8b_layer --out vllm.json
    python run_layers.py --system helion --workload qwen3_8b_layer --out helion.json

Each system runs in its own environment (vLLM pins its own torch), so the two
never share a process.  The vLLM run saves its outputs under ``--refs``; the
Helion run checks its decoder-layer outputs against them before timing (the
Qwen3 MoE and MLA workloads check both systems against an FP32 reference
instead).  Both build their inputs with the same seeded code from the pretuned
kernel modules and record a fingerprint of them, so a check against another
environment's outputs is a check on the same inputs.

Workloads (batch 1, one decode token, see ``WORKLOADS``):

* ``qwen3_8b_layer``: Qwen3-8B BF16 decoder layer at context 1024.  Helion's
  megakernel vs vLLM's compiled ``Qwen3DecoderLayer`` (default backends).
* ``llama32_1b_layer``: the same for Llama-3.2-1B.
* ``qwen3_30b_a3b_moe``: Qwen3-30B-A3B MoE block (RMSNorm, router, top-8 experts,
  residual add) vs vLLM's compiled RMSNorm + ``Qwen3MoeSparseMoeBlock``; vLLM's
  residual add runs in the next layer's fused add + RMSNorm, so it is not part of
  its call (Helion's includes it).  Three routing cases (distinct tokens),
  averaged.
* ``deepseek_v3_mla``: DeepSeek-V3 MLA decode (16 heads, TP8) at context 1024 vs
  FlashInfer's TRT-LLM-gen MLA, called as vLLM's decode CUDA graph calls it
  (``max_seq_len`` = DeepSeek-V3's ``max_model_len``).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from benchmarks.megakernel_sm100 import coldl2  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

LAYER_CONTEXT = 1024
MLA_CONTEXTS = (1024,)
WORKLOADS = (
    "qwen3_8b_layer",
    "llama32_1b_layer",
    "qwen3_30b_a3b_moe",
    "deepseek_v3_mla",
)
_MODULES = {
    "qwen3_8b_layer": "pretuned_kernels.megakernels.qwen3_8b.qwen3_decode_layer_bf16",
    "llama32_1b_layer": (
        "pretuned_kernels.megakernels.llama32_1b_decode_layer_bf16."
        "llama32_1b_decode_layer_bf16"
    ),
    "qwen3_30b_a3b_moe": (
        "pretuned_kernels.megakernels.qwen3_30b_a3b_moe.qwen3_30b_a3b_moe"
    ),
    "deepseek_v3_mla": "pretuned_kernels.megakernels.flash_mla.flash_mla",
}
_KERNELS = {
    "qwen3_8b_layer": "qwen3_decode_layer_bf16",
    "llama32_1b_layer": "llama32_1b_decode_layer_bf16",
}


def fingerprint(tensors: dict[str, torch.Tensor], names: list[str]) -> list[float]:
    """Sums of the named inputs, to confirm two environments built the same data."""
    return [float(tensors[name].double().sum()) for name in names]


def _check_fingerprint(ref: dict[str, Any], actual: list[float], workload: str) -> None:
    for expected, value in zip(ref["fingerprint"], actual, strict=True):
        if not math.isclose(expected, value, rel_tol=1e-9, abs_tol=1e-6):
            raise AssertionError(
                f"{workload}: inputs differ from the reference run's "
                f"({value} vs {expected}); the seeded generators disagree"
            )


def _close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rel_l2: float,
) -> None:
    """Largest absolute and relative L2 error within bounds.

    The bounds sit a few times above a correct run's errors and below those of
    a run that attends over too few KV rows (896 of 1024 for the decoder layers,
    1023 of 1024 for MLA)."""
    delta = actual.float().cpu() - expected.float().cpu()
    max_abs = float(delta.abs().max())
    rel = float(delta.norm() / expected.float().cpu().norm())
    if not (max_abs <= atol and rel <= rel_l2):
        raise AssertionError(
            f"{name}: max abs error {max_abs:.4g} (bound {atol}), relative L2 "
            f"error {rel:.3g} (bound {rel_l2})"
        )


def decoder_layer(
    workload: str, system: str, refs: Path, reps: int
) -> list[dict[str, Any]]:
    module = importlib.import_module(_MODULES[workload])
    base = module._make_inputs(context_len=LAYER_CONTEXT, batch=1)
    names = [
        name
        for name, value in base.items()
        if name not in ("cos_sin", "pre_normed", "ffn_normed")
        and value.is_floating_point()
    ] + ["block_table"]
    inputs = fingerprint(base, names)
    ref_path = refs / f"{workload}.pt"
    close: Callable[[], None] | None = None
    ref: dict[str, Any]
    if system == "vllm":
        vllm_call, tensors, backend, close = module._make_vllm_call(base, LAYER_CONTEXT)
        reset = module._make_reset(tensors, vllm_layout=True)
        cache_rows = lambda: module._cache_rows(tensors, vllm_layout=True)  # noqa: E731
        call = coldl2.Recorded(vllm_call)
        call()
        torch.cuda.synchronize()
        output, residual = call.outputs
        ref = {
            "fingerprint": inputs,
            "cos_sin": base["cos_sin"].cpu(),
            "output": output.cpu(),
            "residual": residual.cpu(),
            "cache_rows": [row.cpu() for row in cache_rows()],
        }
        torch.save(ref, ref_path)
        name = f"vLLM ({backend})"
    else:
        ref = torch.load(ref_path)
        _check_fingerprint(ref, inputs, workload)
        # vLLM's rotary table, so both apply the same rotation.
        base["cos_sin"].copy_(ref["cos_sin"])
        tensors = module._clone_writable(base)
        reset = module._make_reset(tensors)
        cache_rows = lambda: module._cache_rows(tensors)  # noqa: E731
        kernel = getattr(module, _KERNELS[workload])
        call = coldl2.Recorded(lambda: kernel(*module._kernel_args(tensors)))
        call()
        torch.cuda.synchronize()
        name = "Helion"

    def check() -> None:
        # A correct run: max abs 0.016, relative L2 6e-3 (output), 9e-4
        # (residual), 4e-3 (KV rows).
        outputs = call.outputs
        _close("output", outputs[0], ref["output"], atol=0.03, rel_l2=1e-2)
        _close("residual", outputs[-1], ref["residual"], atol=0.03, rel_l2=2e-3)
        for row, expected in zip(cache_rows(), ref["cache_rows"], strict=True):
            _close("kv row", row, expected, atol=0.03, rel_l2=1e-2)

    try:
        check()
        reset()
        timing = coldl2.time_cold(call, reps=reps, reset=reset)
        check()
    finally:
        if close is not None:
            close()
    return [
        {"workload": workload, "case": f"ctx{LAYER_CONTEXT}", "impl": name, **timing}
    ]


def qwen3_moe(system: str, reps: int) -> list[dict[str, Any]]:
    """The three routing cases (distinct tokens), timed in one session."""
    module = importlib.import_module(_MODULES["qwen3_30b_a3b_moe"])
    tensors = module._make_inputs()
    cases = {
        label: residual.clone() for label, residual in module._make_routing_cases()
    }
    calls: dict[str, coldl2.Recorded] = {}
    close = None
    if system == "vllm":
        make_launch, backend, close = module._make_vllm_call(tensors)
        name = f"vLLM ({backend})"

        def make_call(residual: torch.Tensor) -> Callable[[], object]:
            return make_launch(residual, add_residual=False)

    else:
        name = "Helion"

        def make_call(residual: torch.Tensor) -> Callable[[], object]:
            args = module._kernel_args(dict(tensors, residual=residual))
            return lambda: module.qwen3_30b_a3b_moe(*args)

    def check(label: str) -> None:
        case_tensors = dict(tensors, residual=cases[label])
        outputs = calls[label].outputs
        if system == "vllm":
            module._check(
                cases[label] + outputs, case_tensors, "vllm", bf16_logits=True
            )
        else:
            module._check_routing(outputs[3], case_tensors)
            module._check(outputs[0], case_tensors, "helion")

    try:
        for label, residual in cases.items():
            calls[label] = coldl2.Recorded(make_call(residual))
            calls[label]()
            torch.cuda.synchronize()
            check(label)
        timings = coldl2.time_cold_many(dict(calls), reps=reps)
        for label in cases:
            check(label)
    finally:
        if close is not None:
            close()
    return [
        {"workload": "qwen3_30b_a3b_moe", "case": label, "impl": name, **timing}
        for label, timing in timings.items()
    ]


def mla(system: str, reps: int) -> list[dict[str, Any]]:
    module = importlib.import_module(_MODULES["deepseek_v3_mla"])
    calls: dict[str, coldl2.Recorded] = {}
    expected = {}
    for context in MLA_CONTEXTS:
        tensors = module._make_inputs(context, seed=context)
        if system == "vllm":
            call = coldl2.Recorded(module._make_flashinfer_call(tensors))
        else:
            args = module._kernel_args(tensors)
            call = coldl2.Recorded(lambda args=args: module.flash_mla(*args))
        call()
        torch.cuda.synchronize()
        calls[f"ctx{context}"] = call
        expected[f"ctx{context}"] = module.reference(tensors)

    def check() -> None:
        # A correct run: max abs 1e-3, relative L2 2e-3.
        for case, call in calls.items():
            _close("mla", call.outputs, expected[case], atol=5e-3, rel_l2=5e-3)

    check()
    timings = coldl2.time_cold_many(dict(calls), reps=reps)
    check()
    name = "vLLM (FlashInfer TRT-LLM-gen MLA)" if system == "vllm" else "Helion"
    return [
        {"workload": "deepseek_v3_mla", "case": case, "impl": name, **timing}
        for case, timing in timings.items()
    ]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--system", choices=("helion", "vllm"), required=True)
    # One workload per process: coldl2 profiles once per process.
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--refs", type=Path, default=HERE / "results" / "refs")
    parser.add_argument("--reps", type=int, default=200)
    cli = parser.parse_args()
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("the megakernels are pretuned for SM100 (B200)")
    cli.refs.mkdir(parents=True, exist_ok=True)
    if cli.workload in _KERNELS:
        rows = decoder_layer(cli.workload, cli.system, cli.refs, cli.reps)
    elif cli.workload == "qwen3_30b_a3b_moe":
        rows = qwen3_moe(cli.system, cli.reps)
    else:
        rows = mla(cli.system, cli.reps)
    for row in rows:
        print(
            f"{row['workload']:>18s} {row['case']:>9s}  {row['impl']:<40s} "
            f"{row['median_us']:8.2f} us (p10 {row['p10_us']:.2f}, p90 {row['p90_us']:.2f}; "
            f"loop {row['loop_us']:.2f})",
            flush=True,
        )
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(
        json.dumps(
            {"system": cli.system, "gpu": coldl2.gpu_info(), "rows": rows}, indent=1
        )
    )


if __name__ == "__main__":
    main()
