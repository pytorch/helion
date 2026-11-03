"""Cold-L2 timing of the DeepSeek-V3 NVFP4 MoE block at TP4, for Helion or vLLM.

    torchrun --standalone --nproc-per-node=4 deepseek_tp4.py --system vllm --out ...
    torchrun --standalone --nproc-per-node=4 deepseek_tp4.py --system helion --out ...

The block is one decode token through the MoE (router, top-8 of 256 experts plus
the shared expert, rank-local intermediate shards), the all-reduce of the four
ranks' partial outputs, the residual add and RMSNorm:

* Helion: ``deepseek_v3_moe_nvfp4_tp``, one kernel per rank.
* vLLM: vLLM's FlashInfer TRT-LLM NVFP4 MoE (plus its NVFP4 shared expert on a
  side stream), then FlashInfer's fused all-reduce + residual add + RMSNorm with
  vLLM's backend choice and settings (``_make_fused_all_reduce``).

Every rank holds its own expert and shared-expert weights (seeded by rank) and
the same token; the residual is about as large as the MoE output, so the checks
see every rank's contribution and the residual add.  The three routing cases get
their own router tensors and are timed in one profiling session, interleaved.
Before each call every rank flushes L2, then the ranks align with a one-shot
all-reduce of a few bytes (outside the timed span), then the call runs.  The
reported time is the largest of the four ranks' median spans (and likewise for
the profiler-free ``loop_us``); every rank's median is recorded.  The vLLM run
saves each rank's expected outputs and routing; the Helion run checks against
them, since the Helion environment has no vLLM.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import math
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from benchmarks.megakernel_sm100 import coldl2  # noqa: E402
from pretuned_kernels.megakernels._distributed import initialize  # noqa: E402
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4_tp import (  # noqa: E402
    _setup as source,
)
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4_tp import (  # noqa: E402
    deepseek_v3_moe_nvfp4_tp as tp,
)

if TYPE_CHECKING:
    from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4_tp._setup import Shape


_FINGERPRINT = (
    "hidden",
    "router_weight",
    "correction_bias",
    "native_w13",
    "w2",
    "native_shared_w13",
    "shared_w2",
    "alpha2",
)
# Per-case copies: the routing cases differ only in these.
_ROUTING_TENSORS = ("routing_target", "router_weight", "correction_bias")
# Similarity error bound of both outputs (normalized output, residual sum).
_TOLERANCE = 1e-4
# A rank's partial output left out, or the residual add skipped, must miss the
# reference by this multiple of the tolerance (checked in the vLLM run).
_CONTROL_MARGIN = 10


def _similarity(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return source._similarity_error(actual.float().cpu(), expected.float().cpu())


def _check(name: str, actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = _similarity(actual, expected)
    if not math.isfinite(error) or error > _TOLERANCE:
        raise AssertionError(f"{name}: similarity error {error:.3g} > {_TOLERANCE}")
    return error


def _vllm_call(
    tensors: dict[str, torch.Tensor],
    shape: Shape,
    fused: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None],
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
) -> tuple[
    Callable[[], tuple[torch.Tensor, torch.Tensor]],
    Callable[[], torch.Tensor],
    torch.Tensor,
    str,
]:
    """(production call, its MoE part, the routing it chose, the MoE backend).

    The call returns the normalized output and the residual sum."""
    raw_call, moe_backend, routing_replay = source._make_vllm_call(tensors, shape)
    moe = cast("Callable[[], torch.Tensor]", raw_call)
    norm_output = torch.empty_like(residual)

    def call() -> tuple[torch.Tensor, torch.Tensor]:
        summed = moe()
        # The residual sum overwrites the all-reduce input.
        fused(summed, residual, norm_weight, norm_output)
        return norm_output, summed

    return call, moe, routing_replay, moe_backend


# build(label, case tensors, input fingerprint) -> (call, name, check), where
# check() checks the call's latest outputs and returns their similarity errors.
Check = Callable[[], dict[str, float]]
Build = Callable[
    [str, dict[str, torch.Tensor], list[float]],
    tuple[coldl2.Recorded, str, Check],
]


def _vllm_builder(
    rank: int,
    group: dist.ProcessGroup,
    shape: Shape,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    refs: dict[str, Any],
) -> Build:
    """Builds vLLM's call per case, checks it, and records the reference."""
    fused, all_reduce_backend = tp._make_fused_all_reduce(rank, group, shape.hidden)

    def build(
        label: str, case: dict[str, torch.Tensor], inputs: list[float]
    ) -> tuple[coldl2.Recorded, str, Check]:
        vllm_call, moe, routing_replay, moe_backend = _vllm_call(
            case, shape, fused, residual, norm_weight
        )
        partial = moe().float()
        summed = partial.clone()
        dist.all_reduce(summed)
        expected, expected_sum = tp._rms_norm_reference(summed, residual, norm_weight)
        call = coldl2.Recorded(vllm_call)
        call()
        torch.cuda.synchronize()

        def check() -> dict[str, float]:
            actual, actual_sum = call.outputs
            return {
                "output": _check("vLLM output", actual, expected),
                "residual_sum": _check("vLLM sum", actual_sum, expected_sum),
            }

        errors = check()
        # Wrong kernels the checks must catch.
        without_rank, _ = tp._rms_norm_reference(
            summed - partial, residual, norm_weight
        )
        without_residual, _ = tp._rms_norm_reference(
            summed, torch.zeros_like(residual), norm_weight
        )
        controls = {
            "without_this_rank": _similarity(without_rank, expected),
            "without_residual": _similarity(without_residual, expected),
        }
        if min(controls.values()) < _CONTROL_MARGIN * _TOLERANCE:
            raise AssertionError(f"{label}: weak check, controls {controls}")
        refs["cases"][label] = {
            "fingerprint": inputs,
            "expected": expected.cpu(),
            "expected_sum": expected_sum.cpu(),
            "topk_ids": routing_replay.to(torch.int32).cpu(),
            "errors": errors,
            "controls": controls,
            "norms": {
                "moe": float(summed.norm()),
                "residual": float(residual.float().norm()),
            },
        }
        name = (
            f"vLLM (FlashInfer {moe_backend} MoE + {all_reduce_backend} "
            "all-reduce/RMSNorm)"
        )
        return call, name, check

    return build


def _helion_builder(
    group: dist.ProcessGroup,
    device: torch.device,
    shape: Shape,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    refs: dict[str, Any],
) -> Build:
    """Builds Helion's call per case and checks it against the vLLM run's."""
    import torch.distributed._symmetric_memory as symm_mem

    symmetric_output = symm_mem.empty(
        (shape.top_k, shape.hidden), dtype=torch.bfloat16, device=device
    )
    symm_mem.rendezvous(symmetric_output, group.group_name)
    symmetric_shared = symm_mem.empty(
        (1, shape.hidden), dtype=torch.bfloat16, device=device
    )
    symm_mem.rendezvous(symmetric_shared, group.group_name)
    norm_counter = torch.zeros(tp.COUNTER_SHAPE, dtype=torch.int32, device=device)

    def build(
        label: str, case: dict[str, torch.Tensor], inputs: list[float]
    ) -> tuple[coldl2.Recorded, str, Check]:
        ref = refs["cases"][label]
        if not all(
            math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)
            for a, b in zip(ref["fingerprint"], inputs, strict=True)
        ):
            raise AssertionError(f"{label}: inputs differ from the vLLM run's")
        args = (
            *source._kernel_args(case, shape),
            tp.W13_SPLIT_K,
            norm_counter,
            residual,
            norm_weight,
            symmetric_output,
            symmetric_shared,
            group.group_name,
        )

        call = coldl2.Recorded(lambda: tp.deepseek_v3_moe_nvfp4_tp(*args))
        call()
        torch.cuda.synchronize()

        def check() -> dict[str, float]:
            outputs = call.outputs
            torch.testing.assert_close(
                outputs[3].cpu(), ref["topk_ids"], rtol=0, atol=0
            )
            return {
                "output": _check("Helion output", outputs[0], ref["expected"]),
                "residual_sum": _check("Helion sum", outputs[-1], ref["expected_sum"]),
            }

        return call, "Helion", check

    return build


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--system", choices=("helion", "vllm"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--refs", type=Path, default=HERE / "results" / "refs")
    parser.add_argument("--reps", type=int, default=200)
    cli = parser.parse_args()
    rank, local_rank, group = initialize(
        kernel_name="deepseek_v3_moe_nvfp4_tp",
        world_size=tp.WORLD_SIZE,
        signal_pad_bytes=tp.SIGNAL_PAD_BYTES,
    )
    import torch.distributed._symmetric_memory as symm_mem

    try:
        device = torch.device("cuda", local_rank)
        shape = source.Shape(intermediate=2048 // tp.WORLD_SIZE)
        # Rank-seeded weights; every rank routes rank 0's token.
        tensors = source._allocate(shape, seed=17 + rank)
        dist.broadcast(tensors["hidden"], src=0)
        generator = torch.Generator(device=device)
        generator.manual_seed(5)
        residual = 16 * torch.randn(
            1, shape.hidden, device=device, dtype=torch.bfloat16, generator=generator
        )
        norm_weight = (
            1 + 0.1 * torch.randn(shape.hidden, device=device, generator=generator)
        ).to(torch.bfloat16)
        align_buffer = symm_mem.empty(16, dtype=torch.bfloat16, device=device).zero_()
        symm_mem.rendezvous(align_buffer, group.group_name)
        flusher = coldl2.L2Flusher(device)

        def flush_and_align() -> None:
            flusher()
            torch.ops.symm_mem.one_shot_all_reduce(
                align_buffer, "sum", group.group_name
            )

        ref_path = cli.refs / f"deepseek_v3_moe_tp4_rank{rank}.pt"
        if cli.system == "vllm":
            refs: dict[str, Any] = {"cases": {}}
            build = _vllm_builder(rank, group, shape, residual, norm_weight, refs)
        else:
            refs = torch.load(ref_path)
            build = _helion_builder(group, device, shape, residual, norm_weight, refs)
        calls: dict[str, coldl2.Recorded] = {}
        checks: dict[str, Check] = {}
        name = ""
        for label, selected_ids in source.ROUTING_CASES:
            case = dict(tensors)
            for key in _ROUTING_TENSORS:
                case[key] = tensors[key].clone()
            source._set_routing_case(case, selected_ids)
            inputs = [float(case[key].double().sum()) for key in _FINGERPRINT] + [
                float(residual.double().sum()),
                float(norm_weight.double().sum()),
            ]
            calls[label], name, checks[label] = build(label, case, inputs)
            errors = checks[label]()
            if rank == 0:
                print(f"{label}: similarity errors {errors}", flush=True)
        timings = coldl2.time_cold_many(
            dict(calls), reps=cli.reps, flusher=flush_and_align
        )
        for label, check in checks.items():  # the outputs of the last timed replay
            errors = check()
            if rank == 0:
                print(f"{label}: after timing, similarity errors {errors}", flush=True)
        rows = []
        for label, timing in timings.items():
            medians: list[float] = [0.0] * tp.WORLD_SIZE
            dist.all_gather_object(medians, timing["median_us"])
            loops: list[float] = [0.0] * tp.WORLD_SIZE
            dist.all_gather_object(loops, timing["loop_us"])
            rows.append(
                {
                    "workload": "deepseek_v3_moe_tp4",
                    "case": label,
                    "impl": name,
                    "median_us": max(medians),
                    "rank_medians_us": medians,
                    "loop_us": max(loops),
                    "rank_loop_us": loops,
                    "rank0": timing,
                }
            )
            if rank == 0:
                print(
                    f"deepseek_v3_moe_tp4 {label:>9s}  {name}: {max(medians):.2f} us "
                    f"(rank medians {', '.join(f'{m:.2f}' for m in medians)}; "
                    f"loop {max(loops):.2f})",
                    flush=True,
                )
        if cli.system == "vllm":
            cli.refs.mkdir(parents=True, exist_ok=True)
            torch.save(refs, ref_path)
        if rank == 0:
            cli.out.parent.mkdir(parents=True, exist_ok=True)
            cli.out.write_text(
                json.dumps(
                    {"system": cli.system, "gpu": coldl2.gpu_info(), "rows": rows},
                    indent=1,
                )
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
