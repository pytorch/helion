"""Probe static list scheduling on a ragged B>1 Qwen3 decode layer.

The kernel body is derived mechanically from the checked-in pretuned B1
kernel, so its fusion boundary and tuned configuration stay identical.  The
only semantic extension is a per-request context-length input.  Attention
splits wholly beyond a request's context skip their body; a mask handles a
partially occupied final split.

This is a scheduler experiment, not a new pretuned production entry point.
It compares the global list placement with the exact same compiler and source
while only disabling the global proposal.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import json
import linecache
from pathlib import Path
import sys
import textwrap
import types
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pretuned_kernels._bench import bench_pre_captured_cudagraphs
from pretuned_kernels._bench import capture_cuda_graph
from pretuned_kernels._bench import thermal_warmup
from pretuned_kernels.megakernels.qwen3_decode_layer import qwen3_decode_layer as qwen
from pretuned_kernels.megakernels.qwen3_decode_layer._helion_aot_qwen3_decode_layer_cuda_sm100 import (
    CONFIG,
)
import torch

import helion
from helion._compiler import cross_loop_scheduler

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.kernel import CompiledConfig
    from helion.runtime.kernel import Kernel


_NEGATIVE_FLOAT32 = "-3.4028234663852886e38"


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(
            f"expected exactly one occurrence of Qwen source fragment: {old!r}"
        )
    return source.replace(old, new, 1)


def _make_batched_kernel() -> Kernel:
    """Derive a ragged kernel without copying the thousand-line tuned body."""
    source = textwrap.dedent(inspect.getsource(qwen.qwen3_decode_layer.fn))
    source = _replace_once(
        source,
        '@helion.aot_kernel(static_shapes=True, backend="triton")',
        '@helion.kernel(static_shapes=True, autotune_effort="none")',
    )
    source = _replace_once(source, "    eps,\n):", "    eps,\n    context_lens,\n):")
    source = _replace_once(
        source, '            float("-inf"),', f"            {_NEGATIVE_FLOAT32},"
    )

    guarded_begin = source.index("        attention_split_query_head = (")
    guarded_end_marker = "            attention_split_m_i = attention_split_m_ij\n"
    guarded_end = source.index(guarded_end_marker, guarded_begin) + len(
        guarded_end_marker
    )
    guarded_body = source[guarded_begin:guarded_end]
    score_compute = """            attention_split_scores = torch.bmm(
                attention_split_q_blk,
                attention_split_k.transpose(1, 2),
                torch.float32,
            )
"""
    masked_scores = (
        score_compute
        + f"""            attention_split_scores = torch.where(
                attention_split_n[None, None, :]
                < attention_split_context_len[:, None, None],
                attention_split_scores,
                {_NEGATIVE_FLOAT32},
            )
"""
    )
    guarded_body = _replace_once(guarded_body, score_compute, masked_scores)
    guard = """        attention_split_context_len = hl.load(
            context_lens, [attention_split_token]
        )
        if (
            attention_split_split_idx * attention_split_split_context
            < attention_split_context_len
        ):
"""
    source = (
        source[:guarded_begin]
        + guard
        + textwrap.indent(guarded_body, "    ")
        + source[guarded_end:]
    )

    module_name = "_helion_qwen3_batched_scheduler_probe"
    filename = f"<{module_name}>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    module = types.ModuleType(module_name)
    module.__dict__.update(vars(qwen))
    module.__dict__["__name__"] = module_name
    module.__file__ = filename
    sys.modules[module_name] = module
    exec(compile(source, filename, "exec"), module.__dict__)
    return module.qwen3_decode_layer


def _make_inputs(context_lengths: tuple[int, ...]) -> dict[str, torch.Tensor]:
    batch = len(context_lengths)
    if not batch or any(
        length <= 0 or length > qwen.CONTEXT for length in context_lengths
    ):
        raise ValueError(f"context lengths must be in [1, {qwen.CONTEXT}]")
    old_batch = qwen.BATCH
    qwen.BATCH = batch
    try:
        tensors = qwen._make_inputs()
    finally:
        qwen.BATCH = old_batch

    max_blocks = qwen.CONTEXT // qwen.CACHE_BLOCK
    block_counts = tuple(
        (length + qwen.CACHE_BLOCK - 1) // qwen.CACHE_BLOCK
        for length in context_lengths
    )
    physical_blocks = sum(block_counts)
    block_table = torch.zeros((batch, max_blocks), device="cuda", dtype=torch.int32)
    cursor = 0
    for request, block_count in enumerate(block_counts):
        block_table[request, :block_count] = torch.arange(
            cursor,
            cursor + block_count,
            device="cuda",
            dtype=torch.int32,
        )
        cursor += block_count
    lengths = torch.tensor(context_lengths, device="cuda", dtype=torch.int64)
    rows = torch.arange(batch, device="cuda")
    final_logical_blocks = (lengths - 1) // qwen.CACHE_BLOCK
    final_physical_blocks = block_table[rows, final_logical_blocks].to(torch.int64)
    tensors["position"] = lengths - 1
    tensors["block_table"] = block_table
    tensors["slot_mapping"] = (
        final_physical_blocks * qwen.CACHE_BLOCK + (lengths - 1) % qwen.CACHE_BLOCK
    )
    tensors["kv_cache"] = torch.randn(
        (
            physical_blocks,
            qwen.CACHE_BLOCK,
            qwen.KV_HEADS,
            2 * qwen.HEAD_DIM,
        ),
        device="cuda",
        dtype=torch.bfloat16,
    )
    tensors["cos_sin"].normal_()
    tensors["context_lens"] = lengths
    return tensors


def _variant_inputs(
    base: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], tuple[object, ...], Callable[[], None]]:
    tensors = {name: value.clone() for name, value in base.items()}
    context_lens = tensors.pop("context_lens")
    args = (*qwen._kernel_args(tensors), context_lens)
    initial_residual = tensors["residual"].clone()
    slots = tensors["slot_mapping"]
    blocks = slots // qwen.CACHE_BLOCK
    offsets = slots % qwen.CACHE_BLOCK
    initial_cache_slots = tensors["kv_cache"][blocks, offsets].clone()

    def reset() -> None:
        tensors["residual"].copy_(initial_residual)
        tensors["kv_cache"][blocks, offsets] = initial_cache_slots

    return tensors, args, reset


def _compile(
    kernel: Kernel,
    args: tuple[object, ...],
    *,
    multiplier: int,
    request_major: bool,
    global_list: bool,
) -> tuple[CompiledConfig, tuple[dict[str, object], ...]]:
    bound = kernel.bind(args)
    values = deepcopy(CONFIG)
    values["num_sm_multiplier"] = multiplier
    if request_major:
        # Root 5 is the split-attention grid.  This is an ordinary loop-order
        # choice, not a scheduler special case.
        values["loop_orders"][5] = [2, 0, 1]
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)

    original = cross_loop_scheduler._global_unit_list_schedule
    records: list[dict[str, object]] = []

    def traced(*call_args: object, **call_kwargs: object):  # noqa: ANN202
        result = original(*call_args, **call_kwargs)
        source_schedule = call_args[1]
        records.append(
            {
                "accepted": result is not None,
                "input_segments": len(source_schedule.segments),
                "output_segments": None if result is None else len(result.segments),
            }
        )
        return result

    try:
        cross_loop_scheduler._global_unit_list_schedule = (
            traced if global_list else lambda *args, **kwargs: None
        )
        bound._compile_cache.clear()
        compiled = bound.compile_config(config)
    finally:
        cross_loop_scheduler._global_unit_list_schedule = original
    return compiled, tuple(records)


@torch.inference_mode()
def benchmark(
    context_lengths: tuple[int, ...],
    *,
    multiplier: int,
    repetitions: int,
    warmup_ms: int,
) -> dict[str, object]:
    qwen._require_sm100()
    kernel = _make_batched_kernel()
    base = _make_inputs(context_lengths)
    variants: list[
        tuple[str, CompiledConfig, tuple[object, ...], Callable[[], None]]
    ] = []
    plans: dict[str, tuple[dict[str, object], ...]] = {}
    for name, request_major, global_list in (
        ("list_pretuned_order", False, True),
        ("list_request_major", True, True),
        ("local_request_major", True, False),
    ):
        _tensors, args, reset = _variant_inputs(base)
        compiled, records = _compile(
            kernel,
            args,
            multiplier=multiplier,
            request_major=request_major,
            global_list=global_list,
        )
        plans[name] = records
        reset()
        compiled(*args)
        torch.cuda.synchronize()
        variants.append((name, compiled, args, reset))

    reference: tuple[torch.Tensor, ...] | None = None
    for _name, compiled, args, reset in variants:
        reset()
        outputs = compiled(*args)
        torch.cuda.synchronize()
        snapshot = tuple(output.clone() for output in outputs)
        if reference is None:
            reference = snapshot
        else:
            for actual, expected in zip(snapshot, reference, strict=True):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    captured: list[tuple[str, Callable[[], object], Callable[[], None]]] = []
    for name, compiled, args, reset in variants:
        graph, _outputs = capture_cuda_graph(
            lambda compiled=compiled, args=args: compiled(*args),
            reset,
        )
        captured.append((name, graph.replay, reset))

    thermal_warmup(warmup_ms)
    timings = bench_pre_captured_cudagraphs(
        [replay for _name, replay, _reset in captured],
        rep=repetitions,
        resets=[reset for _name, _replay, reset in captured],
    )
    return {
        "workload": "Qwen3-8B FP8 full decode layer, ragged batch",
        "context_lengths": context_lengths,
        "num_sm_multiplier": multiplier,
        "device": torch.cuda.get_device_name(),
        "timings_us": {
            name: latency_ms * 1000
            for (name, _replay, _reset), latency_ms in zip(
                captured, timings, strict=True
            )
        },
        "plans": plans,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-lengths", type=int, nargs="+", default=(2048, 8192))
    parser.add_argument("--multiplier", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--warmup-ms", type=int, default=10_000)
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark(
                tuple(args.context_lengths),
                multiplier=args.multiplier,
                repetitions=args.repetitions,
                warmup_ms=args.warmup_ms,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
