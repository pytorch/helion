"""Exact variable-length top-k with a configuration-dependent reduction hierarchy.

For N=128000 and K=2048, tiles of 16384 need two stages; tiles of 4096
need six. The whole-pipeline objective includes the additional stages.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import torch

import helion
from helion.autotuner.pipeline import autotune_pipeline
from helion.autotuner.pipeline_benchmark import PipelineEvaluator
import helion.language as hl
from helion.runtime.pipeline import PipelineConfig
from helion.runtime.pipeline import PipelineStage

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel


def _threshold(
    values: torch.Tensor, valid: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    bits = values.to(torch.float32).view(torch.int32)
    keys = torch.where(bits < 0, bits ^ 2147483647, bits)
    prefix = torch.full((), -2147483648, dtype=torch.int32, device=values.device)
    positive = torch.sum(valid & (keys >= 0), dtype=torch.int32)
    prefix = torch.where(positive >= k, 0, prefix)
    low_bit = 16 if values.dtype == torch.bfloat16 else 0
    for bit in range(30, low_bit - 1, -1):
        candidate = prefix | (1 << bit)
        count = torch.sum(valid & (keys >= candidate), dtype=torch.int32)
        prefix = torch.where(count >= k, candidate, prefix)
    if low_bit:
        prefix = torch.where(prefix < 0, prefix | 65535, prefix)
    return keys, prefix


@helion.kernel(
    static_shapes=True, ignore_warnings=[helion.exc.TensorOperationInWrapper]
)
def topk_stage(
    x: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    input_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep K candidates per tile, preserving original positions and padding."""
    rows, cols = x.shape
    k = hl.specialize(k)
    assert x.dtype in (torch.float32, torch.bfloat16)
    assert x.is_contiguous() and lengths.is_contiguous()
    assert lengths.shape == (rows,) and lengths.dtype == torch.int32
    assert rows > 0 and 0 < k <= min(cols, 2048)
    block_cols = hl.register_block_size(4096, 32768)
    parts = (cols + block_cols - 1) // block_cols
    padded_k = 1 << (k - 1).bit_length()
    values = torch.empty((rows, parts * k), dtype=x.dtype, device=x.device)
    indices = torch.empty((rows, parts * k), dtype=torch.int32, device=x.device)
    next_lengths = torch.empty_like(lengths)
    flat_x, flat_values, flat_indices = x.view(-1), values.view(-1), indices.view(-1)
    for row, part in hl.grid([rows, parts]):
        length = lengths[row]
        columns = part * block_cols + hl.arange(block_cols)
        valid = (columns < cols) & (columns < length)
        data = hl.load(flat_x, [row * cols + columns], extra_mask=valid)
        local_length = torch.clamp(length - part * block_cols, min=0, max=block_cols)
        wanted = torch.clamp(local_length, max=k)
        if local_length > k:
            keys, prefix = _threshold(data, valid, k)
            above = valid & (keys > prefix)
            equal = valid & (keys == prefix)
            above_count = torch.sum(above, dim=0, dtype=torch.int32)
            tie_rank = hl.cumsum(equal.to(torch.int32), dim=0)
            selected = above | (equal & (tie_rank <= k - above_count))
        else:
            selected = valid
        positions = hl.cumsum(selected.to(torch.int32), dim=0) - 1
        if input_indices is None:
            original_indices = columns.to(torch.int32)
        else:
            original_indices = hl.load(
                input_indices, [row, columns], extra_mask=selected
            )
        output_base = (row * parts + part) * k
        hl.store(flat_values, [output_base + positions], data, extra_mask=selected)
        hl.store(
            flat_indices,
            [output_base + positions],
            original_indices,
            extra_mask=selected,
        )
        padding = hl.arange(padded_k)
        pad_mask = (padding >= wanted) & (padding < k)
        hl.store(
            flat_values, [output_base + padding], float("-inf"), extra_mask=pad_mask
        )
        hl.store(flat_indices, [output_base + padding], -1, extra_mask=pad_mask)
        if part == 0:
            next_lengths[row] = (length // block_cols) * k + torch.clamp(
                length % block_cols, max=k
            )
    return values, indices, next_lengths


def topk_pipeline(
    x: torch.Tensor, lengths: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select exact unordered values/indices over each row's valid prefix."""
    current, current_lengths, current_indices = x, lengths, None
    while True:
        values, indices, current_lengths = topk_stage(
            current, current_lengths, k, current_indices
        )
        if values.shape[1] == k:
            return values, indices
        if values.shape[1] >= current.shape[1]:
            raise ValueError("The candidate top-k hierarchy does not make progress")
        current, current_indices = values, indices


@dataclass
class TopkReference:
    values: torch.Tensor
    x: torch.Tensor
    lengths: torch.Tensor
    k: int


def reference(x: torch.Tensor, lengths: torch.Tensor, k: int) -> TopkReference:
    valid = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
    values = x.masked_fill(~valid, -float("inf")).topk(k, dim=1).values
    return TopkReference(values, x, lengths, k)


def check(actual: object, expected: object) -> None:
    assert isinstance(expected, TopkReference)
    assert isinstance(actual, tuple) and len(actual) == 2
    values, indices = actual
    assert isinstance(values, torch.Tensor) and isinstance(indices, torch.Tensor)
    assert values.shape == indices.shape == expected.values.shape
    assert values.dtype == expected.x.dtype and indices.dtype == torch.int32
    torch.testing.assert_close(
        values.sort(dim=1, descending=True).values, expected.values, atol=0, rtol=0
    )
    valid = (indices >= 0) & (indices < expected.lengths[:, None])
    assert bool(torch.all(valid.sum(dim=1) == expected.lengths.clamp(max=expected.k)))
    assert bool(torch.all(indices[~valid] == -1))
    selected = expected.x.gather(1, indices.clamp(min=0).long())
    torch.testing.assert_close(values[valid], selected[valid], atol=0, rtol=0)
    sorted_indices = indices.sort(dim=1).values
    assert not bool(
        torch.any(
            (sorted_indices[:, 1:] == sorted_indices[:, :-1])
            & (sorted_indices[:, 1:] >= 0)
        )
    )


def make_inputs(
    *,
    rows: int = 24,
    cols: int = 128000,
    k: int = 2048,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    generator = torch.Generator(device=device).manual_seed(17)
    x = torch.randn((rows, cols), device=device, dtype=dtype, generator=generator)
    full = torch.full((rows,), cols, device=device, dtype=torch.int32)
    ragged = (
        full
        - torch.arange(rows, device=device, dtype=torch.int32)
        * max(1, cols // max(2 * rows, 1))
    ).clamp(min=0)
    return [(x, full, k), (x.clone(), ragged, k)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=24)
    parser.add_argument("--cols", type=int, default=128000)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run a bounded finite joint search (no LLM request)",
    )
    args = parser.parse_args()
    inputs = make_inputs(
        rows=args.rows, cols=args.cols, k=args.k, dtype=getattr(torch, args.dtype)
    )
    rows: list[dict[str, Any]] = []
    measurements = []
    for width in (16384, 4096):
        config = helion.Config(block_sizes=[width], num_warps=8, num_stages=1)

        def initial_config(
            kernel: Kernel[object],
            values: tuple[object, ...],
            config: helion.Config = config,
        ) -> helion.Config:
            return config

        evaluator = PipelineEvaluator(
            topk_pipeline,
            inputs,
            reference=reference,
            check=check,
            initial_config=initial_config,
        )
        measured = evaluator.evaluate(PipelineConfig())
        if measured.status != "ok":
            raise RuntimeError(measured.error)
        row = {"tile_width": width, **measured.to_dict()}
        rows.append(row)
        measurements.append(measured)
        print(
            f"tile={width}: stages={[len(trace) for trace in measured.traces]}, graph geomean={measured.aggregate_ms * 1000:.3f} us"
        )
    if args.tune:

        def candidates(stage: PipelineStage) -> list[helion.Config]:
            return [
                helion.Config(block_sizes=[width], num_warps=8, num_stages=1)
                for width in (4096, 8192, 16384)
            ]

        def initial(
            kernel: Kernel[object], values: tuple[object, ...]
        ) -> helion.Config:
            return helion.Config(block_sizes=[16384], num_warps=8, num_stages=1)

        tuned = autotune_pipeline(
            topk_pipeline,
            inputs,
            reference=reference,
            check=check,
            initial_bundle=measurements[0].bundle,
            initial_config=initial,
            config_candidates=candidates,
            algorithm="finite",
            max_evaluations=16,
            coordinate_evaluations=3,
            beam_width=2,
            topology_refinements=2,
            rounds=2,
            final_top_k=2,
            workload_tag=f"topk:{args.rows}:{args.cols}:{args.k}:{args.dtype}:full-ragged-v1",
            benchmark_tag="cuda-hip-graph-event-median-v1",
        )
        rows.append({"joint_search": tuned.to_dict()})
        print(
            f"joint result: stages={[len(trace) for trace in tuned.traces]}, graph geomean={tuned.aggregate_ms * 1000:.3f} us"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
