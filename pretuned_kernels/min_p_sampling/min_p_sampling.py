"""Measured FP32 min-p sampling, shape 16 x 128512 and cutoff 0.05.

Ordinary full-search winner. The adjacent AOT module contains the measured
configuration."""

from __future__ import annotations

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from pretuned_kernels._sampling import capture_checked as _capture_checked
from pretuned_kernels._sampling import cdf_sample as _cdf_sample
from pretuned_kernels._sampling import philox_uniform as _philox_uniform
import torch

import helion
from helion._compiler.rng_utils import philox_rand_ref
import helion.language as hl

STRUCTURAL_POLICY = helion.CuteStructuralPolicy(
    cute_region_fission=True,
    cute_full_slice_matmul_tiling=True,
    cute_segmented_matmul_tiling=True,
    cute_flatten_nested_reductions=True,
    cute_materialize_transformed_operands=True,
)


@helion.aot_kernel(
    backend="cute",
    static_shapes=True,
    cute_structural_policy=STRUCTURAL_POLICY,
    cute_rng_stream="word0",
)
def min_p(
    x: torch.Tensor,
    seed: torch.Tensor,
    cutoff: torch.Tensor | float,
    vocab: hl.constexpr,
    index_dtype: torch.dtype,
) -> torch.Tensor:
    batch = x.numel() // vocab
    block_rows = hl.register_block_size(batch)
    block_cols = hl.register_block_size(vocab)
    parts = (vocab + block_cols - 1) // block_cols
    maxima = torch.empty((batch, parts), dtype=torch.float32, device=x.device)
    masses = torch.empty_like(maxima)
    result = torch.empty((batch,), dtype=index_dtype, device=x.device)
    for row, col in hl.tile([batch, vocab], block_size=[block_rows, block_cols]):
        values = x[row.index[:, None] * vocab + col.index[None, :]].float()
        values = torch.where(col.index[None, :] < vocab, values, 0.0)
        maxima[row, col.id] = values.amax(-1)
    hl.barrier()
    for row, col in hl.tile([batch, vocab], block_size=[block_rows, block_cols]):
        maximum = maxima[row, :].amax(-1)
        if isinstance(cutoff, torch.Tensor):
            row_cutoff = cutoff[row].float()
        else:
            row_cutoff = hl.full([row], cutoff, dtype=torch.float32)
        # Match pinned FlashInfer's rounded FP32 product and inclusive tie.
        threshold = maximum * row_cutoff
        values = x[row.index[:, None] * vocab + col.index[None, :]].float()
        weights = torch.where(
            (col.index[None, :] < vocab) & (values >= threshold[:, None]),
            values,
            0.0,
        )
        masses[row, col.id] = weights.sum(-1)
    hl.barrier()
    for row in hl.tile(batch):
        maximum = maxima[row, :].amax(-1)
        if isinstance(cutoff, torch.Tensor):
            row_cutoff = cutoff[row].float()
        else:
            row_cutoff = hl.full([row], cutoff, dtype=torch.float32)
        threshold = maximum * row_cutoff
        partial_masses = masses[row, :]
        tile_ids = hl.arange(partial_masses.size(-1))
        partial_cdf = hl.cumsum(partial_masses, dim=-1)
        total = partial_cdf.amax(-1)
        uniform = hl.rand([], seed=seed[0], offsets=row.index.to(torch.int64))
        target = uniform * total
        bucket = torch.amin(
            torch.where(partial_cdf > target[:, None], tile_ids[None, :], parts),
            dim=-1,
        )
        last_bucket = torch.amax(
            torch.where(partial_masses > 0.0, tile_ids[None, :], 0), dim=-1
        )
        bucket = torch.where(bucket < parts, bucket, last_bucket)
        prefix_before = torch.amax(
            torch.where(tile_ids[None, :] < bucket[:, None], partial_cdf, 0.0),
            dim=-1,
        )
        residual = (target - prefix_before).clamp_min(0.0)
        token = bucket[:, None] * block_cols + hl.arange(block_cols)[None, :]
        values = hl.load(
            x,
            [row.index[:, None] * vocab + token],
            extra_mask=token < vocab,
        ).float()
        weights = torch.where(
            (token < vocab) & (values >= threshold[:, None]), values, 0.0
        )
        cdf = hl.cumsum(weights, dim=-1)
        selected = torch.amin(
            torch.where((cdf > residual[:, None]) & (weights > 0.0), token, vocab),
            dim=-1,
        )
        last_positive = torch.amax(torch.where(weights > 0.0, token, -1), dim=-1)
        result[row] = torch.where(selected < vocab, selected, last_positive)
    return result


def use_cudagraph() -> bool:
    return True


SHAPES = [(16, 128512, 0.05)]


def min_p_sampling(
    probs: torch.Tensor, seed: torch.Tensor, min_p_cutoff: float = 0.05
) -> torch.Tensor:
    """Return one Int32 sample per row; seed is a current-call device tensor."""
    return min_p(probs.reshape(-1), seed, min_p_cutoff, probs.shape[1], torch.int32)


def _min_p_weights(probs: torch.Tensor, cutoff: float) -> torch.Tensor:
    threshold = probs.amax(-1) * cutoff
    return torch.where(probs >= threshold[:, None], probs, 0).double()


def _min_p_sampling_torch(
    probs: torch.Tensor, seed: torch.Tensor, cutoff: float = 0.05
) -> torch.Tensor:
    uniform = _philox_uniform(
        seed[0], torch.arange(probs.shape[0], device=probs.device, dtype=torch.int64)
    )
    return _cdf_sample(_min_p_weights(probs, cutoff), uniform)


def check_output(
    probs: torch.Tensor, seed: torch.Tensor, cutoff: float, output: torch.Tensor
) -> None:
    from pretuned_kernels._sampling import assert_cdf_accuracy
    from pretuned_kernels._sampling import reference_from_weights

    uniform = philox_rand_ref(
        seed[0], torch.arange(probs.shape[0], device=probs.device, dtype=torch.int64)
    )
    assert_cdf_accuracy(
        output,
        reference_from_weights(_min_p_weights(probs, cutoff), uniform, torch.int32),
    )


def make_inputs(
    shape: tuple[int, int, float], device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, vocab, _ = shape
    generator = torch.Generator(device=device).manual_seed(0)
    probs = torch.randn((batch, vocab), device=device, generator=generator).softmax(-1)
    return probs, torch.tensor([1729], dtype=torch.int64, device=device)


def _make_calls(shape: tuple[int, int, float]) -> tuple:
    probs, seed = make_inputs(shape)
    cutoff = shape[2]
    candidate = _capture_checked(
        lambda: min_p_sampling(probs, seed, cutoff),
        (probs, seed),
        seed,
        lambda output: check_output(probs, seed, cutoff, output),
    )
    reference = _capture_checked(
        lambda: _min_p_sampling_torch(probs, seed, cutoff),
        (probs, seed),
        seed,
        lambda output: check_output(probs, seed, cutoff, output),
    )
    return candidate, [("torch_philox_cdf", reference)], str(shape)


def check_case(shape: tuple[int, int, float]) -> None:
    """Validate both captured public calls used by the benchmark."""
    _make_calls(shape)


def correctness_check() -> None:
    for shape in SHAPES:
        check_case(shape)


def main(verbose: bool = True) -> dict:
    from pretuned_kernels._bench import run_sweep

    return run_sweep(
        SHAPES,
        _make_calls,
        use_cudagraph=False,
        pre_captured_cudagraph=True,
        rep=100,
        thermal_warmup_ms=1000,
        verbose=verbose,
        shape_header="(batch, vocab, min_p)",
    )


if __name__ == "__main__":
    main()
