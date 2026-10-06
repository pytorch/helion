"""Measured ordinary CHAIN recipe, 99 x 1 x 32000, with uniform producer regions.

This is a fixed-config diagnostic, not a full-search winner. The config enables
uniform producer regions. The adjacent AOT module contains the configuration."""

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
def chain(
    draft: torch.Tensor,
    draft_ids: torch.Tensor,
    target: torch.Tensor,
    seed: torch.Tensor,
    accepted_buffer: torch.Tensor | None = None,
    emitted_buffer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, steps, vocab = draft.shape
    draft_flat = draft.reshape(-1)
    target_flat = target.reshape(-1)
    output = torch.empty((batch, steps + 1), dtype=torch.int32, device=draft.device)
    output_flat = output.reshape(-1)
    accepted_count = torch.empty((batch,), dtype=torch.int32, device=draft.device)
    emitted_count = torch.empty((batch,), dtype=torch.int32, device=draft.device)
    mass_grain = 256
    parts = (vocab + mass_grain - 1) // mass_grain
    stops = torch.empty((batch,), dtype=torch.int32, device=draft.device)
    masses = torch.empty((batch, parts), dtype=torch.float32, device=draft.device)
    masses_flat = masses.reshape(-1)
    for row in hl.tile(batch):
        stop = hl.full([row], steps, dtype=torch.int32)
        alive = hl.full([row], True, dtype=torch.bool)
        accepted = hl.full([row], 0, dtype=torch.int32)
        for step in hl.static_range(steps):
            token = draft_ids[row, step]
            q = draft_flat[(row.index * steps + step) * vocab + token]
            p = target_flat[(row.index * (steps + 1) + step) * vocab + token]
            offset = row.index.to(torch.int64) * (steps + 2) + step
            uniform = 1.0 - hl.rand([], seed=seed[0], offsets=offset)
            ok = uniform * q < p
            stop = torch.where(alive & ~ok, step, stop)
            alive = alive & ok
            output[row, step] = torch.where(alive, token, -1)
        output[row, steps] = -1
        for step in hl.static_range(steps):
            token = draft_ids[row, step]
            q = draft_flat[(row.index * steps + step) * vocab + token]
            p = target_flat[(row.index * (steps + 1) + step) * vocab + token]
            offset = row.index.to(torch.int64) * (steps + 2) + step + 1
            uniform = 1.0 - hl.rand([], seed=seed[0], offsets=offset)
            accepted = accepted + ((step < stop) | (uniform * q < p)).to(torch.int32)
        if accepted_buffer is None:
            accepted_count[row] = accepted
        else:
            accepted_buffer[row] = accepted_buffer[row] + accepted
        if emitted_buffer is None:
            emitted_count[row] = stop
        else:
            emitted_buffer[row] = emitted_buffer[row] + stop
        stops[row] = stop
    hl.barrier()
    for job in hl.tile(batch * parts):
        job_row = job.index // parts
        part = job.index % parts
        column = part[:, None] * mass_grain + hl.arange(mass_grain)[None, :]
        stop = stops[job_row]
        safe_stop = stop.clamp_max(steps - 1)
        target_weights = hl.load(
            target_flat,
            [(job_row[:, None] * (steps + 1) + stop[:, None]) * vocab + column],
            extra_mask=column < vocab,
        )
        draft_weights = hl.load(
            draft_flat,
            [(job_row[:, None] * steps + safe_stop[:, None]) * vocab + column],
            extra_mask=column < vocab,
        )
        weights = torch.where(
            stop[:, None] < steps,
            (target_weights - draft_weights).clamp_min(0),
            target_weights,
        )
        weights = torch.where(column < vocab, weights, 0.0)
        masses_flat[job] = weights.sum(-1)
    hl.barrier()
    for row in hl.tile(batch):
        stop = stops[row]
        partial_mass = masses[row, :]
        part_ids = hl.arange(partial_mass.size(-1))
        partial_cdf = hl.cumsum(partial_mass, dim=-1)
        total = partial_cdf.amax(-1)
        sample_offset = (
            row.index.to(torch.int64) * (steps + 2)
            + steps
            + (stop < steps).to(torch.int64)
        )
        uniform = hl.rand([], seed=seed[0], offsets=sample_offset)
        threshold = uniform * total
        bucket = torch.amin(
            torch.where(
                (partial_cdf > threshold[:, None]) & (partial_mass > 0),
                part_ids[None, :],
                parts,
            ),
            dim=-1,
        )
        last_bucket = torch.amax(
            torch.where(partial_mass > 0, part_ids[None, :], 0), dim=-1
        )
        bucket = torch.where(bucket < parts, bucket, last_bucket)
        before = torch.amax(
            torch.where(part_ids[None, :] < bucket[:, None], partial_cdf, 0.0),
            dim=-1,
        )
        residual = (threshold - before).clamp_min(0.0)
        token = bucket[:, None] * mass_grain + hl.arange(mass_grain)[None, :]
        target_weights = hl.load(
            target_flat,
            [(row.index[:, None] * (steps + 1) + stop[:, None]) * vocab + token],
            extra_mask=token < vocab,
        )
        draft_weights = hl.load(
            draft_flat,
            [
                (row.index[:, None] * steps + stop.clamp_max(steps - 1)[:, None])
                * vocab
                + token
            ],
            extra_mask=token < vocab,
        )
        weights = torch.where(
            stop[:, None] < steps,
            (target_weights - draft_weights).clamp_min(0),
            target_weights,
        )
        weights = torch.where(token < vocab, weights, 0.0)
        cdf = hl.cumsum(weights, dim=-1)
        selected = torch.amin(
            torch.where((cdf > residual[:, None]) & (weights > 0), token, vocab),
            dim=-1,
        )
        last_positive = torch.amax(torch.where(weights > 0, token, -1), dim=-1)
        sampled = torch.where(selected < vocab, selected, last_positive)
        output_flat[row.index * (steps + 1) + stop] = sampled
    return (
        output,
        accepted_count if accepted_buffer is None else accepted_buffer,
        emitted_count if emitted_buffer is None else emitted_buffer,
    )


def use_cudagraph() -> bool:
    return True


SHAPES = [(99, 1, 32000)]


def chain_speculative_sampling(
    draft: torch.Tensor,
    draft_ids: torch.Tensor,
    target: torch.Tensor,
    seed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return token IDs, accepted counts, and emitted-draft counts.

    Only the measured default-counter signature is selected by this recipe.
    """
    return chain(draft, draft_ids, target, seed, None, None)


def _chain_details(
    draft: torch.Tensor, ids: torch.Tensor, target: torch.Tensor, seed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, steps, _ = draft.shape
    rows = torch.arange(batch, device=draft.device)
    offsets = torch.arange(
        batch * (steps + 2), device=draft.device, dtype=torch.int64
    ).reshape(batch, steps + 2)
    uniforms = 1 - _philox_uniform(seed[0], offsets)
    q = draft.gather(-1, ids.long().unsqueeze(-1)).squeeze(-1)
    p = target[:, :steps].gather(-1, ids.long().unsqueeze(-1)).squeeze(-1)
    prefix = (uniforms[:, :steps] * q < p).long().cumprod(-1).bool()
    stops = prefix.sum(-1)
    accepted = (
        (
            (torch.arange(steps, device=draft.device)[None, :] < stops[:, None])
            | (uniforms[:, 1 : steps + 1] * q < p)
        )
        .sum(-1)
        .to(torch.int32)
    )
    weights = target[rows, stops]
    weights = torch.where(
        stops[:, None] < steps,
        (weights - draft[rows, stops.clamp_max(steps - 1)]).clamp_min(0),
        weights,
    )
    uniform = _philox_uniform(
        seed[0], rows.to(torch.int64) * (steps + 2) + steps + (stops < steps)
    )
    tokens = torch.full((batch, steps + 1), -1, dtype=torch.int32, device=draft.device)
    tokens[:, :steps] = torch.where(prefix, ids, -1)
    return tokens, accepted, stops, weights, uniform


def _chain_speculative_sampling_torch(
    draft: torch.Tensor, ids: torch.Tensor, target: torch.Tensor, seed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, accepted, stops, weights, uniform = _chain_details(draft, ids, target, seed)
    rows = torch.arange(draft.shape[0], device=draft.device)
    tokens[rows, stops] = _cdf_sample(weights, uniform)
    return tokens, accepted, stops.to(torch.int32)


def check_output(
    draft: torch.Tensor,
    ids: torch.Tensor,
    target: torch.Tensor,
    seed: torch.Tensor,
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    from pretuned_kernels._sampling import assert_cdf_accuracy
    from pretuned_kernels._sampling import reference_from_weights

    tokens, accepted, emitted = output
    batch, steps, _ = draft.shape
    assert tokens.shape == (batch, steps + 1) and accepted.shape == emitted.shape == (
        batch,
    )
    assert all(
        value.dtype == torch.int32 and value.device == draft.device for value in output
    )
    expected, counts, stops, weights, uniform = _chain_details(draft, ids, target, seed)
    torch.testing.assert_close(accepted, counts, rtol=0, atol=0)
    torch.testing.assert_close(emitted, stops.to(torch.int32), rtol=0, atol=0)
    rows = torch.arange(batch, device=draft.device)
    sample = tokens[rows, stops]
    positions = torch.arange(steps + 1, device=draft.device)[None, :]
    fixed = positions != stops[:, None]
    torch.testing.assert_close(tokens[fixed], expected[fixed], rtol=0, atol=0)
    assert_cdf_accuracy(
        sample, reference_from_weights(weights.double(), uniform, torch.int32)
    )


def make_inputs(
    shape: tuple[int, int, int], device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, steps, vocab = shape
    generator = torch.Generator(device=device).manual_seed(0)
    draft = torch.rand((batch, steps, vocab), device=device, generator=generator)
    draft = draft / draft.sum(-1, keepdim=True)
    ids = torch.randint(
        vocab, (batch, steps), device=device, generator=generator, dtype=torch.int32
    )
    target = torch.rand((batch, steps + 1, vocab), device=device, generator=generator)
    target = target / target.sum(-1, keepdim=True)
    return draft, ids, target, torch.tensor([1729], device=device, dtype=torch.int64)


def _make_calls(shape: tuple[int, int, int]) -> tuple:
    args = make_inputs(shape)
    candidate = _capture_checked(
        lambda: chain_speculative_sampling(*args),
        args,
        args[3],
        lambda output: check_output(*args, output),
    )
    reference = _capture_checked(
        lambda: _chain_speculative_sampling_torch(*args),
        args,
        args[3],
        lambda output: check_output(*args, output),
    )
    return candidate, [("torch_philox_chain_cdf", reference)], str(shape)


def check_case(shape: tuple[int, int, int]) -> None:
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
        shape_header="(batch, steps, vocab)",
    )


if __name__ == "__main__":
    main()
