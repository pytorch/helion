"""Seven measured categorical cases: six CDF recipes and one packed Gumbel race.

Ordinary full-search winners; V=128512 at B=16/32/64/128/256/512,
and B=99, V=32000 for Gumbel.
The adjacent AOT module contains the measured configurations."""

from __future__ import annotations

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import math

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


def multiply_u32(
    value: torch.Tensor, multiplier: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # Expose a widened unsigned 32-bit product to normal backend lowering.
    # Signed int64 overflow preserves the low 64 product bits; masking after
    # the arithmetic shift recovers the unsigned high word exactly.
    product = value.to(torch.uint32).to(torch.int64) * multiplier
    return (product >> 32) & 4294967295, product & 4294967295


def philox_words(
    seed: torch.Tensor, subsequence: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """First four words after curand_init(seed, subsequence, offset=0)."""
    c0 = subsequence * 0
    c1 = subsequence * 0
    c2 = subsequence & 4294967295
    c3 = (subsequence >> 32) & 4294967295
    key0 = seed & 4294967295
    key1 = (seed >> 32) & 4294967295
    for _round in range(10):
        high0, low0 = multiply_u32(c0, 3528531795)
        high1, low1 = multiply_u32(c2, 3449720151)
        c0, c1, c2, c3 = high1 ^ c1 ^ key0, low1, high0 ^ c3 ^ key1, low0
        key0 = (key0 + 2654435769) & 4294967295
        key1 = (key1 + 3144134277) & 4294967295
    return c0, c1, c2, c3


def uniform_from_word(word: torch.Tensor) -> torch.Tensor:
    return word.to(torch.float32) * 2.3283064365386963e-10 + 1.1641532182693481e-10


def gumbel_from_word(word: torch.Tensor) -> torch.Tensor:
    uniform = uniform_from_word(word)
    return -0.6931471806 * torch.log2(-torch.log2(uniform * 0.9999998807907104))


def score_from_word(value: torch.Tensor, word: torch.Tensor) -> torch.Tensor:
    """Compute a Gumbel score with explicit SM103 arithmetic.

    Use FP32 FMA and multiplication with flush-to-zero, two approximate base-2
    logarithms, and a final FMA with the input logit.
    """
    uniform = hl.inline_asm_elementwise(
        "fma.rn.ftz.f32 $0, $1, 0f2F800000, 0f2F000000;",
        "=f,f",
        [word.float()],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )
    scaled = hl.inline_asm_elementwise(
        "mul.rn.ftz.f32 $0, $1, 0f3F7FFFFE;",
        "=f,f",
        [uniform],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )
    logarithm = hl.inline_asm_elementwise(
        "lg2.approx.ftz.f32 $0, $1;",
        "=f,f",
        [scaled],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )
    logarithm = hl.inline_asm_elementwise(
        "lg2.approx.ftz.f32 $0, $1;",
        "=f,f",
        [-logarithm],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )
    return hl.inline_asm_elementwise(
        "fma.rn.ftz.f32 $0, $1, 0fBF317218, $2;",
        "=f,f,f",
        [logarithm, value],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


def score_key(score: torch.Tensor, column: torch.Tensor) -> torch.Tensor:
    # Numeric Float32 ordering, treating the two zero signs as equal.
    # The low word makes equal scores select the largest logical index.
    bits = score.view(torch.int32)
    magnitude = bits & 2147483647
    rank = torch.where(bits < 0, -magnitude, magnitude).to(torch.int64)
    return (rank << 32) | column.to(torch.int64)


@helion.aot_kernel(
    backend="cute",
    static_shapes=True,
    cute_structural_policy=STRUCTURAL_POLICY,
    cute_rng_stream="word0",
)
def categorical(
    x: torch.Tensor,
    seed: torch.Tensor,
    vocab: hl.constexpr,
    logits: hl.constexpr,
    index_dtype: torch.dtype,
) -> torch.Tensor:
    batch = x.numel() // vocab
    block_rows = hl.register_block_size(batch)
    block_cols = hl.register_block_size(vocab)
    parts = (vocab + block_cols - 1) // block_cols
    masses = torch.empty((batch, parts), dtype=torch.float32, device=x.device)
    maxima = (
        torch.empty_like(masses)
        if logits
        else torch.empty((0,), dtype=torch.float32, device=x.device)
    )
    result = torch.empty((batch,), dtype=index_dtype, device=x.device)
    for row, col in hl.tile([batch, vocab], block_size=[block_rows, block_cols]):
        values = x[row.index[:, None] * vocab + col.index[None, :]].float()
        if logits:
            values = torch.where(col.index[None, :] < vocab, values, -float("inf"))
            maximum = values.amax(-1)
            weights = torch.where(
                values > -float("inf"),
                torch.exp(values - maximum[:, None]),
                0.0,
            )
            maxima[row, col.id] = maximum
        else:
            weights = torch.where(col.index[None, :] < vocab, values, 0.0)
        masses[row, col.id] = weights.sum(-1)
    hl.barrier()
    for row in hl.tile(batch):
        if logits:
            partial_maxima = maxima[row, :]
            global_maximum = partial_maxima.amax(-1)
            partial_masses = masses[row, :] * torch.exp(
                partial_maxima - global_maximum[:, None]
            )
        else:
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
        selected_values = hl.load(
            x,
            [row.index[:, None] * vocab + token],
            extra_mask=token < vocab,
        ).float()
        if logits:
            selected_weights = torch.exp(selected_values - global_maximum[:, None])
        else:
            selected_weights = selected_values
        selected_weights = torch.where(token < vocab, selected_weights, 0.0)
        cdf = hl.cumsum(selected_weights, dim=-1)
        selected = torch.amin(
            torch.where(
                (cdf > residual[:, None]) & (selected_weights > 0.0), token, vocab
            ),
            dim=-1,
        )
        # A rounded target can reach the last CDF entry. Keep that final
        # rounding interval on the last positive token, never padding.
        last_positive = torch.amax(
            torch.where(selected_weights > 0.0, token, -1), dim=-1
        )
        result[row] = torch.where(selected < vocab, selected, last_positive)
    return result


@helion.aot_kernel(
    backend="cute",
    static_shapes=True,
    cute_structural_policy=STRUCTURAL_POLICY,
    cute_rng_stream="word0",
)
def categorical_v2(
    x: torch.Tensor,
    seed: torch.Tensor,
    vocab: hl.constexpr,
    logits: hl.constexpr,
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
        if logits:
            values = torch.where(col.index[None, :] < vocab, values, -float("inf"))
            maximum = values.amax(-1)
            weights = torch.where(
                values > -float("inf"),
                torch.exp(values - maximum[:, None]),
                0.0,
            )
        else:
            maximum = hl.full([row], 0.0, dtype=torch.float32)
            weights = torch.where(col.index[None, :] < vocab, values, 0.0)
        maxima[row, col.id] = maximum
        masses[row, col.id] = weights.sum(-1)
    hl.barrier()
    for row in hl.tile(batch):
        partial_maxima = maxima[row, :]
        global_maximum = partial_maxima.amax(-1)
        partial_masses = masses[row, :] * torch.exp(
            partial_maxima - global_maximum[:, None]
        )
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
        selected_values = hl.load(
            x,
            [row.index[:, None] * vocab + token],
            extra_mask=token < vocab,
        ).float()
        if logits:
            selected_weights = torch.exp(selected_values - global_maximum[:, None])
        else:
            selected_weights = selected_values
        selected_weights = torch.where(token < vocab, selected_weights, 0.0)
        cdf = hl.cumsum(selected_weights, dim=-1)
        selected = torch.amin(
            torch.where(
                (cdf > residual[:, None]) & (selected_weights > 0.0), token, vocab
            ),
            dim=-1,
        )
        # A rounded target can reach the last CDF entry. Keep that final
        # rounding interval on the last positive token, never padding.
        last_positive = torch.amax(
            torch.where(selected_weights > 0.0, token, -1), dim=-1
        )
        result[row] = torch.where(selected < vocab, selected, last_positive)
    return result


@helion.aot_kernel(
    backend="cute",
    static_shapes=True,
    cute_structural_policy=STRUCTURAL_POLICY,
    fast_math=True,
)
def categorical_gumbel(
    x: torch.Tensor,
    seed: torch.Tensor,
    cutoff: torch.Tensor,
    min_p: hl.constexpr,
    logits: hl.constexpr,
    index_dtype: torch.dtype,
) -> torch.Tensor:
    assert not min_p
    batch, vocab = x.shape
    vector_width = math.gcd(4, vocab)
    groups = vocab // vector_width
    output = torch.empty((batch,), dtype=index_dtype, device=x.device)
    for row in hl.tile(batch):
        best_key = hl.full([row], -9223372036854775808, dtype=torch.int64)
        for packet in hl.tile(groups):
            subsequence = (
                row.index[:, None].to(torch.int64) * vocab
                + packet.index[None, :].to(torch.int64) * vector_width
            )
            words = philox_words(seed[0], subsequence)
            packet_key = hl.full([row, packet], -9223372036854775808, dtype=torch.int64)
            for word_index in hl.static_range(vector_width):
                column = packet.index * vector_width + word_index
                value = hl.load(
                    x,
                    [row.index[:, None], column[None, :]],
                    extra_mask=column[None, :] < vocab,
                ).float()
                if not logits:
                    value = torch.log(value)
                score = score_from_word(value, words[word_index])
                key = score_key(score, column[None, :])
                key = torch.where(column[None, :] < vocab, key, -9223372036854775808)
                packet_key = torch.maximum(packet_key, key)
            best_key = torch.maximum(best_key, packet_key.amax(-1))
        output[row] = best_key & 4294967295
    return output


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _score_reference(x: torch.Tensor, words: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row, col in hl.tile(x.shape):
        out[row, col] = hl.inline_asm_elementwise(
            "{ .reg .f32 u, a, b; cvt.rn.f32.u32 u, $1; "
            "fma.rn.ftz.f32 u, u, 0f2F800000, 0f2F000000; "
            "mul.rn.ftz.f32 a, u, 0f3F7FFFFE; "
            "lg2.approx.ftz.f32 b, a; neg.f32 a, b; "
            "lg2.approx.ftz.f32 b, a; "
            "fma.rn.ftz.f32 $0, b, 0fBF317218, $2; }",
            "=f,r,f",
            [words[row, col], x[row, col]],
            dtype=torch.float32,
            is_pure=True,
            pack=1,
        )
    return out


def use_cudagraph() -> bool:
    return True


SHAPES = [
    (16, 128512, True),
    (32, 128512, False),
    (64, 128512, True),
    (128, 128512, False),
    (256, 128512, True),
    (512, 128512, False),
    (99, 32000, True),
]


def categorical_sampling(
    x: torch.Tensor, seed: torch.Tensor, logits: bool = True
) -> torch.Tensor:
    """Sample Int32[B] from a measured F32 signature and device Int64[1] seed."""
    if tuple(x.shape) == (99, 32000) and logits:
        return categorical_gumbel(
            x, seed, torch.zeros((99,), device=x.device), False, logits, torch.int32
        )
    kernel = (
        categorical_v2 if tuple(x.shape) == (16, 128512) and logits else categorical
    )
    return kernel(x.reshape(-1), seed, x.shape[1], logits, torch.int32)


def _gumbel_torch(x: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
    rows, vocab = x.shape
    packet = math.gcd(4, vocab)
    offsets = torch.arange(rows * vocab, device=x.device, dtype=torch.int64).reshape(
        rows, vocab
    )
    words = philox_words(seed[0], offsets[:, ::packet])
    noise = torch.stack(
        [gumbel_from_word(words[i]) for i in range(packet)], -1
    ).reshape(rows, vocab)
    score = x + noise
    return (vocab - 1 - score.flip(-1).argmax(-1)).to(torch.int32)


def _categorical_sampling_torch(
    x: torch.Tensor, seed: torch.Tensor, logits: bool = True
) -> torch.Tensor:
    """Torch Philox/CDF or Gumbel composition, not torch.multinomial."""
    if tuple(x.shape) == (99, 32000) and logits:
        return _gumbel_torch(x, seed)
    weights = (
        (x.double() - x.double().amax(-1, keepdim=True)).exp() if logits else x.double()
    )
    uniform = _philox_uniform(
        seed[0], torch.arange(x.shape[0], device=x.device, dtype=torch.int64)
    )
    return _cdf_sample(weights, uniform)


def _exact_gumbel_indices(x: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
    from pretuned_kernels.categorical_sampling._gumbel_words import token_words

    words = torch.from_numpy(token_words(int(seed.item()), x.shape[0], x.shape[1])).to(
        device=x.device
    )
    bound = _score_reference.bind((x, words))
    # Qualification-only pointwise instruction oracle: explicit config, no search.
    compiled = bound.compile_config(helion.Config(block_sizes=[1, 256]))
    scores = compiled(x, words)
    return (x.shape[1] - 1 - scores.flip(-1).argmax(-1)).to(torch.int32)


def check_output(
    x: torch.Tensor,
    seed: torch.Tensor,
    logits: bool,
    output: torch.Tensor,
    *,
    torch_math: bool = False,
) -> None:
    from pretuned_kernels._sampling import assert_cdf_accuracy
    from pretuned_kernels._sampling import torch_reference

    assert output.shape == (x.shape[0],) and output.dtype == torch.int32
    assert output.device == x.device and bool(
        ((output >= 0) & (output < x.shape[1])).all()
    )
    if tuple(x.shape) == (99, 32000) and logits:
        # Torch logarithms are a mathematical baseline, not the NVIDIA MUFU oracle.
        expected = (
            _gumbel_torch(x, seed) if torch_math else _exact_gumbel_indices(x, seed)
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    else:
        assert_cdf_accuracy(
            output,
            torch_reference(x.reshape(-1), seed, x.shape[1], logits, torch.int32),
        )


def make_inputs(
    shape: tuple[int, int, bool], device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, vocab, logits = shape
    generator = torch.Generator(device=device).manual_seed(0)
    if batch in (256, 512):
        beta = 0.1 if batch == 256 else 1.0
        x = (
            torch.log(
                -torch.log(
                    torch.rand((batch, vocab), device=device, generator=generator)
                    + 1e-20
                )
                + 1e-20
            )
            / beta
        )
    elif batch == 99:
        x = torch.rand((batch, vocab), device=device, generator=generator)
        x = x / x.sum(-1, keepdim=True)
    else:
        x = torch.randn((batch, vocab), device=device, generator=generator) * (
            1 if logits else 5
        )
    if not logits:
        x = x.softmax(-1)
    return x, torch.tensor([1729], device=device, dtype=torch.int64)


def _make_calls(shape: tuple[int, int, bool]) -> tuple:
    x, seed = make_inputs(shape)
    logits = shape[2]
    candidate = _capture_checked(
        lambda: categorical_sampling(x, seed, logits),
        (x, seed),
        seed,
        lambda output: check_output(x, seed, logits, output),
    )
    reference = _capture_checked(
        lambda: _categorical_sampling_torch(x, seed, logits),
        (x, seed),
        seed,
        lambda output: check_output(x, seed, logits, output, torch_math=True),
    )
    # Keep the reference column stable across the CDF and Gumbel cases.
    return candidate, [("torch_philox", reference)], str(shape)


def check_case(shape: tuple[int, int, bool]) -> None:
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
        shape_header="(batch, vocab, logits)",
    )


if __name__ == "__main__":
    main()
