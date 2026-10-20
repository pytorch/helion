from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch.utils._pytree import tree_flatten
from torch.utils._pytree import tree_map_only

if TYPE_CHECKING:
    from ..runtime.settings import Settings

_FP8_DTYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
    torch.float8_e8m0fnu,
}


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in _FP8_DTYPES


FP8_ACCURACY_MODES = ("ulp", "bitwise")

# The byte every free caching-allocator block is filled with before a run the
# accuracy check judges.  Repeated, it is a finite value of every dtype
# (fp16 -439.5, bf16 and fp32 about -8.0e18, int8 -34), so an output region a
# kernel never writes compares equal between the baseline and a candidate,
# while a value a kernel reads before writing it is far from any answer.
ACCURACY_POISON_BYTE = 0xDE
# The caching allocator serves requests of at most this many bytes from its
# small pool; a free block of the large pool below it is unreachable.
_ALLOCATOR_SMALL_POOL_MAX_BYTES = 1 << 20


def accuracy_poison_value(dtype: torch.dtype) -> torch.Tensor:
    """The scalar ``dtype`` reads from poisoned memory."""
    return torch.full([dtype.itemsize], ACCURACY_POISON_BYTE, dtype=torch.uint8).view(
        dtype
    )[0]


def poison_cached_device_memory(device: torch.device | None = None) -> None:
    """Fill the caching allocator's free blocks on ``device`` with poison.

    A kernel's fresh ``torch.empty`` outputs come from these blocks.  The
    autotuner runs every candidate on the same arguments, so the block a
    candidate recycles usually holds the previous run's output -- exactly the
    answer it should compute -- and a config that reads an output tile before
    writing it, or never writes part of it, would pass the accuracy check on
    inherited values and fail the first time the kernel runs into other
    memory.  Called right before every accuracy-judged run (the baseline and
    each candidate), this hands such a config poison instead.

    Each free block is claimed by an allocation of its exact size (the
    allocator's best fit), filled on the current stream and released; blocks
    of a fully free small-pool segment are claimed in small-pool-sized
    pieces.  CUDA / ROCm allocators only; a no-op elsewhere.
    """
    if not torch.cuda.is_available():
        return
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type != "cuda":
        return
    index = device.index if device.index is not None else torch.cuda.current_device()
    requests: list[int] = []
    for segment in torch.cuda.memory_snapshot():
        if segment["device"] != index:
            continue
        small = segment.get("segment_type") == "small"
        for block in segment["blocks"]:
            if block["state"] != "inactive":
                continue
            size = block["size"]
            if small:
                full, rest = divmod(size, _ALLOCATOR_SMALL_POOL_MAX_BYTES)
                requests.extend([_ALLOCATOR_SMALL_POOL_MAX_BYTES] * full)
                if rest:
                    requests.append(rest)
            elif size > _ALLOCATOR_SMALL_POOL_MAX_BYTES:
                requests.append(size)
    held: list[torch.Tensor] = []
    for size in sorted(requests, reverse=True):
        try:
            held.append(torch.empty(size, dtype=torch.uint8, device=device))
        except torch.OutOfMemoryError:
            break
    for block in held:
        block.fill_(ACCURACY_POISON_BYTE)
    del held


def effective_fp8_accuracy(settings: Settings) -> str:
    """The fp8 comparison mode the autotuner applies under ``settings``.

    ``autotune_fp8_accuracy`` selects it, except that explicit zero
    tolerances (``autotune_baseline_atol=0`` with ``autotune_baseline_rtol=0``)
    ask for exact outputs and select ``"bitwise"``: a kernel that pins 0/0
    keeps getting identical bytes under the default ``"ulp"`` mode instead of
    a silent one-ulp allowance.
    """
    if settings.autotune_baseline_atol == 0 and settings.autotune_baseline_rtol == 0:
        return "bitwise"
    return settings.autotune_fp8_accuracy


def relative_l2_disagreement(
    actual: object, expected: object, max_relative_l2: float
) -> str | None:
    """Why two runs of one kernel under different lowerings disagree (None:
    they agree): a tensor leaf whose non-finite elements differ, whose
    finite elements differ by more than ``max_relative_l2`` of the larger
    norm in relative L2, a non-float leaf that differs, or a structure that
    differs.  Relative L2 is indifferent to the accumulation order and the
    bf16 roundings two correct lowerings differ in (an MMA against SIMT
    code: 1e-3 or so) and names a lowering that lost or misread a term
    (near 1).  Integer and boolean leaves must be equal: an index output
    whose ties two correct lowerings break differently (an argmax over equal
    values) counts as a disagreement, as it does in the accuracy check of
    every candidate.  Leaves that are not tensors are not compared."""
    actual_leaves, actual_spec = tree_flatten(actual)
    expected_leaves, expected_spec = tree_flatten(expected)
    if actual_spec != expected_spec or len(actual_leaves) != len(expected_leaves):
        return "the outputs differ in structure"
    for index, (a, b) in enumerate(zip(actual_leaves, expected_leaves, strict=True)):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
                return f"leaf {index} is a tensor on one side only"
            continue
        if a.shape != b.shape or a.dtype != b.dtype:
            return (
                f"leaf {index}: {tuple(a.shape)} {a.dtype} against "
                f"{tuple(b.shape)} {b.dtype}"
            )
        if a.numel() == 0:
            continue
        if a.dtype == torch.float4_e2m1fn_x2:
            # Packed FP4 has no numeric cast (see ``_assert_close_large``):
            # the two runs must agree byte for byte.
            if not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
                return f"leaf {index} ({a.dtype}) differs"
            continue
        if not (a.is_floating_point() or a.is_complex()):
            if not torch.equal(a, b):
                return f"leaf {index} ({a.dtype}) differs"
            continue
        af = a.detach().to(torch.float32)
        bf = b.detach().to(torch.float32)
        finite = torch.isfinite(bf)
        if not torch.equal(torch.isfinite(af), finite) or not torch.equal(
            torch.isnan(af), torch.isnan(bf)
        ):
            return f"leaf {index}: the non-finite elements differ"
        af = torch.where(finite, af, torch.zeros((), dtype=af.dtype, device=af.device))
        bf = torch.where(finite, bf, torch.zeros((), dtype=bf.dtype, device=bf.device))
        scale = max(
            torch.linalg.vector_norm(af).item(), torch.linalg.vector_norm(bf).item()
        )
        if scale == 0.0:
            continue
        relative_l2 = torch.linalg.vector_norm(af - bf).item() / scale
        if not relative_l2 <= max_relative_l2:
            return (
                f"leaf {index}: relative L2 difference {relative_l2:.3e} exceeds "
                f"{max_relative_l2:.0e}"
            )
    return None


def assert_close(
    actual: object,
    expected: object,
    atol: float,
    rtol: float,
    *,
    scale_atol_by_expected_rms: bool = False,
    fp8_accuracy: str = "bitwise",
) -> None:
    """Like torch.testing.assert_close, with packed floats and large tensor handling.

    With ``scale_atol_by_expected_rms`` (used when the caller did not specify
    an explicit atol), the absolute-tolerance floor for each tensor leaf is
    ``atol * max(1, rms(expected))``: a near-zero element of a large reduction
    legitimately differs across accumulation orders by an amount proportional
    to the global output scale, not to the element's own magnitude, so a fixed
    elementwise atol falsely rejects every config that restructures the
    reduction (e.g. all split_k > 1 matmul configs). The scaling never
    tightens the check.

    ``fp8_accuracy`` selects how fp8 leaves are compared: ``"bitwise"`` views
    them as bytes and applies ``atol``/``rtol`` to the byte values (exact when
    both are 0); ``"ulp"`` compares dequantized values within one fp8 ulp plus
    the tolerances (see ``_assert_close_fp8_ulp``).
    """
    if fp8_accuracy not in FP8_ACCURACY_MODES:
        raise ValueError(
            f"fp8_accuracy must be one of {FP8_ACCURACY_MODES}, got {fp8_accuracy!r}"
        )

    def convert(t: torch.Tensor) -> torch.Tensor:
        if t.dtype in _FP8_DTYPES and fp8_accuracy == "bitwise":
            return t.view(torch.uint8)
        return t

    actual_flat, actual_spec = tree_flatten(
        tree_map_only(torch.Tensor, convert, actual)
    )
    expected_flat, expected_spec = tree_flatten(
        tree_map_only(torch.Tensor, convert, expected)
    )

    if actual_spec != expected_spec:
        raise AssertionError(
            f"Output tree structure mismatch during autotuner accuracy check:\n"
            f"  actual:   {actual_spec} ({len(actual_flat)} leaves)\n"
            f"  expected: {expected_spec} ({len(expected_flat)} leaves)"
        )

    for actual_leaf, expected_leaf in zip(actual_flat, expected_flat, strict=True):
        if isinstance(actual_leaf, torch.Tensor):
            if not isinstance(expected_leaf, torch.Tensor):
                raise AssertionError(
                    "Output leaf type mismatch during autotuner accuracy check: "
                    f"actual is Tensor, expected is {type(expected_leaf).__name__}"
                )
            if actual_leaf.dtype in _FP8_DTYPES or expected_leaf.dtype in _FP8_DTYPES:
                # Only the "ulp" mode keeps fp8 leaves typed; "bitwise" viewed
                # them as uint8 above.
                _assert_close_fp8_ulp(
                    actual_leaf,
                    expected_leaf,
                    atol=atol,
                    rtol=rtol,
                    scale_atol_by_expected_rms=scale_atol_by_expected_rms,
                )
                continue
            _chunked_assert_close(
                actual_leaf,
                expected_leaf,
                atol=atol,
                rtol=rtol,
                scale_atol_by_expected_rms=scale_atol_by_expected_rms,
            )
        elif isinstance(actual_leaf, str):
            if not isinstance(expected_leaf, str):
                raise AssertionError(f"Type mismatch {actual_leaf} vs {expected_leaf}")
            if actual_leaf != expected_leaf:
                raise AssertionError(
                    f"string mismatch {actual_leaf} vs {expected_leaf}"
                )
        else:
            torch.testing.assert_close(actual_leaf, expected_leaf, atol=atol, rtol=rtol)


def _assert_close(actual: object, expected: object, atol: float, rtol: float) -> None:
    assert_close(actual, expected, atol=atol, rtol=rtol)


_RMS_CACHE_ATTR = "_helion_accuracy_rms"
_TPU_ACCURACY_CHUNK_ELEMENTS = 2**26


def _accuracy_chunk_size(t: torch.Tensor, requested: int) -> int:
    if t.device.type == "tpu":
        return max(requested, _TPU_ACCURACY_CHUNK_ELEMENTS)
    return requested


def _finite_sum_of_squares(chunk: torch.Tensor) -> tuple[float, int]:
    """Sum of squares over the finite elements of ``chunk`` and their count."""
    finite = torch.isfinite(chunk)
    squares = torch.where(finite, chunk, torch.zeros_like(chunk)).square()
    return float(squares.sum()), int(finite.sum())


def _chunked_rms(t: torch.Tensor, chunk_size: int) -> float:
    """RMS of the finite elements of ``t`` (0.0 when there are none).

    Masked positions of a baseline output (``-inf`` scores, ``inf`` log-space
    sentinels) are excluded: they carry no information about the output's
    scale, and a single one would otherwise make the RMS non-finite and
    silently drop the caller back to the unscaled atol floor for the whole
    tensor. The comparison itself still requires every non-finite position to
    match exactly (``torch.testing.assert_close`` never tolerates ``inf``
    against a finite element).
    """
    # The autotuner compares hundreds of candidates against one fixed
    # baseline per run; memoize the RMS on the tensor (keyed by its mutation
    # version) instead of re-reading the whole output for every candidate.
    cached = getattr(t, _RMS_CACHE_ATTR, None)
    if cached is not None and cached[0] == t._version:
        return cached[1]
    flat = t.reshape(-1)
    if flat.numel() == 0:
        return 0.0
    if t.device.type == "tpu":
        # XLA fuses the conversion, mask, square, and reduction. Keeping the
        # whole reduction on device avoids one host synchronization per chunk.
        total, count = _finite_sum_of_squares(flat.to(torch.float32))
        rms = math.sqrt(total / count) if count else 0.0
        setattr(t, _RMS_CACHE_ATTR, (t._version, rms))
        return rms
    chunk_size = _accuracy_chunk_size(t, chunk_size)
    # Accumulate in float64 so large-magnitude outputs cannot overflow the
    # sum of squares. MPS has no float64; use float32 there — an overflow to
    # inf makes the RMS non-finite, and the caller then keeps the unscaled
    # atol floor.
    acc_dtype = torch.float32 if t.device.type in ("mps", "tpu") else torch.float64
    total = 0.0
    count = 0
    for start in range(0, flat.numel(), chunk_size):
        chunk_total, chunk_count = _finite_sum_of_squares(
            flat[start : start + chunk_size].to(acc_dtype)
        )
        total += chunk_total
        count += chunk_count
    rms = math.sqrt(total / count) if count else 0.0
    setattr(t, _RMS_CACHE_ATTR, (t._version, rms))
    return rms


def _fp8_ulp(magnitude: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The spacing of ``dtype`` at each non-negative float32 ``magnitude``.

    ``magnitude = m * 2**e`` with ``m`` in [0.5, 1) puts it in the binade whose
    spacing is ``eps * 2**(e - 1)``; below the smallest normal the spacing is
    the subnormal step ``eps * smallest_normal``.  ``frexp`` is exact, unlike a
    ``log2`` of an exact power of two.
    """
    finfo = torch.finfo(dtype)
    _, exponent = torch.frexp(magnitude.clamp_min(finfo.smallest_normal))
    return torch.ldexp(torch.full_like(magnitude, finfo.eps), exponent - 1)


def _unravel_index(index: int, shape: torch.Size) -> tuple[int, ...]:
    result = []
    for size in reversed(shape):
        index, coordinate = divmod(index, max(int(size), 1))
        result.append(coordinate)
    return tuple(reversed(result))


def _assert_close_fp8_ulp(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    chunk_size: int = 2**22,
    scale_atol_by_expected_rms: bool = False,
) -> None:
    """Compare two fp8 tensors on their dequantized values.

    A bitwise fp8 comparison pins every knob that changes the rounding of an
    intermediate the kernel quantizes to fp8: an online softmax that stores
    ``P`` as fp8 moves 5-25% of its outputs by exactly one bucket when the
    reduction blocking (hence the running max) changes, while its accuracy
    against an fp32 reference is unchanged.  The byte view also cannot carry a
    tolerance (a sign flip of a near-zero value reads as a distance >= 128).

    Every element must satisfy ``|a - b| <= ulp(max(|a|, |b|)) + atol + rtol * |b|``
    with ``atol`` scaled by ``max(1, rms(b))`` when requested: one ulp admits
    the single-bucket flips, the tolerances cover near-zero elements of a
    large reduction whose few-ulp distance is small against the output scale.
    NaNs must match NaNs.  ``float8_e8m0fnu`` has no mantissa, so its one ulp
    is one exponent step (a byte distance of one).

    The differing elements must additionally not be systematically
    one-directional: reordering noise flips values up and down in equal
    measure, whereas a rounding-mode or scale difference moves them one way,
    so a sign imbalance beyond half the flips (plus a four-sigma binomial
    margin) is rejected even though each element passes the one-ulp bound.
    """
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Tensor shape mismatch during autotuner accuracy check: "
            f"{tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    if actual.dtype != expected.dtype:
        raise AssertionError(
            "Tensor dtype mismatch during autotuner accuracy check: "
            f"{actual.dtype} != {expected.dtype}"
        )
    if actual.device != expected.device:
        actual = actual.to(expected.device)
    dtype = expected.dtype
    if scale_atol_by_expected_rms:
        rms = _chunked_rms(expected, chunk_size)
        if math.isfinite(rms):
            atol = atol * max(1.0, rms)
    chunk_size = _accuracy_chunk_size(expected, chunk_size)
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    total = expected_flat.numel()
    mismatches = 0
    up = 0
    down = 0
    worst_diff = -1.0
    worst_index = 0
    for start in range(0, total, chunk_size):
        a8 = actual_flat[start : start + chunk_size]
        b8 = expected_flat[start : start + chunk_size]
        a = a8.to(torch.float32)
        b = b8.to(torch.float32)
        both_nan = torch.isnan(a) & torch.isnan(b)
        equal = a == b
        diff = (a - b).abs()
        if dtype is torch.float8_e8m0fnu:
            distance = (
                a8.view(torch.uint8).to(torch.int16)
                - b8.view(torch.uint8).to(torch.int16)
            ).abs()
            within = distance <= 1
        else:
            tolerance = _fp8_ulp(torch.maximum(a.abs(), b.abs()), dtype) + atol
            if rtol:
                tolerance = tolerance + rtol * b.abs()
            within = diff <= tolerance
        bad = ~(equal | both_nan | within)
        n_bad = int(bad.sum())
        if n_bad:
            mismatches += n_bad
            # A NaN against a number has a NaN difference; rank it above every
            # finite excess so the report names it.
            excess = torch.where(bad, diff, torch.zeros_like(diff)).nan_to_num(
                nan=math.inf
            )
            local = int(excess.argmax())
            value = float(excess[local])
            if value > worst_diff:
                worst_diff = value
                worst_index = start + local
        differs = ~(equal | both_nan)
        up += int((differs & (a > b)).sum())
        down += int((differs & (a < b)).sum())
    if mismatches:
        raise AssertionError(
            "fp8 tensors differ by more than one ulp plus the tolerance "
            f"(atol={atol:.3g}, rtol={rtol:.3g}, dtype={dtype})!\n\n"
            f"Mismatched elements: {mismatches} / {total} "
            f"({100.0 * mismatches / max(total, 1):.1f}%)\n"
            f"Greatest absolute difference: {worst_diff} at index "
            f"{_unravel_index(worst_index, expected.shape)}"
        )
    flips = up + down
    if abs(up - down) > 0.5 * flips + 4.0 * math.sqrt(flips) + 1.0:
        raise AssertionError(
            "fp8 one-ulp differences are systematically one-directional "
            f"(dtype={dtype})!\n\n"
            f"Elements above the baseline: {up}, below: {down}, of {total}; "
            "reordering noise flips values both ways, a biased imbalance "
            "indicates a rounding or scale difference."
        )


def _chunked_assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    chunk_size: int = 2**22,
    scale_atol_by_expected_rms: bool = False,
) -> None:
    """Memory-efficient assert_close for large tensors."""
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Tensor shape mismatch during autotuner accuracy check: "
            f"{tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    if torch.float4_e2m1fn_x2 in (actual.dtype, expected.dtype):
        if actual.dtype != expected.dtype:
            raise AssertionError(
                "Tensor dtype mismatch during autotuner accuracy check: "
                f"{actual.dtype} != {expected.dtype}"
            )
        # Packed FP4 has no numeric cast to the RMS accumulator dtype. Check
        # the storage exactly; byte-distance tolerances do not measure FP4
        # numeric error, and input mutation checks must preserve both nibbles.
        actual = actual.view(torch.uint8)
        expected = expected.view(torch.uint8)
        atol = rtol = 0.0
    if scale_atol_by_expected_rms and expected.dtype.is_floating_point:
        rms = _chunked_rms(expected, chunk_size)
        # The RMS skips non-finite baseline elements; it can still overflow
        # on MPS (float32 accumulation), and that must not disable the gate:
        # keep the unscaled floor in that case.
        if math.isfinite(rms):
            atol = atol * max(1.0, rms)
    if (
        actual.device.type == "tpu"
        and actual.dtype == expected.dtype
        and actual.device == expected.device
        and (
            torch.allclose(actual, expected, atol=atol, rtol=rtol)
            if actual.dtype.is_floating_point or actual.dtype.is_complex
            else torch.equal(actual, expected)
        )
    ):
        return
    if actual.numel() <= chunk_size:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return
    if actual.dtype != expected.dtype or actual.device != expected.device:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return
    if actual.device.type == "tpu":
        # ``torch.testing.assert_close`` copies each TPU chunk to the host to
        # construct its diagnostic. Compare on device first and transfer only
        # the scalar result. A larger chunk amortizes TPU dispatch overhead while
        # keeping temporary tensors bounded.
        chunk_size = _accuracy_chunk_size(actual, chunk_size)
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    for start in range(0, actual_flat.numel(), chunk_size):
        actual_chunk = actual_flat[start : start + chunk_size]
        expected_chunk = expected_flat[start : start + chunk_size]
        if actual.device.type == "tpu":
            if actual.dtype.is_floating_point or actual.dtype.is_complex:
                if torch.allclose(actual_chunk, expected_chunk, atol=atol, rtol=rtol):
                    continue
            elif torch.equal(actual_chunk, expected_chunk):
                continue
        torch.testing.assert_close(actual_chunk, expected_chunk, atol=atol, rtol=rtol)
