from __future__ import annotations

import dataclasses
import math
import types
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Literal

import numpy as np
import torch
from torch.utils._pytree import tree_flatten

from .kernel_args import PortableFunction
from .kernel_args import _input_tree_leaves

if TYPE_CHECKING:
    from typing import Callable

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
AtolScale = Literal["rms", "peak"]

# The byte every free caching-allocator block is filled with before a run the
# accuracy check judges.  Repeated, it is a finite value of every dtype
# (fp16 -439.5, bf16 and fp32 about -8.0e18, int8 -34), so an output region a
# kernel never writes compares equal between the baseline and a candidate,
# while a value a kernel reads before writing it is far from any answer.
ACCURACY_POISON_BYTE = 0xDE
# The caching allocator serves requests of at most this many bytes from its
# small pool; a free block of the large pool below it is unreachable.
_ALLOCATOR_SMALL_POOL_MAX_BYTES = 1 << 20
# The allocator's smallest block.
_ALLOCATOR_MIN_BLOCK_BYTES = 512


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
    pieces.  Only the blocks of the stream and pool a fresh allocation is
    routed to (the pool of a probe allocation's segment: the default pool, or
    a ``torch.cuda.use_mem_pool`` pool the caller allocates into) are
    claimed: a fresh output can come from no others, and a request for any
    other free block (one a side stream owns, like each of
    ``steady_graph_dispatch``'s warm-up segments, or one in a CUDA graph's
    private pool) cannot be served from it, so it would allocate a new
    segment instead -- on every call, growing the reservation until the
    device is full.  CUDA / ROCm allocators only; a no-op elsewhere.
    """
    if not torch.cuda.is_available():
        return
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type != "cuda":
        return
    index = device.index if device.index is not None else torch.cuda.current_device()
    # a block-sized probe, poisoned so that its own block is too once freed
    probe = torch.full(
        [_ALLOCATOR_MIN_BLOCK_BYTES],
        ACCURACY_POISON_BYTE,
        dtype=torch.uint8,
        device=device,
    )
    address = probe.data_ptr()
    segments = [s for s in torch.cuda.memory_snapshot() if s["device"] == index]
    del probe
    probe_segment = next(
        (
            (s["stream"], s["segment_pool_id"])
            for s in segments
            if s["address"] <= address < s["address"] + s["total_size"]
        ),
        None,
    )
    if probe_segment is None:
        # not from the caching allocator (PYTORCH_NO_CUDA_MEMORY_CACHING=1):
        # nothing is cached to be recycled
        return
    stream, pool = probe_segment
    requests: list[int] = []
    for segment in segments:
        if segment["stream"] != stream or segment["segment_pool_id"] != pool:
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
    values) counts as a disagreement.  This is stricter than the accuracy
    check of every candidate, which applies its tolerances to integer leaves
    too (``|a - b| <= atol + rtol * |b|``: the default ``rtol=1e-2`` admits
    an index of 101 against 100).  Leaves that are not tensors are not
    compared."""
    return _relative_l2_disagreement(actual, expected, max_relative_l2, {})


def _relative_l2_disagreement(
    actual: object,
    expected: object,
    max_relative_l2: float,
    compared_records: dict[tuple[int, int], tuple[object, object]],
) -> str | None:
    actual_leaves, actual_spec = tree_flatten(actual)
    expected_leaves, expected_spec = tree_flatten(expected)
    if actual_spec != expected_spec or len(actual_leaves) != len(expected_leaves):
        return "the outputs differ in structure"
    for index, (a, b) in enumerate(zip(actual_leaves, expected_leaves, strict=True)):
        a_operands = _callback_tensor_values(a)
        b_operands = _callback_tensor_values(b)
        if a_operands is not None or b_operands is not None:
            if a_operands is None or b_operands is None:
                return f"leaf {index} has a different callback type"
            reason = _relative_l2_disagreement(
                a_operands, b_operands, max_relative_l2, compared_records
            )
            if reason is not None:
                return f"leaf {index} callback operands: {reason}"
            continue
        a_fields = _record_field_values(a)
        b_fields = _record_field_values(b)
        if a_fields is not None or b_fields is not None:
            if type(a) is not type(b) or a_fields is None or b_fields is None:
                return f"leaf {index} has a different dataclass type"
            pair = (id(a), id(b))
            if pair in compared_records:
                continue
            # Pytree flatteners may create temporary records. Keep them alive
            # so a later sibling cannot reuse a visited pair of object IDs.
            compared_records[pair] = (a, b)
            reason = _relative_l2_disagreement(
                a_fields, b_fields, max_relative_l2, compared_records
            )
            if reason is not None:
                return f"leaf {index}: {reason}"
            continue
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
        # float64 leaves keep their range: in float32 two different large
        # values would both overflow to inf and agree.
        compute_dtype = torch.float64 if a.dtype == torch.float64 else torch.float32
        af = a.detach().to(compute_dtype)
        bf = b.detach().to(compute_dtype)
        finite = torch.isfinite(bf)
        if (
            not torch.equal(torch.isfinite(af), finite)
            or not torch.equal(torch.isnan(af), torch.isnan(bf))
            or not torch.equal(torch.isposinf(af), torch.isposinf(bf))
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
    atol_scale: AtolScale | None = None,
    fp8_accuracy: str = "bitwise",
) -> None:
    """Like torch.testing.assert_close, with packed floats and large tensor handling.

    With ``atol_scale="rms"`` (``scale_atol_by_expected_rms`` for short; used
    when the caller did not specify an explicit atol), the absolute-tolerance
    floor for each tensor leaf is
    ``atol * max(1, rms(expected))``: a near-zero element of a large reduction
    legitimately differs across accumulation orders by an amount proportional
    to the global output scale, not to the element's own magnitude, so a fixed
    elementwise atol falsely rejects every config that restructures the
    reduction (e.g. all split_k > 1 matmul configs). The scaling never
    tightens the check.

    With ``atol_scale="peak"`` the floor is instead
    ``atol * max(|expected|)`` over the finite elements of each floating-point
    leaf (see :class:`AccuracyCheck`); unlike the rms scaling it can tighten
    the check, down to exact equality for an all-zero leaf.

    ``fp8_accuracy`` selects how fp8 leaves are compared: ``"bitwise"`` views
    them as bytes and applies ``atol``/``rtol`` to the byte values (exact when
    both are 0); ``"ulp"`` compares dequantized values within one fp8 ulp plus
    the tolerances (see ``_assert_close_fp8_ulp``).
    """
    if scale_atol_by_expected_rms:
        if atol_scale not in (None, "rms"):
            raise ValueError(
                f"scale_atol_by_expected_rms contradicts atol_scale={atol_scale!r}"
            )
        atol_scale = "rms"
    _assert_close_records(
        actual,
        expected,
        atol,
        rtol,
        atol_scale=atol_scale,
        fp8_accuracy=fp8_accuracy,
        compared_records={},
    )


def _assert_close_records(
    actual: object,
    expected: object,
    atol: float,
    rtol: float,
    *,
    atol_scale: AtolScale | None,
    fp8_accuracy: str,
    compared_records: dict[tuple[int, int], tuple[object, object]],
) -> None:
    if fp8_accuracy not in FP8_ACCURACY_MODES:
        raise ValueError(
            f"fp8_accuracy must be one of {FP8_ACCURACY_MODES}, got {fp8_accuracy!r}"
        )
    actual_flat, actual_spec = tree_flatten(actual)
    expected_flat, expected_spec = tree_flatten(expected)

    if actual_spec != expected_spec:
        raise AssertionError(
            f"Output tree structure mismatch during autotuner accuracy check:\n"
            f"  actual:   {actual_spec} ({len(actual_flat)} leaves)\n"
            f"  expected: {expected_spec} ({len(expected_flat)} leaves)"
        )

    for actual_leaf, expected_leaf in zip(actual_flat, expected_flat, strict=True):
        actual_operands = _callback_tensor_values(actual_leaf)
        expected_operands = _callback_tensor_values(expected_leaf)
        if actual_operands is not None or expected_operands is not None:
            if actual_operands is None or expected_operands is None:
                raise AssertionError(
                    "Output leaf type mismatch during autotuner accuracy check: "
                    f"actual is {type(actual_leaf).__name__}, expected is "
                    f"{type(expected_leaf).__name__}"
                )
            _assert_close_records(
                actual_operands,
                expected_operands,
                atol,
                rtol,
                atol_scale=atol_scale,
                fp8_accuracy=fp8_accuracy,
                compared_records=compared_records,
            )
            continue
        actual_fields = _record_field_values(actual_leaf)
        expected_fields = _record_field_values(expected_leaf)
        if actual_fields is not None or expected_fields is not None:
            if (
                type(actual_leaf) is not type(expected_leaf)
                or actual_fields is None
                or expected_fields is None
            ):
                raise AssertionError(
                    "Dataclass type mismatch during autotuner accuracy check"
                )
            pair = (id(actual_leaf), id(expected_leaf))
            if pair in compared_records:
                continue
            # Pytree flatteners may create temporary records. Keep them alive
            # so a later sibling cannot reuse a visited pair of object IDs.
            compared_records[pair] = (actual_leaf, expected_leaf)
            _assert_close_records(
                actual_fields,
                expected_fields,
                atol=atol,
                rtol=rtol,
                atol_scale=atol_scale,
                fp8_accuracy=fp8_accuracy,
                compared_records=compared_records,
            )
            continue
        actual_kind = _leaf_kind(actual_leaf)
        if actual_kind != _leaf_kind(expected_leaf):
            # Two leaves of different kinds would reach a comparison that has
            # no pair for them (a TypeError, which the autotuner does not
            # catch): a mismatch.
            raise AssertionError(
                "Output leaf type mismatch during autotuner accuracy check: "
                f"actual is {type(actual_leaf).__name__}, expected is "
                f"{type(expected_leaf).__name__}"
            )
        if actual_kind == "tensor":
            assert isinstance(actual_leaf, torch.Tensor)
            assert isinstance(expected_leaf, torch.Tensor)
            if actual_leaf.dtype in _FP8_DTYPES or expected_leaf.dtype in _FP8_DTYPES:
                if actual_leaf.dtype != expected_leaf.dtype:
                    # Two fp8 formats (or fp8 against bytes) can share a bit
                    # pattern for different values.
                    raise AssertionError(
                        "Tensor dtype mismatch during autotuner accuracy check: "
                        f"{actual_leaf.dtype} != {expected_leaf.dtype}"
                    )
                if fp8_accuracy == "bitwise":
                    if atol or rtol:
                        # Byte tolerances would let a value one byte from inf
                        # or NaN pass against it.
                        _assert_non_finite_bytes_match(actual_leaf, expected_leaf)
                    _chunked_assert_close(
                        actual_leaf.view(torch.uint8),
                        expected_leaf.view(torch.uint8),
                        atol=atol,
                        rtol=rtol,
                    )
                    continue
                _assert_close_fp8_ulp(
                    actual_leaf,
                    expected_leaf,
                    atol=atol,
                    rtol=rtol,
                    atol_scale=atol_scale,
                )
                continue
            _chunked_assert_close(
                actual_leaf,
                expected_leaf,
                atol=atol,
                rtol=rtol,
                atol_scale=atol_scale,
            )
        elif actual_kind == "str":
            if actual_leaf != expected_leaf:
                raise AssertionError(
                    f"string mismatch {actual_leaf} vs {expected_leaf}"
                )
        elif _is_number(actual_leaf) and _is_number(expected_leaf):
            torch.testing.assert_close(actual_leaf, expected_leaf, atol=atol, rtol=rtol)
        elif actual_leaf != expected_leaf:
            # Leaves torch.testing has no comparison for (a torch.dtype, None)
            # must be equal.
            raise AssertionError(
                "Leaf mismatch during autotuner accuracy check: "
                f"{actual_leaf!r} vs {expected_leaf!r}"
            )


def _is_number(leaf: object) -> bool:
    # The scalars torch.testing.assert_close pairs.
    return isinstance(leaf, (int, float, complex, np.number))


def _assert_non_finite_bytes_match(
    actual: torch.Tensor, expected: torch.Tensor, chunk_size: int = 2**22
) -> None:
    """Every position where either fp8 tensor is inf or NaN holds the same
    byte in both."""
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Tensor shape mismatch during autotuner accuracy check: "
            f"{tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    if actual.device != expected.device:
        actual = actual.to(expected.device)
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    chunk_size = _accuracy_chunk_size(expected, chunk_size)
    for start in range(0, expected_flat.numel(), chunk_size):
        a8 = actual_flat[start : start + chunk_size]
        b8 = expected_flat[start : start + chunk_size]
        non_finite = ~torch.isfinite(a8.to(torch.float32)) | ~torch.isfinite(
            b8.to(torch.float32)
        )
        if not torch.equal(
            a8.view(torch.uint8)[non_finite], b8.view(torch.uint8)[non_finite]
        ):
            raise AssertionError(
                "fp8 tensors differ at an inf or NaN element during autotuner "
                f"accuracy check (dtype={expected.dtype})"
            )


def _leaf_kind(leaf: object) -> str:
    if isinstance(leaf, torch.Tensor):
        return "tensor"
    if isinstance(leaf, str):
        return "str"
    if _is_function_leaf(leaf):
        return "function"
    return "other"


def _is_function_leaf(leaf: object) -> bool:
    return callable(leaf) or isinstance(leaf, PortableFunction)


def _callback_tensor_values(value: object) -> tuple[torch.Tensor, ...] | None:
    # Functions themselves are not writable kernel outputs, but their captured
    # tensors are operands. Reuse cloning's cycle-safe traversal without calling
    # the callback or comparing its code, opaque metadata, or Python identity.
    if not _is_function_leaf(value):
        return None
    if not isinstance(value, (types.FunctionType, PortableFunction)):
        # Arbitrary callable objects retain their existing opaque contract.
        return ()
    return tuple(
        leaf for leaf in _input_tree_leaves(value) if isinstance(leaf, torch.Tensor)
    )


def _record_field_values(value: object) -> dict[str, object] | None:
    if (
        dataclasses.is_dataclass(value)
        and not isinstance(value, type)
        and not _is_function_leaf(value)
    ):
        return {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
        }
    return None


def tensor_dtypes(value: object) -> set[torch.dtype]:
    """Dtypes of tensor leaves visited by the accuracy comparison.

    Registered pytrees keep their flattening contract; otherwise opaque records
    expose only the same declared fields as comparison. Retain visited records
    through this walk so cycles terminate and ephemeral pytree children cannot
    reuse a visited object's identity. No records or pytrees are rebuilt.
    """
    dtypes: set[torch.dtype] = set()
    records: dict[int, object] = {}

    def collect(tree: object) -> None:
        for leaf in tree_flatten(tree)[0]:
            if isinstance(leaf, torch.Tensor):
                dtypes.add(leaf.dtype)
            elif id(leaf) not in records:
                operands = _callback_tensor_values(leaf)
                if operands is not None:
                    dtypes.update(tensor.dtype for tensor in operands)
                    continue
                fields = _record_field_values(leaf)
                if fields is not None:
                    records[id(leaf)] = leaf
                    collect(fields)

    collect(value)
    return dtypes


def _assert_close(actual: object, expected: object, atol: float, rtol: float) -> None:
    assert_close(actual, expected, atol=atol, rtol=rtol)


@dataclasses.dataclass(frozen=True)
class AccuracyCheck:
    """An ``autotune_baseline_accuracy_check_fn`` built from plain tolerances.

    It compares every output (and mutated argument) of a candidate with the
    baseline's as the default check does (:func:`assert_close`), with the
    tolerances given here instead of the ``autotune_baseline_*`` settings:

    - ``atol_scale=None``: ``|actual - expected| <= atol + rtol * |expected|``
      elementwise.
    - ``atol_scale="rms"``: ``atol`` is scaled per tensor to
      ``atol * max(1, rms(expected))``, the default check's scaling:
      ``AccuracyCheck(atol=1e-2, rtol=1e-2, atol_scale="rms")`` is the default
      check of outputs that are not fp8.
    - ``atol_scale="peak"``: ``atol`` is scaled per tensor to
      ``atol * max(|expected|)``.  Rounding error in low-precision outputs and
      accumulation-order differences scale with a tensor's peak; with
      ``rtol=0`` this accepts correct heavy-tailed outputs, such as bf16
      attention gradients whose peak is tens of times their rms, that a fixed
      elementwise tolerance rejects.  An all-zero tensor that is not fp8
      must match exactly (an fp8 one under ``"ulp"`` still gets one ulp).

    Scaling applies to floating-point tensors and ignores non-finite baseline
    elements, which must match exactly.  Integer and bool tensors get the
    unscaled tolerances, as in the default check (exact when ``rtol=0`` and
    ``atol < 1``); packed fp4 tensors must match byte for byte.
    ``fp8_accuracy`` selects how fp8 tensors are compared (see
    :func:`assert_close`) in place of the ``autotune_fp8_accuracy`` setting
    (``HELION_AUTOTUNE_FP8_ACCURACY``), which this check ignores like the
    ``autotune_baseline_*`` tolerances, so that equal checks behave the same
    in every process; as with the settings, ``atol=0`` with ``rtol=0``
    compares them bitwise whatever it says.

    Unlike an arbitrary callback, this check is Helion's own code, so the
    autotune cache keys it by its fields (and ``VERSION``, bumped whenever its
    behavior changes): a config tuned under one check is reused across
    processes under an equal check and never under a different one.
    """

    atol: float
    rtol: float
    atol_scale: AtolScale | None = None
    fp8_accuracy: Literal["ulp", "bitwise"] = "ulp"
    # Bump whenever a behavior reachable from ``assert_close`` changes: cached
    # autotune results are keyed by it, and the loose cache key holds no hash
    # of Helion's source.
    VERSION: ClassVar[int] = 3

    def __post_init__(self) -> None:
        for name in ("atol", "rtol"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"AccuracyCheck {name} must be a non-negative finite number, "
                    f"got {value!r}"
                )
        if self.atol_scale not in (None, "rms", "peak"):
            raise ValueError(
                "AccuracyCheck atol_scale must be None, 'rms' or 'peak', "
                f"got {self.atol_scale!r}"
            )
        if self.fp8_accuracy not in FP8_ACCURACY_MODES:
            raise ValueError(
                f"AccuracyCheck fp8_accuracy must be one of {FP8_ACCURACY_MODES}, "
                f"got {self.fp8_accuracy!r}"
            )

    @property
    def effective_fp8_accuracy(self) -> str:
        """The fp8 mode the check runs: ``"bitwise"`` under zero tolerances."""
        if self.atol == 0 and self.rtol == 0:
            return "bitwise"
        return self.fp8_accuracy

    def cache_policy(self) -> dict[str, object]:
        """Plain data identifying this check for autotune cache keys."""
        return {
            "check": "accuracy_check",
            "version": self.VERSION,
            "atol": float(self.atol),
            "rtol": float(self.rtol),
            "atol_scale": self.atol_scale,
            "fp8_accuracy": self.effective_fp8_accuracy,
        }

    def __call__(self, actual: object, expected: object) -> None:
        assert_close(
            actual,
            expected,
            atol=self.atol,
            rtol=self.rtol,
            atol_scale=self.atol_scale,
            fp8_accuracy=self.effective_fp8_accuracy,
        )


def keyable_accuracy_check(settings: Settings) -> AccuracyCheck | None:
    """The ``autotune_baseline_accuracy_check_fn`` of ``settings`` when the
    autotune cache can key it: an :class:`AccuracyCheck` (not a subclass,
    which may override ``__call__``)."""
    check = settings.autotune_baseline_accuracy_check_fn
    return check if type(check) is AccuracyCheck else None


_RMS_CACHE_ATTR = "_helion_accuracy_rms"
_PEAK_CACHE_ATTR = "_helion_accuracy_peak"
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


def _memoized(t: torch.Tensor, attr: str, compute: Callable[[], float]) -> float:
    """``compute()`` memoized on ``t`` under ``attr``, keyed by its mutation
    version: the autotuner compares hundreds of candidates against one fixed
    baseline per run.  Inference tensors have no version counter, so they are
    not memoized."""
    if t.is_inference():
        return compute()
    cached = getattr(t, attr, None)
    if cached is not None and cached[0] == t._version:
        return cached[1]
    value = compute()
    setattr(t, attr, (t._version, value))
    return value


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
    return _memoized(t, _RMS_CACHE_ATTR, lambda: _compute_rms(t, chunk_size))


def _compute_rms(t: torch.Tensor, chunk_size: int) -> float:
    flat = t.reshape(-1)
    if flat.numel() == 0:
        return 0.0
    if t.device.type == "tpu":
        # XLA fuses the conversion, mask, square, and reduction. Keeping the
        # whole reduction on device avoids one host synchronization per chunk.
        total, count = _finite_sum_of_squares(flat.to(torch.float32))
        return math.sqrt(total / count) if count else 0.0
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
    return math.sqrt(total / count) if count else 0.0


def _chunked_peak(t: torch.Tensor, chunk_size: int) -> float:
    """Largest magnitude of the finite elements of ``t`` (0.0 when there are
    none), memoized on the tensor like ``_chunked_rms``."""
    return _memoized(t, _PEAK_CACHE_ATTR, lambda: _compute_peak(t, chunk_size))


def _compute_peak(t: torch.Tensor, chunk_size: int) -> float:
    flat = t.reshape(-1)
    chunk_size = _accuracy_chunk_size(t, chunk_size)
    peak = 0.0
    for start in range(0, flat.numel(), chunk_size):
        chunk = flat[start : start + chunk_size]
        if chunk.dtype != torch.float64:
            chunk = chunk.to(torch.float32)
        magnitude = torch.where(
            torch.isfinite(chunk), chunk.abs(), torch.zeros_like(chunk)
        )
        peak = max(peak, float(magnitude.max()))
    return peak


def _scaled_atol(
    expected: torch.Tensor,
    atol: float,
    chunk_size: int,
    atol_scale: AtolScale | None,
) -> float:
    """``atol`` scaled by the expected tensor's scale (see ``assert_close``)."""
    if atol_scale == "rms":
        rms = _chunked_rms(expected, chunk_size)
        # The RMS skips non-finite baseline elements; it can still overflow
        # on MPS (float32 accumulation), and that must not disable the gate:
        # keep the unscaled floor in that case.
        if math.isfinite(rms):
            return atol * max(1.0, rms)
    elif atol_scale == "peak":
        return atol * _chunked_peak(expected, chunk_size)
    return atol


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
    atol_scale: AtolScale | None = None,
) -> None:
    """Compare two fp8 tensors on their dequantized values.

    A bitwise fp8 comparison pins every knob that changes the rounding of an
    intermediate the kernel quantizes to fp8: an online softmax that stores
    ``P`` as fp8 moves 5-25% of its outputs by exactly one bucket when the
    reduction blocking (hence the running max) changes, while its accuracy
    against an fp32 reference is unchanged.  The byte view also cannot carry a
    tolerance (a sign flip of a near-zero value reads as a distance >= 128).

    Every element must satisfy ``|a - b| <= ulp(max(|a|, |b|)) + atol + rtol * |b|``
    with ``atol`` scaled by ``max(1, rms(b))`` or ``max(|b|)`` when requested: one ulp admits
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
    atol = _scaled_atol(expected, atol, chunk_size, atol_scale)
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
            # NaN (byte 0xFF) is one byte from the largest finite value.
            within = (distance <= 1) & ~torch.isnan(a) & ~torch.isnan(b)
        else:
            tolerance = _fp8_ulp(torch.maximum(a.abs(), b.abs()), dtype) + atol
            if rtol:
                tolerance = tolerance + rtol * b.abs()
            # An infinite element (whose tolerance is infinite too) must match
            # exactly: ``equal`` covers that.
            within = (diff <= tolerance) & torch.isfinite(a) & torch.isfinite(b)
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
    atol_scale: AtolScale | None = None,
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
    if expected.dtype.is_floating_point:
        atol = _scaled_atol(expected, atol, chunk_size, atol_scale)
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
