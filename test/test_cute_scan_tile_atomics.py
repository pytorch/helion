"""Scan-then-atomic tile kernels on the CuTe backend (``examples/segment_reduction.py``).

GPU-free codegen checks for three lowerings and their seed, plus GPU numerics:

* the exact zero-add elision ``atomic_add(t, i, where(c, x, 0))`` ->
  ``if c: atomic_add(t, i, x)`` (``cute/atomic_ops.py``),
* the vector atomic flush ``red.global.add.v{V}.f32`` along a vectorized tile
  axis (same module),
* the loads-first lane unroll ``cute_lane_unroll`` without vector-loop sinking
  (``cute/unroll_lane_loads.py``),
* the ``cute_scan_tile`` seed and the knob's normalization.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import patch_cute_mma_support
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

pytestmark = skipUnlessBackends(["cute"])
CPU = torch.device("cpu")

VECTOR_ATOMIC = "_cute_red_add_f32_vec("
SCALAR_ATOMIC = "cute.arch.atomic_add("
LOAD_LIST = "_lane_unroll_loads_"


def _combine(
    left_values: torch.Tensor,
    left_indices: torch.Tensor,
    right_values: torch.Tensor,
    right_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.where(
            left_indices == right_indices, left_values + right_values, right_values
        ),
        right_indices,
    )


def _segmented_reduction(
    indices: torch.Tensor, input_data: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """``examples/segment_reduction.py::segmented_reduction_helion``."""
    num_elements, num_features = input_data.shape
    output = torch.zeros(
        (num_nodes, num_features), dtype=input_data.dtype, device=input_data.device
    )
    for tile_e, tile_f in hl.tile([num_elements, num_features]):
        vals = input_data[tile_e, tile_f]
        idxs = indices[tile_e]
        idxs_next = hl.load(
            indices, [tile_e.index + 1], extra_mask=tile_e.index < num_elements - 1
        )
        tuple_in = (vals, idxs.float().unsqueeze(1).expand_as(vals))
        out_vals, _ = hl.associative_scan(_combine, tuple_in, dim=0)
        mask = (idxs != idxs_next) | (
            tile_e.index % tile_e.block_size == tile_e.block_size - 1
        )
        segment_vals = torch.where(mask.unsqueeze(1), out_vals, 0.0)
        hl.atomic_add(output, [idxs, tile_f], segment_vals)
    return output


def _scatter_where(
    indices: torch.Tensor, x: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Row-masked scatter-add into a fresh zeros output."""
    out = torch.zeros((num_nodes, x.size(1)), dtype=x.dtype, device=x.device)
    for tile_e, tile_f in hl.tile(x.shape):
        idxs = indices[tile_e]
        keep = (idxs % 2 == 0).unsqueeze(1)
        hl.atomic_add(out, [idxs, tile_f], torch.where(keep, x[tile_e, tile_f], 0.0))
    return out


def _scatter_where_flipped(
    indices: torch.Tensor, x: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """The zero on the ``where``'s true branch: the predicate is negated."""
    out = torch.zeros((num_nodes, x.size(1)), dtype=x.dtype, device=x.device)
    for tile_e, tile_f in hl.tile(x.shape):
        idxs = indices[tile_e]
        drop = (idxs % 2 == 1).unsqueeze(1)
        hl.atomic_add(out, [idxs, tile_f], torch.where(drop, 0.0, x[tile_e, tile_f]))
    return out


def _scatter_where_into_argument(
    indices: torch.Tensor, x: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """The destination is a kernel input: it may hold -0.0, no elision."""
    for tile_e, tile_f in hl.tile(x.shape):
        idxs = indices[tile_e]
        keep = (idxs % 2 == 0).unsqueeze(1)
        hl.atomic_add(out, [idxs, tile_f], torch.where(keep, x[tile_e, tile_f], 0.0))
    return out


def _scatter_where_negative_zero(
    indices: torch.Tensor, x: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """A ``-0.0`` literal: not the ``+0`` the proof asks for, no elision."""
    out = torch.zeros((num_nodes, x.size(1)), dtype=x.dtype, device=x.device)
    for tile_e, tile_f in hl.tile(x.shape):
        idxs = indices[tile_e]
        keep = (idxs % 2 == 0).unsqueeze(1)
        hl.atomic_add(out, [idxs, tile_f], torch.where(keep, x[tile_e, tile_f], -0.0))
    return out


def _scatter_where_computed_zero(
    indices: torch.Tensor, x: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """The dropped operand is computed, not a literal: no elision."""
    out = torch.zeros((num_nodes, x.size(1)), dtype=x.dtype, device=x.device)
    for tile_e, tile_f in hl.tile(x.shape):
        idxs = indices[tile_e]
        keep = (idxs % 2 == 0).unsqueeze(1)
        vals = x[tile_e, tile_f]
        hl.atomic_add(out, [idxs, tile_f], torch.where(keep, vals, vals * 0.0))
    return out


def _where_add_used_result(x: torch.Tensor) -> torch.Tensor:
    """The atomic's old value is consumed: no elision."""
    out = torch.zeros_like(x)
    seen = torch.empty_like(x)
    for tile_e, tile_f in hl.tile(x.shape):
        vals = x[tile_e, tile_f]
        seen[tile_e, tile_f] = hl.atomic_add(
            out, [tile_e, tile_f], torch.where(vals > 0, vals, 0.0)
        )
    return seen


def _where_add_unused_result(x: torch.Tensor) -> torch.Tensor:
    """``_where_add_used_result`` without the consumer: elided."""
    out = torch.zeros_like(x)
    for tile_e, tile_f in hl.tile(x.shape):
        vals = x[tile_e, tile_f]
        hl.atomic_add(out, [tile_e, tile_f], torch.where(vals > 0, vals, 0.0))
    return out


def _scatter_where_empty(
    indices: torch.Tensor, x: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """A fresh allocation that is not zero-filled: no elision."""
    out = torch.empty((num_nodes, x.size(1)), dtype=x.dtype, device=x.device)
    for tile_e, tile_f in hl.tile(x.shape):
        idxs = indices[tile_e]
        keep = (idxs % 2 == 0).unsqueeze(1)
        hl.atomic_add(out, [idxs, tile_f], torch.where(keep, x[tile_e, tile_f], 0.0))
    return out


def _scatter_where_new_zeros(
    indices: torch.Tensor, x: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """The zeros come from the ``Tensor.new_zeros`` method: elided."""
    out = x.new_zeros((num_nodes, x.size(1)))
    for tile_e, tile_f in hl.tile(x.shape):
        idxs = indices[tile_e]
        keep = (idxs % 2 == 0).unsqueeze(1)
        hl.atomic_add(out, [idxs, tile_f], torch.where(keep, x[tile_e, tile_f], 0.0))
    return out


def _scatter_where_release(
    indices: torch.Tensor, x: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """A release RMW is a fence-carrying write: never skipped."""
    out = torch.zeros((num_nodes, x.size(1)), dtype=x.dtype, device=x.device)
    for tile_e, tile_f in hl.tile(x.shape):
        idxs = indices[tile_e]
        keep = (idxs % 2 == 0).unsqueeze(1)
        hl.atomic_add(
            out,
            [idxs, tile_f],
            torch.where(keep, x[tile_e, tile_f], 0.0),
            sem="release",
        )
    return out


def _scatter_where_ones(
    indices: torch.Tensor, x: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """A fresh allocation filled with ones: no elision."""
    out = torch.ones((num_nodes, x.size(1)), dtype=x.dtype, device=x.device)
    for tile_e, tile_f in hl.tile(x.shape):
        idxs = indices[tile_e]
        keep = (idxs % 2 == 0).unsqueeze(1)
        hl.atomic_add(out, [idxs, tile_f], torch.where(keep, x[tile_e, tile_f], 0.0))
    return out


def _scatter_where_and_store(
    indices: torch.Tensor, x: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """A plain store also writes the zeros output: no elision."""
    out = torch.zeros((num_nodes, x.size(1)), dtype=x.dtype, device=x.device)
    for tile_e, tile_f in hl.tile(x.shape):
        idxs = indices[tile_e]
        keep = (idxs % 2 == 0).unsqueeze(1)
        out[tile_e, tile_f] = x[tile_e, tile_f] * 0.0
        hl.atomic_add(out, [idxs, tile_f], torch.where(keep, x[tile_e, tile_f], 0.0))
    return out


def _scatter_where_int(
    indices: torch.Tensor, x: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """Integers have no signed zero: elided even into a kernel input."""
    for tile_e, tile_f in hl.tile(x.shape):
        idxs = indices[tile_e]
        keep = (idxs % 2 == 0).unsqueeze(1)
        hl.atomic_add(out, [idxs, tile_f], torch.where(keep, x[tile_e, tile_f], 0))
    return out


def _scatter_where_per_element(
    indices: torch.Tensor, x: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """The predicate varies per column: elided per lane, never vectorized."""
    out = torch.zeros((num_nodes, x.size(1)), dtype=x.dtype, device=x.device)
    for tile_e, tile_f in hl.tile(x.shape):
        idxs = indices[tile_e]
        vals = x[tile_e, tile_f]
        hl.atomic_add(out, [idxs, tile_f], torch.where(vals > 0, vals, 0.0))
    return out


def _scatter_release(
    indices: torch.Tensor, x: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Not a relaxed atomic: keeps the scalar form."""
    out = torch.zeros((num_nodes, x.size(1)), dtype=x.dtype, device=x.device)
    for tile_e, tile_f in hl.tile(x.shape):
        idxs = indices[tile_e]
        hl.atomic_add(out, [idxs, tile_f], x[tile_e, tile_f], sem="release")
    return out


def _column_count_beside_copy(x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """A column count that is uniform along the row lane loop: the placement
    pass pins it to the first row lane, so it must stay a scalar atomic."""
    out = torch.empty_like(x)
    for tile_e, tile_f in hl.tile(x.shape):
        hl.atomic_add(counts, [tile_f], 1.0)
        out[tile_e, tile_f] = x[tile_e, tile_f]
    return out


def _load_after_atomic(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """The only load follows the atomic: nothing may move ahead of it."""
    for tile_e, tile_f in hl.tile(x.shape):
        hl.atomic_add(out, [tile_e, tile_f], 1.0)
        x[tile_e, tile_f] = out[tile_e, tile_f]
    return x


def _add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.add(left, right)


def _row_scan(x: torch.Tensor) -> torch.Tensor:
    """A scan along the outer grid block of a row-major tile: seeded."""
    output = torch.empty_like(x)
    for tile_e, tile_f in hl.tile(x.shape):
        output[tile_e, tile_f] = hl.associative_scan(_add, x[tile_e, tile_f], dim=0)
    return output


def _column_scan(input_t: torch.Tensor) -> torch.Tensor:
    """A scan along the inner grid block of a tensor contiguous along the
    outer one (``x.t()``): the strip is not the innermost block."""
    output = torch.empty_like(input_t)
    for tile_f, tile_e in hl.tile(input_t.shape):
        output[tile_f, tile_e] = hl.associative_scan(
            _add, input_t[tile_f, tile_e], dim=1
        )
    return output


def _pointwise_2d(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_e, tile_f in hl.tile(x.shape):
        out[tile_e, tile_f] = x[tile_e, tile_f] * 2
    return out


# GPU numerics tests; everything else runs with CUDA forbidden.
_GPU_TESTS = {
    "test_segmented_reduction_matches_index_add",
    "test_elided_scatter_matches_the_unelided_sum",
}


@pytest.fixture(autouse=True)
def _cpu_only(request: pytest.FixtureRequest) -> Iterator[None]:
    if request.node.originalname in _GPU_TESTS:
        yield
        return
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch("helion.runtime.kernel.target_device_capability", return_value=(10, 0)),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 0),
        ),
    ):
        yield


def _bind(
    fn: Callable[..., torch.Tensor], args: tuple[object, ...], **settings: Any
) -> Any:
    kernel = helion.kernel(fn, backend="cute", static_shapes=True, **settings)
    return kernel.bind(args)


def _codegen(
    fn: Callable[..., torch.Tensor],
    args: tuple[object, ...],
    *,
    fast_math: bool = False,
    **overrides: Any,
) -> str:
    bound = _bind(fn, args, fast_math=fast_math)
    config = bound.config_spec.default_config()
    if overrides:
        config = helion.Config(**{**config.config, **overrides})
    return _kernel_body(bound.to_code(config))


def _kernel_body(code: str) -> str:
    start = code.index("@cute.kernel")
    kernel_def = code.index("\ndef ", start)
    end = code.find("\ndef ", kernel_def + 1)
    if end < 0:
        end = len(code)
    return "\n".join(
        line for line in code[start:end].splitlines() if "# src[" not in line
    )


def _segment_args(
    num_elements: int = 4096, num_features: int = 128, num_nodes: int = 64
) -> tuple[torch.Tensor, torch.Tensor, int]:
    return (
        torch.zeros(num_elements, dtype=torch.int64, device=CPU),
        torch.zeros(num_elements, num_features, device=CPU),
        num_nodes,
    )


def _scatter_args(
    num_elements: int = 256, num_features: int = 128, num_nodes: int = 16
) -> tuple[torch.Tensor, torch.Tensor, int]:
    return (
        torch.zeros(num_elements, dtype=torch.int64, device=CPU),
        torch.zeros(num_elements, num_features, device=CPU),
        num_nodes,
    )


# One thread per 4-wide column strip owning the whole scan axis.
STRIP = {
    "block_sizes": [16, 128],
    "num_threads": [1, 32],
    "cute_vector_widths": [1, 4],
}
# Two strided threads on the scan axis (the study's shape1 winner's layout).
STRIDED = {
    "block_sizes": [32, 32],
    "num_threads": [2, 0],
    "cute_lane_layouts": ["strided", "blocked"],
}


# --------------------------------------------------------------------------
# zero-add elision
# --------------------------------------------------------------------------


def test_segment_reduction_elides_the_zero_adds_under_fast_math() -> None:
    code = _codegen(_segmented_reduction, _segment_args(), fast_math=True, **STRIDED)
    guarded = re.search(r"if v_\d+:\n\s+" + re.escape(SCALAR_ATOMIC), code)
    assert guarded is not None, code
    # The addend is the scan output itself; the dead ``where`` select is gone.
    assert "val=cutlass.Float32(scan_out" in code
    assert "segment_vals" not in code


def test_float_elision_stays_off_without_fast_math() -> None:
    """Flush-to-zero atomics leave no bit-exact proof for floats: the default
    build adds every lane's ``where`` value, scalar or vector."""
    code = _codegen(_segmented_reduction, _segment_args(), **STRIDED)
    assert re.search(r"if .*:\n\s+" + re.escape(SCALAR_ATOMIC), code) is None, code
    assert "val=cutlass.Float32(segment_vals)" in code
    code = _codegen(_segmented_reduction, _segment_args(), **STRIP)
    assert "_tile_atomic_vals_1_0.append(cutlass.Float32(segment_vals" in code
    assert re.search(r"if v_\d+:\n\s+" + re.escape(VECTOR_ATOMIC), code) is None, code
    assert VECTOR_ATOMIC in code


def test_elision_negates_a_zero_on_the_true_branch() -> None:
    code = _codegen(_scatter_where_flipped, _scatter_args(), fast_math=True)
    assert re.search(r"if not v_\d+:\n\s+" + re.escape(SCALAR_ATOMIC), code), code


_UNGUARDED_WHERE_ADD = r"cute.arch.atomic_add\(.*val=cutlass.Float32\(where\)"


def _refusal_cases() -> list[
    tuple[str, Callable[..., torch.Tensor], tuple[object, ...]]
]:
    """One failing proof condition each (``_cute_where_zero_elision``)."""
    indices, x, num_nodes = _scatter_args()
    out = torch.zeros(num_nodes, x.size(1), device=CPU)
    return [
        (
            "caller-provided destination",
            _scatter_where_into_argument,
            (indices, x, out),
        ),
        ("torch.empty destination", _scatter_where_empty, (indices, x, num_nodes)),
        ("torch.ones destination", _scatter_where_ones, (indices, x, num_nodes)),
        ("destination also stored", _scatter_where_and_store, (indices, x, num_nodes)),
        ("-0.0 literal", _scatter_where_negative_zero, (indices, x, num_nodes)),
        ("computed zero", _scatter_where_computed_zero, (indices, x, num_nodes)),
        ("used result", _where_add_used_result, (x,)),
        ("release semantics", _scatter_where_release, (indices, x, num_nodes)),
    ]


@pytest.mark.parametrize(
    ("fn", "args"),
    [pytest.param(fn, args, id=name) for name, fn, args in _refusal_cases()],
)
def test_elision_fails_closed_when_one_proof_condition_is_missing(
    fn: Callable[..., torch.Tensor], args: tuple[object, ...]
) -> None:
    code = _codegen(fn, args, fast_math=True)
    assert not re.search(r"if .*:\n\s+" + re.escape(SCALAR_ATOMIC), code), code
    assert re.search(_UNGUARDED_WHERE_ADD, code), code


def test_elision_admits_a_new_zeros_destination() -> None:
    """``x.new_zeros(...)`` is the same fresh ``+0`` allocation as
    ``torch.zeros``; the default build still adds every lane."""
    code = _codegen(_scatter_where_new_zeros, _scatter_args(), fast_math=True)
    assert re.search(r"if v_\d+:\n\s+" + re.escape(SCALAR_ATOMIC), code), code
    assert not re.search(_UNGUARDED_WHERE_ADD, code), code
    code = _codegen(_scatter_where_new_zeros, _scatter_args())
    assert not re.search(r"if .*:\n\s+" + re.escape(SCALAR_ATOMIC), code), code
    assert re.search(_UNGUARDED_WHERE_ADD, code), code


def test_elision_admits_the_unused_result_form_of_the_used_result_case() -> None:
    _, x, _ = _scatter_args()
    code = _codegen(_where_add_unused_result, (x,), fast_math=True)
    assert re.search(r"if v_\d+:\n\s+" + re.escape(SCALAR_ATOMIC), code), code
    assert not re.search(_UNGUARDED_WHERE_ADD, code), code


def test_integer_adds_elide_without_a_destination_proof_or_fast_math() -> None:
    indices, _, _ = _scatter_args()
    ints = torch.zeros(256, 128, dtype=torch.int32, device=CPU)
    int_out = torch.zeros(16, 128, dtype=torch.int32, device=CPU)
    code = _codegen(_scatter_where_int, (indices, ints, int_out))
    assert re.search(r"if v_\d+:\n\s+" + re.escape(SCALAR_ATOMIC), code), code


# --------------------------------------------------------------------------
# vector atomics
# --------------------------------------------------------------------------


def test_strip_layout_flushes_one_vector_atomic_per_row() -> None:
    code = _codegen(_segmented_reduction, _segment_args(), fast_math=True, **STRIP)
    assert SCALAR_ATOMIC not in code
    assert "_tile_atomic_vals_1_0 = []" in code
    assert "_tile_atomic_vals_1_0.append(cutlass.Float32(scan_out" in code
    flush = re.search(
        r"if v_\d+:\n\s+_cute_red_add_f32_vec\(\(output\.iterator \+ "
        r"cute\.crd2idx\(\(cutlass\.Int32\(idxs\), lane_base_1\), output\.layout\)\)"
        r"\.llvm_ptr, _tile_atomic_vals_1_0\)",
        code,
    )
    assert flush is not None, code
    # The flush sits after the constexpr vector loop, at the lane loop level.
    vloop = code.index("for vec_lane_1 in cutlass.range_constexpr(4):")
    assert flush.start() > vloop
    flush_indent = len(code[: flush.start()].rsplit("\n", 1)[-1])
    vloop_indent = len(code[:vloop].rsplit("\n", 1)[-1])
    assert flush_indent == vloop_indent


def test_two_wide_vectors_flush_a_v2_atomic() -> None:
    code = _codegen(
        _segmented_reduction,
        _segment_args(),
        block_sizes=[16, 64],
        num_threads=[1, 32],
        cute_vector_widths=[1, 2],
    )
    assert VECTOR_ATOMIC in code
    assert "for vec_lane_1 in cutlass.range_constexpr(2):" in code
    assert SCALAR_ATOMIC not in code


def test_vector_atomic_declines_unaligned_release_and_per_lane_sites() -> None:
    # A column extent that is not a multiple of the vector width: the chunk
    # mask is not uniform, so the scalar (elided) atomics stay.
    code = _codegen(
        _segmented_reduction,
        _segment_args(num_features=98),
        block_sizes=[16, 128],
        num_threads=[1, 32],
        cute_vector_widths=[1, 4],
    )
    assert VECTOR_ATOMIC not in code
    assert SCALAR_ATOMIC in code
    # A non-relaxed atomic keeps the scalar form.
    code = _codegen(
        _scatter_release,
        _scatter_args(),
        block_sizes=[16, 128],
        num_threads=[1, 32],
        cute_vector_widths=[1, 4],
    )
    assert VECTOR_ATOMIC not in code
    assert "sem='release'" in code
    # A predicate that varies per column: elided per lane, but a lane-varying
    # guard cannot predicate one vector reduction.
    code = _codegen(
        _scatter_where_per_element,
        _scatter_args(),
        fast_math=True,
        block_sizes=[16, 128],
        num_threads=[1, 32],
        cute_vector_widths=[1, 4],
    )
    assert VECTOR_ATOMIC not in code
    assert re.search(r"if v_\d+:\n\s+" + re.escape(SCALAR_ATOMIC), code), code


def test_vector_atomic_declines_an_atomic_uniform_along_a_lane_loop() -> None:
    # ``counts[tile_f]`` does not vary with the row lane loop; the lane-loop
    # placement pins that atomic to the loop's first lane, which the flush
    # protocol cannot express (it would reduce once per row iteration).
    x = torch.randn(64, 128, dtype=torch.float32)
    counts = torch.zeros(128, dtype=torch.float32)
    code = _codegen(
        _column_count_beside_copy,
        (x, counts),
        block_sizes=[16, 128],
        num_threads=[1, 32],
        cute_vector_widths=[1, 4],
    )
    assert VECTOR_ATOMIC not in code
    assert SCALAR_ATOMIC in code
    assert "lane_0 == 0" in code, code


def test_vector_atomic_needs_an_sm90_target() -> None:
    """``red.global.add.v4.f32`` is sm_90+: below that (or unknown) the site
    keeps the scalar atomic."""
    for capability in ((8, 0), None):
        with patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=capability,
        ):
            code = _codegen(_segmented_reduction, _segment_args(), **STRIP)
        assert VECTOR_ATOMIC not in code, capability
        assert SCALAR_ATOMIC in code, capability


# --------------------------------------------------------------------------
# loads-first lane unroll
# --------------------------------------------------------------------------


def test_lane_unroll_issues_every_load_before_the_first_atomic() -> None:
    code = _codegen(_segmented_reduction, _segment_args(), **STRIP, cute_lane_unroll=16)
    assert code.count("for lane_0 in cutlass.range_constexpr(16):") == 2
    assert "for lane_0 in range(16):" not in code
    load_phase = code.index("for lane_0 in cutlass.range_constexpr(16):")
    compute_phase = code.index(
        "for lane_0 in cutlass.range_constexpr(16):", load_phase + 1
    )
    # Three loads per row move: the row index, the next row index (masked)
    # and the 16-byte value packet, in that order of appearance.
    lists = re.findall(r"(_lane_unroll_loads_\d+) = \[\]", code)
    assert len(lists) == 3
    for name in lists:
        assert code.index(f"{name}.append(") < compute_phase
        assert code.index(f"{name}[", compute_phase) > compute_phase
    assert (
        f"_tile_unroll_vec_1_0 = {lists[2]}[(lane_0) * 1 + lane_1]" in code
        or f"_tile_unroll_vec_1_0 = {lists[2]}[lane_0 * 1 + lane_1]" in code
    )
    # No load is left in the compute phase and no memory effect precedes it.
    compute = code[compute_phase:]
    assert ".load()" not in compute
    assert "cute.arch.load(" not in compute
    assert VECTOR_ATOMIC not in code[:compute_phase]
    assert VECTOR_ATOMIC in compute


def test_partial_lane_unroll_rolls_the_remaining_trips() -> None:
    code = _codegen(
        _segmented_reduction,
        _segment_args(),
        block_sizes=[32, 128],
        num_threads=[1, 32],
        cute_vector_widths=[1, 4],
        cute_lane_unroll=8,
    )
    assert "for lane_0_blk in range(4):" in code
    assert code.count("for lane_0_u in cutlass.range_constexpr(8):") == 2
    assert code.count("lane_0 = lane_0_blk * 8 + lane_0_u") == 2
    # The lists are rebuilt per rolled trip and indexed by the constexpr lane.
    outer = code.index("for lane_0_blk in range(4):")
    assert code.index("_lane_unroll_loads_0 = []") > outer
    assert re.search(r"= _lane_unroll_loads_\d+\[lane_0_u\]", code)


def test_lane_unroll_without_a_lane_loop_or_a_movable_load_changes_nothing() -> None:
    # No lane loop: every row of the tile has its own thread.
    args = _segment_args()
    plain = _codegen(_segmented_reduction, args, num_threads=[0, 32])
    unrolled = _codegen(
        _segmented_reduction, args, num_threads=[0, 32], cute_lane_unroll=8
    )
    assert plain == unrolled
    # A lane loop whose only load follows the atomic: the pass declines and
    # the knob-off kernel is regenerated.
    x = torch.zeros(256, 128, device=CPU)
    out = torch.zeros(256, 128, device=CPU)
    plain = _codegen(
        _load_after_atomic, (x, out), block_sizes=[16, 128], num_threads=[1, 32]
    )
    unrolled = _codegen(
        _load_after_atomic,
        (x, out),
        block_sizes=[16, 128],
        num_threads=[1, 32],
        cute_lane_unroll=16,
    )
    assert plain == unrolled
    assert LOAD_LIST not in unrolled
    assert "for lane_0 in range(16):" in unrolled


def test_lane_unroll_declines_bodies_with_scalar_cache_policy_loads() -> None:
    # The scalar streaming helper is not eligible for lane-load unrolling.
    # Keep the policy and regenerate the ordinary loop unchanged.
    args = _segment_args()
    policies = ["", "", "streaming"]
    plain = _codegen(
        _segmented_reduction, args, **STRIP, load_eviction_policies=policies
    )
    unrolled = _codegen(
        _segmented_reduction,
        args,
        **STRIP,
        load_eviction_policies=policies,
        cute_lane_unroll=16,
    )
    assert "_cute_scalar_policy_load(" in plain
    assert "cutlass.Int64, 'streaming'" in plain
    assert plain == unrolled
    assert LOAD_LIST not in unrolled


def test_lane_unroll_normalizes_to_one_without_a_grid_lane_loop() -> None:
    bound = _bind(_segmented_reduction, _segment_args())
    spec = bound.config_spec

    def normalized(**overrides: Any) -> int:
        config = helion.Config(**{**spec.default_config().config, **overrides})
        return spec.normalized_config(config).config.get("cute_lane_unroll", 1)

    assert normalized(num_threads=[0, 32], cute_lane_unroll=8) == 1
    assert (
        normalized(block_sizes=[16, 128], num_threads=[1, 32], cute_lane_unroll=8) == 8
    )
    assert (
        normalized(block_sizes=[16, 128], num_threads=[16, 128], cute_lane_unroll=8)
        == 1
    )


# --------------------------------------------------------------------------
# seed
# --------------------------------------------------------------------------


def test_scan_tile_seed_targets_the_column_strip_layout() -> None:
    bound = _bind(_segmented_reduction, _segment_args())
    seeds = [
        config.config
        for config in bound.config_spec.compiler_seed_configs
        if config.config.get("cute_lane_unroll", 1) > 1
    ]
    assert seeds[0] == {
        "block_sizes": [16, 128],
        "num_threads": [1, 32],
        "cute_vector_widths": [1, 4],
        "cute_lane_unroll": 16,
    }
    assert {
        "block_sizes": [32, 128],
        "num_threads": [1, 32],
        "cute_vector_widths": [1, 4],
        "cute_lane_unroll": 16,
    } in seeds
    # Seeding opens the unroll depth to the search.
    seen: list[Any] = []

    def record(fragment: Any) -> Any:
        seen.append(fragment)
        return fragment.default()

    bound.config_spec.flat_config(record)
    choices = [
        tuple(fragment.search_choices or ())
        for fragment in seen
        if getattr(fragment, "choices", None) == (1, 2, 4, 8, 16)
    ]
    assert (1, 2, 4, 8, 16) in choices


def _unroll_seeds(bound: Any) -> list[dict[str, Any]]:
    return [
        config.config
        for config in bound.config_spec.compiler_seed_configs
        if config.config.get("cute_lane_unroll", 1) > 1
    ]


def test_scan_tile_seed_skips_kernels_without_a_scan() -> None:
    bound = _bind(_pointwise_2d, (torch.zeros(4096, 128, device=CPU),))
    assert not _unroll_seeds(bound)


def test_scan_tile_seed_skips_layouts_the_vector_protocol_cannot_realise() -> None:
    # 98 columns: no 16-byte packet covers the strip.
    assert not _unroll_seeds(
        _bind(_segmented_reduction, _segment_args(num_features=98))
    )
    # A transposed input: the contiguous strip is the outer grid block.
    transposed = torch.zeros(4096, 128, device=CPU).t()
    assert not _unroll_seeds(_bind(_column_scan, (transposed,)))
    # The same scan with the contiguous strip on the inner block keeps its seed.
    assert _unroll_seeds(_bind(_row_scan, (torch.zeros(4096, 128, device=CPU),)))


# --------------------------------------------------------------------------
# GPU numerics
# --------------------------------------------------------------------------


def _reference(indices: torch.Tensor, x: torch.Tensor, num_nodes: int) -> torch.Tensor:
    out = torch.zeros(num_nodes, x.size(1), dtype=x.dtype, device=x.device)
    return out.index_add_(0, indices, x)


def _index_patterns(num_elements: int, num_nodes: int) -> dict[str, torch.Tensor]:
    """Sorted segment ids whose last row never belongs to node 0.

    The example masks the next-row index of the last row with 0, so a
    partial last tile whose last row is node 0 does not flag it as a
    segment end (the Triton backend drops that sum too); the study inputs
    (sorted random ids) never end on node 0 either.
    """
    generator = torch.Generator().manual_seed(1)
    sorted_random = torch.randint(
        0, num_nodes, (num_elements,), generator=generator
    ).sort()[0]
    sorted_random[-1] = num_nodes - 1
    # Segments of 33 rows straddle every 16- and 32-row tile boundary; the
    # last node ids stay empty.
    spans = torch.arange(num_elements) // 33
    return {
        "sorted_random": sorted_random,
        "single_segment": torch.ones(num_elements, dtype=torch.int64),
        "one_row_segments": torch.arange(num_elements) % num_nodes,
        "tile_spanning": spans.clamp(max=num_nodes - 1),
    }


_GPU_CONFIGS: dict[str, dict[str, Any]] = {
    "strip_16_unrolled": {**STRIP, "cute_lane_unroll": 16},
    "strip_32_partial": {
        "block_sizes": [32, 128],
        "num_threads": [1, 32],
        "cute_vector_widths": [1, 4],
        "cute_lane_unroll": 8,
    },
    "strip_4": {
        "block_sizes": [4, 128],
        "num_threads": [1, 32],
        "cute_vector_widths": [1, 4],
        "cute_lane_unroll": 4,
    },
    "strip_v2": {
        "block_sizes": [16, 64],
        "num_threads": [1, 32],
        "cute_vector_widths": [1, 2],
        "cute_lane_unroll": 16,
    },
    "strided_vector": {
        "block_sizes": [32, 128],
        "num_threads": [2, 32],
        "cute_vector_widths": [1, 4],
        "cute_lane_layouts": ["strided", "blocked"],
        "cute_lane_unroll": 16,
    },
    "strided_scalar": STRIDED,
}


@skipIfNotCUDA()
@pytest.mark.parametrize("fast_math", [False, True])
@pytest.mark.parametrize("name", sorted(_GPU_CONFIGS))
@pytest.mark.parametrize(
    ("num_elements", "num_features", "num_nodes"),
    [(2000, 128, 100), (4096, 128, 64), (1000, 98, 30)],
)
def test_segmented_reduction_matches_index_add(
    name: str, num_elements: int, num_features: int, num_nodes: int, fast_math: bool
) -> None:
    kernel = helion.kernel(
        _segmented_reduction, backend="cute", static_shapes=True, fast_math=fast_math
    )
    x = torch.randn(num_elements, num_features, device=DEVICE)
    for pattern, indices in _index_patterns(num_elements, num_nodes).items():
        indices = indices.to(DEVICE)
        code, out = code_and_output(
            kernel, (indices, x, num_nodes), **_GPU_CONFIGS[name]
        )
        expected = _reference(indices, x, num_nodes)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-3), pattern
        if name.startswith("strip") and num_features % 4 == 0:
            assert VECTOR_ATOMIC in code
        # A partial column tile keeps every load inside the per-element
        # V-loop, which the unroll leaves alone (knob-off code).
        if _GPU_CONFIGS[name].get("cute_lane_unroll", 1) > 1 and num_features % 4 == 0:
            assert LOAD_LIST in code
        # The elision must not depend on the data: a second input through
        # the same config.
        y = torch.randn_like(x)
        _, out = code_and_output(kernel, (indices, y, num_nodes), **_GPU_CONFIGS[name])
        torch.testing.assert_close(
            out, _reference(indices, y, num_nodes), rtol=1e-4, atol=1e-3
        )


@skipIfNotCUDA()
def test_elided_scatter_matches_the_unelided_sum() -> None:
    """``fast_math`` elision vs every add issued, on +0.0, -0.0, NaN, inf and
    a subnormal.  Zeros compare sign-insensitively: a flushed negative
    underflow can leave ``-0.0`` in a cell the unelided kernel would reset
    to ``+0.0`` with its zero add, which is the deviation ``fast_math``
    accepts (``_cute_where_zero_elision``)."""
    indices = torch.randint(0, 16, (256,), device=DEVICE).sort()[0]
    x = torch.randn(256, 128, device=DEVICE)
    # Special values in separate column classes, so the NaNs do not swallow
    # every other class of the node rows they land on.
    x[::3, 0::4] = 0.0
    x[1::7, 1::4] = -0.0
    x[2::11, 2::4] = float("nan")
    x[3::13, 3::4] = 1e-40  # an fp32 subnormal; the atomic flushes it to +0.0
    x[4::17, 0::4] = float("inf")
    elided = helion.kernel(
        _scatter_where, backend="cute", static_shapes=True, fast_math=True
    )
    code, out = code_and_output(elided, (indices, x, 16), **STRIP)
    assert VECTOR_ATOMIC in code
    assert re.search(r"if v_\d+:\n\s+" + re.escape(VECTOR_ATOMIC), code)
    # The shipping default issues every lane's add, zeros included, and the
    # same sum through a destination the proof rejects (ones, not zeros).
    plain = helion.kernel(_scatter_where, backend="cute", static_shapes=True)
    code, default = code_and_output(plain, (indices, x, 16), **STRIP)
    assert re.search(r"if v_\d+:\n", code) is None
    ones = helion.kernel(_scatter_where_ones, backend="cute", static_shapes=True)
    code, from_ones = code_and_output(ones, (indices, x, 16), **STRIP)
    assert re.search(r"if v_\d+:\n", code) is None
    torch.testing.assert_close(out, default, rtol=1e-3, atol=1e-4, equal_nan=True)
    torch.testing.assert_close(
        out, from_ones - 1.0, rtol=1e-3, atol=1e-4, equal_nan=True
    )
    keep = (indices % 2 == 0).unsqueeze(1)
    expected = _reference(indices, torch.where(keep, x, 0.0), 16)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-4, equal_nan=True)
    assert torch.equal(torch.isnan(out), torch.isnan(expected))
    assert torch.isnan(out).any() and torch.isinf(out).any()
