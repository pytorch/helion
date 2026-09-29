"""Dynamic-shape rolled row reductions keep their vector memory path (CPU codegen).

A rolled (``reduction_loops``) reduction over a symbolic extent always carries
a bounds mask, and the mask used to force every load and store back to one
masked scalar access per element and to disable the two-pass register cache.
Now:

* When the bound cache key proves ``extent % V == 0`` (the persistent-vector
  alignment signature records every input size modulo the widest vector), the
  chunk base decides all V lanes: the mask is evaluated once per chunk above
  the constexpr V-loop, and the packet load/store is predicated on it.
* Otherwise the lane body carries a whole-chunk predicate: full chunks still
  use one vector load/store, and only the chunk straddling the extent falls
  back to per-element accesses.
* The trip count is exposed as a host-computed ``cutlass.Constexpr`` so the
  fuser can allocate the register cache exactly for every runtime extent; the
  fused sweeps run under a trace-time check that the exact fragment does not
  exceed the size-hint fragment the binding was admitted with, with the
  pre-fusion sweeps as the fallback branch.  Every fused group that guard
  intersects is guarded whole, so a static group never populates its cache in
  one branch and consumes it outside.
* A packet is only formed when the tensor's base and every non-lane stride
  are V-aligned (cache-key-backed for symbolic strides); a 4100-wide bf16 row
  is 8-byte aligned and an LDG.128 at its second row would fault.  The base is
  proven by provenance: an input's bound pointer residue or a fresh wrapper
  allocation, never a tensor that merely lacks an input source.
* An outer row mask guards the packet pointer even when the roll itself needs
  no bounds mask, so a partially filled row tile never reads rows past the
  tensor.  Static row lengths that do not divide the block take the same
  chunk-level forms instead of scalar accesses.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from unittest.mock import patch

from examples.aot_example import rms_norm_batched
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.cute.fuse_two_pass_loads import fuse_two_pass_loads
from helion._compiler.cute.memory_ops import cute_known_multiple
from helion._compiler.cute.pipeline_inner_loads import pipeline_inner_loads
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

VEC_LOAD = "cute.arch.load("
VEC_STORE = "_cute_store_u16_vec(out.iterator"
TRIPS = "_REDUCTION_TRIPS_1"
# The roll iterates the same ``ceil(n / block)`` trips as ``range(0, n, block)``
# but with a trace-time constant bound the DSL can unroll.
LOOP = (
    "for roffset_1 in range(cutlass.Int32(0), "
    f"cutlass.Int32({TRIPS} * _REDUCTION_BLOCK_1), cutlass.Int32(_REDUCTION_BLOCK_1)):"
)


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU-only test")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


def _rms_norm_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Like the example, but writes into a caller-provided (padded) output."""
    m, n = x.size()
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        rms = torch.sqrt(torch.mean(x_tile * x_tile, dim=-1) + 1e-5)
        out[tile_m, :] = (x_tile / rms[:, None]).to(out.dtype)
    return out


def _rms_norm_specialized_n(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Dynamic row count over a specialized (static) row length."""
    m, n = x.size()
    hl.specialize(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        rms = torch.sqrt(torch.mean(x_tile * x_tile, dim=-1) + eps)
        out[tile_m, :] = (x_tile / rms[:, None]).to(out.dtype)
    return out


def _dynamic_rows_between_static_column_sweeps(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rolled sweeps over dynamic ``n`` interleaved with static sweeps over ``k``.

    The ``k`` group populates its cache between the two ``n`` sweeps and
    consumes it after them, so it straddles the ``n`` group's trace-time guard.
    """
    m, n = x.size()
    _, k = y.size()
    hl.specialize(k)
    block_k = hl.register_block_size(k)
    out_x = torch.empty_like(x)
    out_y = torch.empty_like(y)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        sum_sq = torch.sum(x_tile * x_tile, dim=-1)
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            acc = acc + torch.sum(y[tile_m, tile_k].to(torch.float32), dim=-1)
        out_x[tile_m, :] = (x_tile / sum_sq[:, None]).to(out_x.dtype)
        for tile_k in hl.tile(k, block_size=block_k):
            y_tile = y[tile_m, tile_k].to(torch.float32)
            out_y[tile_m, tile_k] = (y_tile / acc[:, None]).to(out_y.dtype)
    return out_x, out_y


def _rms_norm_of_bf16_bits(x_bits: torch.Tensor) -> torch.Tensor:
    """The bf16 rows live behind an int16 input that owns their storage."""
    x = x_bits.view(torch.bfloat16)
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        rms = torch.sqrt(torch.mean(x_tile * x_tile, dim=-1) + 1e-5)
        out[tile_m, :] = (x_tile / rms[:, None]).to(out.dtype)
    return out


def _rms_norm_into_new_empty(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = x.new_empty(m, n)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        rms = torch.sqrt(torch.mean(x_tile * x_tile, dim=-1) + 1e-5)
        out[tile_m, :] = (x_tile / rms[:, None]).to(out.dtype)
    return out


def _rms_norm_into_new_zeros(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = x.new_zeros(m, n)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        rms = torch.sqrt(torch.mean(x_tile * x_tile, dim=-1) + 1e-5)
        out[tile_m, :] = (x_tile / rms[:, None]).to(out.dtype)
    return out


def _rms_norm_into_empty_strided(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_strided((m, n), (n, 1), dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        rms = torch.sqrt(torch.mean(x_tile * x_tile, dim=-1) + 1e-5)
        out[tile_m, :] = (x_tile / rms[:, None]).to(out.dtype)
    return out


def _dynamic_kernel(
    fn: Callable[..., Any], *, static_shapes: bool = False
) -> helion.Kernel:
    return helion.kernel(
        fn,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )


def _bind(
    kernel: helion.Kernel, arguments: tuple[object, ...]
) -> tuple[Any, tuple[object, ...]]:
    """Bind on CPU; the caller keeps ``arguments`` alive while generating code."""
    return _cpu_bind(kernel, arguments), arguments


def _bind_rms_norm(
    x: torch.Tensor, *, static_shapes: bool = False
) -> tuple[Any, tuple[object, ...]]:
    return _bind(
        _dynamic_kernel(rms_norm_batched.fn, static_shapes=static_shapes), (x, 1e-5)
    )


def _padded_rows(rows: int = 64, width: int = 4100, stride: int = 4104) -> torch.Tensor:
    """``width`` columns of a wider buffer: 16-byte aligned rows, partial chunks."""
    return torch.empty((rows, stride), dtype=torch.bfloat16)[:, :width]


def _reduction_block_id(bound: Any) -> int:
    (block_id,) = [block.block_id for block in bound.env.block_sizes if block.reduction]
    return block_id


def _rolled_config(
    bound: Any,
    *,
    threads: int,
    vec: int,
    chunk: int,
    reload: str = "register",
    rows: int = 1,
) -> helion.Config:
    spec = bound.config_spec
    reduction = _reduction_block_id(bound)
    return spec.normalized_config(
        helion.Config(
            block_sizes=[rows],
            num_threads=[
                threads if block_id == reduction else rows
                for block_id in spec.num_threads.valid_block_ids()
            ],
            reduction_loops=[chunk],
            cute_vector_widths=[
                vec if block_id == reduction else 1
                for block_id in spec.cute_vector_widths.valid_block_ids()
            ],
            cute_reduction_reloads=[reload],
        )
    )


def _straddle_config(bound: Any) -> helion.Config:
    """64-row tiles on 4x32 threads, V=8 rolled ``n`` sweeps, ``k`` tiles of one."""
    spec = bound.config_spec
    reduction = _reduction_block_id(bound)
    return spec.normalized_config(
        helion.Config(
            block_sizes=[64, 1],
            num_threads=[
                {reduction: 32, 0: 4}.get(block_id, 1)
                for block_id in spec.num_threads.valid_block_ids()
            ],
            reduction_loops=[1024],
            cute_vector_widths=[
                8 if block_id == reduction else 1
                for block_id in spec.cute_vector_widths.valid_block_ids()
            ],
            cute_reduction_reloads=["register"],
        )
    )


def _kernel_def(code: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )


def _element_loop_bodies(code: str) -> list[str]:
    """Source of every constexpr per-element V-loop body."""
    return [
        "\n".join(ast.unparse(stmt) for stmt in node.body)
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and ast.unparse(node.iter.func) == "cutlass.range_constexpr"
    ]


def _vector_loads(code: str) -> list[str]:
    return [line for line in code.splitlines() if VEC_LOAD in line]


def _trace_time_branches(code: str) -> list[tuple[str, str, str]]:
    """``(test, body, orelse)`` source of every ``if cutlass.const_expr(...)``."""
    return [
        (
            ast.unparse(node.test),
            "\n".join(ast.unparse(stmt) for stmt in node.body),
            "\n".join(ast.unparse(stmt) for stmt in node.orelse),
        )
        for node in ast.walk(_kernel_def(code))
        if isinstance(node, ast.If)
        and ast.unparse(node.test).startswith("cutlass.const_expr(")
    ]


def _fused_and_fallback(code: str, budget: str) -> tuple[str, str]:
    """The fused group and its pre-fusion fallback, both guarded by ``budget``."""
    branches = _trace_time_branches(code)
    assert [test for test, _body, _orelse in branches] == [budget, budget]
    (_test, declaration, _no_declaration), (_test, fused, fallback) = branches
    assert "cute.make_rmem_tensor(" in declaration
    return fused, fallback


@skipUnlessBackends(["cute"])
def test_divisible_extent_vectorizes_under_a_chunk_level_mask() -> None:
    # The study config of aot_example/rms_norm_batched at 4096x4096 bf16:
    # 256 threads, V=2, 2048-element chunks (two trips per row).
    bound, _arguments = _bind_rms_norm(torch.empty((4096, 4096), dtype=torch.bfloat16))
    assert cute_known_multiple(bound.env, bound.env.block_sizes[1].numel, 8)
    code = bound.to_code(_rolled_config(bound, threads=256, vec=2, chunk=2048))
    kernel = _kernel_def(code)
    # The trip count is a constexpr kernel parameter computed on the host and
    # passed through the launcher.
    assert [arg.arg for arg in kernel.args.args][-2:] == ["_REDUCTION_BLOCK_1", TRIPS]
    assert f"{TRIPS} = (n + _REDUCTION_BLOCK_1 - 1) // _REDUCTION_BLOCK_1" in code
    assert f"_REDUCTION_BLOCK_1, {TRIPS}, block=(256, 1, 1))" in code
    assert "range(cutlass.Int32(0), cutlass.Int32(n)" not in code
    # The register cache is sized by the exact trip count and, like the fused
    # sweeps, only exists while that fragment stays within the size-hint
    # fragment (two trips of 8 elements per thread); the fallback branch is the
    # pre-fusion pair of sweeps.
    cap = f"cutlass.const_expr({TRIPS} * 8 <= 16)"
    assert (
        f"if {cap}:\n        _fuse_cache_0 = cute.make_rmem_tensor({TRIPS} * 8, cutlass.Uint16)"
        in code
    )
    fused, fallback = _fused_and_fallback(code, cap)
    assert fused.count(LOOP) == 2
    assert fallback.count(LOOP) == 2
    assert "_fuse_cache_0" not in fallback
    # The bounds mask is evaluated once per V-chunk, above the V-loop.
    assert fused.count("mask_1 = reduction_lane_base_1 < n") == 1
    assert fused.count("mask_1 = reduction_lane_base_2 < n") == 1
    assert "rindex_1 < n" not in code
    # One guarded LDG.32 packet per lane iteration; the consume sweep reads
    # the register cache instead of reloading (the fallback reloads).
    vector_loads = _vector_loads(fused)
    assert len(vector_loads) == 1
    assert len(_vector_loads(fallback)) == 2
    assert "ir.VectorType.get([2], cutlass.Uint16.mlir_type)" in vector_loads[0]
    assert (
        "if mask_0 and reduction_lane_base_1 < n else x.iterator + cutlass.Int32(0)"
        in vector_loads[0]
    )
    assert "load_1 = cutlass.Uint16(_fuse_cache_0[" in fused
    # One vector store per chunk, predicated on the chunk-level mask.
    assert fused.count(VEC_STORE) == 1
    assert "if mask_0 and mask_1:\n" in fused
    for body in _element_loop_bodies(code):
        assert ".load()" not in body
        assert ".store(" not in body


@skipUnlessBackends(["cute"])
def test_unknown_divisibility_keeps_vector_interior_and_a_scalar_tail() -> None:
    # 4100 % 8 == 4, so V=8 packets are not chunk-uniform; both tensors are
    # views of 4104-wide buffers so their rows stay 16-byte aligned.  Full
    # chunks use one LDG.128/STG.128 and only the straddling chunk goes scalar.
    # 64 threads give two lane iterations, which keeps the load pipeliner out
    # of the shape under test.
    x = _padded_rows()
    out = _padded_rows()
    bound, _arguments = _bind(_dynamic_kernel(_rms_norm_into), (x, out))
    numel = bound.env.block_sizes[1].numel
    assert not cute_known_multiple(bound.env, numel, 8)
    assert cute_known_multiple(bound.env, numel, 4)
    code = bound.to_code(_rolled_config(bound, threads=64, vec=8, chunk=1024))
    assert "reduction_chunk_full_1 = reduction_lane_base_1 + 7 < n" in code
    assert "reduction_chunk_tail" not in code
    # Per-element masks stay inside the V-loop.
    assert "mask_1 = rindex_1 < n" in code
    # The size hint of 4100 admits five trips of 16 elements per thread.
    fused, _fallback = _fused_and_fallback(
        code, f"cutlass.const_expr({TRIPS} * 16 <= 80)"
    )
    vector_loads = _vector_loads(fused)
    assert len(vector_loads) == 1
    assert "ir.VectorType.get([8], cutlass.Uint16.mlir_type)" in vector_loads[0]
    assert "if mask_0 and reduction_lane_base_1 + 7 < n else" in vector_loads[0]
    # Full chunks read the packet; the tail chunk re-reads its live elements.
    assert (
        "load = (cutlass.Uint16(_unroll_vec_0[reduction_vec_lane_1])"
        ".bitcast(cutlass.BFloat16) if reduction_chunk_full_1 else (x.iterator"
    ) in code
    assert (
        f"_fuse_cache_0 = cute.make_rmem_tensor({TRIPS} * 16, cutlass.Uint16)" in code
    )
    # The consume sweep flushes full chunks as one vector and stores the
    # tail chunk's live elements individually under the negated predicate.
    assert fused.count(VEC_STORE) == 1
    assert "if mask_0 and reduction_chunk_full_2:\n" in fused
    assert "if not reduction_chunk_full_2 and (mask_0 and mask_1):" in fused
    assert fused.count(".store(cutlass.BFloat16(") == 1


@skipUnlessBackends(["cute"])
def test_divisibility_fact_selects_between_the_two_forms() -> None:
    # A contiguous 4100-wide row is chunk-uniform for V=4 (4100 % 4 == 0) and
    # its 8200-byte row stride is 8-byte aligned, so LDG.64 packets apply.
    bound, _arguments = _bind_rms_norm(torch.empty((4096, 4100), dtype=torch.bfloat16))
    code = bound.to_code(_rolled_config(bound, threads=256, vec=4, chunk=2048))
    assert "mask_1 = reduction_lane_base_1 < n" in code
    assert "reduction_chunk_full_1" not in code
    assert "ir.VectorType.get([4], cutlass.Uint16.mlir_type)" in code
    assert VEC_STORE in code


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_row_mask_guards_the_packet_of_an_unmasked_roll(static_shapes: bool) -> None:
    # 13 rows in tiles of 4 and a 4096-wide row that divides the block: the
    # roll needs no bounds mask, but rows 13-15 of the last tile do not exist,
    # so the row mask must select the packet pointer (not just gate the
    # extracted values).  The dynamic variant specializes ``n`` and keeps
    # ``m`` symbolic.
    fn = rms_norm_batched.fn if static_shapes else _rms_norm_specialized_n
    x = torch.empty((13, 4096), dtype=torch.bfloat16)
    bound, _arguments = _bind(
        _dynamic_kernel(fn, static_shapes=static_shapes), (x, 1e-5)
    )
    code = bound.to_code(_rolled_config(bound, rows=4, threads=64, vec=8, chunk=1024))
    assert TRIPS not in code
    assert "mask_1" not in code
    row_bound = "13" if static_shapes else "m"
    assert f"and offsets_0 < {row_bound}" in code
    vector_loads = _vector_loads(code)
    assert len(vector_loads) == 1
    assert " if mask_0 else x.iterator + cutlass.Int32(0)" in vector_loads[0]
    assert "_fuse_cache_0 = cute.make_rmem_tensor(64, cutlass.Uint16)" in code
    assert code.count(VEC_STORE) == 1
    assert "if mask_0:\n" in code


@skipUnlessBackends(["cute"])
def test_static_extents_not_a_multiple_of_the_block_take_the_chunk_forms() -> None:
    # A static row length that does not divide the block used to keep every
    # access scalar; it now takes the same chunk-level forms as a symbolic
    # extent, without a constexpr trip count or a trace-time cap.
    bound, _arguments = _bind_rms_norm(
        torch.empty((64, 4100), dtype=torch.bfloat16), static_shapes=True
    )
    code = bound.to_code(_rolled_config(bound, threads=256, vec=4, chunk=2048))
    assert TRIPS not in code
    assert "const_expr" not in code
    assert "mask_1 = reduction_lane_base_1 < 4100" in code
    assert "ir.VectorType.get([4], cutlass.Uint16.mlir_type)" in code
    assert code.count(VEC_STORE) == 1
    # 4100 of 4104 columns for both tensors (``empty_like`` of the view would
    # allocate contiguous 8200-byte rows, which correctly stay scalar).
    x = _padded_rows()
    out = _padded_rows()
    bound, _arguments = _bind(
        _dynamic_kernel(_rms_norm_into, static_shapes=True), (x, out)
    )
    code = bound.to_code(_rolled_config(bound, threads=64, vec=8, chunk=1024))
    assert TRIPS not in code
    assert "const_expr" not in code
    assert "reduction_chunk_full_1 = reduction_lane_base_1 + 7 < 4100" in code
    assert "ir.VectorType.get([8], cutlass.Uint16.mlir_type)" in code
    assert "if not reduction_chunk_full_2 and" in code
    assert code.count(VEC_STORE) == 1


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_contiguous_rows_not_a_multiple_of_v_stay_scalar(static_shapes: bool) -> None:
    # Contiguous 4100-wide rows are only 8-byte aligned: a V=8 packet at the
    # second row would fault, so the packet is refused whatever the mask says.
    bound, _arguments = _bind_rms_norm(
        torch.empty((64, 4100), dtype=torch.bfloat16), static_shapes=static_shapes
    )
    code = bound.to_code(_rolled_config(bound, threads=128, vec=8, chunk=1024))
    assert VEC_LOAD not in code
    assert "_cute_store_u16_vec" not in code


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_misaligned_row_stride_of_a_view_stays_scalar(static_shapes: bool) -> None:
    # 4096 columns of a 4100-wide buffer: the extent divides V=8 but the row
    # stride does not, so V=8 stays scalar while V=4 (8-byte rows) vectorizes.
    x = torch.empty((64, 4100), dtype=torch.bfloat16)[:, :4096]
    bound, _arguments = _bind_rms_norm(x, static_shapes=static_shapes)
    code = bound.to_code(_rolled_config(bound, threads=128, vec=8, chunk=1024))
    assert VEC_LOAD not in code
    code = bound.to_code(_rolled_config(bound, threads=256, vec=4, chunk=1024))
    assert "ir.VectorType.get([4], cutlass.Uint16.mlir_type)" in code


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_misaligned_base_stays_scalar(static_shapes: bool) -> None:
    # The bound residue of the view's base pointer (2 bytes into a 16-byte
    # aligned buffer) refuses every packet width.
    x = torch.empty((64, 4104), dtype=torch.bfloat16)[:, 1:4097]
    bound, _arguments = _bind_rms_norm(x, static_shapes=static_shapes)
    for vec in (8, 4, 2):
        code = bound.to_code(_rolled_config(bound, threads=256, vec=vec, chunk=1024))
        assert VEC_LOAD not in code, vec


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("static_shapes", [False, True])
@pytest.mark.parametrize(
    "fn",
    [_rms_norm_into_new_empty, _rms_norm_into_new_zeros, _rms_norm_into_empty_strided],
)
def test_wrapper_allocations_prove_the_output_base(
    fn: Callable[..., Any], static_shapes: bool
) -> None:
    # ``Tensor.new_*`` and ``torch.empty_strided`` allocate fresh storage like
    # ``torch.empty``, so the output's base is proven and its rows keep one
    # vector store per chunk (twice in dynamic mode: fused branch and
    # fallback).  Only the row stride proof differs between the modes.
    x = torch.empty((64, 4096), dtype=torch.bfloat16)
    bound, _arguments = _bind(_dynamic_kernel(fn, static_shapes=static_shapes), (x,))
    code = bound.to_code(_rolled_config(bound, threads=128, vec=8, chunk=1024))
    assert VEC_LOAD in code
    assert code.count(VEC_STORE) == (1 if static_shapes else 2)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_dtype_punning_view_reads_its_owners_keyed_alignment(
    static_shapes: bool,
) -> None:
    # The bf16 rows are a ``view(torch.bfloat16)`` of an int16 input.  Every
    # packet dtype keys its inputs' pointer/stride residues, so with static
    # shapes the zero-offset view that this one input owns reads the input's
    # bound base residue and keeps its packet; the symbolic view has no owner
    # proof and stays scalar.  The freshly allocated output proves its base
    # either way and its row stride from the static extent or the keyed size
    # residue, so its vector stores stay.
    x_bits = torch.empty((64, 4096), dtype=torch.int16)
    bound, _arguments = _bind(
        _dynamic_kernel(_rms_norm_of_bf16_bits, static_shapes=static_shapes), (x_bits,)
    )
    code = bound.to_code(_rolled_config(bound, threads=128, vec=8, chunk=1024))
    assert (VEC_LOAD in code) is static_shapes
    assert "_cute_store_u16_vec" in code


@skipUnlessBackends(["cute"])
def test_a_static_group_straddling_the_guard_is_guarded_whole() -> None:
    # The static ``k`` group's populate sweep runs between the two rolled ``n``
    # sweeps and its consume sweep after them.  Guarding only the ``n`` span
    # would trace the original populate sweep in the fallback while the
    # consume sweep still read ``_fuse_cache_1``; the guard covers the whole
    # ``k`` group, and the fallback reloads both groups.
    x = torch.empty((64, 4096), dtype=torch.bfloat16)
    y = torch.empty((64, 256), dtype=torch.bfloat16)
    bound, _arguments = _bind(
        _dynamic_kernel(_dynamic_rows_between_static_column_sweeps), (x, y)
    )
    code = bound.to_code(_straddle_config(bound))
    kernel = _kernel_def(code)
    # Four trips of 32 elements per thread at the 4096 size hint.
    budget = "cutlass.const_expr(_REDUCTION_TRIPS_2 * 32 <= 128)"
    fused, fallback = _fused_and_fallback(code, budget)
    assert (
        "_fuse_cache_0 = cute.make_rmem_tensor(_REDUCTION_TRIPS_2 * 32, cutlass.Uint16)"
        in code
    )
    assert "_fuse_cache_1 = cute.make_rmem_tensor(64, cutlass.BFloat16)" in code
    for branch in (fused, fallback):
        assert branch.count("for roffset_2 in range(") == 2
        assert branch.count("for tile_offset_2 in range(") == 2
    assert "] = load_1" in fused and "load_3 = _fuse_cache_1[" in fused
    assert "_fuse_cache_" not in fallback
    (guard,) = [
        stmt
        for stmt in kernel.body
        if isinstance(stmt, ast.If)
        and ast.unparse(stmt.test) == budget
        and any(isinstance(inner, ast.For) for inner in stmt.body)
    ]
    after_guard = "\n".join(
        ast.unparse(stmt) for stmt in kernel.body[kernel.body.index(guard) + 1 :]
    )
    assert "for " not in after_guard
    assert "_fuse_cache_" not in after_guard


@skipUnlessBackends(["cute"])
def test_static_extent_codegen_is_unchanged() -> None:
    bound, _arguments = _bind_rms_norm(
        torch.empty((4096, 4096), dtype=torch.bfloat16), static_shapes=True
    )
    code = bound.to_code(_rolled_config(bound, threads=256, vec=2, chunk=2048))
    assert TRIPS not in code
    assert "mask_1" not in code
    assert "const_expr" not in code
    assert "_fuse_cache_0 = cute.make_rmem_tensor(16, cutlass.Uint16)" in code
    vector_loads = _vector_loads(code)
    assert len(vector_loads) == 1
    assert " if " not in vector_loads[0]


@skipUnlessBackends(["cute"])
def test_gmem_reload_keeps_the_constexpr_bound_without_a_cache() -> None:
    bound, _arguments = _bind_rms_norm(torch.empty((4096, 4096), dtype=torch.bfloat16))
    code = bound.to_code(
        _rolled_config(bound, threads=256, vec=2, chunk=2048, reload="gmem")
    )
    assert code.count(LOOP) == 2
    assert "_fuse_cache_" not in code
    assert "const_expr" not in code
    # Both sweeps still load one packet per chunk.
    assert code.count(VEC_LOAD) == 2


@skipUnlessBackends(["cute"])
def test_dynamic_single_lane_row_is_software_pipelined() -> None:
    # 128 threads x V=8 cover a 1024-element chunk in one lane iteration, the
    # shape the load pipeliner handles; a symbolic end no longer blocks it
    # because the guarded packet makes the speculative prefetch safe, and the
    # chunk-level mask between the lane base and the packet is kept in place.
    bound, _arguments = _bind_rms_norm(torch.empty((4096, 4096), dtype=torch.bfloat16))
    code = bound.to_code(_rolled_config(bound, threads=128, vec=8, chunk=1024))
    assert "_pipe_load_0 = cute.arch.load(" in code
    assert "_unroll_vec_0 = _pipe_load_0" in code
    prefetch = next(
        line for line in code.splitlines() if "_pipe_load_0 = cute.arch.load(" in line
    )
    assert "if mask_0 and _pipe_lane_base_0 < n else" in prefetch
    # The guard reads the lane base, so the prefetch base is not clamped.
    assert "else reduction_lane_base_1" not in code
    body = ast.unparse(_kernel_def(code))
    assert body.index("_unroll_vec_0 = _pipe_load_0") < body.index(
        "mask_1 = reduction_lane_base_1 < n"
    )


@skipUnlessBackends(["cute"])
def test_pipelined_tail_form_keeps_its_guard_and_predicates() -> None:
    # Same single-lane-iteration shape with an extent that is not a multiple
    # of V: the prefetch is guarded on its last lane and the whole-chunk
    # predicate is re-evaluated after the snapshot, before the tail select.
    x = _padded_rows()
    out = _padded_rows()
    bound, _arguments = _bind(_dynamic_kernel(_rms_norm_into), (x, out))
    code = bound.to_code(_rolled_config(bound, threads=128, vec=8, chunk=1024))
    prefetch = next(
        line for line in code.splitlines() if "_pipe_load_0 = cute.arch.load(" in line
    )
    assert "if mask_0 and _pipe_lane_base_0 + 7 < n else" in prefetch
    assert "else reduction_lane_base_1" not in code
    body = ast.unparse(_kernel_def(code))
    assert body.index("_unroll_vec_0 = _pipe_load_0") < body.index(
        "reduction_chunk_full_1 = reduction_lane_base_1 + 7 < n"
    )
    assert (
        "load = (cutlass.Uint16(_unroll_vec_0[reduction_vec_lane_1])"
        ".bitcast(cutlass.BFloat16) if reduction_chunk_full_1 else (x.iterator"
    ) in code
    assert "if not reduction_chunk_full_2 and (mask_0 and mask_1):" in code


def _pipeline(body: str, *, hint: int = 4) -> str:
    fused = pipeline_inner_loads(
        ast.parse(body).body,
        {"_REDUCTION_BLOCK_1": 1024},
        dynamic_trip_counts={"roffset_1": (hint, "_REDUCTION_TRIPS_1")},
    )
    return ast.unparse(ast.Module(body=fused, type_ignores=[]))


def _sweep(pointer: str, *predicates: str) -> str:
    lines = [
        "for roffset_1 in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(_REDUCTION_BLOCK_1)):",
        "    for reduction_lane_1 in range(1):",
        "        reduction_lane_base_1 = roffset_1 + tid * 8 + reduction_lane_1 * 1024",
        *(f"        {predicate}" for predicate in predicates),
        f"        _unroll_vec_0 = cute.arch.load({pointer}, ir.VectorType.get([8], cutlass.Uint16.mlir_type))",
        "        for reduction_vec_lane_1 in cutlass.range_constexpr(8):",
        "            acc = acc + cutlass.Float32(cutlass.Uint16(_unroll_vec_0[reduction_vec_lane_1]).bitcast(cutlass.BFloat16))",
    ]
    return "\n".join(lines) + "\n"


def test_pipeliner_clamps_unless_the_packet_guard_reads_the_lane_base() -> None:
    clamp = " < cutlass.Int32(n) else reduction_lane_base_1"
    # An unguarded packet over a symbolic end is prefetched behind a clamp
    # that compares against that end.
    code = _pipeline(_sweep("x.iterator + reduction_lane_base_1"))
    assert "_pipe_load_0 = cute.arch.load(" in code
    assert clamp in code
    # A select that only reads an outer row mask says nothing about the
    # swept extent: still clamped.
    code = _pipeline(
        _sweep("(x.iterator + reduction_lane_base_1 if mask_0 else x.iterator)")
    )
    assert "_pipe_load_0 = cute.arch.load(" in code
    assert clamp in code
    # A guard on the lane base rebases with the prefetch and needs no clamp.
    code = _pipeline(
        _sweep(
            "(x.iterator + reduction_lane_base_1 if mask_0 and reduction_lane_base_1 + 7 < n else x.iterator)"
        )
    )
    assert "_pipe_load_0 = cute.arch.load(" in code
    assert clamp not in code
    assert "if mask_0 and _pipe_lane_base_0 + 7 < n else x.iterator" in code
    # ...unless its fall-through anchor depends on the lane base too.
    code = _pipeline(
        _sweep(
            "(x.iterator + reduction_lane_base_1 if reduction_lane_base_1 + 7 < n else x.iterator + reduction_lane_base_1)"
        )
    )
    assert "_pipe_load_0 = cute.arch.load(" in code
    assert clamp in code
    # A single-trip size hint is not worth a prefetch.
    code = _pipeline(_sweep("x.iterator + reduction_lane_base_1"), hint=1)
    assert "_pipe_load_0" not in code


def test_pipeliner_keeps_predicates_the_packet_does_not_read() -> None:
    # Pure scalar predicates between the lane base and the packet are
    # re-evaluated after the snapshot; a packet reading one of them cannot be
    # prefetched (its guard would use the previous iteration's value).
    code = _pipeline(
        _sweep(
            "(x.iterator + reduction_lane_base_1 if reduction_lane_base_1 + 7 < n else x.iterator)",
            "chunk_full_1 = reduction_lane_base_1 + 7 < n",
        )
    )
    assert "_unroll_vec_0 = _pipe_load_0" in code
    assert code.index("_unroll_vec_0 = _pipe_load_0") < code.index(
        "chunk_full_1 = reduction_lane_base_1 + 7 < n"
    )
    code = _pipeline(
        _sweep(
            "(x.iterator + reduction_lane_base_1 if chunk_full_1 else x.iterator)",
            "chunk_full_1 = reduction_lane_base_1 + 7 < n",
        )
    )
    assert "_pipe_load_0" not in code


def test_fuser_sizes_and_caps_the_cache_from_a_constexpr_trip_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HELION_FUSER_MODE", raising=False)
    body = ast.parse(
        """
for roffset_1 in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(_REDUCTION_BLOCK_1)):
    for reduction_lane_1 in range(4):
        reduction_lane_base_1 = roffset_1 + tid * 2 + reduction_lane_1 * 512
        _unroll_vec_0 = cute.arch.load(x.iterator + reduction_lane_base_1, ir.VectorType.get([2], cutlass.Uint16.mlir_type))
        for reduction_vec_lane_1 in cutlass.range_constexpr(2):
            acc = acc + cutlass.Float32(cutlass.Uint16(_unroll_vec_0[reduction_vec_lane_1]).bitcast(cutlass.BFloat16))
for roffset_1 in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(_REDUCTION_BLOCK_1)):
    for reduction_lane_1 in range(4):
        reduction_lane_base_2 = roffset_1 + tid * 2 + reduction_lane_1 * 512
        _unroll_vec_1 = cute.arch.load(x.iterator + reduction_lane_base_2, ir.VectorType.get([2], cutlass.Uint16.mlir_type))
        for reduction_vec_lane_2 in cutlass.range_constexpr(2):
            out_val = cutlass.Uint16(_unroll_vec_1[reduction_vec_lane_2]).bitcast(cutlass.BFloat16)
"""
    ).body
    kwargs: dict[str, Any] = {
        "constexpr_values": {"_REDUCTION_BLOCK_1": 2048},
        "tensor_dtypes": {"x": "cutlass.BFloat16"},
        "reload_modes": {1: "register"},
    }
    untouched = fuse_two_pass_loads([*body], **kwargs)
    assert "_fuse_cache_" not in ast.unparse(
        ast.Module(body=untouched, type_ignores=[])
    )
    fused = fuse_two_pass_loads(
        [*body], dynamic_trip_counts={"roffset_1": (2, "_REDUCTION_TRIPS_1")}, **kwargs
    )
    module = ast.Module(body=fused, type_ignores=[])
    # The cache and the fused sweeps exist only while the exact fragment stays
    # within the size-hint fragment (two trips of 8 elements); otherwise the
    # original sweeps run.
    cap = "cutlass.const_expr(_REDUCTION_TRIPS_1 * 8 <= 16)"
    declaration, group = [
        node
        for node in module.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == cap
    ]
    assert [ast.unparse(stmt) for stmt in declaration.body] == [
        "_fuse_cache_0 = cute.make_rmem_tensor(_REDUCTION_TRIPS_1 * 8, cutlass.Uint16)"
    ]
    fast = "\n".join(ast.unparse(stmt) for stmt in group.body)
    fallback = "\n".join(ast.unparse(stmt) for stmt in group.orelse)
    assert fast.count("cute.arch.load(") == 1
    # The size hint is not a single-trip proof: the slot keeps its trip index.
    assert (
        "_fuse_cache_0[((roffset_1 - cutlass.Int32(0)) // cutlass.Int32(_REDUCTION_BLOCK_1) * 4 + reduction_lane_1) * 2 + 0] = _unroll_vec_0[0]"
        in fast
    )
    assert "out_val = cutlass.Uint16(_fuse_cache_0[" in fast
    assert fallback.count("cute.arch.load(") == 2
    assert "_fuse_cache_0" not in fallback
    assert "_unroll_vec_1 = cute.arch.load(" in fallback
