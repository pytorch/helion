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
from types import MethodType
from types import SimpleNamespace
from typing import Any
from typing import Callable
from typing import cast
from unittest.mock import patch

import pytest
import sympy
import torch

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.fuse_two_pass_loads import fuse_two_pass_loads
from helion._compiler.cute.memory_ops import cute_known_multiple
from helion._compiler.cute.pipeline_inner_loads import pipeline_inner_loads
from helion._compiler.cute.reduction_emitter import CuteLoopedReductionEmitter
from helion._compiler.host_function import HostFunction
from helion._compiler.reduction_strategy import LoopedReductionStrategy
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import import_example
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_native_compile
import helion.language as hl

rms_norm_batched = import_example("aot_example").rms_norm_batched

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.cpu_render(reduction_smem_budget_bytes=232448),
]

VEC_LOAD = "cute.arch.load("
VEC_STORE = "_cute_store_u16_vec(out.iterator"
TRIPS = "_REDUCTION_TRIPS_1"
# The roll iterates the same ``ceil(n / block)`` trips as ``range(0, n, block)``
# but with a trace-time constant bound the DSL can unroll.
LOOP = (
    "for roffset_1 in range(cutlass.Int32(0), "
    f"cutlass.Int32({TRIPS} * _REDUCTION_BLOCK_1), cutlass.Int32(_REDUCTION_BLOCK_1)):"
)


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
    # The pins of this file count packet loads and stores; the exact
    # division hoist (test_cute_exact_division.py) would add a redo copy of
    # every normalizing sweep.
    return helion.kernel(
        fn,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
        cute_exact_division=False,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )


def _bind(
    kernel: helion.Kernel, arguments: tuple[object, ...]
) -> tuple[Any, tuple[object, ...]]:
    """Bind on CPU; the caller keeps ``arguments`` alive while generating code."""
    return cpu_bind(kernel, arguments), arguments


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


def _assert_roll_ranges_cover_width(
    code: str, *, width: int, chunk: int, cluster_n: int
) -> None:
    rolls = [
        node
        for node in ast.walk(_kernel_def(code))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("roffset_")
    ]
    assert len(rolls) == 2
    for loop in rolls:
        assert isinstance(loop.iter, ast.Call)
        covered = []
        for rank in range(cluster_n):
            scope = {
                "__builtins__": {},
                "cutlass": SimpleNamespace(Int32=int),
                "cute": SimpleNamespace(
                    arch=SimpleNamespace(block_idx=lambda rank=rank: (0, rank, 0))
                ),
                "_REDUCTION_BLOCK_1": chunk,
                TRIPS: (width + chunk - 1) // chunk,
            }
            controls = [
                eval(compile(ast.Expression(arg), "<roll range>", "eval"), scope)
                for arg in loop.iter.args
            ]
            for offset in range(*controls):
                covered.extend(range(offset, min(offset + chunk, width)))
        assert sorted(covered) == list(range(width))


def test_guarded_cluster_keeps_single_cta_reductions_dynamic() -> None:
    width, chunk = 8192, 2048
    bound, _arguments = _bind_rms_norm(torch.empty((4, width), dtype=torch.bfloat16))
    assert "input_tensor_metadata" in bound.env.compiler_fact_specialization_facts
    plain_config = _rolled_config(bound, threads=128, vec=8, chunk=chunk, reload="gmem")
    cluster_config = helion.Config.from_dict(
        {**plain_config.config, "cute_cluster_n": 2}
    )
    with forbid_native_compile():
        plain = bound.to_code(plain_config)
        clustered = bound.to_code(cluster_config)
        plain_again = bound.to_code(plain_config)
    assert TRIPS in [arg.arg for arg in _kernel_def(plain).args.args]
    assert "_cute_grouped_reduce_cluster(" not in plain
    assert ast.dump(ast.parse(plain)) == ast.dump(ast.parse(plain_again))
    assert not bound.env.specialized_vars
    assert "_cute_grouped_reduce_cluster(" in clustered
    assert "_helion_cute_cluster_shape = (1, 2, 1)" in clustered
    assert TRIPS not in clustered
    assert "mask_1" not in clustered
    _assert_roll_ranges_cover_width(plain, width=width, chunk=chunk, cluster_n=1)
    _assert_roll_ranges_cover_width(clustered, width=width, chunk=chunk, cluster_n=2)


@pytest.mark.parametrize("width,cluster_n", [(6144, 4), (8196, 2)])
def test_guarded_cluster_declines_partial_slices(width: int, cluster_n: int) -> None:
    chunk = 2048
    bound, _arguments = _bind_rms_norm(torch.empty((4, width), dtype=torch.bfloat16))
    plain_config = _rolled_config(bound, threads=128, vec=8, chunk=chunk, reload="gmem")
    config = helion.Config.from_dict(
        {**plain_config.config, "cute_cluster_n": cluster_n}
    )
    with forbid_native_compile():
        code = bound.to_code(config)
    assert TRIPS in [arg.arg for arg in _kernel_def(code).args.args]
    assert "_cute_grouped_reduce_cluster(" not in code
    assert "_helion_cute_cluster_shape" not in code
    assert "mask_1" in code
    _assert_roll_ranges_cover_width(code, width=width, chunk=chunk, cluster_n=1)


@pytest.mark.parametrize("unbacked", [False, True])
def test_cluster_proof_refuses_unguarded_or_unbacked_extents(unbacked: bool) -> None:
    # Even an exact-looking hint must not retire the mask without a replayable
    # proof. The real specialization helper rejects both cases before using it.
    extent = sympy.Symbol("u0" if unbacked else "s0", integer=True)
    env = SimpleNamespace(
        backend=SimpleNamespace(name="cute"),
        block_sizes=[SimpleNamespace(numel=extent)],
        config_spec=SimpleNamespace(cute_indexed_reduction_block_ids=set()),
        compiler_fact_specialization_facts={"input_tensor_metadata"}
        if unbacked
        else set(),
        specialize_expr=lambda expr: expr,
    )
    env.specialized_multiple = MethodType(CompileEnvironment.specialized_multiple, env)
    fn = SimpleNamespace(
        config=helion.Config(cute_cluster_n=2),
        cute_state=SimpleNamespace(lanes=SimpleNamespace(simt_cluster_n=1)),
    )
    strategy = cast("Any", object.__new__(LoopedReductionStrategy))
    strategy._fn = lambda: fn
    strategy.block_ids = [0]
    strategy._thread_count = 128
    strategy._loop_block_size = 2048
    strategy._mask_var = "mask_0"
    strategy._cute_rolled_cluster_n = 1
    strategy._cute_rolled_cluster_checked = False
    state = SimpleNamespace(
        codegen=SimpleNamespace(
            current_grid_state=SimpleNamespace(thread_axis_sizes={})
        )
    )
    host = SimpleNamespace(device_ir=SimpleNamespace(has_atomic_ops=lambda: False))
    with (
        patch.object(CompileEnvironment, "current", return_value=env),
        patch.object(HostFunction, "current", return_value=host),
    ):
        CuteLoopedReductionEmitter(strategy)._maybe_apply_cute_rolled_cluster(state)
    assert strategy._cute_rolled_cluster_n == 1
    assert fn.cute_state.lanes.simt_cluster_n == 1
    assert strategy._mask_var == "mask_0"


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


def test_divisibility_fact_selects_between_the_two_forms() -> None:
    # A contiguous 4100-wide row is chunk-uniform for V=4 (4100 % 4 == 0) and
    # its 8200-byte row stride is 8-byte aligned, so LDG.64 packets apply.
    bound, _arguments = _bind_rms_norm(torch.empty((4096, 4100), dtype=torch.bfloat16))
    code = bound.to_code(_rolled_config(bound, threads=256, vec=4, chunk=2048))
    assert "mask_1 = reduction_lane_base_1 < n" in code
    assert "reduction_chunk_full_1" not in code
    assert "ir.VectorType.get([4], cutlass.Uint16.mlir_type)" in code
    assert VEC_STORE in code


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


@pytest.mark.parametrize("static_shapes", [False, True])
def test_misaligned_base_stays_scalar(static_shapes: bool) -> None:
    # The bound residue of the view's base pointer (2 bytes into a 16-byte
    # aligned buffer) refuses every packet width.
    x = torch.empty((64, 4104), dtype=torch.bfloat16)[:, 1:4097]
    bound, _arguments = _bind_rms_norm(x, static_shapes=static_shapes)
    for vec in (8, 4, 2):
        code = bound.to_code(_rolled_config(bound, threads=256, vec=vec, chunk=1024))
        assert VEC_LOAD not in code, vec


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


# GPU numerics for dynamic-shape rolled row reductions on the vector path.
#
# Companion to ``test_cute_dynamic_vector_reduction.py``: runs the dynamic-shape
# ``rms_norm_batched`` example under configs whose vector width divides the row
# extent (chunk-uniform mask) and padded-row configs where it does not (vector
# interior plus a per-element tail, pipelined or not), checks that inputs with
# different size, stride, or base residues bind and run separately through one
# kernel object, covers the V-misaligned strided views whose packets used to
# fault, the row-mask guard of a partially filled row tile, static extents that
# do not divide the block, rows shorter than a chunk, the trace-time fragment
# budget's fallback for long rows (also when a static group straddles the guard),
# and a max reduction.


pytest.importorskip("cutlass")


def _rms_norm_into_specialized_n(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Dynamic row count into a caller-provided output over a specialized row length."""
    m, n = x.size()
    hl.specialize(n)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        rms = torch.sqrt(torch.mean(x_tile * x_tile, dim=-1) + 1e-5)
        out[tile_m, :] = (x_tile / rms[:, None]).to(out.dtype)
    return out


def _row_max(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], device=x.device, dtype=torch.float32)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.amax(x[tile_m, :].to(torch.float32), dim=-1)
    return out


def _kernel(fn=rms_norm_batched.fn, *, static_shapes: bool = False) -> helion.Kernel:
    return helion.kernel(
        fn,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )


def _reference(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x32 = x.float()
    return (x32 / torch.sqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + eps)).to(
        x.dtype
    )


@pytest.mark.cuda
@onlyBackends(["cute"])
class TestCuteDynamicVectorReduction(TestCase):
    @pytest.mark.cuda
    @skipIfNotCUDA()
    def test_divisible_extent_matches_reference(self) -> None:
        # Study config: 256 threads, V=2, 2048-element chunks; 4096 % 2 == 0.
        x = torch.randn(64, 4096, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            _kernel(),
            (x, 1e-5),
            block_sizes=[1],
            num_threads=[1, 256],
            reduction_loops=[2048],
            cute_vector_widths=[2, 1],
            cute_reduction_reloads=["register"],
        )
        torch.testing.assert_close(out, _reference(x), atol=1e-2, rtol=1e-2)
        self.assertIn("mask_1 = reduction_lane_base_1 < n", code)
        self.assertIn("_REDUCTION_TRIPS_1: cutlass.Constexpr", code)
        self.assertIn("_cute_store_u16_vec(out.iterator", code)

    @pytest.mark.cuda
    @skipIfNotCUDA()
    def test_divisible_v8_row_is_pipelined(self) -> None:
        x = torch.randn(64, 4096, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            _kernel(),
            (x, 1e-5),
            block_sizes=[1],
            num_threads=[1, 128],
            reduction_loops=[1024],
            cute_vector_widths=[8, 1],
            cute_reduction_reloads=["register"],
        )
        torch.testing.assert_close(out, _reference(x), atol=1e-2, rtol=1e-2)
        self.assertIn("_pipe_load_0 = cute.arch.load(", code)
        self.assertIn("ir.VectorType.get([8], cutlass.Uint16.mlir_type)", code)

    @pytest.mark.cuda
    @skipIfNotCUDA()
    def test_padded_rows_use_vector_interior_and_scalar_tail(self) -> None:
        # 4100 % 8 == 4 with 16-byte aligned (4104-wide) rows: the last V=8
        # chunk of every row straddles the extent.
        x = torch.randn(64, 4104, device=DEVICE, dtype=torch.bfloat16)[:, :4100]
        out = torch.zeros(64, 4104, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            _kernel(_rms_norm_into),
            (x, out[:, :4100]),
            block_sizes=[1],
            num_threads=[1, 64],
            reduction_loops=[1024],
            cute_vector_widths=[8, 1],
            cute_reduction_reloads=["register"],
        )
        torch.testing.assert_close(result, _reference(x), atol=1e-2, rtol=1e-2)
        # The padding columns are never written.
        self.assertEqual(out[:, 4100:].abs().sum().item(), 0.0)
        self.assertIn("reduction_chunk_full_1 = reduction_lane_base_1 + 7 < n", code)
        self.assertIn("if not reduction_chunk_full_2 and (mask_0 and mask_1):", code)
        self.assertIn("_cute_store_u16_vec(out.iterator", code)

    @pytest.mark.cuda
    @skipIfNotCUDA()
    def test_rows_narrower_than_a_chunk_and_partial_grid(self) -> None:
        # Fewer fp32 elements than one 1024-wide chunk, a row count that leaves
        # the last thread block partially masked, and rows that are only
        # 8-byte aligned (1030 * 4 bytes): V=2 packets apply, V=4 must not.
        x = torch.randn(13, 1030, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            _kernel(),
            (x, 1e-5),
            block_sizes=[4],
            num_threads=[4, 64],
            reduction_loops=[1024],
            cute_vector_widths=[2, 1],
            cute_reduction_reloads=["register"],
        )
        torch.testing.assert_close(out, _reference(x), atol=1e-4, rtol=1e-4)
        self.assertIn("ir.VectorType.get([2], cutlass.Uint32.mlir_type)", code)
        code, out = code_and_output(
            _kernel(),
            (x, 1e-5),
            block_sizes=[4],
            num_threads=[4, 64],
            reduction_loops=[1024],
            cute_vector_widths=[4, 1],
            cute_reduction_reloads=["register"],
        )
        torch.testing.assert_close(out, _reference(x), atol=1e-4, rtol=1e-4)
        self.assertNotIn("cute.arch.load(", code)

    @pytest.mark.cuda
    @skipIfNotCUDA()
    def test_fp32_rows_vectorize_with_v4(self) -> None:
        x = torch.randn(32, 2048, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            _kernel(),
            (x, 1e-5),
            block_sizes=[1],
            num_threads=[1, 256],
            reduction_loops=[1024],
            cute_vector_widths=[4, 1],
            cute_reduction_reloads=["register"],
        )
        torch.testing.assert_close(out, _reference(x), atol=1e-4, rtol=1e-4)
        self.assertIn("ir.VectorType.get([4], cutlass.Uint32.mlir_type)", code)
        self.assertIn("_cute_store_u32_vec(out.iterator", code)

    @pytest.mark.cuda
    @skipIfNotCUDA()
    def test_misaligned_views_fall_back_to_scalar_and_run(self) -> None:
        # Regression: these strided views used to emit LDG.128 packets at
        # 8-byte or 2-byte aligned addresses and fault, for static shapes too.
        config = helion.Config(
            block_sizes=[1],
            num_threads=[1, 128],
            reduction_loops=[1024],
            cute_vector_widths=[8, 1],
            cute_reduction_reloads=["register"],
        )
        for static_shapes in (True, False):
            stride_view = torch.randn(64, 4100, device=DEVICE, dtype=torch.bfloat16)
            base_view = torch.randn(64, 4104, device=DEVICE, dtype=torch.bfloat16)
            for x in (stride_view[:, :4096], base_view[:, 1:4097]):
                bound = _kernel(static_shapes=static_shapes).bind((x, 1e-5))
                bound.set_config(config)
                self.assertNotIn("cute.arch.load(", bound.to_code(config))
                torch.testing.assert_close(
                    bound(x, 1e-5), _reference(x), atol=1e-2, rtol=1e-2
                )

    @pytest.mark.cuda
    @skipIfNotCUDA()
    def test_size_residue_rebinds_and_both_shapes_run(self) -> None:
        # ``n % 8`` is part of the bound cache key: 4096 (chunk-uniform V=8)
        # and 4100 (8-byte rows, scalar) bind separately and both stay
        # correct in either call order. Compiler seed guards also specialize
        # the row-size class, so 2056 uses its own bound.
        kernel = _kernel()
        config = helion.Config(
            block_sizes=[1],
            num_threads=[1, 128],
            reduction_loops=[1024],
            cute_vector_widths=[8, 1],
            cute_reduction_reloads=["register"],
        )
        bounds = []
        for width in (4096, 4100, 4096, 2056):
            x = torch.randn(8, width, device=DEVICE, dtype=torch.bfloat16)
            bound = kernel.bind((x, 1e-5))
            bound.set_config(config)
            torch.testing.assert_close(
                bound(x, 1e-5), _reference(x), atol=1e-2, rtol=1e-2
            )
            bounds.append(bound)
        self.assertIs(bounds[0], bounds[2])
        self.assertIsNot(bounds[0], bounds[3])
        self.assertIsNot(bounds[0], bounds[1])

    @pytest.mark.cuda
    @skipIfNotCUDA()
    def test_partial_row_tile_never_reads_rows_past_the_tensor(self) -> None:
        # 13 rows in tiles of 4 over a 4096-wide row that divides the block:
        # the roll needs no bounds mask, so only the row mask keeps the packets
        # of rows 13-15 off memory.  Input and output are views of larger
        # buffers whose rows 13-15 are NaN: the values a row-masked lane loads
        # are gated and its stores are predicated, so no NaN may reach rows
        # 0-12 of the output and rows 13-15 of the output buffer must stay
        # untouched.  A packet pointer that ignored the row mask would read
        # past the tensor instead, which is observable only as a fault, so the
        # pointer selection is asserted on the generated code.
        for static_shapes, fn in (
            (True, _rms_norm_into),
            (False, _rms_norm_into_specialized_n),
        ):
            big = torch.full(
                (16, 4096), float("nan"), device=DEVICE, dtype=torch.bfloat16
            )
            big[:13] = torch.randn(13, 4096, device=DEVICE, dtype=torch.bfloat16)
            out_buffer = torch.full(
                (16, 4096), float("nan"), device=DEVICE, dtype=torch.bfloat16
            )
            x = big[:13]
            code, out = code_and_output(
                _kernel(fn, static_shapes=static_shapes),
                (x, out_buffer[:13]),
                block_sizes=[4],
                num_threads=[4, 64],
                reduction_loops=[1024],
                cute_vector_widths=[8, 1],
                cute_reduction_reloads=["register"],
            )
            packet_loads = [
                line for line in code.splitlines() if "cute.arch.load(" in line
            ]
            self.assertTrue(packet_loads)
            for line in packet_loads:
                self.assertIn(" if mask_0 else x.iterator + cutlass.Int32(0)", line)
            self.assertNotIn("mask_1", code)
            self.assertTrue(torch.isfinite(out).all().item())
            torch.testing.assert_close(out, _reference(x), atol=1e-2, rtol=1e-2)
            self.assertTrue(torch.isnan(out_buffer[13:]).all().item())

    @pytest.mark.cuda
    @skipIfNotCUDA()
    def test_static_extents_not_a_multiple_of_the_block_take_the_chunk_forms(
        self,
    ) -> None:
        # Static row lengths that do not divide the block used to keep every
        # access scalar; they now take the chunk-level forms.
        x = torch.randn(64, 4100, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            _kernel(static_shapes=True),
            (x, 1e-5),
            block_sizes=[1],
            num_threads=[1, 256],
            reduction_loops=[2048],
            cute_vector_widths=[4, 1],
            cute_reduction_reloads=["register"],
        )
        self.assertIn("mask_1 = reduction_lane_base_1 < 4100", code)
        self.assertIn("ir.VectorType.get([4], cutlass.Uint16.mlir_type)", code)
        torch.testing.assert_close(out, _reference(x), atol=1e-2, rtol=1e-2)
        x = torch.randn(64, 4104, device=DEVICE, dtype=torch.bfloat16)[:, :4100]
        padded = torch.zeros(64, 4104, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            _kernel(_rms_norm_into, static_shapes=True),
            (x, padded[:, :4100]),
            block_sizes=[1],
            num_threads=[1, 64],
            reduction_loops=[1024],
            cute_vector_widths=[8, 1],
            cute_reduction_reloads=["register"],
        )
        self.assertIn("reduction_chunk_full_1 = reduction_lane_base_1 + 7 < 4100", code)
        self.assertIn("if not reduction_chunk_full_2 and", code)
        torch.testing.assert_close(result, _reference(x), atol=1e-2, rtol=1e-2)
        self.assertEqual(padded[:, 4100:].abs().sum().item(), 0.0)

    @pytest.mark.cuda
    @skipIfNotCUDA()
    def test_fragment_cap_falls_back_to_reloads_for_long_rows(self) -> None:
        # Bound at n=4096 (two trips) the register cache holds TRIPS * 32
        # elements under a trace-time budget of the 64-element size-hint
        # fragment.  The same bound kernel run at n=73728 (36 trips) takes the
        # pre-fusion sweeps instead of growing the dynamically indexed local
        # array past the footprint the binding was admitted with.
        kernel = _kernel()
        config = helion.Config(
            block_sizes=[1],
            num_threads=[1, 64],
            reduction_loops=[2048],
            cute_vector_widths=[8, 1],
            cute_reduction_reloads=["register"],
        )
        x = torch.randn(4, 4096, device=DEVICE, dtype=torch.bfloat16)
        bound = kernel.bind((x, 1e-5))
        bound.set_config(config)
        code = bound.to_code(config)
        self.assertIn("if cutlass.const_expr(_REDUCTION_TRIPS_1 * 32 <= 64):", code)
        self.assertIn(
            "_fuse_cache_0 = cute.make_rmem_tensor(_REDUCTION_TRIPS_1 * 32, cutlass.Uint16)",
            code,
        )
        torch.testing.assert_close(bound(x, 1e-5), _reference(x), atol=1e-2, rtol=1e-2)
        long_x = torch.randn(4, 73728, device=DEVICE, dtype=torch.bfloat16)
        self.assertIsNot(kernel.bind((long_x, 1e-5)), bound)
        torch.testing.assert_close(
            bound(long_x, 1e-5), _reference(long_x), atol=1e-2, rtol=1e-2
        )

    @pytest.mark.cuda
    @skipIfNotCUDA()
    def test_static_group_straddling_the_guard_runs_whole_in_both_branches(
        self,
    ) -> None:
        # The static ``k`` group populates its cache between the rolled ``n``
        # sweeps and consumes it after them.  The guard covers the whole ``k``
        # group, so the fallback (8 or 72 trips over the four-trip budget of
        # the 4096 binding) reloads both groups instead of consuming a cache
        # that only the fused branch fills.
        kernel = _kernel(_dynamic_rows_between_static_column_sweeps)
        y = torch.rand(64, 256, device=DEVICE, dtype=torch.bfloat16) + 0.5
        x = torch.randn(64, 4096, device=DEVICE, dtype=torch.bfloat16)
        bound = kernel.bind((x, y))
        config = _straddle_config(bound)
        bound.set_config(config)
        code = bound.to_code(config)
        self.assertIn("if cutlass.const_expr(_REDUCTION_TRIPS_2 * 32 <= 128):", code)
        self.assertIn(
            "_fuse_cache_1 = cute.make_rmem_tensor(64, cutlass.BFloat16)", code
        )
        y32 = y.float()
        expected_y = (y32 / y32.sum(dim=-1, keepdim=True)).to(y.dtype)

        def check(x: torch.Tensor) -> None:
            out_x, out_y = bound(x, y)
            x32 = x.float()
            expected_x = (x32 / (x32 * x32).sum(dim=-1, keepdim=True)).to(x.dtype)
            torch.testing.assert_close(
                out_x.float(), expected_x.float(), atol=1e-7, rtol=2e-2
            )
            torch.testing.assert_close(
                out_y.float(), expected_y.float(), atol=1e-6, rtol=2e-2
            )

        # The binding shape takes the fused branch.  A compiler seed of this
        # nested tile nest keys exact input metadata, so ``kernel.bind`` would
        # rebind longer rows; the bound kernel itself is compiled for dynamic
        # shapes and every fact behind its code (size residues, alignment,
        # disjoint storage) still holds, so it is launched directly and takes
        # the fallback.
        check(x)
        bound_key = kernel.specialization_key((x, y))
        for width in (8192, 73728):
            x = torch.randn(64, width, device=DEVICE, dtype=torch.bfloat16)
            (changed,) = [
                entry
                for entry, other in zip(
                    bound_key, kernel.specialization_key((x, y)), strict=True
                )
                if entry != other
            ]
            self.assertEqual(changed[0], "compiler_seed_results")
            check(x)

    @pytest.mark.cuda
    @skipIfNotCUDA()
    def test_rows_shorter_than_a_chunk_reuse_the_bound_kernel(self) -> None:
        # A 520-wide row is shorter than the 1024 block a 4096-wide binding
        # chose: one partial trip whose chunk predicates carry the whole roll.
        # 520 shares the ``n % 8 == 0`` residue (chunk-level mask); 516 of a
        # 520-wide buffer shares the residue-4, aligned-stride class of 4100 of
        # 4104 (vector interior with a per-element tail).
        kernel = _kernel(_rms_norm_into)
        config = helion.Config(
            block_sizes=[4],
            num_threads=[4, 64],
            reduction_loops=[1024],
            cute_vector_widths=[8, 1],
            cute_reduction_reloads=["register"],
        )
        for width, stride, short_width, short_stride, predicate in (
            (4096, 4096, 520, 520, "mask_1 = reduction_lane_base_1 < n"),
            (
                4100,
                4104,
                516,
                520,
                "reduction_chunk_full_1 = reduction_lane_base_1 + 7 < n",
            ),
        ):
            x = torch.randn(13, stride, device=DEVICE, dtype=torch.bfloat16)[:, :width]
            out = torch.zeros(13, stride, device=DEVICE, dtype=torch.bfloat16)
            bound = kernel.bind((x, out[:, :width]))
            bound.set_config(config)
            self.assertIn(predicate, bound.to_code(config))
            torch.testing.assert_close(
                bound(x, out[:, :width]), _reference(x), atol=1e-2, rtol=1e-2
            )
            short = torch.randn(13, short_stride, device=DEVICE, dtype=torch.bfloat16)[
                :, :short_width
            ]
            short_out = torch.zeros(
                13, short_stride, device=DEVICE, dtype=torch.bfloat16
            )
            self.assertIsNot(kernel.bind((short, short_out[:, :short_width])), bound)
            torch.testing.assert_close(
                bound(short, short_out[:, :short_width]),
                _reference(short),
                atol=1e-2,
                rtol=1e-2,
            )
            self.assertEqual(short_out[:, short_width:].abs().sum().item(), 0.0)

    @pytest.mark.cuda
    @skipIfNotCUDA()
    def test_pipelined_tail_form_matches_reference(self) -> None:
        # 128 threads x V=8 cover the 1024 chunk in one lane iteration, so the
        # tail form is software pipelined: the prefetch is guarded on its last
        # lane and the whole-chunk predicate follows the snapshot.
        x = torch.randn(64, 4104, device=DEVICE, dtype=torch.bfloat16)[:, :4100]
        padded = torch.zeros(64, 4104, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            _kernel(_rms_norm_into),
            (x, padded[:, :4100]),
            block_sizes=[1],
            num_threads=[1, 128],
            reduction_loops=[1024],
            cute_vector_widths=[8, 1],
            cute_reduction_reloads=["register"],
        )
        self.assertIn("_pipe_load_0 = cute.arch.load(", code)
        self.assertIn("if mask_0 and _pipe_lane_base_0 + 7 < n else", code)
        self.assertIn("reduction_chunk_full_1 = reduction_lane_base_1 + 7 < n", code)
        torch.testing.assert_close(result, _reference(x), atol=1e-2, rtol=1e-2)
        self.assertEqual(padded[:, 4100:].abs().sum().item(), 0.0)

    @pytest.mark.cuda
    @skipIfNotCUDA()
    def test_alignment_facts_rebind_the_same_kernel(self) -> None:
        # One kernel object sees an aligned input (vector path), a view whose
        # row stride is not a multiple of V, and a view whose base is 2 bytes
        # off; each alignment class binds separately, the misaligned ones stay
        # scalar, and the aligned input returns to its original binding.
        kernel = _kernel()
        config = helion.Config(
            block_sizes=[1],
            num_threads=[1, 128],
            reduction_loops=[1024],
            cute_vector_widths=[8, 1],
            cute_reduction_reloads=["register"],
        )
        aligned = torch.randn(64, 4096, device=DEVICE, dtype=torch.bfloat16)
        stride_view = torch.randn(64, 4100, device=DEVICE, dtype=torch.bfloat16)[
            :, :4096
        ]
        base_view = torch.randn(64, 4104, device=DEVICE, dtype=torch.bfloat16)[
            :, 1:4097
        ]
        bounds = []
        vectorized = []
        for x in (aligned, stride_view, base_view, aligned):
            bound = kernel.bind((x, 1e-5))
            bound.set_config(config)
            bounds.append(bound)
            vectorized.append("cute.arch.load(" in bound.to_code(config))
            torch.testing.assert_close(
                bound(x, 1e-5), _reference(x), atol=1e-2, rtol=1e-2
            )
        self.assertEqual(vectorized, [True, False, False, True])
        self.assertIs(bounds[0], bounds[3])
        self.assertEqual(len({id(bound) for bound in bounds[:3]}), 3)

    @pytest.mark.cuda
    @skipIfNotCUDA()
    def test_row_max_reduction_matches_reference(self) -> None:
        # A max reduction over both chunk forms; every element is negative so
        # a masked lane leaking its zero placeholder would win the max.
        kernel = _kernel(_row_max)
        for x, predicate in (
            (
                torch.randn(13, 4096, device=DEVICE, dtype=torch.bfloat16) - 5.0,
                "mask_1 = reduction_lane_base_1 < n",
            ),
            (
                (torch.randn(13, 4104, device=DEVICE, dtype=torch.bfloat16) - 5.0)[
                    :, :4100
                ],
                "reduction_chunk_full_1 = reduction_lane_base_1 + 7 < n",
            ),
        ):
            bound = kernel.bind((x,))
            config = _rolled_config(bound, rows=4, threads=64, vec=8, chunk=1024)
            bound.set_config(config)
            code = bound.to_code(config)
            self.assertIn("ir.VectorType.get([8], cutlass.Uint16.mlir_type)", code)
            self.assertIn(predicate, code)
            torch.testing.assert_close(bound(x), x.float().amax(dim=-1))
