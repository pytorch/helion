"""Schedule knobs of the CuTe register tile (CPU codegen).

``cute_register_tile_warp_fold`` folds the multi-warp shared-memory column
reduce within each warp first (``_cute_grouped_reduce_shared_columns(...,
fold='warp')``); ``cute_register_tile_store_first`` emits the per-lane consume
nest (the state stores) before the reductions' cross-thread combines.  Both
are off by default and render the knob-free schedule byte for byte; a knob the
tile cannot honour is ignored (see ``RegisterTileOptions``).
"""

from __future__ import annotations

import ast
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import cast

import pytest
import torch

from test._cute_register_tile_kernels import col_scale
from test._cute_register_tile_kernels import col_stats
from test._cute_register_tile_kernels import col_sum
from test._cute_register_tile_kernels import column_config

import helion
from helion._compiler.cute.autotune.heuristics.cute import CuteRegisterTileHeuristic
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.register_tile_admission import RegisterTileUnsupported
from helion._testing import DEVICE
from helion._testing.cute_codegen import cpu_bind
from helion.experimental.linear_attention import recurrent_step_correction_fused
from helion.experimental.linear_attention import recurrent_step_fused

if TYPE_CHECKING:
    from helion._compiler.generate_ast import GenerateAST

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.cpu_render(reduction_smem_budget_bytes=232448),
]

BH, D, DV = 8, 128, 128
COLUMN_REDUCE = "_cute_grouped_reduce_shared_columns("
WARP_FOLD = "fold='warp'"
STATE_STORE = "_cute_store_u32_vec(new_state.iterator"
KNOBS: dict[str, Any] = {
    "cute_register_tile_warp_fold": True,
    "cute_register_tile_store_first": True,
}


def _bind() -> tuple[Any, tuple[torch.Tensor, ...]]:
    """Bind the fused recurrent step on CPU; the caller keeps ``arguments``
    alive while generating code (the alignment proof reads the live inputs)."""
    kernel = helion.kernel(
        recurrent_step_fused.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    arguments = (
        torch.empty((BH, D), dtype=torch.float32),
        torch.empty((BH, D), dtype=torch.float32),
        torch.empty((BH, DV), dtype=torch.float32),
        torch.empty((BH, D, DV), dtype=torch.float32),
        torch.empty((BH,), dtype=torch.float32),
    )
    return cpu_bind(kernel, arguments), arguments


def _config(
    bound: Any,
    *,
    dv_threads: int,
    reduction_threads: int,
    dv_block: int = 32,
    vec: int = 4,
    **extra: Any,
) -> helion.Config:
    spec = bound.config_spec
    (reduction,) = [
        block.block_id for block in bound.env.block_sizes if block.reduction
    ]
    (dv,) = spec.block_sizes.valid_block_ids()
    threads = {dv: dv_threads, reduction: reduction_threads}
    return spec.normalized_config(
        helion.Config(
            block_sizes=[dv_block],
            num_threads=[
                threads.get(block_id, 0)
                for block_id in spec.num_threads.valid_block_ids()
            ],
            reduction_loops=[None],
            cute_vector_widths=[
                vec if block_id == dv else 1
                for block_id in spec.cute_vector_widths.valid_block_ids()
            ],
            **extra,
        )
    )


def _kernel_statements(code: str) -> list[str]:
    """The kernel body's top-level statements, unparsed, in program order."""
    module = ast.parse(code)
    (kernel,) = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    ]
    return [ast.unparse(node) for node in kernel.body]


def _first_statement(statements: list[str], needle: str) -> int:
    return next(i for i, source in enumerate(statements) if needle in source)


def _owner_loops(statements: list[str]) -> list[int]:
    return [
        i
        for i, source in enumerate(statements)
        if source.startswith("for synthetic_lane_")
    ]


def test_knobs_off_render_the_knob_free_schedule() -> None:
    bound, _arguments = _bind()
    knob_free = bound.to_code(_config(bound, dv_threads=8, reduction_threads=16))
    explicit_off = bound.to_code(
        _config(
            bound,
            dv_threads=8,
            reduction_threads=16,
            cute_register_tile_warp_fold=False,
            cute_register_tile_store_first=False,
        )
    )
    assert explicit_off == knob_free
    assert WARP_FOLD not in knob_free
    statements = _kernel_statements(knob_free)
    # Knob-free order: accumulate nest, combine, consume nest (state store).
    assert _first_statement(statements, COLUMN_REDUCE) < _first_statement(
        statements, STATE_STORE
    )


def test_warp_fold_requests_the_fold_of_the_multi_warp_column_reduce() -> None:
    bound, _arguments = _bind()
    code = bound.to_code(
        _config(
            bound,
            dv_threads=8,
            reduction_threads=16,
            cute_register_tile_warp_fold=True,
        )
    )
    assert code.count(COLUMN_REDUCE) == 1
    assert f"pre=8, group_span=128, group_count=1, {WARP_FOLD})" in code
    statements = _kernel_statements(code)
    # The fold alone keeps the combine ahead of the consume nest.
    assert _first_statement(statements, COLUMN_REDUCE) < _first_statement(
        statements, STATE_STORE
    )
    # Everything else is the knob-free render.
    knob_free = bound.to_code(_config(bound, dv_threads=8, reduction_threads=16))
    assert code.replace(f", {WARP_FOLD})", ")") == knob_free


def test_store_first_emits_the_consume_nest_before_the_combine() -> None:
    bound, _arguments = _bind()
    code = bound.to_code(
        _config(
            bound,
            dv_threads=8,
            reduction_threads=8,
            cute_register_tile_store_first=True,
        )
    )
    statements = _kernel_statements(code)
    accumulate, consume = _owner_loops(statements)
    combine = _first_statement(statements, COLUMN_REDUCE)
    out_store = _first_statement(statements, "_cute_store_u32_vec(out.iterator")
    assert accumulate < consume < combine < out_store
    assert STATE_STORE in statements[consume]
    assert "_lane_stash" in statements[accumulate]
    assert WARP_FOLD not in code
    # The same statements as the knob-free render, reordered.
    knob_free = _kernel_statements(
        bound.to_code(_config(bound, dv_threads=8, reduction_threads=8))
    )
    assert sorted(statements) == sorted(knob_free)
    assert statements != knob_free


def test_both_knobs_compose() -> None:
    bound, _arguments = _bind()
    code = bound.to_code(_config(bound, dv_threads=8, reduction_threads=16, **KNOBS))
    statements = _kernel_statements(code)
    _accumulate, consume = _owner_loops(statements)
    assert consume < _first_statement(statements, COLUMN_REDUCE)
    assert f"pre=8, group_span=128, group_count=1, {WARP_FOLD})" in code


def test_store_first_keeps_the_combine_first_when_a_store_needs_the_result() -> None:
    # ``y = x * (1 / (sum(x * x) + 1))``: the per-lane store reads the
    # reduction result, so the consume nest cannot move ahead of the combine
    # and the knob renders the knob-free schedule.
    arguments = (torch.empty((128, 1024), dtype=torch.float32),)
    bound = cpu_bind(col_scale, arguments)
    knob_free = bound.to_code(
        column_config(bound, block=32, tile_threads=8, reduction_threads=8)
    )
    code = bound.to_code(
        column_config(
            bound,
            block=32,
            tile_threads=8,
            reduction_threads=8,
            cute_register_tile_store_first=True,
        )
    )
    assert code == knob_free
    statements = _kernel_statements(code)
    assert _first_statement(statements, COLUMN_REDUCE) < _first_statement(
        statements, "_cute_store_u32_vec(y.iterator"
    )


def test_warp_fold_is_ignored_by_the_consecutive_lane_column_reduce() -> None:
    # One vector thread under 64 consecutive reduction threads (``pre == 1``):
    # that branch of the helper already folds each warp with a shuffle, so the
    # knob adds nothing and the render is the knob-free one.
    arguments = (torch.empty((128, 1024), dtype=torch.float32),)
    bound = cpu_bind(col_sum, arguments)
    knob_free = bound.to_code(
        column_config(bound, block=4, tile_threads=1, reduction_threads=64)
    )
    code = bound.to_code(
        column_config(
            bound,
            block=4,
            tile_threads=1,
            reduction_threads=64,
            cute_register_tile_warp_fold=True,
        )
    )
    assert code == knob_free
    assert "pre=1, group_span=64" in code


def test_knobs_are_dropped_without_a_register_tile() -> None:
    bound, _arguments = _bind()
    spec = bound.config_spec
    looped = spec.normalized_config(
        helion.Config(block_sizes=[32], reduction_loops=[64], **KNOBS)
    )
    assert looped.config["cute_register_tile_warp_fold"] is False
    assert looped.config["cute_register_tile_store_first"] is False
    # The rolled-lane retry re-normalizes with the register tile withheld.
    tile = _config(bound, dv_threads=8, reduction_threads=16, **KNOBS)
    assert tile.config["cute_register_tile_warp_fold"] is True
    retried = spec.normalized_config(tile, _cute_register_tiles=False)
    assert retried.config["cute_register_tile_warp_fold"] is False
    assert retried.config["cute_register_tile_store_first"] is False
    assert {
        key: value for key, value in retried.config.items() if key not in KNOBS
    } == {key: value for key, value in tile.config.items() if key not in KNOBS}


def test_seed_alternates_carry_the_knobs_and_the_finer_tile() -> None:
    bound, _arguments = _bind()
    spec = bound.config_spec
    env = bound.env
    device_ir = bound.host_function.device_ir
    primary = CuteRegisterTileHeuristic.get_seed_config(env, device_ir)
    seeds = CuteRegisterTileHeuristic.get_seed_configs(env, device_ir)
    assert primary is not None and seeds is not None
    assert seeds[0].config == primary.config
    assert "cute_register_tile_warp_fold" not in primary.config
    (reduction,) = [
        block.block_id for block in bound.env.block_sizes if block.reduction
    ]
    thread_ids = spec.num_threads.valid_block_ids()
    reduction_threads = [
        seed.num_threads[thread_ids.index(reduction)] for seed in seeds
    ]
    assert reduction_threads == [8, 8, 16]
    for seed in seeds[1:]:
        assert seed.config["cute_register_tile_warp_fold"] is True
        assert seed.config["cute_register_tile_store_first"] is True
    # The loader plants the alternates, which makes both knobs searchable.
    planted = [
        seed
        for seed in spec.compiler_seed_configs
        if seed.config.get("cute_register_tile_warp_fold") is True
    ]
    assert len(planted) == 2
    fields = spec._flat_fields()
    for key in KNOBS:
        assert fields[key].search_choices == (False, True)
    # The autotune-off default is still the knob-free primary.
    default = spec.default_config()
    assert default.config.get("cute_register_tile_warp_fold") in (None, False)
    assert WARP_FOLD not in bound.to_code(default)


# ---- two reduction levels: the delta-rule correction step -------------------

CORRECTION_OWNER = "for synthetic_lane_2 in cutlass.range_constexpr("


def _bind_correction(*, diagonal: bool = True) -> tuple[Any, tuple[torch.Tensor, ...]]:
    """Bind the delta-rule step on CPU (``alpha`` a ``[BH, D]`` row by
    default, as the kda decode step calls it)."""
    kernel = helion.kernel(
        recurrent_step_correction_fused.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    arguments = (
        torch.empty((BH, D), dtype=torch.float32),
        torch.empty((BH, D), dtype=torch.float32),
        torch.empty((BH, DV), dtype=torch.float32),
        torch.empty((BH, D, DV), dtype=torch.float32),
        torch.empty((BH, D) if diagonal else (BH,), dtype=torch.float32),
        torch.empty((BH,), dtype=torch.float32),
    )
    return cpu_bind(kernel, arguments), arguments


def _combines(statements: list[str]) -> list[int]:
    return [i for i, source in enumerate(statements) if COLUMN_REDUCE in source]


def test_correction_step_knob_free_schedule_stores_behind_the_last_combine() -> None:
    # ``kts = sum(k * s)`` feeds the state that ``o = sum(q * s)`` reduces:
    # the first nest loads the tile once and accumulates ``kts``; behind its
    # combine the second nest applies the correction (stashing the corrected
    # tile) and accumulates ``o``; the state store nest reads that stash
    # behind the ``o`` combine; the output tail comes last.
    bound, _arguments = _bind_correction()
    code = bound.to_code(_config(bound, dv_threads=8, reduction_threads=16))
    assert code.count("cute.arch.load(state.iterator") == 1
    assert code.count(STATE_STORE) == 1
    assert WARP_FOLD not in code
    statements = _kernel_statements(code)
    first, second, consume = _owner_loops(statements)
    kts_combine, o_combine = _combines(statements)
    out_store = _first_statement(statements, "_cute_store_u32_vec(out.iterator")
    assert first < kts_combine < second < o_combine < consume < out_store
    assert "cute.arch.load(" in statements[first]
    assert "cute.arch.load(" not in statements[second]
    assert "kts_lane_results[" in statements[second]
    assert "_lane_stash.append(" in statements[second]
    assert STATE_STORE in statements[consume]
    assert "_lane_stash[" in statements[consume]
    assert "kts" not in statements[consume]
    assert "o_vec_lane_results[" in statements[out_store]


def test_correction_step_store_first_joins_the_second_accumulate_nest() -> None:
    # ``store_first``: the state store needs ``kts`` but not ``o``, so it
    # joins the level-1 nest ahead of the ``o`` combine and the corrected tile
    # is never stashed; the warp fold applies to both combines.
    bound, _arguments = _bind_correction()
    code = bound.to_code(
        _config(
            bound,
            dv_threads=8,
            reduction_threads=16,
            cute_register_tile_store_first=True,
        )
    )
    statements = _kernel_statements(code)
    first, second = _owner_loops(statements)
    kts_combine, o_combine = _combines(statements)
    assert first < kts_combine < second < o_combine
    assert STATE_STORE in statements[second]
    assert "kts_lane_results[" in statements[second]
    assert "_lane_stash.append(" not in statements[second]
    folded = bound.to_code(_config(bound, dv_threads=8, reduction_threads=16, **KNOBS))
    assert folded.count(WARP_FOLD) == 2
    assert folded.replace(f", {WARP_FOLD})", ")") == code


def test_correction_step_default_is_the_two_level_register_tile() -> None:
    # The chained body seeds half the lanes per thread (16 reduction threads
    # over 128 rows) and that seed is the autotune-off default.
    bound, _arguments = _bind_correction()
    code = bound.to_code(bound.config_spec.default_config())
    assert "block=(8, 16, 1)" in code
    assert code.count(CORRECTION_OWNER) == 3
    assert code.count(COLUMN_REDUCE) == 2


def test_register_budget_counts_distinct_values() -> None:
    # Sixteen lanes x 4 columns per thread: the decayed tile and ``beta * k
    # * v`` are held once per element (64 each), ``beta * k`` and ``q`` once
    # per lane (16 each; the unrolled duplicates merge in the compiler), so
    # the tile fits the budget that a per-lane-per-element count (256) would
    # have exceeded, and the eight-thread geometry forms the register tile.
    bound, _arguments = _bind_correction()
    code = bound.to_code(_config(bound, dv_threads=8, reduction_threads=8, **KNOBS))
    assert "_lane_stash" in code
    assert "block=(8, 8, 1)" in code
    assert code.count(CORRECTION_OWNER) == 2


def test_rolled_fallback_passes_through_a_fragment_capacity_decline() -> None:
    # Once the computed-fragment owner has declined a root for capacity, the
    # generic lowering's refusals report the overflow as InvalidConfig.  The
    # register tile's regeneration signal is not a refusal: it must still
    # reach ``generate_ast``'s rolled retry unchanged.
    bound, _arguments = _bind()
    codegen = cast(
        "GenerateAST",
        SimpleNamespace(
            device_function=SimpleNamespace(
                cute_state=SimpleNamespace(
                    root_plans=SimpleNamespace(
                        fragment_capacity_decline=(262144, 232448)
                    )
                )
            )
        ),
    )
    failure = RegisterTileUnsupported("a load inside a post-combine value")
    refusal = helion.exc.BackendUnsupported("cute", "lane restores")
    with bound.env:
        assert CuteBackend().explain_root_failure(codegen, failure) is failure
        converted = CuteBackend().explain_root_failure(codegen, refusal)
    assert isinstance(converted, helion.exc.InvalidConfig)
    assert "262144" in str(converted) and "232448" in str(converted)


def _packed_pid_axis(code: str) -> tuple[int, int]:
    """``(axis, pack)`` of the pid that carries the packed sub-group."""
    (match,) = re.finditer(
        r"pid_\d+ = .* \* (\d+) \+ cutlass\.Int32\(cute\.arch\.thread_idx\(\)\[(\d)\]\)$",
        code,
        flags=re.MULTILINE,
    )
    return int(match.group(2)), int(match.group(1))


def _launch_block(code: str) -> tuple[int, ...]:
    (match,) = re.finditer(r"block=\((\d+), (\d+), (\d+)\)", code)
    return tuple(int(dim) for dim in match.groups())


@pytest.mark.parametrize("pack", [2, 4])
@pytest.mark.parametrize(
    "reduction_loop", [None, 32], ids=["register-tile", "looped-reduction"]
)
def test_tiles_per_block_raises_the_sub_group_axis(
    pack: int, reduction_loop: int | None
) -> None:
    # The BH axis is one element per tile and the body only reads
    # ``tile_bh.id``: no graph distributes over it, so the strategy marks it
    # inactive.  Packing still makes it a thread axis (the pid varies per
    # sub-group): the knob stays on the surface, the pid carries the
    # sub-group on an axis of its own, and the launch raises that axis.
    # Before the fix the pid read an axis the launch never raised (or the
    # axis the DV tile owned), and every sub-group but one wrote garbage.
    bound, _arguments = _bind()
    spec = bound.config_spec
    (bh,) = [block.block_id for block in bound.env.block_sizes if block.size == BH]
    assert spec.cute_tile_pack_counts == {bh: BH}
    assert "cute_tiles_per_block" in spec.default_config().config
    extra: dict[str, Any] = {**KNOBS, "cute_tiles_per_block": pack}
    if reduction_loop is None:
        config = _config(bound, dv_threads=8, reduction_threads=8, **extra)
    else:
        (reduction,) = [
            block.block_id for block in bound.env.block_sizes if block.reduction
        ]
        (dv,) = spec.block_sizes.valid_block_ids()
        threads = {dv: 4, reduction: 8}
        config = spec.normalized_config(
            helion.Config(
                block_sizes=[4],
                num_threads=[
                    threads.get(block_id, 0)
                    for block_id in spec.num_threads.valid_block_ids()
                ],
                reduction_loops=[reduction_loop],
                cute_tiles_per_block=pack,
            )
        )
    assert config.config["cute_tiles_per_block"] == pack
    code = bound.to_code(config)
    assert ("_lane_stash" in code) == (reduction_loop is None)
    axis, rendered_pack = _packed_pid_axis(code)
    assert rendered_pack == pack
    block = _launch_block(code)
    assert block[axis] == pack
    # The DV tile and the D reduction index on axes of their own (a linear
    # lane of a cross-thread reduction legitimately folds every axis in).
    other_axes = {
        int(match)
        for line in code.splitlines()
        if re.match(r"\s*(indices|lane_base|rindex)_\d+ = ", line)
        for match in re.findall(r"cute\.arch\.thread_idx\(\)\[(\d)\]", line)
    }
    assert other_axes and axis not in other_axes, (axis, other_axes)
    dv_block = 32 if reduction_loop is None else 4
    assert f"({BH // pack * (DV // dv_block)},)" in code


# GPU numerics of the register-tile schedule knobs (see
# ``test_cute_register_tile_schedule.py`` for the codegen shape).


BH_DEVICE = 256


def _check_recurrent_step(
    *,
    dtype: torch.dtype,
    dv_block: int,
    dv_threads: int,
    reduction_threads: int,
    vec: int = 4,
    **extra: Any,
) -> str:
    """Run the fused recurrent step on two input sets (a stale output or new
    state cannot pass twice) and return the generated code."""
    kernel = helion.kernel(
        recurrent_step_fused.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    torch.manual_seed(0)
    q = torch.randn(BH_DEVICE, D, device=DEVICE).to(dtype)
    k = torch.randn(BH_DEVICE, D, device=DEVICE).to(dtype)
    v = torch.randn(BH_DEVICE, DV, device=DEVICE).to(dtype)
    alpha = torch.rand(BH_DEVICE, device=DEVICE).to(dtype)
    initial = torch.randn(BH_DEVICE, D, DV, device=DEVICE).to(dtype)
    state = initial.clone()
    bound = kernel.bind((q, k, v, state, alpha))
    config = _config(
        bound,
        dv_block=dv_block,
        dv_threads=dv_threads,
        reduction_threads=reduction_threads,
        vec=vec,
        **extra,
    )
    code = bound.to_code(config)
    assert "_lane_stash" in code
    compiled = bound.compile_config(config)
    rtol, atol = (1e-3, 1e-3) if dtype is torch.float32 else (2e-2, 2e-1)
    for _ in range(2):
        out, new_state = compiled(q, k, v, state, alpha)
        torch.cuda.synchronize()
        expected_state = (
            alpha.float()[:, None, None] * initial.float()
            + k.float()[:, :, None] * v.float()[:, None, :]
        )
        expected_out = (q.float()[:, :, None] * expected_state).sum(1)
        torch.testing.assert_close(
            new_state.float(), expected_state, rtol=rtol, atol=atol
        )
        torch.testing.assert_close(out.float(), expected_out, rtol=rtol, atol=atol)
        assert torch.equal(state, initial)
        q.copy_(torch.rand_like(q, dtype=torch.float32).mul_(2).sub_(1).to(dtype))
        v.copy_(torch.rand_like(v, dtype=torch.float32).mul_(2).sub_(1).to(dtype))
    return code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("dv_block", "dv_threads", "reduction_threads", "extra"),
    (
        pytest.param(32, 8, 8, KNOBS, id="two-warps-both-knobs"),
        pytest.param(32, 8, 16, KNOBS, id="four-warps-both-knobs"),
        pytest.param(32, 8, 32, KNOBS, id="eight-warps-both-knobs"),
        pytest.param(64, 16, 8, KNOBS, id="wide-tile-both-knobs"),
        pytest.param(
            32,
            8,
            16,
            {"cute_register_tile_warp_fold": True},
            id="four-warps-warp-fold",
        ),
        pytest.param(
            32,
            8,
            16,
            {"cute_register_tile_store_first": True},
            id="four-warps-store-first",
        ),
        pytest.param(
            32,
            8,
            16,
            {**KNOBS, "pid_type": "persistent_blocked"},
            id="four-warps-both-knobs-persistent",
        ),
    ),
)
def test_schedule_knobs_match_reference(
    dv_block: int, dv_threads: int, reduction_threads: int, extra: dict[str, Any]
) -> None:
    code = _check_recurrent_step(
        dtype=torch.float32,
        dv_block=dv_block,
        dv_threads=dv_threads,
        reduction_threads=reduction_threads,
        **extra,
    )
    assert code.count(COLUMN_REDUCE) == 1
    assert (WARP_FOLD in code) == bool(extra.get("cute_register_tile_warp_fold"))
    store_first = bool(extra.get("cute_register_tile_store_first"))
    assert (code.index(STATE_STORE) < code.index(COLUMN_REDUCE)) == store_first


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
def test_bf16_register_tile_with_both_knobs() -> None:
    code = _check_recurrent_step(
        dtype=torch.bfloat16,
        dv_block=64,
        dv_threads=8,
        reduction_threads=16,
        vec=8,
        **KNOBS,
    )
    assert WARP_FOLD in code
    assert code.count("_cute_store_u16_vec(new_state.iterator") == 1


def _col_sum_reference(x: torch.Tensor) -> torch.Tensor:
    return x.sum(0)


def _col_scale_reference(x: torch.Tensor) -> torch.Tensor:
    return x / ((x * x).sum(0) + 1.0)[None, :]


def _col_stats_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([x.sum(0), (x * x).sum(0)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("kernel", "reference", "reduction_threads"),
    (
        pytest.param(col_sum, _col_sum_reference, 16, id="col-sum-four-warps"),
        pytest.param(col_sum, _col_sum_reference, 32, id="col-sum-eight-warps"),
        pytest.param(col_stats, _col_stats_reference, 16, id="col-stats-four-warps"),
        # The scale's store needs the sum: store_first is ignored, the fold is not.
        pytest.param(col_scale, _col_scale_reference, 16, id="col-scale-four-warps"),
    ),
)
def test_column_kernels_with_both_knobs_match_reference(
    kernel: helion.Kernel,
    reference: Callable[[torch.Tensor], torch.Tensor],
    reduction_threads: int,
) -> None:
    torch.manual_seed(0)
    x = torch.randn(128, 16384, device=DEVICE)
    bound = kernel.bind((x,))
    config = column_config(
        bound,
        block=32,
        tile_threads=8,
        reduction_threads=reduction_threads,
        **KNOBS,
    )
    code = bound.to_code(config)
    assert WARP_FOLD in code
    compiled = bound.compile_config(config)
    for _ in range(2):
        out = compiled(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, reference(x), rtol=1e-4, atol=1e-3)
        x.uniform_(-1, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
def test_diagonal_decay_with_both_knobs_matches_reference() -> None:
    # ``alpha[idx, :]`` (a [D] row) scales the rows of the state tile: the
    # extra per-lane load joins the accumulate nest like ``k`` and ``q``.
    kernel = helion.kernel(
        recurrent_step_fused.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    torch.manual_seed(0)
    q = torch.randn(BH_DEVICE, D, device=DEVICE)
    k = torch.randn(BH_DEVICE, D, device=DEVICE)
    v = torch.randn(BH_DEVICE, DV, device=DEVICE)
    alpha = torch.rand(BH_DEVICE, D, device=DEVICE)
    initial = torch.randn(BH_DEVICE, D, DV, device=DEVICE)
    state = initial.clone()
    bound = kernel.bind((q, k, v, state, alpha))
    config = _config(bound, dv_block=32, dv_threads=8, reduction_threads=16, **KNOBS)
    code = bound.to_code(config)
    assert WARP_FOLD in code
    assert code.index(STATE_STORE) < code.index(COLUMN_REDUCE)
    compiled = bound.compile_config(config)
    out, new_state = compiled(q, k, v, state, alpha)
    torch.cuda.synchronize()
    expected_state = alpha[:, :, None] * initial + k[:, :, None] * v[:, None, :]
    expected_out = (q[:, :, None] * expected_state).sum(1)
    torch.testing.assert_close(new_state, expected_state, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out, expected_out, rtol=1e-3, atol=1e-3)
    assert torch.equal(state, initial)


OWNER_LOOP = "for synthetic_lane_2 in cutlass.range_constexpr("


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("diagonal", "extra"),
    (
        pytest.param(False, {}, id="scalar-decay-knob-free"),
        pytest.param(False, KNOBS, id="scalar-decay-both-knobs"),
        pytest.param(True, {}, id="diagonal-decay-knob-free"),
        pytest.param(True, KNOBS, id="diagonal-decay-both-knobs"),
    ),
)
def test_correction_step_chains_two_register_tile_levels(
    diagonal: bool, extra: dict[str, Any]
) -> None:
    # The delta-rule step feeds ``kts = (k * s).sum(0)`` into the state it
    # then reduces again (``o = (q * s).sum(0)``): the tile is loaded once,
    # ``kts`` is combined, a second accumulate nest applies the correction
    # from the stashed tile and accumulates ``o``.  With ``store_first`` the
    # state store joins that nest (ahead of the ``o`` combine); without it the
    # corrected tile is stashed and stored behind the last combine.
    kernel = helion.kernel(
        recurrent_step_correction_fused.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    torch.manual_seed(0)
    q = torch.randn(BH_DEVICE, D, device=DEVICE)
    k = torch.randn(BH_DEVICE, D, device=DEVICE)
    v = torch.randn(BH_DEVICE, DV, device=DEVICE)
    alpha = torch.rand((BH_DEVICE, D) if diagonal else (BH_DEVICE,), device=DEVICE)
    beta = torch.rand(BH_DEVICE, device=DEVICE)
    initial = torch.randn(BH_DEVICE, D, DV, device=DEVICE)
    state = initial.clone()
    bound = kernel.bind((q, k, v, state, alpha, beta))
    spec = bound.config_spec
    assert spec.cute_register_tile_reduction_blocks
    config = _config(bound, dv_block=32, dv_threads=8, reduction_threads=16, **extra)
    for key, value in extra.items():
        assert config.config[key] is value
    code = bound.to_code(config)
    assert code.count("cute.arch.load(state.iterator") == 1
    assert code.count(COLUMN_REDUCE) == 2
    assert (WARP_FOLD in code) == bool(extra.get("cute_register_tile_warp_fold"))
    first_combine = code.index(COLUMN_REDUCE)
    second_combine = code.index(COLUMN_REDUCE, first_combine + 1)
    assert first_combine < code.index(STATE_STORE)
    store_first = bool(extra.get("cute_register_tile_store_first"))
    assert (code.index(STATE_STORE) < second_combine) == store_first
    assert code.count(OWNER_LOOP) == (2 if store_first else 3)
    compiled = bound.compile_config(config)
    a = alpha[:, :, None] if diagonal else alpha[:, None, None]
    for _ in range(2):
        out, new_state = compiled(q, k, v, state, alpha, beta)
        torch.cuda.synchronize()
        decayed = a * state
        kts = (k[:, :, None] * decayed).sum(1)
        expected_state = (
            decayed
            - beta[:, None, None] * k[:, :, None] * kts[:, None, :]
            + beta[:, None, None] * k[:, :, None] * v[:, None, :]
        )
        expected_out = (q[:, :, None] * expected_state).sum(1)
        torch.testing.assert_close(new_state, expected_state, rtol=1e-3, atol=1e-2)
        torch.testing.assert_close(out, expected_out, rtol=1e-3, atol=1e-2)
        assert torch.equal(state, initial)
        q.copy_(torch.rand_like(q).mul_(2).sub_(1))
        v.copy_(torch.rand_like(v).mul_(2).sub_(1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
@pytest.mark.parametrize("pack", [2, 4])
@pytest.mark.parametrize(
    "reduction_loop", [None, 32], ids=["register-tile", "looped-reduction"]
)
def test_tiles_per_block_matches_reference(
    pack: int, reduction_loop: int | None
) -> None:
    # ``cute_tiles_per_block`` on the recurrent step: the one-element BH_DEVICE axis
    # is only read through ``tile_bh.id`` (no graph distributes over it), yet
    # packing makes it a thread axis.  Every sub-group must own its own tile
    # (``test_cute_register_tile_schedule.py`` checks the rendered axes).
    kernel = helion.kernel(
        recurrent_step_fused.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    torch.manual_seed(0)
    q = torch.randn(BH_DEVICE, D, device=DEVICE)
    k = torch.randn(BH_DEVICE, D, device=DEVICE)
    v = torch.randn(BH_DEVICE, DV, device=DEVICE)
    alpha = torch.rand(BH_DEVICE, device=DEVICE)
    state = torch.randn(BH_DEVICE, D, DV, device=DEVICE)
    bound = kernel.bind((q, k, v, state, alpha))
    spec = bound.config_spec
    if reduction_loop is None:
        config = _config(
            bound,
            dv_block=32,
            dv_threads=8,
            reduction_threads=8,
            **KNOBS,
            cute_tiles_per_block=pack,
        )
    else:
        (reduction,) = [
            block.block_id for block in bound.env.block_sizes if block.reduction
        ]
        (dv,) = spec.block_sizes.valid_block_ids()
        threads = {dv: 4, reduction: 8}
        config = spec.normalized_config(
            helion.Config(
                block_sizes=[4],
                num_threads=[
                    threads.get(block_id, 0)
                    for block_id in spec.num_threads.valid_block_ids()
                ],
                reduction_loops=[reduction_loop],
                cute_tiles_per_block=pack,
            )
        )
    assert config.config["cute_tiles_per_block"] == pack
    code = bound.to_code(config)
    dv_block = 32 if reduction_loop is None else 4
    grid = BH_DEVICE // pack * (DV // dv_block)
    assert f"_launcher(_helion_recurrent_step_fused, ({grid},)" in code
    out, new_state = bound.compile_config(config)(q, k, v, state, alpha)
    torch.cuda.synchronize()
    expected_state = alpha[:, None, None] * state + k[:, :, None] * v[:, None, :]
    expected_out = (q[:, :, None] * expected_state).sum(1)
    torch.testing.assert_close(new_state, expected_state, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out, expected_out, rtol=1e-3, atol=1e-3)
