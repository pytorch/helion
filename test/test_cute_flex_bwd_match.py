"""CPU tests of the FlexAttention BACKWARD matcher (``cute/flex_bwd_match.py``).

Every test binds ``examples/flex_attention.py::helion_flex_attention_backward_kernel``
on the CPU with real ``create_block_mask`` BlockMasks and asserts the plan the
matcher produces, the regions it captures (score / mask / Jacobian, in the
renderer's transposed frame), the autotune surface it feeds and the loops it
claims. Nothing here compiles or launches CuTe.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import re
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import noop_mask

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")
from test._cute_aux import _cpu_codegen

from helion._compiler.cute import flex_match
from helion._compiler.cute.backend import _attention_config_envelope_ok
from helion._compiler.cute.cute_flash_bwd_flex import _fit_smem_budget
from helion._compiler.cute.cute_flash_bwd_flex import additive_bias_region
from helion._compiler.cute.cute_flash_bwd_flex import fbwd_aux_smem_candidates
from helion._compiler.cute.flex_bwd_match import FlexBwdMatch
from helion._compiler.cute.flex_bwd_match import match_flex_attention_bwd
from helion._testing import EXAMPLES_DIR
from helion._testing import TestCase
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.runtime.kernel import Kernel

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    import pathlib

    from helion._compiler.host_function import HostFunction
    from helion.runtime.kernel import BoundKernel

pytestmark = skipUnlessBackends(["cute"])
_aten = torch.ops.aten
_CPU = torch.device("cpu")


@functools.cache
def _example():
    return import_path(EXAMPLES_DIR / "flex_attention.py")


def _cute_kernel(kernel: Kernel) -> Kernel:
    return Kernel(kernel.fn, settings=kernel.settings.copy(backend="cute"))


@functools.cache
def _bwd_kernel() -> Kernel:
    return _cute_kernel(_example().helion_flex_attention_backward_kernel)


def _tensors(
    batch: int = 2,
    heads: int = 4,
    q_seq: int = 512,
    kv_seq: int = 512,
    head_dim: int = 64,
    *,
    kv_heads: int | None = None,
) -> dict[str, torch.Tensor]:
    kv_heads = heads if kv_heads is None else kv_heads
    dtype = torch.bfloat16
    return {
        "q": torch.randn(batch, heads, q_seq, head_dim, dtype=dtype),
        "k": torch.randn(batch, kv_heads, kv_seq, head_dim, dtype=dtype),
        "v": torch.randn(batch, kv_heads, kv_seq, head_dim, dtype=dtype),
        "do": torch.randn(batch, heads, q_seq, head_dim, dtype=dtype),
        "lse": torch.randn(batch, heads, q_seq, dtype=torch.float32),
        "out": torch.randn(batch, heads, q_seq, head_dim, dtype=dtype),
    }


def _args(
    t: dict[str, torch.Tensor],
    mask_mod: Callable[..., Any] | None,
    score_mod: Callable[..., Any] | None = None,
    score_mod_bwd: Callable[..., Any] | None = None,
    *,
    block_size: int | tuple[int, int] = 128,
    scale: float = 0.125,
) -> tuple[Any, ...]:
    q_seq, kv_seq = t["q"].shape[2], t["k"].shape[2]
    block_mask = create_block_mask(
        mask_mod or noop_mask,
        None,
        None,
        q_seq,
        kv_seq,
        device=str(_CPU),
        BLOCK_SIZE=block_size,
    )
    return _example().flex_attention_backward_args(
        t["q"],
        t["k"],
        t["v"],
        t["do"],
        t["lse"],
        t["out"],
        score_mod,
        score_mod_bwd,
        block_mask,
        scale,
        t["q"].shape[1] != t["k"].shape[1],
    )


@contextlib.contextmanager
def _as_sm100() -> Iterator[None]:
    with (
        _cpu_codegen(),
        patch("helion._compiler.backend._attention_flash_supported", return_value=True),
        patch(
            "helion._compiler.backend._attention_flash_gate_enabled", return_value=True
        ),
        patch.object(flex_match, "FLEX_PLAN_ENABLED", True),
    ):
        yield


def _bind(args: tuple[Any, ...]) -> BoundKernel:
    with _as_sm100():
        return _bwd_kernel()._bind_isolated(args)


def _host(bound: BoundKernel) -> HostFunction:
    host_function = bound.host_function
    assert host_function is not None
    return host_function


def _match(bound: BoundKernel, reasons: list[str] | None = None) -> FlexBwdMatch | None:
    host_function = _host(bound)
    device_ir = host_function.device_ir
    with bound.env:
        return match_flex_attention_bwd(
            device_ir.graphs,
            device_ir.grid_block_ids[0],
            env=bound.env,
            host_function=host_function,
            reasons=reasons,
        )


def _match_or_fail(bound: BoundKernel) -> FlexBwdMatch:
    reasons: list[str] = []
    match = _match(bound, reasons)
    assert match is not None, reasons
    return match


def _causal(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> torch.Tensor:
    return q_idx >= kv_idx


def _targets(graph: torch.fx.Graph) -> set[object]:
    return {node.target for node in graph.nodes if node.op == "call_function"}


# -- the plan ----------------------------------------------------------------


def test_causal_mask_plan_and_surface() -> None:
    t = _tensors()
    bound = _bind(_args(t, _causal))
    match = _match_or_fail(bound)
    plan = match.plan
    assert (plan.batch, plan.heads, plan.kv_heads) == (2, 4, 4)
    assert (plan.q_seq, plan.kv_seq, plan.head_dim, plan.head_dim_v) == (
        512,
        512,
        64,
        64,
    )
    assert plan.group == 1 and plan.io_dtype is torch.bfloat16
    assert plan.score_region is None and plan.grad_region is None
    assert plan.mask_region is not None
    # The mask reads the renderer's row (KV) and lane (Q) coordinates only.
    mask_targets = _targets(plan.mask_region.graph)
    assert _aten.ge.Tensor in mask_targets
    assert plan.walk.q_block == 128 and plan.walk.kv_block == 128
    assert plan.walk.table_cols == 4 and plan.walk.table_heads == 1
    # The scale is a kernel argument: a host scalar symbol, not a literal.
    assert plan.scale_value is None and plan.scale_arg is not None
    assert plan.lse_prescaled, "the example consumes the preprocess kernel's lse2"
    assert plan.log2_factor == pytest.approx(1.44269504)
    # Owned loops: the q-head loop, the two outer list loops and the inner
    # Q-tile loop the two walks share (one block id).
    assert len(match.loop_block_ids) == 4
    assert (
        match.h_block_id in match.loop_block_ids
        and match.q_block_id in match.loop_block_ids
    )
    assert match.full_nest.masked is False and match.part_nest.masked is True
    # The surface detector pinned the KV tile and the Q tile to 128.
    spec = bound.config_spec
    assert spec.cute_flash_bwd_search_enabled
    assert spec._cute_flash_bwd_flex and spec._cute_flash_bwd_persistent_allowed
    config = spec.default_config()
    assert config.config["cute_flash_bwd_regs"] == "120/152/88"
    assert config.config["cute_flash_bwd_exp2_f32"] == 1
    # The persistent CLC walk, the L2-sized batch-head group and the
    # statistics-through-MMA mode are knobs; the walk starts non-persistent,
    # the group at the widest fit (2 batch-heads of 512 rows at head dim 64),
    # the statistics on the register path.
    assert config.config["cute_flash_bwd_persistent"] == 0
    assert config.config["cute_flash_bwd_tile_group"] == 8
    assert config.config["cute_flash_bwd_stats_mma"] == 0
    # The polynomial exp2 share is offered (the example runs fast_math) but
    # starts at 0 (every element on MUFU).
    assert config.config["cute_flash_bwd_exp2_emu"] == 0
    # dS^T through TMEM for the dK MMA is offered at head dim 64 (off first).
    assert config.config["cute_flash_bwd_ds_tmem"] == 0
    # A mask without a score_mod: the mask form and the hook knobs are searched,
    # the additive-bias fold (nothing to fold) is not.
    fields = spec._flat_fields()
    assert "cute_flash_bwd_mask_form" in fields and "cute_flash_bwd_hook_cols" in fields
    assert "cute_flash_bwd_bias_fold" not in fields
    # No gathered aux table: nothing to stage, no staging knob.
    assert fbwd_aux_smem_candidates(plan) == ()
    assert "cute_flash_bwd_aux_smem" not in spec._flat_fields()
    # The fused body owns its epilogue: the store-split knob is not searched
    # (its graph rewrite would defeat the matcher) and its envelope rejects it.
    assert "epilogue_subtile" not in spec._flat_fields()
    subtiled = spec.default_config()
    subtiled.config["epilogue_subtile"] = 2
    assert not _attention_config_envelope_ok(subtiled)


def test_runtime_scale_and_grouped_heads() -> None:
    t = _tensors(heads=8, kv_heads=2, head_dim=128)
    bound = _bind(_args(t, _causal, scale=0.0883883))
    match = _match_or_fail(bound)
    assert match.plan.group == 4 and match.plan.head_dim == 128
    assert match.plan.max_steps == 4 * 4
    # At head dim 128 the dQ accumulator aliases the dP columns: no TMEM dS^T.
    assert "cute_flash_bwd_ds_tmem" not in bound.config_spec._flat_fields()


def test_additive_bias_score_mod_is_an_lse_fold() -> None:
    bias = torch.randn(4, 1023, dtype=torch.float32)

    def score_mod(score, b, h, q_idx, kv_idx):
        return score + hl.load(bias, [h, q_idx - kv_idx + 511])

    t = _tensors()
    bound = _bind(_args(t, _causal, score_mod))
    match = _match_or_fail(bound)
    plan = match.plan
    assert plan.score_region is not None and plan.grad_region is None
    assert len(plan.aux_buffers) == 1 and plan.aux_buffers[0].shape == (4, 1023)
    g = additive_bias_region(plan)
    assert g is not None
    assert "cute_flash_bwd_bias_fold" in bound.config_spec._flat_fields()
    # The fold's region is the bias alone: the score placeholder is unused.
    score_ph = next(n for n in g.graph.nodes if n.op == "placeholder")
    assert score_ph.target == "score" and not score_ph.users
    assert g.output.target is hl.load


def test_gathered_table_stages_in_smem() -> None:
    # graphormer-style: a per-(b, q, kv) index table selects a row of a small
    # per-head bias table (a data-dependent gather behind an affine load).
    index = torch.randint(0, 7, (2, 512, 512), dtype=torch.int32)
    bias = torch.randn(4, 7, dtype=torch.float32)

    def score_mod(score, b, h, q_idx, kv_idx):
        dist = hl.load(index, [b, q_idx, kv_idx])
        return score + hl.load(bias, [h, dist])

    t = _tensors()
    bound = _bind(_args(t, _causal, score_mod))
    match = _match_or_fail(bound)
    plan = match.plan
    # Only the gathered table is a candidate (its 28 floats + the sentinel,
    # 16-byte rounded); the affine index table stays in global memory.
    assert fbwd_aux_smem_candidates(plan) == (128,)
    spec = bound.config_spec
    assert spec._flat_fields()["cute_flash_bwd_aux_smem"].choices == (0, 4096)
    assert spec.default_config().config["cute_flash_bwd_aux_smem"] == 0
    # The fitter keeps the staged region when it fits (and drops it first).
    staged = _fit_smem_budget(
        plan, stats_mma=False, raw_score=False, device=_CPU, aux_smem=4096
    )
    assert staged.aux_bytes == 128
    plain = _fit_smem_budget(plan, stats_mma=False, raw_score=False, device=_CPU)
    assert plain.aux_bytes == 0 and (plain.q_stage, plain.do_stage) == (
        staged.q_stage,
        staged.do_stage,
    )


def test_softcap_jacobian_region() -> None:
    cap = 20.0

    def score_mod(score, b, h, q_idx, kv_idx):
        return cap * torch.tanh(score / cap)

    def score_mod_bwd(grad, score, b, h, q_idx, kv_idx):
        th = torch.tanh(score / cap)
        return grad * (1.0 - th * th)

    t = _tensors()
    bound = _bind(_args(t, None, score_mod, score_mod_bwd))
    match = _match_or_fail(bound)
    plan = match.plan
    assert plan.score_region is not None and plan.grad_region is not None
    assert additive_bias_region(plan) is None
    # ``grad * J`` becomes ``J``: the Jacobian region has the tanh but no
    # multiply by the (unit) gradient's literal.
    targets = _targets(plan.grad_region.graph)
    assert _aten.tanh.default in targets
    for node in plan.grad_region.graph.nodes:
        if node.target is _aten.mul.Tensor:
            assert all(not isinstance(a, float) for a in node.args), node.args


def test_ragged_lengths_match() -> None:
    t = _tensors(q_seq=192, kv_seq=320)
    match = _match_or_fail(_bind(_args(t, _causal)))
    assert match.plan.ragged_m and match.plan.ragged_n
    assert match.plan.num_q_tiles == 2 and match.plan.num_kv_tiles == 3


def test_declines_head_dim_192() -> None:
    t = _tensors(head_dim=192)
    t["v"] = torch.randn(2, 4, 512, 128, dtype=torch.bfloat16)
    t["do"] = torch.randn(2, 4, 512, 128, dtype=torch.bfloat16)
    t["out"] = torch.randn(2, 4, 512, 128, dtype=torch.bfloat16)
    reasons: list[str] = []
    assert _match(_bind(_args(t, _causal)), reasons) is None
    assert any("head dims" in r for r in reasons), reasons


def test_q_block_256_pads_the_step_table() -> None:
    # Ragged M with 256-row Q blocks: the last block's second 128-row tile
    # lies past M; the table and the hook bounds cover it.
    t = _tensors(q_seq=300, kv_seq=512)
    bound = _bind(_args(t, _causal, block_size=(256, 128)))
    match = _match_or_fail(bound)
    plan = match.plan
    assert plan.walk.q_block == 256
    assert plan.num_q_tiles == 3 and plan.num_q_tiles_padded == 4
    assert plan.max_steps == plan.group * 4


def test_declines_dk_store_with_another_factor(tmp_path: pathlib.Path) -> None:
    # The emitter scales dK by the softmax scale: a store with any other
    # factor must not match (it would silently get the wrong dK).
    source = (EXAMPLES_DIR / "flex_attention.py").read_text()
    old = "(dk_acc * scale)[None, None, :, :].to(dk.dtype)"
    assert source.count(old) == 1
    path = tmp_path / "flex_attention_dk_factor.py"
    path.write_text(
        source.replace(old, "(dk_acc * 2.0)[None, None, :, :].to(dk.dtype)")
    )
    kernel = _cute_kernel(import_path(path).helion_flex_attention_backward_kernel)
    t = _tensors()
    with _as_sm100():
        bound = kernel._bind_isolated(_args(t, _causal))
    reasons: list[str] = []
    assert _match(bound, reasons) is None
    assert any("dK store factor" in r for r in reasons), reasons


def test_declines_kv_block_other_than_128() -> None:
    t = _tensors()
    reasons: list[str] = []
    assert _match(_bind(_args(t, _causal, block_size=(128, 64))), reasons) is None
    assert any("KB" in r for r in reasons), reasons


# -- strided operands ---------------------------------------------------------


def _transposed_tensors(
    batch: int = 2,
    heads: int = 4,
    q_seq: int = 512,
    kv_seq: int = 512,
    head_dim: int = 64,
    *,
    kv_heads: int | None = None,
) -> dict[str, torch.Tensor]:
    """``[B, H, S, D]`` views of ``[B, S, H, D]`` allocations (the layout
    every model produces); LSE stays contiguous."""
    kv_heads = heads if kv_heads is None else kv_heads
    dtype = torch.bfloat16

    def view(b: int, s: int, h: int, d: int) -> torch.Tensor:
        return torch.randn(b, s, h, d, dtype=dtype).transpose(1, 2)

    return {
        "q": view(batch, q_seq, heads, head_dim),
        "k": view(batch, kv_seq, kv_heads, head_dim),
        "v": view(batch, kv_seq, kv_heads, head_dim),
        "do": view(batch, q_seq, heads, head_dim),
        "lse": torch.randn(batch, heads, q_seq, dtype=torch.float32),
        "out": view(batch, q_seq, heads, head_dim),
    }


def _wrapper_plan(code: str) -> dict[str, Any]:
    import ast

    for line in code.splitlines():
        if "._helion_cute_wrapper_plans = " in line:
            (plan,) = ast.literal_eval(line.split(" = ", 1)[1])
            return plan
    raise AssertionError("no wrapper plan in the generated module")


def _host_lines(plan: dict[str, Any]) -> str:
    from helion.runtime.cute import launcher

    body: list[str] = []
    call_args: list[str] = []
    launcher._append_cute_wrapper_plan(body, call_args, plan, num_sm=148)
    return "\n".join(body)


def test_contiguous_operands_keep_the_rank3_maps() -> None:
    """Contiguous operands: row-major strides recorded, ``operand_strides``
    ``None``, the rank-3 ``(seq, D, B*H)`` render (the render-unchanged pin
    of the strided work)."""
    t = _tensors()
    bound = _bind(_args(t, _causal))
    match = _match_or_fail(bound)
    plan = match.plan
    assert plan.strides == ((131072, 32768, 64, 1),) * 6
    assert plan.operand_strides is None
    with _as_sm100():
        code = bound.to_code(bound.config_spec.default_config())
    wrapper = _wrapper_plan(code)
    assert "flex_strides" not in wrapper
    assert "tQgQ = tQgQ_qdl[None, None, 0, fbwd_bhq]" in code
    assert "tKgK_kdl[None, None, 0, fbwd_bhk]" in code
    assert "fbwd_hq = " not in code
    assert (
        "_fbwd_mQ = cute.make_tensor(arg0.iterator, cute.make_layout((512, 64, 8), "
        "stride=(64, 1, 32768)))"
    ) in _host_lines(wrapper)


@pytest.mark.parametrize(("kv_heads", "head_dim"), [(4, 64), (2, 128)])
def test_transposed_operands_take_the_rank4_maps(kv_heads: int, head_dim: int) -> None:
    """Transposed Q/K/V/dO (dK/dV ``empty_like`` inherit the layout): the
    matcher records the six strides, the surface is enabled, the render
    indexes rank-4 ``(seq, D, H, B)`` views by head and batch (the Q head of
    a step decoded from its collapsed index) and the launcher builds the maps
    from the strides; dQ / LSE / delta stay flat."""
    t = _transposed_tensors(head_dim=head_dim, kv_heads=kv_heads)
    bound = _bind(_args(t, _causal))
    match = _match_or_fail(bound)
    plan = match.plan
    q_st = (512 * 4 * head_dim, head_dim, 4 * head_dim, 1)
    kv_st = (512 * kv_heads * head_dim, head_dim, kv_heads * head_dim, 1)
    assert plan.strides == (q_st, q_st, kv_st, kv_st, kv_st, kv_st)
    assert plan.operand_strides == plan.strides
    spec = bound.config_spec
    assert spec.cute_flash_bwd_search_enabled
    # The same knob space (and seed) as the contiguous specialization.
    contiguous = _bind(_args(_tensors(head_dim=head_dim, kv_heads=kv_heads), _causal))
    assert sorted(spec._flat_fields()) == sorted(contiguous.config_spec._flat_fields())
    assert spec.default_config() == contiguous.config_spec.default_config()
    with _as_sm100():
        code = bound.to_code(spec.default_config())
    assert "fbwd_sSteps" in code
    wrapper = _wrapper_plan(code)
    assert wrapper["flex_strides"] == plan.strides
    assert (wrapper["flex_batch"], wrapper["flex_heads"], wrapper["flex_kv_heads"]) == (
        2,
        4,
        kv_heads,
    )
    assert "fbwd_hq = fbwd_bhq - fbwd_b * 4" in code
    assert "tQgQ = tQgQ_qdl[None, None, 0, fbwd_hq, fbwd_b]" in code
    assert "tDOgDO = tDOgDO_qdl[None, None, 0, fbwd_hq, fbwd_b]" in code
    assert "tKgK_kdl[None, None, 0, fbwd_hk, fbwd_b]" in code
    assert "tVgV = tVgV_kdl[None, None, 0, fbwd_hk, fbwd_b]" in code
    assert "_fbwd_mdVt[None, None, fbwd_hk, fbwd_b]" in code
    assert "_fbwd_mdKt[None, None, fbwd_hk, fbwd_b]" in code
    assert "tQgQ_qdl[None, None, 0, fbwd_bhq]" not in code
    assert "tKgK_kdl[None, None, 0, fbwd_bhk]" not in code
    assert "[None, None, fbwd_bhk]" not in code
    # The fp32 dQ reduce-add keeps the flat contiguous map.
    assert "_fbwd_mDQt[None, None, fbwd_bhq]" in code
    host = _host_lines(wrapper)
    assert (
        f"_fbwd_mQ = cute.make_tensor(arg0.iterator, cute.make_layout((512, {head_dim}, 4, 2), "
        f"stride=({4 * head_dim}, 1, {head_dim}, {q_st[0]})))"
    ) in host
    assert (
        f"_fbwd_mdK = cute.make_tensor(arg7.iterator, cute.make_layout((512, {head_dim}, {kv_heads}, 2), "
        f"stride=({kv_heads * head_dim}, 1, {head_dim}, {kv_st[0]})))"
    ) in host
    assert (
        "_fbwd_mDQ2 = cute.make_tensor(arg6.iterator, cute.make_layout((512, " in host
    )


def test_packed_qkv_slices_are_admitted() -> None:
    qkv = torch.randn(2, 512, 4, 192, dtype=torch.bfloat16)
    t = _tensors()
    t["q"] = qkv[..., :64].transpose(1, 2)
    t["k"] = qkv[..., 64:128].transpose(1, 2)
    t["v"] = qkv[..., 128:].transpose(1, 2)
    bound = _bind(_args(t, _causal))
    plan = _match_or_fail(bound).plan
    assert (
        plan.strides[0] == plan.strides[2] == plan.strides[3] == (393216, 192, 768, 1)
    )
    # dO and dK / dV (``empty_like`` of the slice views) are dense.
    assert plan.strides[1] == (131072, 32768, 64, 1)
    assert plan.strides[4] == plan.strides[5] == (131072, 64, 256, 1)
    assert bound.config_spec.cute_flash_bwd_search_enabled


def test_unaligned_strides_decline_and_log(caplog: pytest.LogCaptureFixture) -> None:
    """A dO view whose outer strides are not multiples of 16 bytes leaves the
    fused path; the first decline reason is logged at DEBUG level."""
    t = _tensors()
    t["do"] = torch.randn(2, 512, 4, 68, dtype=torch.bfloat16)[..., :64].transpose(1, 2)
    assert tuple(t["do"].stride()) == (139264, 68, 272, 1)
    with caplog.at_level(logging.DEBUG, logger="helion._compiler.cute.flex_bwd_match"):
        bound = _bind(_args(t, _causal))
    reasons: list[str] = []
    assert _match(bound, reasons) is None
    assert reasons[0] == (
        "flex_bwd: dO strides (139264, 68, 272, 1) are not a unit inner stride "
        "with non-zero 16-byte aligned outer strides"
    )
    assert not bound.config_spec.cute_flash_bwd_search_enabled
    assert any(
        record.getMessage() == f"flex backward envelope declined: {reasons[0]}"
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


def test_expanded_operands_decline() -> None:
    """Expanded (zero-stride) K/V are not TMA layouts: declined, so the
    generic path serves them as before."""
    t = _tensors()
    t["k"] = torch.randn(1, 4, 512, 64, dtype=torch.bfloat16).expand(2, 4, 512, 64)
    t["v"] = torch.randn(2, 1, 512, 64, dtype=torch.bfloat16).expand(2, 4, 512, 64)
    assert t["k"].stride()[0] == 0 and t["v"].stride()[1] == 0
    bound = _bind(_args(t, _causal))
    reasons: list[str] = []
    assert _match(bound, reasons) is None
    assert reasons[0] == (
        "flex_bwd: K strides (0, 32768, 64, 1) are not a unit inner stride "
        "with non-zero 16-byte aligned outer strides"
    )
    assert not bound.config_spec.cute_flash_bwd_search_enabled


def _without_source_comments(code: str) -> str:
    """The generated module without its ``# src[file:line]`` comments (line
    numbers of the example source, not generated code)."""
    return re.sub(r"^[ \t]*# src\[.*\n", "", code, flags=re.MULTILINE)


# unittest runners ignore pytestmark.
@onlyBackends(["cute"])
class TestContiguousRenderPin(TestCase):
    """The exact text of one contiguous backward render (the
    ``.expected_cute`` journal next to this file): the strided-operand
    support must not change the code contiguous inputs produce."""

    def test_contiguous_backward_render(self) -> None:
        bound = _bind(_args(_tensors(), _causal))
        with _as_sm100():
            code = bound.to_code(bound.config_spec.default_config())
        assert "'flex_strides'" not in code
        self.assertExpectedJournal(_without_source_comments(code))
