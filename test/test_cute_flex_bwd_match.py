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
import os
import re
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import noop_mask

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")
from test._cute_aux import _cpu_codegen

import helion
from helion._compiler.cute import flex_bwd_match as flex_bwd_match_module
from helion._compiler.cute import flex_match
from helion._compiler.cute.backend import _attention_config_envelope_ok
from helion._compiler.cute.cute_flash_bwd_flex import _fit_smem_budget
from helion._compiler.cute.cute_flash_bwd_flex import additive_bias_region
from helion._compiler.cute.cute_flash_bwd_flex import fbwd_aux_smem_candidates
from helion._compiler.cute.cute_flash_bwd_flex import fbwd_table_step_choices
from helion._compiler.cute.flex_bwd_match import FlexBwdMatch
from helion._compiler.cute.flex_bwd_match import flex_bwd_tmem_plan
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
    from pathlib import Path

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
    block_mask: BlockMask | None = None,
) -> tuple[Any, ...]:
    q_seq, kv_seq = t["q"].shape[2], t["k"].shape[2]
    if block_mask is None:
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


def _causal_block_mask(q_seq: int, kv_seq: int, block: int = 128) -> BlockMask:
    """``create_block_mask(_causal)`` from the block indices alone (the dense
    mask it would materialize does not fit for a 128K problem): KV block
    ``j`` is full for Q block ``i > j`` and partial on the diagonal."""
    nq, nk = -(-q_seq // block), -(-kv_seq // block)
    qi = torch.arange(nq)[:, None]
    kj = torch.arange(nk)[None, :]

    def lists(keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = keep.sum(1).to(torch.int32)
        # Kept blocks first, ascending.
        index = torch.argsort(~keep, dim=1, stable=True).to(torch.int32)
        return count[None, None], index[None, None]

    part_cnt, part_idx = lists(qi == kj)
    full_cnt, full_idx = lists(qi > kj)
    return BlockMask.from_kv_blocks(
        part_cnt,
        part_idx,
        full_cnt,
        full_idx,
        BLOCK_SIZE=(block, block),
        mask_mod=_causal,
        seq_lengths=(q_seq, kv_seq),
    )


def _long_tensors(
    heads: int, kv_heads: int, seq: int, head_dim: int
) -> dict[str, torch.Tensor]:
    """Uninitialized (the values never matter here) tensors of a long problem."""
    dtype = torch.bfloat16
    return {
        "q": torch.empty(1, heads, seq, head_dim, dtype=dtype),
        "k": torch.empty(1, kv_heads, seq, head_dim, dtype=dtype),
        "v": torch.empty(1, kv_heads, seq, head_dim, dtype=dtype),
        "do": torch.empty(1, heads, seq, head_dim, dtype=dtype),
        "lse": torch.empty(1, heads, seq, dtype=torch.float32),
        "out": torch.empty(1, heads, seq, head_dim, dtype=dtype),
    }


def _render(bound: BoundKernel, **overrides: object) -> str:
    config = bound.config_spec.default_config()
    if overrides:
        config = helion.Config(**{**config.config, **overrides})
    with _as_sm100():
        return bound.to_code(config)


def _fit(plan: Any, **kwargs: Any) -> Any:
    smem = _fit_smem_budget(
        plan, stats_mma=False, raw_score=False, device=_CPU, **kwargs
    )
    assert smem is not None
    return smem


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


def _tensors_dv(
    head_dim: int, head_dim_v: int, **kwargs: Any
) -> dict[str, torch.Tensor]:
    """``_tensors`` with a separate value head dim (V / dO / O columns)."""
    t = _tensors(head_dim=head_dim, **kwargs)
    kv_heads, kv_seq = t["k"].shape[1], t["k"].shape[2]
    batch, heads, q_seq = t["q"].shape[0], t["q"].shape[1], t["q"].shape[2]
    t["v"] = torch.randn(batch, kv_heads, kv_seq, head_dim_v, dtype=torch.bfloat16)
    t["do"] = torch.randn(batch, heads, q_seq, head_dim_v, dtype=torch.bfloat16)
    t["out"] = torch.randn(batch, heads, q_seq, head_dim_v, dtype=torch.bfloat16)
    return t


@pytest.mark.parametrize(
    ("head_dim", "head_dim_v"),
    [
        (192, 192),
        (72, 64),
        (64, 8),
        # TMEM plans on paper whose tiles do not fit shared memory.
        (160, 160),
        (144, 144),
        (96, 160),
        (80, 144),
        (64, 256),
        (176, 128),
        (256, 192),
    ],
)
def test_declines_unsupported_head_dims(head_dim: int, head_dim_v: int) -> None:
    """Pairs whose accumulators do not fit TMEM, whose tiles do not fit shared
    memory, or widths that are not multiples of 16 are declined by the
    matcher (nothing admitted can fail in the emitter)."""
    reasons: list[str] = []
    assert (
        _match(_bind(_args(_tensors_dv(head_dim, head_dim_v), _causal)), reasons)
        is None
    )
    assert any("head dims" in r for r in reasons), reasons


def test_head_dim_192_128_matches() -> None:
    match = _match_or_fail(_bind(_args(_tensors_dv(192, 128), _causal)))
    assert (match.plan.head_dim, match.plan.head_dim_v) == (192, 128)


def test_head_dim_256_pair_matches() -> None:
    """(256, 256) takes the 2-CTA pair plan; the matcher also captured the
    forward-frame regions its dQ clusters render."""
    match = _match_or_fail(_bind(_args(_tensors_dv(256, 256), _causal)))
    plan = match.plan
    assert (plan.head_dim, plan.head_dim_v) == (256, 256)
    tmem = flex_bwd_tmem_plan(256, 256)
    assert tmem is not None and tmem.mode == "pair"
    assert plan.regions_q_captured
    assert plan.mask_region_q is not None and plan.score_region_q is None
    assert plan.mask_region is not None


@pytest.mark.skipif(
    os.environ.get("HELION_TEST_PAIR_PTX") != "1",
    reason="2-3 minutes: compiles the pair kernel through the CuTe DSL and "
    "assembles its PTX; set HELION_TEST_PAIR_PTX=1",
)
def test_head_dim_256_pair_assembles_without_gpu(tmp_path: Path) -> None:
    """The ptxas rule the DSL hides ("NVVM backend compilation failed" is all
    it says): one cta_group per kernel. Compile the pair kernel through the
    DSL on the CPU, keep the PTX and assemble it with nvJitLink, which
    reports ptxas's own message (the mixed cta_group::1 / ::2 rejection,
    for one)."""
    t = _tensors_dv(256, 256, batch=1, heads=2, q_seq=256, kv_seq=256)
    args = _args(t, _causal)
    bound = _bind(args)
    code = _render(bound)
    assert "fbwd_is_dq" in code
    path = tmp_path / "pair_256.py"
    path.write_text(code)
    namespace: dict[str, object] = {}
    exec(compile(code, str(path), "exec"), namespace)
    host = namespace["helion_flex_attention_backward_kernel"]
    with patch.dict(os.environ, {"CUTE_DSL_ARCH": "sm_100a"}), _as_sm100():
        host(*args, _launcher=_ptx_dump_launcher(tmp_path))  # pyrefly: ignore[not-callable]
    ptx_files = sorted(tmp_path.glob("*.ptx"))
    assert ptx_files, list(tmp_path.iterdir())
    assert_ptx_assembles(ptx_files[-1].read_bytes())


def _ptx_dump_launcher(dump_dir: Path) -> Callable[..., None]:
    """A compile-only launcher (``test_cute_flash_gated._compile_only_launcher``
    with the DSL's ``--keep-ptx`` into ``dump_dir``: the keep / dump settings
    are per-compile options, the environment variables are read at import)."""

    def launcher(
        cute_kernel: object, grid: tuple[int, ...], *args: object, **kwargs: object
    ) -> None:
        import importlib

        import cutlass.cute as cute

        from helion.runtime.cute import launcher as cute_launcher

        cuda_driver = importlib.import_module("cuda.bindings.driver")
        block = kwargs.pop("block", (256, 1, 1))
        assert isinstance(block, tuple)
        with patch.object(
            cute_launcher, "_validate_cute_launcher_tensor", lambda arg: None
        ):
            launch = cute_launcher._build_cute_schema_and_args(
                cute_kernel, tuple(args), (int(grid[0]), 1, 1)
            )
        jit_func = cute_launcher._create_cute_wrapper(
            cute_kernel, launch.schema, block, num_sm=148
        )
        cute.compile(
            jit_func,
            *launch.launch_args,
            cuda_driver.CUstream(0),
            options=f"--keep-ptx --dump-dir={dump_dir}",
        )

    return launcher


def assert_ptx_assembles(ptx: bytes) -> None:
    """Assemble ``ptx`` for sm_100a with nvJitLink and fail with ptxas's
    error log when it is rejected."""
    from cuda.bindings import nvjitlink

    handle = nvjitlink.create(2, ["-arch=sm_100a", "-O3"])
    try:
        nvjitlink.add_data(handle, nvjitlink.InputType.PTX, ptx, len(ptx), "k.ptx")
        nvjitlink.complete(handle)
    except Exception as error:
        size = nvjitlink.get_error_log_size(handle)
        log = bytearray(size)
        if size:
            nvjitlink.get_error_log(handle, log)
        raise AssertionError(
            f"ptxas rejected the PTX: {log.decode(errors='replace')}"
        ) from error
    finally:
        nvjitlink.destroy(handle)


def test_head_dim_256_pair_render() -> None:
    """The pair kernel's shape: (2, 1, 1) clusters, the dK/dV clusters (one
    per KV block) and the dQ clusters (one per Q block of every batch-head,
    Q-major, their KV lists derived on the host from the Q-side tables),
    alternating in launch order when their counts are equal and otherwise
    the dK/dV range first, the roles' TMEM plans (dK/dV: dK 0, dV 128,
    S 256 / 384, dP 320 / 448 by step parity; dQ: S^T 0 / 128, dP^T 64 /
    192, the accumulator 256), every MMA a pair MMA with its own
    instruction-descriptor register per role, and the dQ clusters' hooks in
    the forward frame (the Q tile is their item-level row tile, the KV tile
    the per-step one)."""
    code = _render_plan(_tensors_dv(256, 256))
    assert "_helion_cute_cluster_shape = (2, 1, 1)" in code
    # 32 KV-block clusters + 32 Q-block clusters: four CTAs per KV block,
    # the roles interleaved.
    assert "[0] * 4,)" in code
    assert (
        "fbwd_is_dq = cute.arch.make_warp_uniform(fbwd_cluster % 2) >= cutlass.Int32(1)"
        in code
    )
    assert "fbwd_tile_id = fbwd_cluster // 2" in code
    assert "fbwd_qtile = fbwd_cluster // 2" in code
    assert "fbwd_bhq_t = fbwd_qt_id // 4" in code
    # 16 Q blocks against 32 KV blocks: the dQ clusters follow the dK/dV
    # range (still four CTAs per KV block, the last 16 clusters empty).
    ragged = _render_plan(_tensors_dv(256, 256, q_seq=256, kv_seq=512))
    assert "[0] * 4,)" in ragged
    assert (
        "fbwd_is_dq = cute.arch.make_warp_uniform(fbwd_cluster // 32) >= cutlass.Int32(1)"
        in ragged
    )
    assert "fbwd_tile_id = cutlass.min(fbwd_cluster, cutlass.Int32(31))" in ragged
    assert "fbwd_qtile = fbwd_cluster - 32" in ragged
    # The derived KV-side lists: one host statement ahead of the launch
    # builds the tuple, the four extra arguments index it.
    build = "fbwd_kv_tables = _helion_flash_flex_bwd_kv_tables("
    assert code.count(build) == 1 and "q_blocks=4, kv_blocks=4)" in code
    assert code.index(build) < code.index("_launcher(")
    for index in range(4):
        assert f"fbwd_kv_tables[{index}]" in code
    assert (
        "fbwd_cfq = cutlass.Int32(_fbwd_kv_full_cnt[fbwd_tb_q, fbwd_th_q, fbwd_qrow])"
        in code
    )
    assert "flash_bwd_flex_pair_shared_storage(cutlass.BFloat16, " in code
    for off, name in ((256, "tS_t"), (320, "tdP_t"), (128, "tDV_t"), (0, "tDK_t")):
        assert f"{name} = cute.make_tensor(fbwd_tmem_ptr + {off}, " in code, name
    assert "tS_t = cute.make_tensor(fbwd_tmem_ptr + 0, " in code
    assert "tdP_t = cute.make_tensor(fbwd_tmem_ptr + 64, " in code
    assert code.count("tDQ_t = cute.make_tensor(fbwd_tmem_ptr + 256, ") == 2
    # The step-parity S / dP buffers of both roles.
    for off in (256, 0):
        assert (
            f"tSb_t = cute.make_tensor(fbwd_tmem_ptr + {off} + 128 * fbwd_b, " in code
        )
    assert "mbarrier_init(fbwd_stq_empty_ptr + fbwd_bi, 16)" in code
    # dK/dV role: S(0) before the loop, then dP(i) / S(i+1) / dV(i) / dK(i);
    # dQ role: S^T(i) / dP^T(i) / dQ(i-1) and the tail dQ, accumulating.
    gemms = [line for line in code.splitlines() if "gemm_ptx_dyn_qk(" in line]
    assert len(gemms) == 9 and all("cta_group=2" in line for line in gemms)
    assert "cta_group=1" not in code and "CtaGroup.ONE" not in code
    assert "fbwd_j = fbwd_steps - 1" in code
    assert code.count("zero_init=fbwd_dq_zero") == 2
    assert "num_stages=2" in code and "fbwd_st_kt2 + 2048 * fbwd_kte.index" in code
    # The dK/dV clusters' P has its own staging buffer (the storage fits
    # exactly up to 128 step-table entries): the dS write waits for the
    # previous dK (sds_free), the P write for the previous dV (pv_done); one
    # LSE / delta staging copy per parity behind a 256-thread barrier.
    assert "flash_bwd_flex_pair_shared_storage(cutlass.BFloat16, 4, True)" in code
    assert "fbwd_sP = storage.sP.get_tensor(" in code
    assert (
        "fbwd_st_p = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 7)"
        in code
    )
    assert "mbarrier_init(fbwd_pv_done_ptr, 1)" in code
    assert code.count("mbar_spin_wait(fbwd_pv_done_ptr, fbwd_i - 1 & 1, 10000000)") == 2
    assert code.count("named_barrier_wait_unaligned(3, 256)") == 2
    assert "named_barrier_wait_unaligned(4, 128)" not in code
    assert code.count("fbwd_sLSE.iterator + (128 - fbwd_par + 0 + fbwd_stat_j)") == 1
    # Past 128 entries P shares the dS buffer again (the dS write then waits
    # for the dV MMA) and the LSE / delta staging keeps one copy per parity.
    long = _render_plan(
        _tensors_dv(256, 256, batch=1, heads=1, q_seq=32768, kv_seq=32768)
    )
    assert "flash_bwd_flex_pair_shared_storage(cutlass.BFloat16, 256, False)" in long
    assert "fbwd_sP = fbwd_sdS_nk" in long and "storage.sP" not in long
    assert long.count("mbar_spin_wait(fbwd_pv_done_ptr, fbwd_i & 1, 10000000)") == 2
    assert long.count("named_barrier_wait_unaligned(3, 256)") == 2
    for name in (
        "helion_fbwd_ss_idesc",
        "helion_fbwd_pv_idesc",
        "helion_fbwdq_ss_idesc",
        "helion_fbwdq_pv_idesc",
    ):
        mma = "ss" if "ss" in name else "pv"
        assert code.count(f"declare_ptx_idesc(_fbwd_{mma}_mma.op, '{name}')") == 1
    # The dQ clusters: the tile's LSE / delta per row once, dS staged
    # K-major, the accumulator drained once per tile through the K / V
    # tiles as (32 x 16) boxes.
    assert "fbwd_lse_frg.fill(_fbwd_mLSE[fbwd_bhq_t * 512 + fbwd_qrow_t])" in code
    assert "mbar_spin_wait(fbwd_dv_done_ptr, 0, 10000000)" in code
    assert "fbwd_gdq[None, None, fbwd_gdq_row, fbwd_my_c0 + 7]" in code
    assert "fbwd_sdq16b[None, None, fbwd_slot0 + 7]" in code
    # No persistent walk, statistics MMAs, staged aux tables or chunked
    # table on the pair plan.
    fields = _bind(_args(_tensors_dv(256, 256), _causal)).config_spec._flat_fields()
    assert fields["cute_flash_bwd_persistent"].choices == (0,)
    for knob in (
        "cute_flash_bwd_stats_mma",
        "cute_flash_bwd_ds_tmem",
        "cute_flash_bwd_table_steps",
        "cute_flash_bwd_aux_smem",
        "cute_flash_bwd_s_first",
    ):
        assert knob not in fields, knob


@pytest.mark.parametrize(
    ("head_dim", "head_dim_v", "exact"),
    [(64, 128, False), (96, 64, True), (32, 48, True)],
)
def test_ds_tmem_dk_mma_has_its_own_width(
    head_dim: int, head_dim_v: int, exact: bool
) -> None:
    """dS^T through TMEM at D != Dv: the dK MMA takes a dK-shaped TS atom
    (N = D) instead of the dV MMA's (N = Dv); its descriptor comes from the
    exact-N twin when D is padded."""
    bound = _bind(_args(_tensors_dv(head_dim, head_dim_v), _causal))
    code = _render(bound, cute_flash_bwd_ds_tmem=1)
    op = "_fbwd_dskts_mma_x" if exact else "_fbwd_dskts_mma"
    assert f"declare_ptx_idesc({op}.op, 'helion_fbwd_dskts_idesc')" in code
    assert "tTSrDS = fbwd_dskts.make_fragment_A(tDS)" in code
    assert "tTSrQt = fbwd_dskts.make_fragment_B(sQt)" in code
    assert code.count("'helion_fbwd_dskts_idesc', zero_init=fbwd_dk_zero") == 2
    assert "'flex_ds_tmem_dk': True" in code
    # With D == Dv the dV MMA's atom still serves.
    same = _render(
        _bind(_args(_tensors(head_dim=64), _causal)), cute_flash_bwd_ds_tmem=1
    )
    assert (
        "_fbwd_dskts" not in same and "tTSrDS = fbwd_tst.make_fragment_A(tDS)" in same
    )


@pytest.mark.parametrize(
    ("head_dim", "head_dim_v", "chunks"), [(96, 64, 3), (80, 64, 3)]
)
def test_dq_drain_loads_only_stored_chunks(
    head_dim: int, head_dim_v: int, chunks: int
) -> None:
    """Plain plans whose padded dQ partition would reach past TMEM column 512
    load only the ceil(D / 32) chunks the drain stores."""
    code = _render_plan(_tensors_dv(head_dim, head_dim_v))
    assert "tDQ_t = cute.make_tensor(fbwd_tmem_ptr + 416, " in code
    assert (
        f"for fbwd_c in cutlass.range_constexpr({chunks}):\n"
        "                cute.copy(fbwd_tiled_dq_ld, tRDtDQ[None, fbwd_c, None, None], tRDrDQ[None, fbwd_c, None, None])"
        in code
    )
    assert "cute.copy(fbwd_tiled_dq_ld, tRDtDQ, tRDrDQ)" not in code


# -- head dims: padded smem tiles, exact MMA extents, TMEM plans -------------


def _render_plan(t: dict[str, torch.Tensor]) -> str:
    code = _render(_bind(_args(t, _causal)))
    assert "fbwd_sSteps" in code
    return code


@pytest.mark.parametrize(
    ("head_dim", "head_dim_v", "expect"),
    [
        (
            96,
            96,
            {
                "storage": "flash_bwd_shared_storage(128, 2, 1, 1, cutlass.BFloat16, 4, 8, 0, True, True, 0)",
                "tmem": (128, 256, 352, 128),
                "k_caps": ("k_tiles=6", "k_tiles=6"),
                "exact_mmas": ("_fbwd_ts_mma_x", "_fbwd_dsk_mma_x", "_fbwd_dq_mma_x"),
                "dq_boxes": 3,
                "epi_chunks": (6, 6),
            },
        ),
        (
            80,
            80,
            {
                "storage": "flash_bwd_shared_storage(128, 2, 1, 1, cutlass.BFloat16, 4, 8, 0, True, True, 0)",
                "tmem": (128, 256, 352, 128),
                "k_caps": ("k_tiles=5", "k_tiles=5"),
                "exact_mmas": ("_fbwd_ts_mma_x", "_fbwd_dsk_mma_x", "_fbwd_dq_mma_x"),
                "dq_boxes": 3,
                "epi_chunks": (5, 5),
            },
        ),
        (
            32,
            32,
            {
                "storage": "flash_bwd_shared_storage(64, 4, 4, 1, cutlass.BFloat16, 4, 8, 0, True, True, 0)",
                "tmem": (128, 256, 288, 320),
                "k_caps": ("k_tiles=2", "k_tiles=2"),
                "exact_mmas": ("_fbwd_ts_mma_x", "_fbwd_dsk_mma_x", "_fbwd_dq_mma_x"),
                "dq_boxes": 1,
                "epi_chunks": (2, 2),
            },
        ),
        (
            64,
            128,
            {
                "storage": "flash_bwd_shared_storage(64, 4, 1, 1, cutlass.BFloat16, 4, 8, 0, True, True, 0, head_dim_v=128)",
                "tmem": (128, 256, 384, 448),
                "k_caps": (),
                "exact_mmas": (),
                "dq_boxes": 2,
                "epi_chunks": (8, 4),
            },
        ),
        (
            192,
            128,
            {
                "storage": "flash_bwd_shared_storage(192, 1, 1, 1, cutlass.BFloat16, 4, 8, 0, True, True, 0, head_dim_v=128, flex_mbars=9)",
                "tmem": (64, 192, 320, 0),
                "k_caps": (),
                "exact_mmas": (),
                # dQ runs as three 64-column chunk MMAs (TMEM slots 128 / 64 /
                # 128, K^T offset by whole atoms) drained one by one.
                "dq_boxes": None,
                "dq_chunks": ((128, 0), (64, 1024), (128, 2048)),
                "epi_chunks": (8, 12),
            },
        ),
    ],
)
def test_head_dim_render_plan(
    head_dim: int, head_dim_v: int, expect: dict[str, Any]
) -> None:
    """The smem storage (padded widths, ring depths that fit), the TMEM plan
    (dP / dV / dK / dQ column offsets), the exact K-tile caps of the S / dP
    MMAs, the exact-N instruction-descriptor MMAs, the dQ drain's 32-column
    box count and the dV / dK epilogue's 16-column chunk counts."""
    code = _render_plan(_tensors_dv(head_dim, head_dim_v))
    assert expect["storage"] in code
    dp_off, dv_off, dk_off, dq_off = expect["tmem"]
    assert f"tdP_t = cute.make_tensor(fbwd_tmem_ptr + {dp_off}, " in code
    assert f"tDV_t = cute.make_tensor(fbwd_tmem_ptr + {dv_off}, " in code
    assert f"tDK_t = cute.make_tensor(fbwd_tmem_ptr + {dk_off}, " in code
    assert f"tDQ_t = cute.make_tensor(fbwd_tmem_ptr + {dq_off}, " in code
    s_lines = [
        line for line in code.splitlines() if "gemm_ptx_dyn_qk(fbwd_s_addr" in line
    ]
    dp_lines = [
        line for line in code.splitlines() if "gemm_ptx_dyn_qk(fbwd_dp_addr" in line
    ]
    assert len(s_lines) == 2 and len(dp_lines) == 2
    if expect["k_caps"]:
        s_cap, dp_cap = expect["k_caps"]
        assert all(line.rstrip().endswith(f"{s_cap})") for line in s_lines), s_lines
        assert all(line.rstrip().endswith(f"{dp_cap})") for line in dp_lines), dp_lines
    else:
        assert not any("k_tiles" in line for line in s_lines + dp_lines)
    for name in ("_fbwd_ts_mma_x", "_fbwd_dsk_mma_x", "_fbwd_dq_mma_x"):
        assert (name in code) == (name in expect["exact_mmas"]), name
    if expect["dq_boxes"] is not None:
        assert f"for fbwd_c in cutlass.range_constexpr({expect['dq_boxes']}):" in code
    else:
        for col, kt_off in expect["dq_chunks"]:
            assert (
                f"gemm_ptx_dyn_qk(fbwd_dqa_addr + {col}, fbwd_st_dsmn, fbwd_dsmn_base, "
                f"tDQrDS[None, None, None, 0].layout, fbwd_st_kt + {kt_off}, " in code
            )
        reduce = code.split("if warp_idx < 4:")[1].split("if (warp_idx >= 4)")[0]
        # Chunks 0 and 2 share the original barrier pair, chunk 1 has its own
        # (flex_mbar slots 7 / 8).
        assert reduce.count("mbar_spin_wait(fbwd_dq_full_ptr") == 2
        assert reduce.count("mbar_spin_wait(fbwd_dq_full_b_ptr") == 1
        assert reduce.count("cute.copy(fbwd_tiled_dq_ld, tRDtDQ_B, tRDrDQ)") == 1
        assert "fbwd_dq_full_b_ptr = storage.flex_mbar.data_ptr() + 7" in code
        assert "fbwd_dq_empty_b_ptr = storage.flex_mbar.data_ptr() + 8" in code
        assert "'flex_dq_chunk': 64" in code
        # The MMA warp: per loop step chunk 2 issues once chunk 0 drained
        # (ahead of S, so its drain overlaps the S MMA and the S read), S
        # waits for chunk 1's drain and dP for chunk 2's; the tail drains 0
        # and 1 before its chunk 2; chunk commits to match.
        mma = code.split("if warp_idx == 12:")[1].split("if warp_idx < 4:")[0]
        assert mma.count("mbar_spin_wait(fbwd_dq_empty_ptr") == 3
        assert mma.count("mbar_spin_wait(fbwd_dq_empty_b_ptr") == 2
        assert mma.count("commit(fbwd_dq_full_ptr)") == 4
        assert mma.count("commit(fbwd_dq_full_b_ptr)") == 2
        head, loop = mma.split("for fbwd_i in cutlass.range(fbwd_steps - 1, unroll=1):")
        # A one-tile CTA keeps the plain prologue: dO and V awaited, then S0
        # (the persistent walk's order is pinned separately).
        assert (
            head.index("fbwd_doe = fbwd_do_cons.wait_and_advance()")
            < head.index("fbwd_ve = fbwd_v_cons.wait_and_advance()")
            < head.index("gemm_ptx_dyn_qk(fbwd_s_addr")
            < head.index("gemm_ptx_dyn_qk(fbwd_dp_addr")
        )
        chunk2 = loop.index("fbwd_st_kt + 2048")
        s_issue = loop.index("gemm_ptx_dyn_qk(fbwd_s_addr")
        dp_issue = loop.index("gemm_ptx_dyn_qk(fbwd_dp_addr")
        empty_waits = [
            i
            for i in range(len(loop))
            if loop.startswith("mbar_spin_wait(fbwd_dq_empty", i)
        ]
        # Loop: 0 (before chunk 2), 1 (before S), 2 (before dP); tail: 0, 1
        # (before its chunk 2).
        assert len(empty_waits) == 5
        assert (
            empty_waits[0]
            < chunk2
            < empty_waits[1]
            < s_issue
            < empty_waits[2]
            < dp_issue
        )
        # The dS staging's last reader is chunk 2: the staging-free barrier
        # (dk_done) commits right after it, not after dK.
        dk_done = loop.index("commit(fbwd_dk_done_ptr)")
        assert chunk2 < dk_done < empty_waits[1]
        assert (
            loop.index("helion_fbwd_dsk_idesc")
            < loop.index("fbwd_st_kt + 1024")
            < dk_done
        )
    # Epilogue chunks per warpgroup: dV (h=0) then dK (h=1), in emission order.
    chunk_counts = []
    for block in code.split("fbwd_dkv_shape = ")[1:]:
        body = block.split("fbwd_zrow = ")[0]
        chunk_counts.append(body.count("cute.copy(fbwd_tiled_epi_ld, tEPItT[None, "))
    assert tuple(c for c in chunk_counts if c) == expect["epi_chunks"], chunk_counts
    assert ("fbwd_sread_ptr" in code) == (dq_off == 0)


def test_head_dim_192_persistent_tile_boundary_order() -> None:
    """The persistent alias_s walk's tile boundary: the TMA warp issues the
    first step's Q and dO before K and V (their stages free before K's; V's
    frees last, after the dV epilogue), and the MMA warp puts S0 in the pipe
    before it waits for dO, V and the previous tile's chunk-2 drain."""
    bound = _bind(_args(_tensors_dv(192, 128), _causal))
    code = _render(bound, cute_flash_bwd_persistent=1)
    assert "fbwd_clc_hw" in code
    # The single dO slot doubles as the dV staging (V is released by the MMA
    # warp after the last dP), so the stages free as V, Q, K, dO: the TMA
    # warp issues the first step's loads in that order (Q first: S0's).
    tma = code.split("if warp_idx == 13:")[1].split("if warp_idx == 12:")[0]
    assert (
        tma.index("cute.copy(_fbwd_tma_q,")
        < tma.index("cute.copy(_fbwd_tma_v,")
        < tma.index("cute.copy(_fbwd_tma_k,")
        < tma.index("cute.copy(_fbwd_tma_do,")
    )
    mma = code.split("if warp_idx == 12:")[1].split("if warp_idx < 4:")[0]
    head, loop = mma.split("for fbwd_i in cutlass.range(fbwd_steps - 1, unroll=1):")
    assert (
        head.index("gemm_ptx_dyn_qk(fbwd_s_addr")
        < head.index("fbwd_doe = fbwd_do_cons.wait_and_advance()")
        < head.index("fbwd_ve = fbwd_v_cons.wait_and_advance()")
        < head.index("mbar_spin_wait(fbwd_dq_empty_ptr")
        < head.index("mbar_spin_wait(fbwd_sread_ptr")
        < head.index("gemm_ptx_dyn_qk(fbwd_dp_addr")
    )
    # The tile's last dO is handed back by the dV epilogue (it stages through
    # the slot); every other dO by the MMA warp. V is released once the last
    # dV is in the pipe (after dv_done's commit, before the tail).
    assert "if fbwd_steps > 1:\n                    fbwd_doe.release()" in head
    assert (
        "if fbwd_i + 2 < fbwd_steps:\n                        fbwd_doe.release()"
        in loop
    )
    assert mma.count("fbwd_ve.release()") == 1
    assert (
        loop.index("commit(fbwd_dv_done_ptr)")
        < loop.index("fbwd_ve.release()")
        < loop.rindex("helion_fbwd_dsk_idesc")
    )
    wg0 = code.split("if (warp_idx >= 4) & (warp_idx < 8):")[1].split(
        "if (warp_idx >= 8)"
    )[0]
    assert "fbwd_sEpi = storage.sdO.get_tensor(_fbwd_epil.outer" in wg0
    assert "fbwd_kv_empty_ptr = storage.do_mbar_ptr.data_ptr() + 1" in wg0
    assert "storage.sV.get_tensor(_fbwd_epil" not in code


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


# -- the step table: one table per walk, or chunks of it ---------------------


def test_whole_walk_in_one_table_by_default() -> None:
    # 512 rows, 4 Q tiles, one q head per kv head: 4 steps per KV tile.
    t = _tensors()
    bound = _bind(_args(t, _causal))
    plan = _match_or_fail(bound).plan
    assert plan.max_steps == 4
    smem = _fit(plan)
    assert smem.table_steps == plan.max_steps
    assert _fit(plan, persistent=True).table_steps == plan.max_steps
    code = _render(bound)
    assert "flash_bwd_shared_storage(64, 4, 4, 1, cutlass.BFloat16, 4, 8," in code
    assert "fbwd_nchunks" not in code and "named_barrier_wait_unaligned(7" not in code
    # No cap below the walk: the knob is not searched.
    assert fbwd_table_step_choices(plan, _CPU) == (0,)
    assert "cute_flash_bwd_table_steps" not in bound.config_spec._flat_fields()
    # An explicit cap below the walk chunks it: a 3-step table, every role
    # wrapped in the chunk loop with the 448-thread boundary barrier.
    assert _fit(plan, table_cap=3).table_steps == 3
    chunked = _render(bound, cute_flash_bwd_table_steps=3)
    assert "flash_bwd_shared_storage(64, 4, 4, 1, cutlass.BFloat16, 3, 8," in chunked
    assert chunked.count("fbwd_nchunks = (fbwd_steps + 2) // 3") == 5
    assert chunked.count("named_barrier_wait_unaligned(7, 448)") == 10
    assert "if fbwd_chunk == fbwd_nchunks - 1:" in chunked
    assert "if (fbwd_i > 0) | (fbwd_chunk > 0):" in chunked
    # Persistent: the double buffer holds two 3-step tables and the length
    # words; the readers load the list counts for the chunk fills.
    persistent = _render(
        bound, cute_flash_bwd_table_steps=3, cute_flash_bwd_persistent=1
    )
    assert "flash_bwd_shared_storage(64, 4, 4, 1, cutlass.BFloat16, 8, 8," in persistent
    assert "fbwd_sSteps[6 + fbwd_tbuf]" in persistent
    assert "if (fbwd_i == 0) & (fbwd_chunk == 0) & (fbwd_nz_it > 0):" in persistent
    assert persistent.count("_fbwd_full_cnt[fbwd_tb, fbwd_th0, fbwd_n_tile]") == 5


def test_table_caps_below_the_walk_are_searched() -> None:
    # 8 q heads per kv head, 128 Q tiles: 1024 steps per KV tile.
    t = _long_tensors(8, 1, 16384, 64)
    bound = _bind(_args(t, _causal, block_mask=_causal_block_mask(16384, 16384)))
    plan = _match_or_fail(bound).plan
    assert plan.max_steps == 1024 and _fit(plan).table_steps == 1024
    assert fbwd_table_step_choices(plan, _CPU) == (0, 512)
    fields = bound.config_spec._flat_fields()
    assert fields["cute_flash_bwd_table_steps"].choices == (0, 512)
    assert bound.config_spec.default_config().config["cute_flash_bwd_table_steps"] == 0
    assert _fit(plan, table_cap=512).table_steps == 512
    assert "fbwd_nchunks = (fbwd_steps + 511) // 512" in _render(
        bound, cute_flash_bwd_table_steps=512
    )


@pytest.mark.parametrize(
    ("heads", "seq", "head_dim"), [(8, 131072, 128), (8, 131072, 64), (4, 262144, 128)]
)
def test_long_walk_chunks_the_step_table(heads: int, seq: int, head_dim: int) -> None:
    # Group-8 (or 4) GQA at 128K (256K) tokens: 8192 steps per KV tile, twice
    # what the shared memory holds; the walk runs in chunks instead of
    # declining (the old bound was ~4266 steps at head dim 128).
    t = _long_tensors(heads, 1, seq, head_dim)
    bound = _bind(_args(t, _causal, block_mask=_causal_block_mask(seq, seq)))
    plan = _match_or_fail(bound).plan
    assert plan.max_steps == 8192
    smem = _fit(plan)
    assert 1024 <= smem.table_steps < plan.max_steps and smem.table_steps % 32 == 0
    assert smem.dq_boxes == 4
    code = _render(bound)
    assert (
        f"fbwd_nchunks = (fbwd_steps + {smem.table_steps - 1}) // {smem.table_steps}"
        in code
    )
    assert "named_barrier_wait_unaligned(7, 448)" in code
    # The persistent walk's double buffer halves the chunk.
    smem_p = _fit(plan, persistent=True)
    assert 1024 <= smem_p.table_steps <= smem.table_steps // 2 + 32
    persistent = _render(bound, cute_flash_bwd_persistent=1)
    assert f"fbwd_sSteps[{2 * smem_p.table_steps} + fbwd_tbuf]" in persistent
    assert "fbwd_tissued_ptr" in persistent
    # Caps at or above the automatic table (2112 steps per persistent buffer)
    # would render the automatic plan: not offered.
    assert fbwd_table_step_choices(plan, _CPU) == (0, 512, 1024, 2048)
    assert smem_p.table_steps > 2048


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


def test_env_kill_switch_covers_the_flex_backward() -> None:
    # HELION_CUTE_FLASH_BWD=0 disables the dense fused backward; the flex
    # backward surface and its loop detector honour it too.
    t = _tensors()
    with patch.dict(os.environ, {"HELION_CUTE_FLASH_BWD": "0"}):
        bound = _bind(_args(t, _causal))
        assert not bound.config_spec.cute_flash_bwd_search_enabled
        assert "cute_flash_bwd_regs" not in bound.config_spec._flat_fields()
    bound = _bind(_args(t, _causal))
    assert bound.config_spec.cute_flash_bwd_search_enabled


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


def test_head_dim_256_pair_declines_strided_operands() -> None:
    """The pair kernel's maps are the contiguous (seq, D, B*H) ones: at head
    dim 256 the ``[B, S, H, D]`` views the rank-4 maps serve elsewhere leave
    the fused path."""
    bound = _bind(_args(_transposed_tensors(head_dim=256), _causal))
    reasons: list[str] = []
    assert _match(bound, reasons) is None
    assert reasons[0] == "flex_bwd: the pair kernel takes contiguous operands only"
    assert not bound.config_spec.cute_flash_bwd_search_enabled


def test_head_dim_256_pair_declines_without_forward_frame_regions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pair kernel's dQ clusters render the hooks in the forward frame;
    when that second capture fails (the first, in the transposed frame,
    succeeded) the pair plan is declined rather than rendered without
    them."""
    original = flex_bwd_match_module._capture_bwd_regions
    frames: list[tuple[object, object]] = []

    def forward_frame_capture_fails(ctx, *args: object, **kwargs: object) -> object:
        # The matcher swaps the axis roles for the forward-frame capture.
        frame = (ctx.m_id, ctx.kv_id)
        if frames and frame == frames[0][::-1]:
            return None
        frames.append(frame)
        return original(ctx, *args, **kwargs)

    monkeypatch.setattr(
        flex_bwd_match_module, "_capture_bwd_regions", forward_frame_capture_fails
    )
    bound = _bind(_args(_tensors_dv(256, 256), _causal))
    reasons: list[str] = []
    assert _match(bound, reasons) is None
    assert reasons[0] == "flex_bwd: the pair kernel needs the forward-frame regions"
    assert frames and not bound.config_spec.cute_flash_bwd_search_enabled


def _without_source_comments(code: str) -> str:
    """The generated module without its ``# src[file:line]`` comments (line
    numbers of the example source, not generated code)."""
    return re.sub(r"^[ \t]*# src\[.*\n", "", code, flags=re.MULTILINE)


def _assert_s_first_order(code: str, *, on: bool) -> None:
    """The first step's load order in the TMA warp's step block (the last
    ``warp_idx == 13`` region: a one-tile CTA issues K in an earlier prologue
    block) and the MMA warp's first-step wait order. ``fbwd_aug_row_fill`` is
    the stats gather of the statistics-MMA form; without it the knob only
    moves dO ahead of V."""
    tma = code.split("if warp_idx == 13:")[-1].split("if warp_idx == 12:")[0]
    first_do = tma.index("cute.copy(_fbwd_tma_do,")
    first_v = tma.index("cute.copy(_fbwd_tma_v,")
    gather = tma.find("fbwd_aug_row_fill(")
    mma = code.split("if warp_idx == 12:")[-1].split("if warp_idx < 4:")[0]
    head = mma.split("for fbwd_i in cutlass.range(fbwd_steps - 1, unroll=1):")[0]
    s0 = head.index("gemm_ptx_dyn_qk(fbwd_s_addr")
    do_wait = head.index("fbwd_doe = fbwd_do_cons.wait_and_advance()")
    v_wait = head.index("fbwd_ve = fbwd_v_cons.wait_and_advance()")
    dp0 = head.index("gemm_ptx_dyn_qk(fbwd_dp_addr")
    if on:
        assert first_do < first_v and (gather < 0 or first_do < gather < first_v)
        assert s0 < do_wait < v_wait < dp0
    else:
        assert first_v < first_do and (gather < 0 or gather < first_v)
        assert do_wait < v_wait < s0 < dp0


def test_s_first_knob_surface() -> None:
    """``cute_flash_bwd_s_first`` is offered on the plain TMEM plans with
    "auto" (follows the form: on for the persistent walk, off for a one-tile
    CTA) as the default and the two explicit values as the searched set; the
    alias_s plan (its own measured boundary orders) does not offer it, and a
    config that carries it there or under the 2-CTA family reads "auto"
    (one identity per render)."""
    bound = _bind(_args(_tensors(), _causal))
    fields = bound.config_spec._flat_fields()
    knob = fields["cute_flash_bwd_s_first"]
    assert knob.choices == ("auto", 0, 1) and knob.search_choices == (0, 1)
    assert bound.config_spec.default_config().config["cute_flash_bwd_s_first"] == "auto"
    plain_off = _render(bound)
    plain_auto = _render(bound, cute_flash_bwd_s_first="auto")
    assert plain_auto == plain_off  # one-tile CTA: auto reads off
    _assert_s_first_order(plain_off, on=False)
    _assert_s_first_order(_render(bound, cute_flash_bwd_s_first=1), on=True)
    persistent_on = _render(bound, cute_flash_bwd_persistent=1)
    assert persistent_on == _render(
        bound, cute_flash_bwd_persistent=1, cute_flash_bwd_s_first=1
    )
    alias_s = _bind(_args(_tensors_dv(192, 128), _causal))
    assert "cute_flash_bwd_s_first" not in alias_s.config_spec._flat_fields()
    config = helion.Config(
        **{**alias_s.config_spec.default_config().config, "cute_flash_bwd_s_first": 1}
    )
    alias_s.config_spec.normalize(config)
    assert config.config["cute_flash_bwd_s_first"] == "auto"
    pair = _bind_d128(None, q_seq=512, kv_seq=512)
    config = helion.Config(
        **{
            **pair.config_spec.default_config().config,
            "cute_flash_bwd_two_cta": 1,
            "cute_flash_bwd_s_first": 1,
        }
    )
    pair.config_spec.normalize(config)
    assert config.config["cute_flash_bwd_s_first"] == "auto"


def test_kv_stage_knob_surface() -> None:
    """``cute_flash_bwd_kv_stage`` (a second K / V ring slot so the persistent
    walk loads the next tile's K / V while the current tile computes) is
    offered where the storage holds the slots beside the statistics form: at
    d64 the Q / dO rings give up depth for it (4 / 3 -> 3 / 2, 217,088 B);
    at d128 the one-slot q1 / do1 storage is already 201 KB of 227, and the
    alias_s plan keeps its own boundary schedule. A one-tile CTA and the
    2-CTA family read 1."""
    bound = _bind(_args(_tensors(), _causal))
    fields = bound.config_spec._flat_fields()
    assert fields["cute_flash_bwd_kv_stage"].choices == (1, 2)
    assert bound.config_spec.default_config().config["cute_flash_bwd_kv_stage"] == 1
    plan = _match(bound).plan
    from helion._compiler.cute.cute_flash_bwd_flex import _fbwd_fit_kwargs
    from helion._compiler.cute.cute_flash_bwd_flex import _fit_smem_budget

    two = _fit_smem_budget(
        plan,
        stats_mma=True,
        raw_score=False,
        device=_CPU,
        persistent=True,
        kv_stage=2,
        **_fbwd_fit_kwargs(plan),
    )
    assert two is not None
    assert (two.q_stage, two.do_stage, two.kv_stage, two.aug_stages) == (3, 2, 2, 3)
    plain_two = _fit_smem_budget(
        plan, stats_mma=False, raw_score=False, device=_CPU, persistent=True, kv_stage=2
    )
    assert plain_two is not None and (plain_two.q_stage, plain_two.do_stage) == (4, 3)
    for other in (
        _bind(_args(_tensors(head_dim=128), _causal)),
        _bind(_args(_tensors_dv(192, 128), _causal)),
    ):
        assert "cute_flash_bwd_kv_stage" not in other.config_spec._flat_fields()
    for extra in ({}, {"cute_flash_bwd_persistent": 1, "cute_flash_bwd_two_cta": 1}):
        config = helion.Config(
            **{
                **bound.config_spec.default_config().config,
                **extra,
                "cute_flash_bwd_kv_stage": 2,
            }
        )
        bound.config_spec.normalize(config)
        assert config.config["cute_flash_bwd_kv_stage"] == 1, extra
    assert _render(
        bound, cute_flash_bwd_persistent=1, cute_flash_bwd_kv_stage=1
    ) == _render(bound, cute_flash_bwd_persistent=1)


def _assert_kv_ring(code: str, *, slots: int) -> None:
    storage = code.split("flash_bwd_shared_storage(")[1].split(")")[0].split(", ")
    assert storage[3] == str(slots), storage[:4]
    stages = code.count("PipelineTmaUmma.create(num_stages=2,")
    assert (stages >= 2) == (slots == 2)
    if slots == 2:
        assert (
            "fbwd_kv_empty_ptr = storage.v_mbar_ptr.data_ptr() + 2 + fbwd_nz_it % 2"
            in code
        )
        assert "fbwd_epi_slot0 = 1 * (fbwd_nz_it % 2)" in code
        assert "'kv_stage': 2" in code
    else:
        assert "fbwd_kv_empty_ptr = storage.v_mbar_ptr.data_ptr() + 1\n" in code
        assert "'kv_stage': 1" in code


# unittest runners ignore pytestmark.
@onlyBackends(["cute"])
class TestContiguousRenderPin(TestCase):
    """The exact text of one contiguous backward render (the
    ``.expected_cute`` journal next to this file): the strided-operand
    support must not change the code contiguous inputs produce."""

    def test_contiguous_persistent_s_first_render(self) -> None:
        """The persistent plain walk (the statistics-MMA form the campaign's
        winners run) with ``cute_flash_bwd_s_first`` at its "auto" default
        (on for the persistent walk): the tile's first S MMA is issued before
        the dO / V waits and the first dO load goes out behind Q, ahead of
        the stats gather, V last."""
        bound = _bind(_args(_tensors(), _causal))
        code = _render(bound, cute_flash_bwd_persistent=1, cute_flash_bwd_stats_mma=1)
        _assert_s_first_order(code, on=True)
        self.assertExpectedJournal(_without_source_comments(code))

    def test_contiguous_persistent_s_first_off_render(self) -> None:
        """The same walk with the knob off: the dO / V waits ahead of S and
        the first dO load behind the stats gather and V."""
        bound = _bind(_args(_tensors(), _causal))
        code = _render(
            bound,
            cute_flash_bwd_persistent=1,
            cute_flash_bwd_stats_mma=1,
            cute_flash_bwd_s_first=0,
        )
        _assert_s_first_order(code, on=False)
        self.assertExpectedJournal(_without_source_comments(code))

    def test_contiguous_persistent_kv_stage_render(self) -> None:
        """The persistent statistics-MMA walk with two K / V ring slots: the
        storage and the K / V pipelines two deep (the Q / dO rings 3 / 2),
        the dV epilogue staged through the current tile's V slot and its
        arrive on that slot's empty barrier, the plan's depth for the host
        layouts."""
        bound = _bind(_args(_tensors(), _causal))
        code = _render(
            bound,
            cute_flash_bwd_persistent=1,
            cute_flash_bwd_stats_mma=1,
            cute_flash_bwd_kv_stage=2,
        )
        _assert_kv_ring(code, slots=2)
        _assert_kv_ring(
            _render(bound, cute_flash_bwd_persistent=1, cute_flash_bwd_stats_mma=1),
            slots=1,
        )
        self.assertExpectedJournal(_without_source_comments(code))

    def test_contiguous_backward_render(self) -> None:
        bound = _bind(_args(_tensors(), _causal))
        with _as_sm100():
            code = bound.to_code(bound.config_spec.default_config())
        assert "'flex_strides'" not in code
        self.assertExpectedJournal(_without_source_comments(code))


# -- the 2-CTA kernel (head dim 128) -------------------------------------------


def _bind_d128(mask_mod: Callable[..., Any] | None, **shape: int) -> BoundKernel:
    t = _tensors(**{"head_dim": 128, **shape})
    return _bind(_args(t, mask_mod))


def test_two_cta_legality() -> None:
    """The pair kernel takes head dims (128, 128), an even KV tile count and
    contiguous maps; the knob otherwise renders the 1-CTA kernel."""
    from helion._compiler.cute.cute_flash_bwd_flex_2cta import flex_bwd_two_cta_allowed
    from helion._compiler.cute.cute_flash_bwd_flex_2cta import (
        flex_bwd_two_cta_table_steps,
    )

    bound = _bind_d128(None, q_seq=512, kv_seq=512)
    plan = _match_or_fail(bound).plan
    assert flex_bwd_two_cta_allowed(plan, _CPU)
    # The merged walk of a pair is at most both tiles' lists end to end.
    assert flex_bwd_two_cta_table_steps(plan, _CPU) == 2 * plan.max_steps
    code = _render(bound, cute_flash_bwd_two_cta=1)
    assert "flash_bwd_flex_2cta_shared_storage" in code
    assert "'flex_two_cta': True" in code
    assert "cta_group=2" in code
    # The dQ drain: the dense kernel's (32 x 32) boxes, the pair's 64 Q rows
    # per CTA.
    assert "(32, 32)" in code
    # A masked walk: the pair's two KV tiles list different Q blocks; the
    # kernel merges them (the absent flag at bit 25 of an entry) and the
    # masked steps run the mask hook.
    causal = _bind_d128(_causal, q_seq=512, kv_seq=512)
    assert flex_bwd_two_cta_allowed(_match_or_fail(causal).plan, _CPU)
    code = _render(causal, cute_flash_bwd_two_cta=1)
    assert "flash_bwd_flex_2cta_shared_storage" in code
    assert "fbwd_absent = fbwd_e >> 25 & 1" in code
    assert "if fbwd_masked != 0:" in code
    assert "cute.arch.vote_ballot_sync(fbwd_keep)" in code
    # An odd KV tile count has no pair for the last tile.
    odd = _bind_d128(None, q_seq=512, kv_seq=384)
    assert not flex_bwd_two_cta_allowed(_match_or_fail(odd).plan, _CPU)
    assert "flash_bwd_flex_2cta_shared_storage" not in _render(
        odd, cute_flash_bwd_two_cta=1
    )
    d64 = _bind(_args(_tensors(q_seq=512, kv_seq=512), None))
    assert "flash_bwd_flex_2cta_shared_storage" not in _render(
        d64, cute_flash_bwd_two_cta=1
    )


def test_two_cta_table_chunks_the_walk() -> None:
    """A walk longer than the pair's step table (its shared memory is at the
    SM100 limit) runs in chunks: 8 q heads x 512 Q tiles do not fit, so the
    prologue writes the CTA's whole merged walk to its region of the walk
    scratch (a kernel argument the launch allocates, ``2 * max_steps``
    entries per CTA) and the roles wrap their step loops in the chunk loop
    that copies the next chunk out of it at the working warps' barrier; the
    ``cute_flash_bwd_table_steps`` cap chunks a short walk the same way; a
    walk that fits the table needs no scratch."""
    from helion._compiler.cute.cute_flash_bwd_flex_2cta import flex_bwd_two_cta_allowed
    from helion._compiler.cute.cute_flash_bwd_flex_2cta import (
        flex_bwd_two_cta_table_steps,
    )

    bound = _bind_d128(None, heads=8, kv_heads=1, q_seq=65536, kv_seq=512)
    plan = _match_or_fail(bound).plan
    assert plan.max_steps == 4096
    steps = flex_bwd_two_cta_table_steps(plan, _CPU)
    assert 0 < steps < plan.max_steps and steps % 32 == 0
    assert flex_bwd_two_cta_allowed(plan, _CPU)
    code = _render(bound, cute_flash_bwd_two_cta=1)
    assert f"fbwd_nchunks = (fbwd_steps + {steps - 1}) // {steps}" in code
    assert "named_barrier_wait_unaligned(7, 448)" in code
    assert "if fbwd_chunk == fbwd_nchunks - 1:" in code
    # The follower's MMA warp walks the chunk loop too (the barrier counts it).
    assert "if (fbwd_rank != 0) & (fbwd_steps > 0):" in code
    # The walk scratch: 2 batches x 1 kv head x 4 KV tiles of CTAs, each
    # 2 * max_steps entries, allocated by the launch from the Q operand.
    assert "_helion_flash_flex_bwd_walk_scratch(query, ctas=8, stride=8192)" in code
    assert "from helion.runtime.cute.flash_flex import flex_bwd_walk_scratch" in code
    assert "fbwd_woff = cutlass.Int32(cute.arch.block_idx()[0]) * 8192" in code
    assert "_fbwd_walk_ws[fbwd_woff + fbwd_gbase + fbwd_j] = fbwd_entry" in code
    assert "if fbwd_gbase + fbwd_j < 8192:" in code
    assert "fbwd_steps = cutlass.min(fbwd_gbase, cutlass.Int32(8192))" in code
    # The refill is a copy of the chunk's window out of the scratch by the
    # 448 working threads (uniform trip count, the thread's share behind a
    # predicate); only the prologue builds the bitmaps (four set loops).
    assert re.search(
        r"named_barrier_wait_unaligned\(7, 448\)\n\s+"
        r"for fbwd_ib in cutlass\.range\(0, fbwd_csteps, 448, unroll=1\):\n\s+"
        r"fbwd_ci = fbwd_ib \+ tidx\n\s+"
        r"if fbwd_ci < fbwd_csteps:\n\s+"
        rf"fbwd_sSteps\[fbwd_ci\] = cutlass\.Int32\(_fbwd_walk_ws\[fbwd_woff \+ fbwd_chunk \* {steps} \+ fbwd_ci\]\)",
        code,
    )
    assert code.count("cute.arch.atomic_or(") == 4
    # The bitmap loops run over uniform bounds with a per-thread predicate:
    # shared atomics in a loop whose trip count is per thread make ptxas
    # guard the whole chunked kernel as divergent.
    assert "for fbwd_jb in cutlass.range(0, fbwd_cf0, 512, unroll=1):" in code
    assert "for fbwd_jb in cutlass.range(tidx" not in code
    # A barrier precedes every head's bitmap clear: the previous head's L0
    # fill reads the bitmaps (rank 1's status of each entry) in a per-thread
    # loop with no barrier after it, and a head whose odd tile lists nothing
    # runs no other barrier before the next head's clear. Without it the
    # clearing warps zeroed a bitmap a slower warp was still reading and that
    # warp dropped the step.
    assert re.search(
        r"cute\.arch\.sync_threads\(\)\n\s+"
        r"for fbwd_wb in cutlass\.range\(0, 51, 512, unroll=1\):",
        code,
    )
    # The bitmaps hold the 512 Q blocks plus a sentinel for the blocks a
    # BlockMask built for a longer Q lists past the sequence (17 words each,
    # 51 in all); every index into them is clamped to the sentinel, so such
    # entries never reach the mbarrier words after the bitmaps.
    assert "storage.sBits.get_tensor(cute.make_layout(51))" in code
    assert (
        "fbwd_x = cutlass.min(cutlass.Int32(_fbwd_full_idx[fbwd_tb, fbwd_th, fbwd_pt0, fbwd_j]), cutlass.Int32(512))"
        in code
    )
    assert "fbwd_sBits.iterator + (0 + fbwd_x // 32)" in code
    assert "fbwd_xc = cutlass.min(fbwd_xb, cutlass.Int32(512))" in code
    assert "fbwd_sBits[17 + fbwd_xc // 32]" in code
    assert "fbwd_sBits[34 + fbwd_xc // 32]" in code
    assert "fbwd_sBits[0 + fbwd_xc // 32]" in code
    assert "fbwd_xb // 32" not in code
    # The pair's step table is its own: a ``cute_flash_bwd_table_steps``
    # cap at or above its automatic table renders that table and reads 0
    # under the knob (one autotune identity); a smaller cap chunks the walk
    # and stays; the 1-CTA kernel keeps every cap of its legal set.
    spec = bound.config_spec
    assert spec._cute_flash_bwd_two_cta_table_steps == steps
    for cap, expect in (
        (1024, 0),
        (steps, 0),
        (steps + 4, 0),
        (steps - 32, steps - 32),
        (5, 5),
        (0, 0),
    ):
        config = spec.default_config()
        config.config.update(
            {"cute_flash_bwd_two_cta": 1, "cute_flash_bwd_table_steps": cap}
        )
        spec.normalize(config)
        assert config.config["cute_flash_bwd_table_steps"] == expect, cap
    config = spec.default_config()
    config.config["cute_flash_bwd_table_steps"] = 1024
    spec.normalize(config)
    assert config.config["cute_flash_bwd_table_steps"] == 1024
    # A cap that is not a multiple of 32 and does not fit resolves to the
    # largest multiple of 32 at or below it that does (``steps`` here), not
    # to the one below it.
    for cap in (steps + 1, steps + 31):
        assert flex_bwd_two_cta_table_steps(plan, _CPU, cap) >= steps
    short = _bind_d128(_causal, heads=4, kv_heads=2, q_seq=8192, kv_seq=1024)
    assert flex_bwd_two_cta_table_steps(_match_or_fail(short).plan, _CPU, 64) == 64
    code = _render(short, cute_flash_bwd_two_cta=1, cute_flash_bwd_table_steps=64)
    assert "fbwd_nchunks = (fbwd_steps + 63) // 64" in code
    # A walk that fits the table (64 merged steps; the 227 KB leave ~160 at
    # this shape) renders without the chunk loop.
    tiny = _bind_d128(_causal, heads=4, kv_heads=2, q_seq=2048, kv_seq=1024)
    tiny_plan = _match_or_fail(tiny).plan
    assert flex_bwd_two_cta_table_steps(tiny_plan, _CPU) == 2 * tiny_plan.max_steps
    whole = _render(tiny, cute_flash_bwd_two_cta=1)
    assert "fbwd_nchunks" not in whole
    # ... and fills the table directly: no scratch argument, no copy.
    assert "_fbwd_walk_ws" not in whole and "fbwd_walk_ws" not in whole
    assert "_helion_flash_flex_bwd_walk_scratch" not in whole
    assert "fbwd_sSteps[fbwd_gbase + fbwd_j] = fbwd_entry" in whole
    assert f"if fbwd_gbase + fbwd_j < {2 * tiny_plan.max_steps}:" in whole
    assert "fbwd_sSteps[fbwd_ci] = " not in whole


def test_two_cta_search_surface_and_seed() -> None:
    """The knob is searched where the pair kernel is legal (head dims
    (128, 128), an even KV tile count), its seed carries the pair's register
    split, and the knobs the pair has no form of read 0 under it."""
    bound = _bind_d128(_causal, heads=4, kv_heads=2, q_seq=512, kv_seq=512)
    spec = bound.config_spec
    assert spec._cute_flash_bwd_two_cta_allowed
    assert spec._flat_fields()["cute_flash_bwd_two_cta"].choices == (0, 1)
    seeds = [
        c
        for c in spec.autotune_seed_configs()
        if c.config.get("cute_flash_bwd_two_cta") == 1
    ]
    assert [seed.config["cute_flash_bwd_regs"] for seed in seeds] == [
        "88/168/88",
        "104/152/104",
    ]
    assert all(seed.config["cute_flash_bwd_persistent"] == 0 for seed in seeds)
    # Under the pair the knob's generic default (the 1-CTA kernel's split)
    # reads as the first seed split; an explicit other split and the 1-CTA
    # kernel's default are untouched.
    from helion._compiler.cute.cute_flash_bwd_flex import FBWD_REG_SPLITS

    pinned = spec.default_config()
    pinned.config["cute_flash_bwd_two_cta"] = 1
    assert pinned.config["cute_flash_bwd_regs"] == FBWD_REG_SPLITS[0]
    spec.normalize(pinned, _fix_invalid=True)
    assert pinned.config["cute_flash_bwd_regs"] == "88/168/88"
    other_split = spec.default_config()
    other_split.config.update(
        {"cute_flash_bwd_two_cta": 1, "cute_flash_bwd_regs": "104/152/104"}
    )
    spec.normalize(other_split, _fix_invalid=True)
    assert other_split.config["cute_flash_bwd_regs"] == "104/152/104"
    single = spec.default_config()
    spec.normalize(single, _fix_invalid=True)
    assert single.config["cute_flash_bwd_regs"] == FBWD_REG_SPLITS[0]
    assert "setmaxregister_increase(168)" in _render(bound, cute_flash_bwd_two_cta=1)
    config = spec.default_config()
    config.config.update(
        {
            "cute_flash_bwd_two_cta": 1,
            "cute_flash_bwd_persistent": 1,
            "cute_flash_bwd_stats_mma": 1,
        }
    )
    spec.normalize(config)
    assert config.config["cute_flash_bwd_persistent"] == 0
    assert config.config["cute_flash_bwd_stats_mma"] == 0
    assert config.config["cute_flash_bwd_two_cta"] == 1
    # A 1-CTA config keeps its knobs.
    config = spec.default_config()
    config.config["cute_flash_bwd_persistent"] = 1
    spec.normalize(config)
    assert config.config["cute_flash_bwd_persistent"] == 1
    # The whole merged walk fits the pair's table here: every cap renders it.
    plan = _match_or_fail(bound).plan
    assert spec._cute_flash_bwd_two_cta_table_steps == 2 * plan.max_steps
    config = spec.default_config()
    config.config.update(
        {"cute_flash_bwd_two_cta": 1, "cute_flash_bwd_table_steps": 512}
    )
    spec.normalize(config)
    assert config.config["cute_flash_bwd_table_steps"] == 0
    for other in (
        _bind(_args(_tensors(q_seq=512, kv_seq=512), _causal)),
        _bind_d128(_causal, q_seq=512, kv_seq=384),
    ):
        assert not other.config_spec._cute_flash_bwd_two_cta_allowed
        assert other.config_spec._cute_flash_bwd_two_cta_table_steps == 0
        assert other.config_spec._flat_fields()["cute_flash_bwd_two_cta"].choices == (
            0,
        )
        assert not other.config_spec.cute_flash_bwd_seed_configs()
        # Pinned anyway, the knob renders the 1-CTA kernel, whose knobs all
        # apply: it reads 0 and strips nothing.
        config = other.config_spec.default_config()
        config.config.update(
            {
                "cute_flash_bwd_two_cta": 1,
                "cute_flash_bwd_persistent": 1,
                "cute_flash_bwd_stats_mma": 1,
            }
        )
        other.config_spec.normalize(config)
        assert config.config["cute_flash_bwd_two_cta"] == 0
        assert config.config["cute_flash_bwd_persistent"] == 1
        assert config.config["cute_flash_bwd_stats_mma"] == 1


# unittest runners ignore pytestmark.
@onlyBackends(["cute"])
class TestTwoCtaRenderPin(TestCase):
    """The exact text of one 2-CTA backward render (dense walk, GQA, head
    dim 128; the ``.expected_cute`` journal next to this file)."""

    def test_two_cta_backward_render(self) -> None:
        bound = _bind_d128(None, heads=4, kv_heads=2, q_seq=512, kv_seq=512)
        with _as_sm100():
            config = bound.config_spec.default_config()
            code = bound.to_code(
                helion.Config(**{**config.config, "cute_flash_bwd_two_cta": 1})
            )
        assert "flash_bwd_flex_2cta_shared_storage" in code
        self.assertExpectedJournal(_without_source_comments(code))
