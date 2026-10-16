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
from helion._compiler.cute.cute_flash_bwd_flex import additive_bias_region
from helion._compiler.cute.flex_bwd_match import FlexBwdMatch
from helion._compiler.cute.flex_bwd_match import match_flex_attention_bwd
from helion._testing import EXAMPLES_DIR
from helion._testing import import_path
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.runtime.kernel import Kernel

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

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
    assert spec._cute_flash_bwd_flex and not spec._cute_flash_bwd_persistent_allowed
    config = spec.default_config()
    assert config.config["cute_flash_bwd_regs"] == "120/152/88"
    assert config.config["cute_flash_bwd_exp2_f32"] == 1


def test_runtime_scale_and_grouped_heads() -> None:
    t = _tensors(heads=8, kv_heads=2, head_dim=128)
    bound = _bind(_args(t, _causal, scale=0.0883883))
    match = _match_or_fail(bound)
    assert match.plan.group == 4 and match.plan.head_dim == 128
    assert match.plan.max_steps == 4 * 4


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
    # The fold's region is the bias alone: the score placeholder is unused.
    score_ph = next(n for n in g.graph.nodes if n.op == "placeholder")
    assert score_ph.target == "score" and not score_ph.users
    assert g.output.target is hl.load


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


def test_declines_kv_block_other_than_128() -> None:
    t = _tensors()
    reasons: list[str] = []
    assert _match(_bind(_args(t, _causal, block_size=(128, 64))), reasons) is None
    assert any("KB" in r for r in reasons), reasons
