"""CPU tests of the FlexAttention-envelope matcher (``cute/flex_match.py``).

Every test binds ``examples/flex_attention.py`` (or a textual variant of it)
on the CPU with real ``create_block_mask`` BlockMasks and asserts the plan the
matcher produces, the autotune surface it feeds and the loop ownership it
records at codegen. Nothing here compiles or launches CuTe.
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
from torch.nn.attention.flex_attention import _create_empty_block_mask
from torch.nn.attention.flex_attention import _identity
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import noop_mask

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from test._cute_aux import _cpu_codegen

import helion
from helion._compiler import backend as _compiler_backend
from helion._compiler.cute import cute_flash
from helion._compiler.cute import flex_match
from helion._compiler.cute.attention_plan import causal_score_plan
from helion._compiler.cute.attention_plan import dense_score_plan
from helion._compiler.cute.flex_match import FlexMatch
from helion._compiler.cute.flex_match import flex_host_scalar_symbol
from helion._compiler.cute.flex_match import match_flex_envelope
from helion._compiler.variable_origin import ClosureOrigin
from helion._compiler.variable_origin import NameOrigin
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
# Every BlockMask here is built on the CPU on purpose: these tests exercise
# the matcher on the traced IR and never launch a kernel.
_CPU = torch.device("cpu")


def _cute_kernel(kernel: Kernel) -> Kernel:
    """``kernel`` on the CuTe backend with every other example setting kept."""
    return Kernel(kernel.fn, settings=kernel.settings.copy(backend="cute"))


@functools.cache
def _flex_kernel() -> Kernel:
    module = import_path(EXAMPLES_DIR / "flex_attention.py")
    return _cute_kernel(module.helion_flex_attention_kernel)


def _variant_kernel(
    tmp_path: pathlib.Path, name: str, *edits: tuple[str, str]
) -> Kernel:
    """A textual variant of the example kernel (each edit must apply once).

    The backward kernels further down the file repeat some of the forward
    kernel's lines, so the edits apply to the forward part only."""
    source = (EXAMPLES_DIR / "flex_attention.py").read_text()
    cut = source.find("\ndef _flex_attention_bwd_preprocess")
    head, tail = (source, "") if cut < 0 else (source[:cut], source[cut:])
    for old, new in edits:
        assert head.count(old) == 1, old
        head = head.replace(old, new)
    source = head + tail
    path = tmp_path / f"flex_attention_{name}.py"
    path.write_text(source)
    return _cute_kernel(import_path(path).helion_flex_attention_kernel)


def _qkv(
    batch: int = 2,
    heads: int = 4,
    q_seq: int = 1024,
    kv_seq: int = 1024,
    head_dim: int = 64,
    *,
    kv_heads: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kv_heads = heads if kv_heads is None else kv_heads
    q = torch.randn(batch, heads, q_seq, head_dim, dtype=torch.float16)
    k = torch.randn(batch, kv_heads, kv_seq, head_dim, dtype=torch.float16)
    v = torch.randn(batch, kv_heads, kv_seq, head_dim, dtype=torch.float16)
    return q, k, v


def _flex_args(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    score_mod: Callable[..., Any],
    block_mask: Any,
    *,
    mask_mod: Callable[..., Any] | None = None,
    scale: float = 0.125,
    enable_gqa: bool = False,
) -> tuple[Any, ...]:
    """The kernel arguments the example wrapper builds (both lists always)."""
    full_num = block_mask.full_kv_num_blocks
    full_idx = block_mask.full_kv_indices
    if full_num is None or full_idx is None:
        full_num = torch.zeros_like(block_mask.kv_num_blocks)
        full_idx = torch.zeros_like(block_mask.kv_indices)
    return (
        q,
        k,
        v,
        score_mod,
        block_mask.kv_num_blocks,
        block_mask.kv_indices,
        full_num,
        full_idx,
        block_mask.mask_mod if mask_mod is None else mask_mod,
        block_mask.BLOCK_SIZE[0],
        block_mask.BLOCK_SIZE[1],
        scale,
        enable_gqa,
    )


@contextlib.contextmanager
def _as_sm100(*, flex: bool) -> Iterator[None]:
    """CPU codegen with the flash gate forced on; ``flex`` sets the flag."""
    with (
        _cpu_codegen(),
        patch("helion._compiler.backend._attention_flash_supported", return_value=True),
        patch(
            "helion._compiler.backend._attention_flash_gate_enabled", return_value=True
        ),
        patch.object(flex_match, "FLEX_PLAN_ENABLED", flex),
    ):
        yield


def _bind(kernel: Kernel, args: tuple[Any, ...], *, flex: bool = False) -> BoundKernel:
    """Bind on the CPU as if on SM100; ``flex`` turns the integration on."""
    with _as_sm100(flex=flex):
        # ``_bind_isolated``: a fresh BoundKernel per test, independent of the
        # process-wide bind cache (the flag above changes the surface).
        return kernel._bind_isolated(args)


def _host(bound: BoundKernel) -> HostFunction:
    host_function = bound.host_function
    assert host_function is not None
    return host_function


def _match(bound: BoundKernel, reasons: list[str] | None = None) -> FlexMatch | None:
    host_function = _host(bound)
    device_ir = host_function.device_ir
    with bound.env:
        return match_flex_envelope(
            device_ir.graphs,
            device_ir.grid_block_ids[0],
            env=bound.env,
            host_function=host_function,
            reasons=reasons,
        )


def _match_or_fail(bound: BoundKernel) -> FlexMatch:
    reasons: list[str] = []
    match = _match(bound, reasons)
    assert match is not None, reasons
    return match


def _targets(graph: torch.fx.Graph) -> set[object]:
    return {node.target for node in graph.nodes if node.op == "call_function"}


def _make_sliding_causal(window: int) -> Callable[..., torch.Tensor]:
    def sliding_causal(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> torch.Tensor:
        return (q_idx >= kv_idx) & (q_idx - kv_idx <= window)

    return sliding_causal


def _causal(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> torch.Tensor:
    return q_idx >= kv_idx


# -- the plan ----------------------------------------------------------------


def test_mask_only_sliding_causal_plan() -> None:
    """(a) mask_mod only, broadcast ``(1, 1, R)`` tables: the scale is peeled,
    the mask region is captured, there is no score region."""
    q, k, v = _qkv()
    mask_mod = _make_sliding_causal(256)
    block_mask = create_block_mask(
        mask_mod, None, None, 1024, 1024, BLOCK_SIZE=128, device=_CPU
    )
    bound = _bind(_flex_kernel(), _flex_args(q, k, v, _identity, block_mask))
    match = _match_or_fail(bound)
    plan = match.plan

    assert plan.is_flex and plan.has_list_walk
    assert plan.modifier_kinds == ("flex",)
    assert plan.has_lowering()
    assert plan.scale_peeled
    assert plan.score_region is None
    assert plan.mask_region is not None
    assert not plan.flex_has_score_region and plan.flex_has_mask_region
    assert plan.aux_buffers == ()
    # The window is a closure int: a host scalar the mask reads.
    assert len(plan.host_scalars) == 1
    host_function = _host(bound)
    window_symbol = flex_host_scalar_symbol(host_function, plan.host_scalars[0])
    assert window_symbol is not None
    assert isinstance(host_function.expr_to_origin[window_symbol].origin, ClosureOrigin)
    # The runtime scale is a kernel argument; the multiplier is log2(e).
    assert plan.scale_arg is not None
    scale_symbol = flex_host_scalar_symbol(host_function, plan.scale_arg)
    assert scale_symbol is not None
    origin = host_function.expr_to_origin[scale_symbol].origin
    assert isinstance(origin, NameOrigin) and origin.name == "scale"
    assert plan.qk_scale_log2 == pytest.approx(flex_match.LOG2E)
    assert plan.lse_scale == pytest.approx(0.6931471805599453)

    walk = plan.kv_walk
    assert walk is not None
    assert (walk.q_block, walk.kv_block) == (128, 128)
    assert (walk.table_batch, walk.table_heads) == (1, 1)
    assert not walk.trivial
    assert walk.part_cnt_name == "block_mask_kv_num_blocks"
    assert walk.part_idx_name == "block_mask_kv_indices"
    assert walk.full_cnt_name == "block_mask_full_kv_num_blocks"
    assert walk.full_idx_name == "block_mask_full_kv_indices"
    assert match.table_names == (
        "block_mask_kv_num_blocks",
        "block_mask_kv_indices",
        "block_mask_full_kv_num_blocks",
        "block_mask_full_kv_indices",
    )

    mask = plan.mask_region
    assert _targets(mask.graph) == {
        _aten.ge.Tensor,
        _aten.sub.Tensor,
        _aten.le.Scalar,
        _aten.bitwise_and.Tensor,
    }
    assert [node.target for node in mask.graph.nodes if node.op == "placeholder"] == [
        "score",
        "b_idx",
        "h_idx",
        "q_idx",
        "kv_idx",
        "host_scalar_0",
    ]
    assert mask.output.meta["val"].dtype == torch.bool
    assert (match.q_seq, match.kv_seq, match.num_kv) == (1024, 1024, 8)
    assert (match.batch, match.heads, match.kv_heads) == (2, 4, 4)
    assert (match.lse_name, match.lse_log_base) == ("lse", "log")
    assert match.loop_block_ids == (5, 1, 6)
    assert match.kv_block_id == 1 and match.q_block_id == 0
    assert (match.b_block_id, match.h_block_id) == (2, 3)


def test_score_mod_closure_table_and_mask() -> None:
    """(b) a score_mod closing over a table read through ``hl.load`` plus a
    mask: the scale stays in the score region, the table is an aux buffer
    with a lane-contiguous index, the closure int is a host scalar."""
    q, k, v = _qkv()
    table = torch.randn(2048)
    offset = 1024

    def rel_score_mod(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return score + hl.load(table, [kv_idx - q_idx + offset])

    block_mask = create_block_mask(
        _make_sliding_causal(256), None, None, 1024, 1024, BLOCK_SIZE=128, device=_CPU
    )
    bound = _bind(_flex_kernel(), _flex_args(q, k, v, rel_score_mod, block_mask))
    match = _match_or_fail(bound)
    plan = match.plan

    assert not plan.scale_peeled
    assert plan.flex_has_score_region and plan.flex_has_mask_region
    assert plan.score_region is not None and plan.mask_region is not None
    assert plan.qk_scale_log2 == pytest.approx(flex_match.LOG2E)
    assert len(plan.aux_buffers) == 1
    (aux,) = plan.aux_buffers
    assert aux.name.endswith("__closure__[1].cell_contents")
    assert aux.dtype == torch.float32
    assert aux.shape == (2048,) and aux.strides == (1,)
    assert aux.index_class == "column_affine"
    # Host scalars: the closure offset, the scale (owned by the region) and
    # the mask's window, each resolvable to a registered symbol.
    assert len(plan.host_scalars) == 3
    assert plan.scale_arg in plan.host_scalars
    host_function = _host(bound)
    for name in plan.host_scalars:
        assert flex_host_scalar_symbol(host_function, name) is not None
    score = plan.score_region
    nodes = [node for node in score.graph.nodes if node.op == "call_function"]
    assert nodes[0].target is _aten.mul.Tensor
    assert nodes[0].args[0].target == "score"
    assert nodes[0].args[1].op == "placeholder"
    loads = [node for node in nodes if node.target is hl.load]
    assert len(loads) == 1
    assert loads[0].meta["flex_index_class"] == "column_affine"
    assert loads[0].meta["flex_source_name"] == aux.name
    assert score.output.meta["val"].dtype == torch.float32
    # Both regions share one placeholder list (unused entries allowed), and
    # every placeholder carries metadata: int32 coordinates, the aux fake,
    # SymInt / SymFloat host scalars (the renderer picks the scalar kind
    # from it, including for a region that never reads the scalar).
    for region in (plan.score_region, plan.mask_region):
        placeholders = [n for n in region.graph.nodes if n.op == "placeholder"]
        assert [n.target for n in placeholders] == [
            "score",
            "b_idx",
            "h_idx",
            "q_idx",
            "kv_idx",
            "aux_0",
            "host_scalar_0",
            "host_scalar_1",
            "host_scalar_2",
        ]
        values = [n.meta["val"] for n in placeholders]
        assert values[0].dtype == torch.float32
        assert all(v.dtype == torch.int32 for v in values[1:5])
        assert tuple(values[5].shape) == (2048,)
        scale_slot = 6 + plan.host_scalars.index(plan.scale_arg)
        assert isinstance(values[scale_slot], torch.SymFloat)
        for slot in range(6, 9):
            if slot != scale_slot:
                assert isinstance(values[slot], torch.SymInt)


def test_dense_noop_mask_and_trivial_empty_mask() -> None:
    """(c) an all-full ``noop_mask`` BlockMask is a real walk without regions;
    ``_create_empty_block_mask`` is the trivial walk (``KB >= N``)."""
    q, k, v = _qkv()
    dense = create_block_mask(noop_mask, 2, 4, 1024, 1024, BLOCK_SIZE=128, device=_CPU)
    assert int(dense.kv_num_blocks.sum()) == 0
    bound = _bind(_flex_kernel(), _flex_args(q, k, v, _identity, dense))
    plan = _match_or_fail(bound).plan
    assert plan.score_region is None and plan.mask_region is None
    assert plan.scale_peeled and plan.has_list_walk
    assert not plan.flex_has_score_region and not plan.flex_has_mask_region
    walk = plan.kv_walk
    assert walk is not None
    assert (walk.table_batch, walk.table_heads, walk.trivial) == (2, 4, False)

    empty = _create_empty_block_mask(q, k)
    bound = _bind(_flex_kernel(), _flex_args(q, k, v, _identity, empty))
    plan = _match_or_fail(bound).plan
    walk = plan.kv_walk
    assert walk is not None
    assert walk.trivial
    assert walk.q_block == walk.kv_block == 1 << 30
    assert (walk.table_batch, walk.table_heads) == (1, 1)
    assert plan.is_flex and not plan.has_list_walk
    assert plan.mask_region is None and plan.score_region is None


def test_ragged_and_unequal_lengths() -> None:
    """(d) ``M != N`` and lengths that are not multiples of 128."""
    q, k, v = _qkv(q_seq=300, kv_seq=640)
    block_mask = create_block_mask(_causal, 2, 4, 300, 640, BLOCK_SIZE=128, device=_CPU)
    bound = _bind(_flex_kernel(), _flex_args(q, k, v, _identity, block_mask))
    match = _match_or_fail(bound)
    assert (match.q_seq, match.kv_seq) == (300, 640)
    assert (match.plan.q_seq, match.plan.kv_seq) == (300, 640)
    assert match.num_kv == 5
    assert match.plan.mask_region is not None
    assert _targets(match.plan.mask_region.graph) == {_aten.ge.Tensor}


def test_grouped_query_heads() -> None:
    """(e) ``enable_gqa``: K/V indexed at ``h // g`` gives ``kv_group = g``."""
    q, k, v = _qkv(kv_heads=2)
    block_mask = create_block_mask(
        _causal, None, None, 1024, 1024, BLOCK_SIZE=128, device=_CPU
    )
    bound = _bind(
        _flex_kernel(), _flex_args(q, k, v, _identity, block_mask, enable_gqa=True)
    )
    match = _match_or_fail(bound)
    assert match.plan.kv_group == 2
    assert (match.heads, match.kv_heads) == (4, 2)


def test_row_col_gather() -> None:
    """(f) ``hl.load(t, [b, q_idx, kv_idx])`` is a per-(row, col) gather."""
    q, k, v = _qkv()
    bias = torch.randn(2, 1024, 1024)

    def gather_mod(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return score + hl.load(bias, [b, q_idx, kv_idx])

    block_mask = create_block_mask(
        _causal, None, None, 1024, 1024, BLOCK_SIZE=128, device=_CPU
    )
    bound = _bind(_flex_kernel(), _flex_args(q, k, v, gather_mod, block_mask))
    plan = _match_or_fail(bound).plan
    (aux,) = plan.aux_buffers
    assert aux.index_class == "row_col"
    assert aux.shape == (2, 1024, 1024)
    assert aux.strides == (1024 * 1024, 1024, 1)
    assert not plan.scale_peeled


def test_mask_tables_and_head_slopes() -> None:
    """A mask that reads a per-position table on both sides, plus a score_mod
    with a per-head slope: index classes per load and per buffer."""
    q, k, v = _qkv()
    document = torch.repeat_interleave(torch.arange(4, dtype=torch.int32), 256)
    slopes = torch.rand(4)

    def torch_document_mask(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return (q_idx >= kv_idx) & (document[q_idx] == document[kv_idx])

    def document_mask(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return (q_idx >= kv_idx) & (
            hl.load(document, [q_idx]) == hl.load(document, [kv_idx])
        )

    def alibi(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return score - hl.load(slopes, [h]) * (q_idx - kv_idx).abs()

    block_mask = create_block_mask(
        torch_document_mask, None, None, 1024, 1024, BLOCK_SIZE=128, device=_CPU
    )
    bound = _bind(
        _flex_kernel(),
        _flex_args(q, k, v, alibi, block_mask, mask_mod=document_mask),
    )
    plan = _match_or_fail(bound).plan
    assert plan.score_region is not None and plan.mask_region is not None
    by_name = {aux.name: aux for aux in plan.aux_buffers}
    assert len(by_name) == 2
    slope_aux = next(aux for aux in plan.aux_buffers if aux.shape == (4,))
    doc_aux = next(aux for aux in plan.aux_buffers if aux.shape == (1024,))
    assert slope_aux.index_class == "head"
    # One buffer read both row-uniformly and lane-contiguously: per-load
    # classes on the copied nodes, "general" on the buffer.
    assert doc_aux.index_class == "general"
    mask_loads = [
        node
        for node in plan.mask_region.graph.nodes
        if node.op == "call_function" and node.target is hl.load
    ]
    assert sorted(node.meta["flex_index_class"] for node in mask_loads) == [
        "column_affine",
        "row_uniform",
    ]
    assert _aten.abs.default in _targets(plan.score_region.graph)
    # The score region first applies the scale, the aux order is score-first.
    assert plan.aux_buffers[0] is slope_aux


def test_zeros_like_accumulation_with_dependent_gathers() -> None:
    """The graphormer-edge form: ``torch.zeros_like(score)`` (an ``aten.full``
    whose shape arguments are symbolic size nodes) accumulating
    ``where(length > n, c_n, 0.0)`` terms through two-level gathers."""
    batch, heads, seq, path_len, edge_types = 2, 4, 256, 3, 8
    q, k, v = _qkv(batch, heads, seq, seq)
    path_lengths = torch.randint(0, path_len + 1, (batch, seq, seq), dtype=torch.int32)
    path_types = torch.randint(
        0, edge_types, (batch, seq, seq, path_len), dtype=torch.int32
    )
    edge_bias_flat = torch.randn(heads, path_len * edge_types)

    # Literal path positions and offsets, as the harness-generated source has
    # them (a closure int would trace as a symbolic host scalar).
    def edge_mod(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        length = hl.load(path_lengths, [b, q_idx, kv_idx])
        total = torch.zeros_like(score)
        et0 = hl.load(path_types, [b, q_idx, kv_idx, 0])
        c0 = hl.load(edge_bias_flat, [h, et0 + 0])
        total = total + torch.where(length > 0, c0, 0.0)
        et1 = hl.load(path_types, [b, q_idx, kv_idx, 1])
        c1 = hl.load(edge_bias_flat, [h, et1 + 8])
        total = total + torch.where(length > 1, c1, 0.0)
        et2 = hl.load(path_types, [b, q_idx, kv_idx, 2])
        c2 = hl.load(edge_bias_flat, [h, et2 + 16])
        total = total + torch.where(length > 2, c2, 0.0)
        return score + total / torch.clamp(length, min=1).to(torch.float32)

    block_mask = create_block_mask(
        noop_mask, batch, heads, seq, seq, BLOCK_SIZE=128, device=_CPU
    )
    bound = _bind(_flex_kernel(), _flex_args(q, k, v, edge_mod, block_mask))
    plan = _match_or_fail(bound).plan
    assert plan.score_region is not None and plan.mask_region is None
    assert not plan.scale_peeled
    classes = {aux.shape: aux.index_class for aux in plan.aux_buffers}
    # path_lengths[b, q, kv]: lane-contiguous per row; path_types[b, q, kv, n]:
    # a constant last index with the lane in a leading dim (strided gather);
    # edge_bias_flat[h, et + c]: indexed by a loaded value.
    assert classes == {
        (batch, seq, seq): "row_col",
        (batch, seq, seq, path_len): "general",
        (heads, path_len * edge_types): "value",
    }
    graph = plan.score_region.graph
    fulls = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target is _aten.full.default
    ]
    # One typed literal zero with its shape argument dropped; no size node.
    assert len(fulls) == 1
    assert fulls[0].args[0] == [] and fulls[0].args[1] == 0.0
    assert fulls[0].meta["val"].dtype == torch.float32
    assert _aten.sym_size.int not in _targets(graph)
    assert {_aten.where.self, _aten.clamp_min.default, _aten.gt.Scalar} <= _targets(
        graph
    )
    assert len([n for n in graph.nodes if n.target is hl.load]) == 1 + 2 * path_len


# -- declines ------------------------------------------------------------------


def test_declines_kv_block_other_than_128() -> None:
    q, k, v = _qkv()
    block_mask = create_block_mask(
        _causal, None, None, 1024, 1024, BLOCK_SIZE=(128, 256), device=_CPU
    )
    bound = _bind(_flex_kernel(), _flex_args(q, k, v, _identity, block_mask))
    reasons: list[str] = []
    assert _match(bound, reasons) is None
    assert reasons == ["flex: KB must be 128 (got 256)"]


def test_declines_reduction_inside_score_mod() -> None:
    q, k, v = _qkv()

    def centered(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return score - score.amax(-1, keepdim=True)

    block_mask = create_block_mask(
        _causal, None, None, 1024, 1024, BLOCK_SIZE=128, device=_CPU
    )
    bound = _bind(_flex_kernel(), _flex_args(q, k, v, centered, block_mask))
    reasons: list[str] = []
    assert _match(bound, reasons) is None
    assert reasons == ["flex: region has non-pointwise op aten.amax.default"]


def test_declines_full_body_with_a_different_score_region(
    tmp_path: pathlib.Path,
) -> None:
    # The full-list body adds a constant the partial body does not.
    kernel = _variant_kernel(
        tmp_path,
        "two_mods",
        (
            """                    score = score_mod(bcast_qk, bcast_b, bcast_h, bcast_m, bcast_n)
                    qk = score
                    qk *= log_2_e
""",
            """                    score = score_mod(bcast_qk, bcast_b, bcast_h, bcast_m, bcast_n)
                    qk = score + 1.0
                    qk *= log_2_e
""",
        ),
    )

    def tanh_mod(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return torch.tanh(score)

    q, k, v = _qkv()
    block_mask = create_block_mask(
        _causal, None, None, 1024, 1024, BLOCK_SIZE=128, device=_CPU
    )
    bound = _bind(kernel, _flex_args(q, k, v, tanh_mod, block_mask))
    reasons: list[str] = []
    assert _match(bound, reasons) is None
    assert reasons == [
        "flex: the full body's score region differs from the partial one"
    ]


def test_declines_non_unit_batch_tile(tmp_path: pathlib.Path) -> None:
    kernel = _variant_kernel(
        tmp_path,
        "wide_batch",
        ("block_size=[1, 1, block_m]", "block_size=[2, 1, block_m]"),
    )
    q, k, v = _qkv()
    block_mask = create_block_mask(
        _causal, None, None, 1024, 1024, BLOCK_SIZE=128, device=_CPU
    )
    bound = _bind(kernel, _flex_args(q, k, v, _identity, block_mask))
    reasons: list[str] = []
    assert _match(bound, reasons) is None
    assert reasons == ["flex: root axis 2 is not pinned to block size 1"]


def test_zeros_like_score_in_a_score_mod_is_admitted() -> None:
    """``torch.zeros_like(score)`` (attention-gym graphormer-edge) lowers to
    ``aten.full`` sized by ``sym_size_int(score, d)``: shape metadata, not
    region values. The region keeps a shapeless literal and no size query
    (the builder used to assert on the size node)."""
    q, k, v = _qkv()
    table = torch.randn(1024, dtype=torch.float32)

    def score_mod(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        total = torch.zeros_like(score)
        total = total + hl.load(table, [kv_idx])
        return score + total

    block_mask = create_block_mask(
        _causal, 2, 4, 1024, 1024, BLOCK_SIZE=128, device=_CPU
    )
    bound = _bind(_flex_kernel(), _flex_args(q, k, v, score_mod, block_mask))
    plan = _match_or_fail(bound).plan
    assert plan.score_region is not None
    nodes = list(plan.score_region.graph.nodes)
    fulls = [node for node in nodes if node.target is _aten.full.default]
    assert fulls and all(node.args[0] == [] for node in fulls)
    assert not any(node.target is _aten.sym_size.int for node in nodes)


# -- integration -----------------------------------------------------------------


def test_flag_off_keeps_the_generic_path() -> None:
    """With ``FLEX_PLAN_ENABLED`` patched off the flex kernel keeps the generic
    surface (the flag is on by default now that the list-walk emitter is in)."""
    q, k, v = _qkv()
    block_mask = create_block_mask(
        _causal, None, None, 1024, 1024, BLOCK_SIZE=128, device=_CPU
    )
    bound = _bind(_flex_kernel(), _flex_args(q, k, v, _identity, block_mask))
    assert flex_match.FLEX_PLAN_ENABLED
    assert not bound.config_spec.cute_flash_search_enabled


def test_surface_and_codegen_claim_every_walk_loop() -> None:
    """With the flag on: the surface records the flex facts and pins only the
    tunable (tile_m, tile_n) ids; codegen matches once, records all loop
    ids of both walks and renders the fa4 list-walk body."""
    q, k, v = _qkv()
    block_mask = create_block_mask(
        _causal, 2, 4, 1024, 1024, BLOCK_SIZE=128, device=_CPU
    )
    bound = _bind(_flex_kernel(), _flex_args(q, k, v, _identity, block_mask), flex=True)
    spec = bound.config_spec
    assert spec.cute_flash_search_enabled
    assert spec._cute_flash_score_modifier_kinds == ("flex",)
    assert spec._cute_flash_has_list_walk
    assert not spec._cute_flash_flex_has_score_region
    assert spec._cute_flash_flex_has_mask_region
    assert not spec._cute_flash_has_score_modifiers
    assert (spec._cute_flash_q_seq, spec._cute_flash_kv_seq) == (1024, 1024)
    assert spec._cute_flash_num_kv == 8
    assert spec._cute_flash_num_bh == 8
    assert spec._cute_flash_tensor_4d_heads == 4
    assert spec._cute_flash_block_size_targets == {0: 128, 1: 128}
    for block_id in (0, 1):
        block_spec = spec.block_sizes.block_id_lookup(block_id)
        assert (block_spec.autotuner_min, block_spec.max_size) == (128, 128)

    captured: dict[str, Any] = {}
    real_codegen = cute_flash.codegen_attention_flash

    def spy(cg: Any) -> bool:
        state = cg.device_function.cute_state
        captured["block_ids"] = list(state.attention_flash_block_ids or [])
        captured["plan"] = state.attention_flash_score_plan
        captured["match"] = state.attention_flex_match
        captured["tensor_plan"] = cute_flash.flash_attention_tensor_plan(
            cg.device_function
        )
        return real_codegen(cg)

    config = spec.default_config()
    assert config.block_sizes == [128, 128]
    with (
        _cpu_codegen(),
        patch.object(flex_match, "FLEX_PLAN_ENABLED", True),
        patch.object(cute_flash, "codegen_attention_flash", spy),
    ):
        code = bound.to_code(config)
    # The list-walk emitter rendered the fa4 pair body over the tables.
    assert "flash_flex_status" in code
    assert "'topology': 'fa4'" in code
    match = captured["match"]
    assert isinstance(match, FlexMatch)
    assert sorted(captured["block_ids"]) == sorted(match.loop_block_ids) == [1, 5, 6]
    assert captured["plan"] is match.plan
    tensor_plan = captured["tensor_plan"]
    assert tensor_plan is not None
    assert [arg.name for arg in tensor_plan.kv_walk_args] == list(match.table_names)
    assert tensor_plan.aux_args == ()
    assert (tensor_plan.tensor_4d_batch, tensor_plan.tensor_4d_heads) == (2, 4)
    assert (tensor_plan.batch, tensor_plan.seq) == (8, 1024)
    assert (tensor_plan.q_seq, tensor_plan.kv_seq) == (1024, 1024)
    assert tensor_plan.flex_scale_expr == "cutlass.Float32(scale)"
    assert tensor_plan.flex_scale_arg_name == "scale"
    assert tensor_plan.flex_scalar_exprs == ()
    assert tensor_plan.flex_scalar_arg_names == ()
    assert tuple(tensor_plan.q_arg.fake_value.shape) == (2, 4, 1024, 64)
    assert tensor_plan.lse_arg is not None and tensor_plan.lse_log_base == "log"


def test_graph_output_plan_entry_handles_flex_plans() -> None:
    """The shared ``*_from_graphs`` entry delegates flex plans to the matcher."""
    q, k, v = _qkv()
    block_mask = create_block_mask(
        _causal, None, None, 1024, 1024, BLOCK_SIZE=128, device=_CPU
    )
    bound = _bind(_flex_kernel(), _flex_args(q, k, v, _identity, block_mask))
    match = _match_or_fail(bound)
    host_function = _host(bound)
    device_ir = host_function.device_ir
    with bound.env, host_function:
        graph_plan = cute_flash._flash_graph_output_plan_from_graphs(
            device_ir.graphs,
            root_block_ids=device_ir.grid_block_ids[0],
            kv_block_id=match.kv_block_id,
            score_plan=match.plan,
        )
        assert cute_flash.flash_attention_graph_lse_plan_valid_from_graphs(
            device_ir.graphs,
            root_block_ids=device_ir.grid_block_ids[0],
            kv_block_id=match.kv_block_id,
            score_plan=match.plan,
        )
    assert graph_plan is not None
    assert graph_plan.kv_walk_names == match.table_names
    assert (graph_plan.batch, graph_plan.seq) == (8, 1024)
    assert (graph_plan.tensor_4d_batch, graph_plan.tensor_4d_heads) == (2, 4)
    assert (graph_plan.q_name, graph_plan.o_name, graph_plan.lse_name) == (
        "query",
        "out",
        "lse",
    )


@pytest.mark.parametrize("name", ["attention", "causal_attention"])
def test_collapsed_envelope_surface_unchanged(name: str) -> None:
    """The dense / causal examples keep exactly their plan whether or not the
    flex integration is on."""
    module = import_path(EXAMPLES_DIR / "attention.py")
    kernel = _cute_kernel(getattr(module, name))
    args = tuple(torch.randn(2, 8, 1024, 64, dtype=torch.float16) for _ in range(3))
    surfaces = []
    for flex in (False, True):
        bound = _bind(kernel, args, flex=flex)
        host_function = _host(bound)
        device_ir = host_function.device_ir
        with _as_sm100(flex=flex), bound.env, host_function:
            surfaces.append(_compiler_backend.detect_flash_search_surface(device_ir))
        assert bound.config_spec.cute_flash_search_enabled
    before, after = surfaces
    assert before is not None
    assert before == after
    expected_kinds = ("dense",) if name == "attention" else ("causal_mask",)
    assert before.score_modifier_kinds == expected_kinds
    assert not before.has_list_walk
    assert before.q_seq is None and before.kv_seq is None
    expected_plan = (
        dense_score_plan(64) if name == "attention" else causal_score_plan(64)
    )
    assert expected_plan.modifier_kinds == expected_kinds


@pytest.mark.parametrize("q_seq", (64, 100, 130))
def test_short_query_lengths_enable_the_search(q_seq: int) -> None:
    """A query length below (or just past) one 128-row tile caps the generic
    block range at its extent; the flex surface still pins the 128x128
    tiles (the emitter row-guards the ragged tile) and the default config
    renders the fa4 list-walk body. Mirrors attention-gym's jetspec-tree
    (M=64 query rows over N=2112 keys, a bool table read per score)."""
    q, k, v = _qkv(q_seq=q_seq, kv_seq=2112)
    ancestor = torch.rand(256, 64) < 0.5

    def tree_mask(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        prefix, num_nodes = 640, 64
        tree_idx = (kv_idx - prefix).clamp(0, num_nodes - 1)
        tree_visible = (kv_idx >= prefix) & (kv_idx < prefix + num_nodes)
        return (kv_idx < prefix) | (tree_visible & hl.load(ancestor, [q_idx, tree_idx]))

    def tree_mask_eager(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        prefix, num_nodes = 640, 64
        tree_idx = (kv_idx - prefix).clamp(0, num_nodes - 1)
        tree_visible = (kv_idx >= prefix) & (kv_idx < prefix + num_nodes)
        return (kv_idx < prefix) | (tree_visible & ancestor[q_idx, tree_idx])

    # ``create_block_mask`` evaluates its mask eagerly (no ``hl.load``); the
    # kernel receives the ``hl.load`` form.
    block_mask = create_block_mask(
        tree_mask_eager, None, None, q_seq, 2112, BLOCK_SIZE=128, device=_CPU
    )
    bound = _bind(
        _flex_kernel(),
        _flex_args(q, k, v, _identity, block_mask, mask_mod=tree_mask),
        flex=True,
    )
    spec = bound.config_spec
    assert spec.cute_flash_search_enabled
    assert spec._cute_flash_block_size_targets == {0: 128, 1: 128}
    assert (spec._cute_flash_q_seq, spec._cute_flash_kv_seq) == (q_seq, 2112)
    assert spec._cute_flash_num_kv == 17
    for block_id in (0, 1):
        block_spec = spec.block_sizes.block_id_lookup(block_id)
        fragment = block_spec._fragment(spec)
        assert (fragment.low, fragment.high, fragment.default()) == (128, 128, 128)
    # ``register_block_size(min(256, 128))`` declared the 128 ceiling; the
    # query extent only capped the generic range.
    q_spec = spec.block_sizes.block_id_lookup(0)
    assert q_spec.declared_max_size == 128 and q_spec.size_hint == q_seq

    config = spec.default_config()
    assert config.block_sizes == [128, 128]
    captured: dict[str, Any] = {}
    real_codegen = cute_flash.codegen_attention_flash

    def spy(cg: Any) -> bool:
        state = cg.device_function.cute_state
        captured["block_ids"] = list(state.attention_flash_block_ids or [])
        captured["match"] = state.attention_flex_match
        return real_codegen(cg)

    with (
        _as_sm100(flex=True),
        patch.object(cute_flash, "codegen_attention_flash", spy),
    ):
        code = bound.to_code(config)
    assert "flash_flex_status" in code
    assert "'topology': 'fa4'" in code
    match = captured["match"]
    assert isinstance(match, FlexMatch)
    assert match.q_seq == q_seq
    assert sorted(captured["block_ids"]) == sorted(match.loop_block_ids)


def test_declared_ceiling_below_128_keeps_the_generic_path(
    tmp_path: pathlib.Path,
) -> None:
    """Only the dimension's extent is overshot: a ``register_block_size``
    ceiling below the 128-row pin still declines the surface (and the
    default config stays inside it, so codegen agrees)."""
    kernel = _variant_kernel(
        tmp_path,
        "declared64",
        (
            "block_m = hl.register_block_size(min(256, block_mask_m))",
            "block_m = hl.register_block_size(64)",
        ),
    )
    q, k, v = _qkv(q_seq=64, kv_seq=2112)
    block_mask = create_block_mask(
        _causal, None, None, 64, 2112, BLOCK_SIZE=128, device=_CPU
    )
    bound = _bind(kernel, _flex_args(q, k, v, _identity, block_mask), flex=True)
    assert _match(bound) is not None
    spec = bound.config_spec
    assert not spec.cute_flash_search_enabled
    q_spec = spec.block_sizes.block_id_lookup(0)
    assert (q_spec.declared_max_size, q_spec.max_size) == (64, 64)
    assert spec.default_config().block_sizes[0] <= 64


# -- strided operands ---------------------------------------------------------


def _transposed(*shape: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """A ``[B, H, S, D]`` view of a ``[B, S, H, D]`` allocation (the layout
    every model produces)."""
    batch, heads, seq, head_dim = shape
    return torch.randn(batch, seq, heads, head_dim, dtype=dtype).transpose(1, 2)


def _packed_qkv(
    batch: int, heads: int, seq: int, head_dim: int, *, pad: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q/K/V as head-dim slices of one ``[B, S, H, 3 * D (+ pad)]`` buffer."""
    qkv = torch.randn(batch, seq, heads, 3 * head_dim + pad, dtype=torch.float16)
    q = qkv[..., :head_dim].transpose(1, 2)
    k = qkv[..., head_dim : 2 * head_dim].transpose(1, 2)
    v = qkv[..., 2 * head_dim : 3 * head_dim].transpose(1, 2)
    return q, k, v


def _wrapper_plan(code: str) -> dict[str, Any]:
    """The flash wrapper plan embedded in a generated module."""
    import ast

    for line in code.splitlines():
        if "._helion_cute_wrapper_plans = " in line:
            (plan,) = ast.literal_eval(line.split(" = ", 1)[1])
            return plan
    raise AssertionError("no wrapper plan in the generated module")


def _host_lines(plan: dict[str, Any]) -> list[str]:
    """The launcher's host-wrapper lines for ``plan`` (built on the CPU)."""
    from helion.runtime.cute import launcher

    body: list[str] = []
    call_args: list[str] = []
    launcher._append_cute_wrapper_plan(body, call_args, plan, num_sm=148)
    return body


def test_contiguous_operands_keep_the_rank3_maps() -> None:
    """Contiguous Q/K/V/O record row-major strides, ``operand_strides`` is
    ``None`` and the render is the rank-3 ``(S, D, B*H)`` path: no
    ``operand_strides`` entry in the wrapper plan, the collapsed batch head
    indexes the TMA views (the render-unchanged pin of the strided work)."""
    q, k, v = _qkv()
    block_mask = create_block_mask(
        _causal, 2, 4, 1024, 1024, BLOCK_SIZE=128, device=_CPU
    )
    bound = _bind(_flex_kernel(), _flex_args(q, k, v, _identity, block_mask), flex=True)
    match = _match_or_fail(bound)
    assert (
        match.q_strides == match.k_strides == match.o_strides == (262144, 65536, 64, 1)
    )
    assert match.operand_strides is None
    with _as_sm100(flex=True):
        code = bound.to_code(bound.config_spec.default_config())
    plan = _wrapper_plan(code)
    assert "operand_strides" not in plan
    assert "tQgQ = tQgQ_qdl[None, None, 0, flash_bh]" in code
    assert "flash_head, flash_batch]" not in code
    assert any(
        "_flash_mQ = cute.make_tensor(arg0.iterator, cute.make_layout((1024, 64, 8), "
        "stride=(64, 1, 65536)))" in line
        for line in _host_lines(plan)
    )


@pytest.mark.parametrize("kv_heads", [4, 2])
def test_transposed_operands_take_the_rank4_maps(kv_heads: int) -> None:
    """``[B, S, H, D].transpose(1, 2)`` Q/K/V (and the O ``empty_like``
    inherits the layout): the matcher records the strides, the surface is
    the same as the contiguous one, and the render indexes rank-4
    ``(S, D, H, B)`` TMA views by head and batch (K/V by the grouped head)
    built by the launcher from the recorded strides."""
    q = _transposed(2, 4, 1024, 64)
    k = _transposed(2, kv_heads, 1024, 64)
    v = _transposed(2, kv_heads, 1024, 64)
    block_mask = create_block_mask(
        _causal, 2, 4, 1024, 1024, BLOCK_SIZE=128, device=_CPU
    )
    bound = _bind(
        _flex_kernel(),
        _flex_args(q, k, v, _identity, block_mask, enable_gqa=kv_heads != 4),
        flex=True,
    )
    match = _match_or_fail(bound)
    assert match.q_strides == match.o_strides == (262144, 64, 256, 1)
    assert (
        match.k_strides == match.v_strides == (kv_heads * 65536, 64, kv_heads * 64, 1)
    )
    assert match.operand_strides == (
        match.q_strides,
        match.k_strides,
        match.v_strides,
        match.o_strides,
    )
    spec = bound.config_spec
    assert spec.cute_flash_search_enabled
    assert spec._cute_flash_block_size_targets == {0: 128, 1: 128}
    # The same knob space (and seed) as the contiguous specialization.
    contiguous = _bind(
        _flex_kernel(),
        _flex_args(
            *_qkv(kv_heads=kv_heads), _identity, block_mask, enable_gqa=kv_heads != 4
        ),
        flex=True,
    )
    assert sorted(spec._flat_fields()) == sorted(contiguous.config_spec._flat_fields())
    config = spec.default_config()
    assert config == contiguous.config_spec.default_config()
    with _as_sm100(flex=True):
        code = bound.to_code(config)
    assert "flash_flex_status" in code
    plan = _wrapper_plan(code)
    assert plan["operand_strides"] == match.operand_strides
    assert (plan["tensor_4d_batch"], plan["tensor_4d_heads"]) == (2, 4)
    assert "flash_head = flash_bh % 4" in code
    assert "tQgQ = tQgQ_qdl[None, None, 0, flash_head, flash_batch]" in code
    kv_head = "flash_head" if kv_heads == 4 else "flash_head // 2"
    assert f"tKgK = tKgK_kdl[None, None, 0, {kv_head}, flash_batch]" in code
    assert f"tVgV = tVgV_dkl[None, 0, None, {kv_head}, flash_batch]" in code
    assert "tQgQ_qdl[None, None, 0, flash_bh]" not in code
    host = "\n".join(_host_lines(plan))
    assert (
        "_flash_mQ = cute.make_tensor(arg0.iterator, cute.make_layout((1024, 64, 4, 2), "
        "stride=(256, 1, 64, 262144)))"
    ) in host
    assert (
        f"_flash_mK = cute.make_tensor(arg1.iterator, cute.make_layout((1024, 64, {kv_heads}, 2), "
        f"stride=({kv_heads * 64}, 1, 64, {kv_heads * 65536})))"
    ) in host
    assert (
        f"_flash_mV = cute.make_tensor(arg2.iterator, cute.make_layout((64, 1024, {kv_heads}, 2), "
        f"stride=(1, {kv_heads * 64}, 64, {kv_heads * 65536})))"
    ) in host
    assert (
        "_flash_mO = cute.make_tensor(arg3.iterator, cute.make_layout((1024, 64, 4, 2), "
        "stride=(256, 1, 64, 262144)))"
    ) in host


def test_strided_epilogues_address_o_by_its_strides() -> None:
    """Every output path of the transposed case addresses O through the
    rank-4 view: the TMA store, the staged store and the split-KV merge
    (which builds its tile from the recorded strides)."""
    q, k, v = (_transposed(2, 4, 1024, 64) for _ in range(3))
    block_mask = create_block_mask(
        _causal, 2, 4, 1024, 1024, BLOCK_SIZE=128, device=_CPU
    )
    bound = _bind(_flex_kernel(), _flex_args(q, k, v, _identity, block_mask), flex=True)
    base = bound.config_spec.default_config().config
    renders: dict[str, str] = {}
    with _as_sm100(flex=True):
        for name, overrides in (
            (
                "tma",
                {
                    "cute_flash_epi_tma": True,
                    "cute_flash_epi_stg": False,
                    "cute_flash_kv_split": 1,
                },
            ),
            (
                "stg",
                {
                    "cute_flash_epi_tma": False,
                    "cute_flash_epi_stg": True,
                    "cute_flash_kv_split": 1,
                },
            ),
            (
                "split",
                {
                    "cute_flash_epi_tma": False,
                    "cute_flash_epi_stg": True,
                    "cute_flash_kv_split": 2,
                },
            ),
        ):
            renders[name] = bound.to_code(helion.Config(**{**base, **overrides}))
    assert "tOgO_tma[None, flash_m_tile0, 0, flash_head, flash_batch]" in renders["tma"]
    assert (
        "flash_mO_cur = _flash_mOt[None, None, flash_head, flash_batch]"
        in renders["stg"]
    )
    assert (
        "cute.assume(flash_batch * 262144 + flash_head * 64 + flash_m_tile0 * 32768, "
        "divby=8), cute.make_layout((128, 64), stride=(256, 1)))"
    ) in renders["split"]
    for code in renders.values():
        assert "flash_bh * 1024" not in code


def test_packed_qkv_slices_are_admitted() -> None:
    """Head-dim slices of one packed QKV buffer: outer strides of ``3 * D``
    (16-byte aligned) are admitted; the O ``empty_like`` of the slice view is
    a dense ``[B, S, H, D]`` allocation."""
    q, k, v = _packed_qkv(2, 4, 1024, 64)
    block_mask = create_block_mask(
        _causal, 2, 4, 1024, 1024, BLOCK_SIZE=128, device=_CPU
    )
    bound = _bind(_flex_kernel(), _flex_args(q, k, v, _identity, block_mask), flex=True)
    match = _match_or_fail(bound)
    assert (
        match.q_strides == match.k_strides == match.v_strides == (786432, 192, 768, 1)
    )
    assert match.o_strides == (262144, 64, 256, 1)
    assert bound.config_spec.cute_flash_search_enabled
    with _as_sm100(flex=True):
        code = bound.to_code(bound.config_spec.default_config())
    assert _wrapper_plan(code)["operand_strides"] == match.operand_strides


@pytest.mark.parametrize("case", ["misaligned", "non_unit_inner"])
def test_unaligned_strides_decline_and_log(
    case: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A head-dim slice whose outer stride is not a multiple of 16 bytes, or a
    non-unit inner stride, leaves the fused path (the TMA descriptors cannot
    describe it); the first decline reason is logged at DEBUG level."""
    if case == "misaligned":
        q, k, v = _packed_qkv(2, 4, 1024, 64, pad=4)
        expected_strides = (802816, 196, 784, 1)
    else:
        q, k, v = (
            torch.randn(2, 4, 1024, 128, dtype=torch.float16)[..., ::2]
            for _ in range(3)
        )
        expected_strides = (524288, 131072, 128, 2)
    assert tuple(q.stride()) == expected_strides
    block_mask = create_block_mask(
        _causal, 2, 4, 1024, 1024, BLOCK_SIZE=128, device=_CPU
    )
    with caplog.at_level(logging.DEBUG, logger="helion._compiler.cute.flex_match"):
        bound = _bind(
            _flex_kernel(), _flex_args(q, k, v, _identity, block_mask), flex=True
        )
    reasons: list[str] = []
    assert _match(bound, reasons) is None
    assert reasons[0] == (
        f"flex: Q strides {expected_strides} are not a unit inner stride with "
        "non-zero 16-byte aligned outer strides"
    )
    assert not bound.config_spec.cute_flash_search_enabled
    assert any(
        record.getMessage() == f"flex forward envelope declined: {reasons[0]}"
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


def test_expanded_operands_decline() -> None:
    """A zero stride on a dimension wider than one (``expand``) is not a TMA
    layout: declined (the generic path serves it), in K/V and in O alike."""
    q, _, _ = _qkv()
    k = torch.randn(1, 4, 1024, 64, dtype=torch.float16).expand(2, 4, 1024, 64)
    v = torch.randn(2, 1, 1024, 64, dtype=torch.float16).expand(2, 4, 1024, 64)
    assert k.stride()[0] == 0 and v.stride()[1] == 0
    block_mask = create_block_mask(
        _causal, 2, 4, 1024, 1024, BLOCK_SIZE=128, device=_CPU
    )
    bound = _bind(_flex_kernel(), _flex_args(q, k, v, _identity, block_mask), flex=True)
    reasons: list[str] = []
    assert _match(bound, reasons) is None
    assert reasons[0] == (
        "flex: K strides (0, 65536, 64, 1) are not a unit inner stride with "
        "non-zero 16-byte aligned outer strides"
    )
    assert not bound.config_spec.cute_flash_search_enabled
    # A broadcast batch of size one keeps its (arbitrary) stride admissible.
    q1 = torch.randn(1, 4, 1024, 64, dtype=torch.float16)
    k1 = torch.randn(1, 4, 1024, 64, dtype=torch.float16)
    block_mask = create_block_mask(
        _causal, 1, 4, 1024, 1024, BLOCK_SIZE=128, device=_CPU
    )
    bound = _bind(
        _flex_kernel(), _flex_args(q1, k1, k1.clone(), _identity, block_mask), flex=True
    )
    assert _match_or_fail(bound).operand_strides is None


def _without_source_comments(code: str) -> str:
    """The generated module without its ``# src[file:line]`` comments (line
    numbers of the example source, not generated code)."""
    return re.sub(r"^[ \t]*# src\[.*\n", "", code, flags=re.MULTILINE)


# unittest runners ignore pytestmark.
@onlyBackends(["cute"])
class TestContiguousRenderPin(TestCase):
    """The exact text of one contiguous forward render (the
    ``.expected_cute`` journal next to this file): the strided-operand
    support must not change the code contiguous inputs produce."""

    def test_contiguous_forward_render(self) -> None:
        q, k, v = _qkv()
        block_mask = create_block_mask(
            _causal, 2, 4, 1024, 1024, BLOCK_SIZE=128, device=_CPU
        )
        bound = _bind(
            _flex_kernel(), _flex_args(q, k, v, _identity, block_mask), flex=True
        )
        with _as_sm100(flex=True):
            code = bound.to_code(bound.config_spec.default_config())
        assert "'operand_strides'" not in code
        self.assertExpectedJournal(_without_source_comments(code))


# -- long KV lengths ---------------------------------------------------------------


def _synthetic_causal_block_mask(seq: int) -> Any:
    """A causal 128-block BlockMask built from its tables (the dense
    evaluation of a 128K x 128K mask is too slow on the CPU): the diagonal
    block partial, the blocks below full."""
    from torch.nn.attention.flex_attention import BlockMask

    rows = -(-seq // 128)
    ar = torch.arange(rows, dtype=torch.int32)
    kv_num = torch.ones(1, 1, rows, dtype=torch.int32)
    kv_idx = torch.zeros(1, 1, rows, rows, dtype=torch.int32)
    kv_idx[0, 0, :, 0] = ar
    full_num = ar[None, None].clone()
    full_idx = ar[None, None, None, :].expand(1, 1, rows, rows).contiguous()
    return BlockMask.from_kv_blocks(
        kv_num, kv_idx, full_num, full_idx, BLOCK_SIZE=128, mask_mod=_causal
    )


def test_long_kv_render_does_not_grow_with_the_sequence() -> None:
    """A 128K causal flex forward (1024 KV tiles, 32 bitmask words) renders
    the same code as a 16K one up to the numeric literals: the item words
    are walked through a cursor, not materialised in registers."""
    renders = {}
    for seq in (16384, 131072):
        q = torch.randn(1, 2, seq, 64, dtype=torch.bfloat16)
        k = torch.randn(1, 2, seq, 64, dtype=torch.bfloat16)
        v = torch.randn(1, 2, seq, 64, dtype=torch.bfloat16)
        block_mask = _synthetic_causal_block_mask(seq)
        bound = _bind(
            _flex_kernel(), _flex_args(q, k, v, _identity, block_mask), flex=True
        )
        assert bound.config_spec.cute_flash_search_enabled
        config = bound.config_spec.default_config()
        assert config.config[cute_flash.FLASH_FLEX_ITEM_WORDS_KEY] is True
        with _as_sm100(flex=True):
            renders[seq] = bound.to_code(config)
    short, long = renders[16384], renders[131072]
    assert short.count("\n") == long.count("\n")
    assert re.sub(r"\d+", "N", short) == re.sub(r"\d+", "N", long)
    assert "while flash_flex_rem == 0 and flash_flex_cw > flash_flex_low:" in long
    assert "flash_flex_rem1" not in long
    # The words argument carries the header: 4 + 4 * words entries per item.
    assert "_helion_flash_flex_item_words(" in long


# -- the dense flex plan ------------------------------------------------------
#
# torch's empty BlockMask (``_create_empty_block_mask``: one KV block over the
# whole sequence, ``noop_mask``) with the identity ``score_mod`` is dense
# attention dressed as flex.  The matcher records it as ``flex_dense`` on the
# dense surface's geometry; the surface, the resolver and the bodies then treat
# the plan as the dense attention plan (its families, seeds and knobs), while
# the operands, the LSE and the runtime scale stay the flex envelope's.


def _empty_mask_args(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    score_mod: Callable[..., Any] = _identity,
    enable_gqa: bool = False,
) -> tuple[Any, ...]:
    return _flex_args(
        q, k, v, score_mod, _create_empty_block_mask(q, k), enable_gqa=enable_gqa
    )


def test_empty_block_mask_is_the_dense_flex_plan() -> None:
    q, k, v = _qkv()
    plan = _match_or_fail(_bind(_flex_kernel(), _empty_mask_args(q, k, v))).plan
    assert plan.is_flex and plan.flex_dense and not plan.flex_envelope
    assert not plan.has_list_walk
    assert not plan.flex_has_mask_region and not plan.flex_has_score_region
    # Every surface and body sees the dense plan.
    assert plan.modifier_kinds == ("dense",)
    assert plan.has_lowering()
    assert cute_flash.flash_tmem_rowmax_score_plan_supported(plan)
    # The runtime scale and the natural-log LSE stay the envelope's.
    assert plan.scale_arg is not None and plan.scale_peeled
    assert plan.lse_scale == pytest.approx(0.6931471805599453)


def test_all_full_noop_block_mask_is_a_list_walk_not_a_dense_plan() -> None:
    """A real 128-block BlockMask walks its tables (the data decide which
    tiles are visible), even when every block happens to be full."""
    q, k, v = _qkv()
    dense = create_block_mask(noop_mask, 2, 4, 1024, 1024, BLOCK_SIZE=128, device=_CPU)
    bound = _bind(_flex_kernel(), _flex_args(q, k, v, _identity, dense))
    plan = _match_or_fail(bound).plan
    assert plan.has_list_walk and not plan.flex_dense and plan.flex_envelope
    assert plan.modifier_kinds == ("flex",)


def test_dense_flex_plan_admits_grouped_query_heads() -> None:
    q, k, v = _qkv(kv_heads=2)
    bound = _bind(_flex_kernel(), _empty_mask_args(q, k, v, enable_gqa=True))
    plan = _match_or_fail(bound).plan
    assert plan.flex_dense and plan.kv_group == 2


def _tanh_mod(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
    return torch.tanh(score)


@pytest.mark.parametrize(
    "case",
    ["mask_region", "score_region", "ragged_length", "unequal_lengths", "strided"],
)
def test_dense_flex_plan_fails_closed_off_the_dense_geometry(case: str) -> None:
    """Trivial walks that are not dense attention on the dense geometry keep
    the flex envelope (the body that hosts them today): a mask region on the
    one block, a score region, a length that is not a multiple of the
    128-row tile, unequal Q / K/V lengths, strided operands."""
    if case == "mask_region":
        q, k, v = _qkv(q_seq=64, kv_seq=64)
        block_mask = create_block_mask(
            _causal, 2, 4, 64, 64, BLOCK_SIZE=128, device=_CPU
        )
        args = _flex_args(q, k, v, _identity, block_mask)
    elif case == "score_region":
        q, k, v = _qkv()
        args = _empty_mask_args(q, k, v, score_mod=_tanh_mod)
    elif case == "ragged_length":
        q, k, v = _qkv(q_seq=1088, kv_seq=1088)
        args = _empty_mask_args(q, k, v)
    elif case == "unequal_lengths":
        q, k, v = _qkv(q_seq=1024, kv_seq=2048)
        args = _empty_mask_args(q, k, v)
    else:
        q = _transposed(2, 4, 1024, 64)
        k = _transposed(2, 4, 1024, 64)
        v = _transposed(2, 4, 1024, 64)
        args = _empty_mask_args(q, k, v)
    plan = _match_or_fail(_bind(_flex_kernel(), args)).plan
    assert plan.is_flex and not plan.has_list_walk
    assert not plan.flex_dense and plan.flex_envelope
    assert plan.modifier_kinds == ("flex",)
    assert plan.flex_has_mask_region == (case == "mask_region")
    assert plan.flex_has_score_region == (case == "score_region")


@functools.cache
def _attention_kernel() -> Kernel:
    module = import_path(EXAMPLES_DIR / "attention.py")
    return _cute_kernel(module.attention)


def _bind_dense_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> BoundKernel:
    with _as_sm100(flex=False):
        return _attention_kernel()._bind_isolated((q, k, v))


def _flash_choices(spec: Any) -> dict[str, tuple[object, ...]]:
    """The searched values of every flash knob of a surface."""
    with patch.dict(os.environ, {}, clear=True):
        fragments = spec._cute_flash_autotune_fragments()
    return {
        key: tuple(
            getattr(fragment, "search_choices", None)
            or getattr(fragment, "choices", ())
        )
        for key, fragment in fragments.items()
    }


def _seeded_families(spec: Any) -> set[tuple[object, object]]:
    with patch.dict(os.environ, {}, clear=True):
        seeds = spec.compiler_seed_configs or ()
    return {
        (
            seed.config.get(cute_flash.FLASH_PIPELINE_FAMILY_KEY),
            seed.config.get(cute_flash.FLASH_PERSISTENT_KEY),
        )
        for seed in seeds
    }


def test_dense_flex_plan_takes_the_dense_surface() -> None:
    """At the campaign's (1, 4, 256, 64) cell the dense flex plan gets the
    dense attention kernel's surface: the dense facts, every family of the
    dense surface (the row programs on this small grid, the persistent fa4
    bodies), the same searched values knob for knob, the dense seeds (a row
    program among them), a complete coverage design and every seed
    transferring."""
    q, k, v = _qkv(batch=1, heads=4, q_seq=256, kv_seq=256, head_dim=64)
    bound = _bind(_flex_kernel(), _empty_mask_args(q, k, v), flex=True)
    spec = bound.config_spec
    assert spec.cute_flash_search_enabled
    assert spec._cute_flash_score_modifier_kinds == ("dense",)
    assert not spec._cute_flash_has_list_walk
    assert not spec._cute_flash_flex_trivial_walk
    assert not spec._cute_flash_flex_has_mask_region
    assert not spec._cute_flash_flex_has_score_region
    assert not spec._cute_flash_has_score_modifiers
    assert spec._cute_flash_plain_row_body
    assert spec._cute_flash_tmem_rowmax_compatible
    # The LSE output and the runtime scale: no standard output-only lowering.
    assert not spec._cute_flash_standard_dense_output
    assert not spec._cute_flash_supports_tensor_4d_tma
    assert spec._cute_flash_block_size_targets == {0: 128, 1: 128}
    assert (spec._cute_flash_num_kv, spec._cute_flash_num_bh) == (2, 4)
    choices = _flash_choices(spec)
    families = choices[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
    assert "row_mma" in families and "fa4" in families and "ws_overlap" in families
    assert set(choices[cute_flash.FLASH_PERSISTENT_KEY]) == {True, False}
    assert set(choices[cute_flash.FLASH_ROW_WARPS_KEY]) == {2, 4, 8}
    # Knob for knob the dense attention kernel's search at the same cell,
    # less the 4-D TMA families: the attention example's reshape-view
    # operands prove them, the flex emitter reads its 4-D operands directly.
    dense = _bind_dense_attention(q, k, v).config_spec
    assert dense._cute_flash_score_modifier_kinds == ("dense",)
    dense_choices = _flash_choices(dense)
    assert "fa4_tma_4d" in dense_choices[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
    assert families == tuple(
        family
        for family in dense_choices[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
        if not str(family).endswith("_tma_4d")
    )
    assert {
        key: value
        for key, value in choices.items()
        if key != cute_flash.FLASH_PIPELINE_FAMILY_KEY
    } == {
        key: value
        for key, value in dense_choices.items()
        if key != cute_flash.FLASH_PIPELINE_FAMILY_KEY
    }
    with patch.dict(os.environ, {}, clear=True):
        seeds = list(spec.compiler_seed_configs or ())
    assert seeds
    assert all(seed.config["block_sizes"] == [128, 128] for seed in seeds)
    families_seeded = _seeded_families(spec)
    assert ("row_mma", False) in families_seeded
    assert ("fa4", True) in families_seeded
    assert families_seeded == {
        seeded
        for seeded in _seeded_families(dense)
        if not str(seeded[0]).endswith("_tma_4d")
    }
    from helion.autotuner.config_generation import ConfigGeneration

    with patch.dict(os.environ, {}, clear=True):
        generation = ConfigGeneration(spec)
        generation.validate_flash_structural_coverage()
        assert generation.flash_structural_coverage_uncovered_values() == []
        assert generation.flash_structural_coverage_uncovered_interactions() == []
        messages: list[str] = []
        pairs = generation.seed_flat_config_pairs(messages.append)
    assert not messages
    # Every seed transfers (two of the dense surface's seeds may fold into
    # one flat config on this small grid, as on the attention example's).
    assert len(pairs) >= len(seeds) - 1


_FLEX_BODY_MARKERS = (
    "flash_flex_status",
    "_flash_mFlexPartCnt",
    "_flash_mFlexFullIdx",
    "flash_flex_needs_mask",
    "_flash_mFlexItems",
    "flash_flex_cp0",
)


def _seed_of(spec: Any, family: str, persistent: bool) -> helion.Config:
    with patch.dict(os.environ, {}, clear=True):
        seeds = spec.compiler_seed_configs or ()
    return next(
        seed
        for seed in seeds
        if seed.config.get(cute_flash.FLASH_PIPELINE_FAMILY_KEY) == family
        and seed.config.get(cute_flash.FLASH_PERSISTENT_KEY) is persistent
    )


def test_dense_flex_plan_renders_the_dense_bodies() -> None:
    """The dense flex plan renders the dense bodies: the persistent fa4 pair
    body (the default) and the persistent ws_overlap body, with none of the
    list-walk / hook markers, and the runtime scale handed to the launcher
    (``scale_idx``: ``_flash_scale_log2 = _flash_scale * log2(e)``)."""
    q, k, v = _qkv(batch=1, heads=4, q_seq=256, kv_seq=256, head_dim=64)
    bound = _bind(_flex_kernel(), _empty_mask_args(q, k, v), flex=True)
    spec = bound.config_spec
    default = spec.default_config()
    assert default.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "fa4"
    assert default.config[cute_flash.FLASH_PERSISTENT_KEY] is True
    with _as_sm100(flex=True):
        code = bound.to_code(default)
    assert not any(marker in code for marker in _FLEX_BODY_MARKERS)
    plan = _wrapper_plan(code)
    assert plan["kind"] == "helion_flash"
    assert (plan["topology"], plan["persistent"]) == ("fa4", True)
    assert plan["scale_peeled"] is True
    assert "scale_idx" in plan and "lse_idx" in plan
    assert (plan["seq"], plan["batch"], plan["head_dim"]) == (256, 4, 64)
    host = "\n".join(_host_lines(plan))
    assert (
        "_flash_scale_log2 = _flash_scale * cutlass.Float32(1.4426950408889634)" in host
    )
    with _as_sm100(flex=True):
        code = bound.to_code(_seed_of(spec, "ws_overlap", True))
    assert not any(marker in code for marker in _FLEX_BODY_MARKERS)
    plan = _wrapper_plan(code)
    assert (plan["topology"], plan["persistent"]) == ("ws_overlap", True)
    assert "scale_idx" in plan


def test_dense_flex_plan_renders_the_row_programs_with_the_runtime_scale() -> None:
    """The row programs (``row_mma``) host the dense flex plan: the kernel
    reads the runtime scale as a scalar parameter (``rm_scale = scale *
    log2(e)``, the tcgen05 wrapper's derivation) instead of baking the
    plan's multiplier, writes the natural-log LSE, and the never-read
    BlockMask tables leave the kernel signature."""
    q, k, v = _qkv(batch=1, heads=4, q_seq=256, kv_seq=256, head_dim=64)
    bound = _bind(_flex_kernel(), _empty_mask_args(q, k, v), flex=True)
    seed = _seed_of(bound.config_spec, "row_mma", False)
    with _as_sm100(flex=True):
        code = bound.to_code(seed)
    assert not any(marker in code for marker in _FLEX_BODY_MARKERS)
    plan = _wrapper_plan(code)
    assert plan["kind"] == "helion_flash_row_mma"
    assert (plan["row_warps"], plan["row_tile_m"]) == (4, 8)
    # The operand names resolve to launcher argument positions; the LSE is
    # among them.
    assert "lse_idx" in plan
    assert (
        "rm_scale = cutlass.Float32(scale) * cutlass.Float32(1.4426950408889634)"
        in code
    )
    assert "rm_scale = cutlass.Float32(1.44269504)" not in code
    (kernel_def,) = [
        line
        for line in code.splitlines()
        if line.startswith("def _helion_helion_flex_attention_kernel(")
    ]
    assert kernel_def == (
        "def _helion_helion_flex_attention_kernel(query, key, value, out, lse, scale):"
    )


def test_trivial_walk_with_a_mask_region_keeps_the_flex_surface() -> None:
    """``M == N == 64`` under a causal BlockMask: the one 128-wide block
    covers the sequence (a trivial walk) but the mask region runs on it, so
    the plan keeps the flex envelope's surface and body (the trivial-walk
    pins: no persistent list walk, no row programs)."""
    q, k, v = _qkv(q_seq=64, kv_seq=64)
    block_mask = create_block_mask(_causal, 2, 4, 64, 64, BLOCK_SIZE=128, device=_CPU)
    bound = _bind(_flex_kernel(), _flex_args(q, k, v, _identity, block_mask), flex=True)
    spec = bound.config_spec
    assert spec._cute_flash_score_modifier_kinds == ("flex",)
    assert spec._cute_flash_flex_trivial_walk and not spec._cute_flash_has_list_walk
    assert spec._cute_flash_flex_has_mask_region
    choices = _flash_choices(spec)
    assert choices[cute_flash.FLASH_PERSISTENT_KEY] == (False,)
    assert "row_mma" not in choices[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
    with _as_sm100(flex=True):
        code = bound.to_code(spec.default_config())
    assert "flash_flex_needs_mask" in code and "flash_flex_status" in code
