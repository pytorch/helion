"""CPU tests of the FlexAttention-envelope matcher (``cute/flex_match.py``).

Every test binds ``examples/flex_attention.py`` (or a textual variant of it)
on the CPU with real ``create_block_mask`` BlockMasks and asserts the plan the
matcher produces, the autotune surface it feeds and the loop ownership it
records at codegen. Nothing here compiles or launches CuTe.
"""

from __future__ import annotations

import contextlib
import functools
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
from helion._testing import import_path
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
