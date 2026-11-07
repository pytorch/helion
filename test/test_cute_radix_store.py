"""Mixed-radix (fused view/permute) destinations of tcgen05 tile stores."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.cute.radix_store import RadixStoreLayout
from helion._compiler.cute.radix_store import prove_radix_store_layout
from helion._compiler.cute.radix_store import proven_radix_store_layout
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import gb300_target
from helion._testing.cute_radix_store import HEAD
from helion._testing.cute_radix_store import ONE_CTA_CONFIG
from helion._testing.cute_radix_store import PAIR_CONFIG
from helion._testing.cute_radix_store import SEQ
from helion._testing.cute_radix_store import block_size_index_mm
from helion._testing.cute_radix_store import colliding_mm
from helion._testing.cute_radix_store import fanout_head_split_mm
from helion._testing.cute_radix_store import flattened_head_split_mm
from helion._testing.cute_radix_store import head_split_args
from helion._testing.cute_radix_store import head_split_mm
from helion._testing.cute_radix_store import literal_reshape_mm
from helion._testing.cute_radix_store import overflowing_head_split_mm
from helion._testing.cute_radix_store import pure_view_mm
from helion._testing.cute_radix_store import seq384_head_split_mm
from helion._testing.cute_radix_store import short_head_split_mm
from helion._testing.cute_radix_store import skewed_mm
from helion._testing.cute_radix_store import transposed_store_mm
from helion._testing.cute_radix_store import wrapped_constant_mm
from helion.exc import BackendUnsupported
from helion.language import memory_ops

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


@pytest.fixture(autouse=True)
def _gb300():
    with gb300_target(), forbid_cuda_work():
        yield


def _layout(kernel, args) -> RadixStoreLayout | None:
    bound = kernel._bind_isolated(args)
    graphs = bound.host_function.device_ir.graphs
    store = [
        node
        for graph in graphs
        for node in graph.graph.nodes
        if node.op == "call_function" and node.target is memory_ops.store
    ][-1]
    with bound.env, bound.host_function:
        return prove_radix_store_layout(
            store, block_ids=(0, 1), extents=(args[0].shape[0], args[1].shape[1])
        )


def _head_split_layout(batch: int, heads: int) -> RadixStoreLayout:
    return RadixStoreLayout(
        rows=((SEQ, HEAD), (batch, heads * SEQ * HEAD)),
        cols=((HEAD, 1), (heads, SEQ * HEAD)),
    )


def test_head_split_layout_in_either_spelling():
    # 4 heads, and BERT-base's 12 heads over three sequences (non-power-of-two
    # N = 768 and M = 1536).
    for batch, heads in ((2, 4), (3, 12)):
        expected = _head_split_layout(batch, heads)
        for kernel in (head_split_mm, flattened_head_split_mm):
            assert _layout(kernel, head_split_args(batch, heads)) == expected
    layout = _head_split_layout(2, 4)
    assert layout.extents == (2 * SEQ, 4 * HEAD)
    assert layout.cute_layout_args() == (
        "((512, 2), (64, 4))",
        "((64, 131072), (1, 32768))",
    )
    assert layout.tma_store_supported(
        rows_per_store=128, cols_per_store=32, element_bytes=2
    )
    # A 128-row TMA box does not fit a 64-row innermost run.
    assert not RadixStoreLayout(
        rows=((64, 64), (16, 4096)), cols=((64, 1),)
    ).tma_store_supported(rows_per_store=128, cols_per_store=32, element_bytes=2)


def test_refuses_collisions_skew_and_config_dependent_indices():
    # An empty problem stores nothing to lay out.
    assert _layout(head_split_mm, head_split_args(batch=0)) is None
    args = head_split_args()
    # Not injective; not separable.
    assert _layout(colliding_mm, args) is None
    assert _layout(skewed_mm, args) is None
    # A literal reshape extent and a block size in the index arithmetic
    # depend on the config the proof does not see: refused, not crashed.
    assert _layout(literal_reshape_mm, args) is None
    assert _layout(block_size_index_mm, args) is None
    # The kernel's int32 index arithmetic wraps where an int64 proof would not.
    assert _layout(overflowing_head_split_mm, args) is None
    # A constant outside int32 is narrowed by torch, not by the int64 proof.
    assert (
        _layout(
            wrapped_constant_mm,
            (
                torch.empty((1024, 128), dtype=torch.bfloat16),
                torch.empty((128, 256), dtype=torch.bfloat16),
                torch.empty((1024, 256), dtype=torch.bfloat16),
            ),
        )
        is None
    )


def test_head_split_renders_the_hierarchical_tma_store():
    source = head_split_mm._bind_isolated(head_split_args()).to_code(PAIR_CONFIG)
    assert (
        "'d_hier_layout': ['((512, 2), (64, 4))', '((64, 131072), (1, 32768))']"
        in source
    )
    assert "tcgen05_tma_store_atom" in source
    # The per-element index arithmetic (y % 512, x % 64, ...) the store no
    # longer reads is dead-code eliminated.
    assert "_cute_python_mod" not in source
    # Every tile is full: the [M, N] extents, not the destination's trailing
    # (512, 64), decide the edge path (two epilogue warp groups need none).
    two_groups = helion.Config.from_dict(
        PAIR_CONFIG.config | {"tcgen05_epi_warp_groups": 2}
    )
    assert "d_hier_layout" in head_split_mm._bind_isolated(head_split_args()).to_code(
        two_groups
    )


def test_pure_view_keeps_its_plain_layout():
    args = (
        torch.empty((1024, 256), dtype=torch.bfloat16),
        torch.empty((256, 256), dtype=torch.bfloat16).T,
        torch.empty((2, 512, 4, 64), dtype=torch.bfloat16),
    )
    assert _layout(pure_view_mm, args) == RadixStoreLayout(
        rows=((1024, 256),), cols=((256, 1),)
    )
    source = pure_view_mm._bind_isolated(args).to_code(PAIR_CONFIG)
    assert "'d_hier_layout': ['(1024, 256)', '(256, 1)']" in source


def test_transposed_destination_takes_its_row_major_view():
    # ``out[x, y] = acc[y, x]`` into an (N, M) view with strides (1, N): the
    # proved view is the row-major [M, N] the TMA store needs.
    args = (
        torch.empty((1024, 256), dtype=torch.bfloat16),
        torch.empty((256, 256), dtype=torch.bfloat16).T,
        torch.empty((1024, 256), dtype=torch.bfloat16).T,
    )
    assert _layout(transposed_store_mm, args) == RadixStoreLayout(
        rows=((1024, 256),), cols=((256, 1),)
    )
    source = transposed_store_mm._bind_isolated(args).to_code(PAIR_CONFIG)
    assert "'d_hier_layout': ['(1024, 256)', '(256, 1)']" in source


def test_cta_tiles_must_partition_the_runs():
    # 384-token sequences: a 256-row pair tile neither divides nor spans the
    # 384-row runs (local_tile would partition it silently wrong); 128-row
    # one-CTA tiles divide them.
    args = head_split_args(2, 4, seq=384)
    layout = _layout(seq384_head_split_mm, args)
    assert layout == RadixStoreLayout(
        rows=((384, HEAD), (2, 4 * 384 * HEAD)), cols=((HEAD, 1), (4, 384 * HEAD))
    )
    assert not layout.tiles(256, 256)
    assert layout.tiles(128, 256)
    # A tile spanning whole runs carries the quotient to the next mode.
    assert RadixStoreLayout(rows=((128, 64), (8, 8192)), cols=((64, 1),)).tiles(256, 64)
    bound = seq384_head_split_mm._bind_isolated(args)
    with pytest.raises(BackendUnsupported, match="hierarchical"):
        bound.to_code(PAIR_CONFIG)
    assert (
        "'d_hier_layout': ['((384, 2), (64, 4))', '((64, 98304), (1, 24576))']"
        in bound.to_code(ONE_CTA_CONFIG)
    )


def test_box_crossing_runs_and_fanout_are_refused():
    args = (
        torch.empty((1024, 256), dtype=torch.bfloat16),
        torch.empty((256, 256), dtype=torch.bfloat16).T,
        torch.empty((16, 4, 64, 64), dtype=torch.bfloat16),
    )
    assert _layout(short_head_split_mm, args) == RadixStoreLayout(
        rows=((64, 64), (16, 4 * 64 * 64)), cols=((64, 1), (4, 64 * 64))
    )
    with pytest.raises(BackendUnsupported, match="hierarchical"):
        short_head_split_mm._bind_isolated(args).to_code(PAIR_CONFIG)
    # A second (plain) store of the accumulator keeps the analysis off the
    # hierarchical path: no store is admitted through a layout, and the
    # one-CTA config (which allows two outputs) refuses the 4-D store.
    a, b, out = head_split_args()
    fanout = fanout_head_split_mm._bind_isolated(
        (a, b, torch.empty((a.shape[0], b.shape[1]), dtype=torch.bfloat16), out)
    )
    with pytest.raises(BackendUnsupported, match="fused epilogues"):
        fanout.to_code(ONE_CTA_CONFIG)
    stores = [
        node
        for graph in fanout.host_function.device_ir.graphs
        for node in graph.graph.nodes
        if node.op == "call_function" and node.target is memory_ops.store
    ]
    assert len(stores) == 2
    assert all(proven_radix_store_layout(store) is None for store in stores)
